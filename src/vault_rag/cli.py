"""Command-line interface for vault-rag."""

from __future__ import annotations

import json
import os
import signal
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, NoReturn, Protocol, cast

import typer  # pyright: ignore[reportMissingImports]
import uvicorn  # pyright: ignore[reportMissingImports]

from . import __version__
from .app import AppFactory
from .domain import JsonValue, LineRange, SourceKind
from .evaluation import (  # pyright: ignore[reportMissingImports]
    EvaluationReport,
    ScaleCase,
    ScaleReport,
    load_evaluation,
    run_evaluation,
    run_postgres_scale,
)
from .indexing import IndexReport
from .presentation import (
    error_exit_code,
    error_payload,
    index_payload,
    read_payload,
    redact_text,
    search_payload,
    serialize_json,
)
from .retrieval import (
    ReadRequest,
    ReadResponse,
    SearchFilters,
    SearchMode,
    SearchRequest,
    SearchResponse,
)
from .service.http import create_app as create_http_app
from .service.observability import configure_structured_event_logging
from .service.runtime import (
    build_api_runtime,
    build_runtime,
    build_worker_runtime,
    check_database,
    cleanup_database,
    migrate_database,
)

_MAX_METADATA_CHARACTERS = 2_000


class FactoryBuilder(Protocol):
    def __call__(self, config_path: Path | None, environ: Mapping[str, str]) -> AppFactory: ...


@dataclass(slots=True)
class _CliState:
    config_path: Path | None = None


def _parse_frontmatter_filters(values: list[str] | None) -> dict[str, JsonValue]:
    """Parse repeatable KEY=JSON options without accepting duplicate keys."""
    parsed: dict[str, JsonValue] = {}
    for item in values or ():
        if len(item) > _MAX_METADATA_CHARACTERS or "=" not in item:
            raise ValueError("frontmatter filters must use bounded KEY=JSON values")
        raw_key, raw_value = item.split("=", 1)
        key = raw_key.strip()
        if not key or key in parsed:
            raise ValueError("frontmatter filter names must be non-empty and unique")
        try:
            value = cast(JsonValue, json.loads(raw_value))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("frontmatter filter values must be valid JSON") from exc
        parsed[key] = value
    return parsed


def _emit_json(value: JsonValue, environ: Mapping[str, str]) -> None:
    del environ
    typer.echo(serialize_json(value))


def _emit_error(error: Exception, json_output: bool, environ: Mapping[str, str]) -> NoReturn:
    payload = error_payload(error, environ)
    exit_code = error_exit_code(error)
    error_value = cast(dict[str, JsonValue], payload["error"])
    if json_output:
        typer.echo(serialize_json(payload))
    else:
        typer.echo(
            f"Error [{error_value['code']}]: {error_value['message']}",
            err=True,
        )
    raise typer.Exit(exit_code)


def _echo_index_text(report: IndexReport, environ: Mapping[str, str]) -> None:
    typer.echo(
        f"Sources: {report.total_sources} "
        f"(added {report.added_sources}, changed {report.changed_sources}, "
        f"unchanged {report.unchanged_sources}, deleted {report.deleted_sources})"
    )
    typer.echo(f"Chunks: {report.ready_chunks} ready, {report.pending_chunks} pending")
    for item in report.diagnostics:
        safe_message = redact_text(item.message, environ)
        typer.echo(f"- {item.vault_id}/{item.path} [{item.category}]: {safe_message}")


def _echo_search_text(response: SearchResponse, environ: Mapping[str, str]) -> None:
    if response.degraded.semantic_search:
        typer.echo(redact_text(f"Semantic search degraded: {response.degraded.reason}", environ))
    if not response.hits:
        typer.echo("No results.")
        return
    for hit in response.hits:
        typer.echo(hit.ref.citation)
        typer.echo(hit.text)
        if hit.continuation is not None:
            typer.echo(
                f"[continued: {hit.continuation.remaining_characters} characters remain; "
                f"next offset {hit.continuation.next_offset}]"
            )


def _echo_read_text(response: ReadResponse, environ: Mapping[str, str]) -> None:
    del environ
    typer.echo(response.ref.citation)
    typer.echo(response.text, nl=not response.text.endswith("\n"))
    if response.continuation is not None:
        typer.echo(
            f"[continued: {response.continuation.remaining_characters} characters remain; "
            f"next offset {response.continuation.next_offset}]"
        )


def _evaluation_payload(report: EvaluationReport) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], report.model_dump(mode="json"))


def _echo_evaluation_text(report: EvaluationReport) -> None:
    typer.echo(f"Evaluation: {'PASS' if report.passed else 'FAIL'}")
    typer.echo(f"Mode: {report.mode.value}")
    typer.echo(f"Cases: {report.case_count}")
    typer.echo(f"Recall@5: {report.recall_at_5:.6f}")
    typer.echo(f"MRR: {report.mean_reciprocal_rank:.6f}")
    typer.echo(f"Identifier rank-1: {report.identifier_rank_1:.6f}")
    typer.echo(f"Invalid citations: {report.invalid_citations}")
    typer.echo(f"Degraded cases: {report.degraded_cases}")
    if report.degradation_reasons:
        typer.echo(f"Degradation reasons: {', '.join(report.degradation_reasons)}")
    typer.echo(f"Latency ms: p50 {report.p50_latency_ms:.3f}, p95 {report.p95_latency_ms:.3f}")
    visible = report.cases[:100]
    for result in visible:
        rank = "miss" if result.rank is None else str(result.rank)
        typer.echo(
            f"- {result.id} [{result.kind.value}/{result.mode.value}]: rank={rank}, "
            f"latency_ms={result.latency_ms:.3f}, invalid={result.invalid_citations}, "
            f"degraded={result.degraded}"
        )
        for citation in result.citations:
            typer.echo(f"  {citation.citation} [{'valid' if citation.valid else 'invalid'}]")
    if len(report.cases) > len(visible):
        typer.echo(f"[{len(report.cases) - len(visible)} additional cases omitted from text]")


def _scale_payload(report: ScaleReport) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], report.model_dump(mode="json"))


def _echo_scale_text(report: ScaleReport) -> None:
    typer.echo(
        f"PostgreSQL exact scale: {'PASS' if report.errors == 0 and report.no_hnsw else 'FAIL'}"
    )
    typer.echo(
        f"Vectors: {report.case.vectors}; concurrency: {report.case.concurrency}; "
        f"queries: {report.completed_queries}; errors: {report.errors}"
    )
    typer.echo(
        f"Latency ms: p50 {report.p50_latency_ms:.3f}, p95 {report.p95_latency_ms:.3f}, "
        f"p99 {report.p99_latency_ms:.3f}; throughput/s: {report.throughput_per_second:.3f}"
    )


def _echo_status_text(payload: dict[str, JsonValue], environ: Mapping[str, str]) -> None:
    def echo_safe(value: str) -> None:
        typer.echo(value)

    echo_safe(f"Profile: {payload['profile']}")
    echo_safe(f"Vaults: {', '.join(cast(list[str], payload['vault_ids']))}")
    echo_safe(f"Policy: {payload['effective_egress_policy']}")
    echo_safe(f"Model: {payload['model']} ({payload['endpoint_class']})")
    echo_safe(f"Database: {payload['database_path']}")
    schema = cast(dict[str, JsonValue], payload["schema"])
    echo_safe(
        f"Schema: {schema['actual']} (expected {schema['expected']}, "
        f"compatible {schema['compatible']})"
    )
    fingerprints = cast(dict[str, JsonValue], payload["fingerprints"])
    echo_safe(f"Fingerprint state: {fingerprints['state']}")
    counts = cast(dict[str, JsonValue], payload["counts"])
    echo_safe(
        f"Counts: {counts['sources']} sources, {counts['chunks']} chunks, "
        f"{counts['ready']} ready, {counts['pending']} pending"
    )
    echo_safe(f"Last successful reconciliation: {payload['last_successful_reconciliation']}")
    echo_safe(f"Index age seconds: {payload['index_age_seconds']}")
    semantic = cast(dict[str, JsonValue], payload["semantic_degradation"])
    echo_safe(
        f"Semantic degradation: {semantic['semantic_search']}"
        + (f" ({semantic['reason']})" if semantic["reason"] is not None else "")
    )


def _echo_doctor_text(payload: dict[str, JsonValue], environ: Mapping[str, str]) -> None:
    typer.echo(f"Doctor: {'healthy' if payload['healthy'] else 'degraded'}")
    for raw in cast(list[JsonValue], payload["checks"]):
        item = cast(dict[str, JsonValue], raw)
        name = str(item["name"])
        label = "FTS5" if name == "fts5" else name.replace("_", " ").title()
        typer.echo(redact_text(f"- {label}: {item['status']} — {item['message']}", environ))


def create_app(factory_builder: FactoryBuilder = AppFactory.from_environment) -> typer.Typer:
    """Create an isolated CLI application with injectable application wiring."""
    root = typer.Typer(no_args_is_help=True)
    profile_app = typer.Typer(help="Inspect configured profiles.")
    database_app = typer.Typer(help="Manage PostgreSQL schema and bounded cleanup.")
    root.add_typer(database_app, name="db")
    root.add_typer(profile_app, name="profile")

    @root.callback()
    def main(
        context: typer.Context,
        config_path: Annotated[Path | None, typer.Option("--config")] = None,
    ) -> None:
        """Vault-scoped hybrid retrieval for Obsidian and agent clients."""
        context.obj = _CliState(config_path)

    def factory(context: typer.Context) -> AppFactory:
        state = cast(_CliState, context.obj)
        return factory_builder(state.config_path, os.environ)

    @root.command()
    def version(
        json_output: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Print the package version."""
        if json_output:
            typer.echo(serialize_json({"version": __version__}))
        else:
            typer.echo(__version__)

    @root.command()
    def register(
        context: typer.Context,
        vault: Annotated[Path, typer.Argument()],
        replace: Annotated[bool, typer.Option("--replace")] = False,
        json_output: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Atomically register one manifest-owned vault."""
        environ = os.environ
        try:
            with factory(context) as services:
                payload = services.register(vault, replace=replace)
        except Exception as exc:
            _emit_error(exc, json_output, environ)
        if json_output:
            _emit_json(payload, environ)
        else:
            typer.echo(f"Registered {payload['vault_id']}: {payload['path']}")

    @profile_app.command("list")
    def profile_list(
        context: typer.Context,
        json_output: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """List profile names and explicit vault allowlists."""
        environ = os.environ
        try:
            with factory(context) as services:
                payload = services.profile_list()
        except Exception as exc:
            _emit_error(exc, json_output, environ)
        if json_output:
            _emit_json(payload, environ)
        else:
            profiles = cast(list[JsonValue], payload["profiles"])
            if not profiles:
                typer.echo("No profiles configured.")
            for raw in profiles:
                item = cast(dict[str, JsonValue], raw)
                rendered = f"{item['name']}: {', '.join(cast(list[str], item['vaults']))}"
                if item["vaults_truncated"]:
                    rendered += f" [{item['vaults_truncated']} additional vaults omitted]"
                typer.echo(rendered)
            if payload["profiles_truncated"]:
                typer.echo(f"[{payload['profiles_truncated']} additional profiles omitted]")

    @root.command()
    def index(
        context: typer.Context,
        profile: Annotated[str, typer.Option("--profile")] = "",
        rebuild: Annotated[bool, typer.Option("--rebuild")] = False,
        json_output: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Reconcile one explicit profile with its persistent index."""
        environ = os.environ
        try:
            with factory(context) as services:
                resolved = services.resolve_profile(profile)
                report = services.indexer(resolved).run(rebuild=rebuild)
        except Exception as exc:
            _emit_error(exc, json_output, environ)
        payload = index_payload(report, environ)
        if json_output:
            _emit_json(payload, environ)
        else:
            _echo_index_text(report, environ)
        if report.pending_chunks:
            raise typer.Exit(4)

    @root.command()
    def search(
        context: typer.Context,
        query: Annotated[str, typer.Argument()],
        profile: Annotated[str, typer.Option("--profile")] = "",
        limit: Annotated[int, typer.Option("--limit", min=1, max=100)] = 10,
        mode: Annotated[SearchMode, typer.Option("--mode", case_sensitive=False)] = (
            SearchMode.HYBRID
        ),
        vault_ids: Annotated[list[str] | None, typer.Option("--vault")] = None,
        path_prefix: Annotated[str | None, typer.Option("--path-prefix")] = None,
        source_kind: Annotated[SourceKind | None, typer.Option("--source-kind")] = None,
        frontmatter: Annotated[list[str] | None, typer.Option("--frontmatter")] = None,
        json_output: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Search one explicit profile using the selected retrieval mode."""
        environ = os.environ
        try:
            filters = SearchFilters(
                vault_ids=tuple(vault_ids or ()),
                path_prefix=path_prefix,
                source_kind=source_kind,
                frontmatter=_parse_frontmatter_filters(frontmatter),
            )
            with factory(context) as services:
                resolved = services.resolve_profile(profile)
                response = services.retrieval(resolved).search(
                    SearchRequest(query=query, filters=filters, limit=limit, mode=mode)
                )
        except Exception as exc:
            _emit_error(exc, json_output, environ)
        if json_output:
            _emit_json(search_payload(resolved.name, response), environ)
        else:
            _echo_search_text(response, environ)

    @root.command()
    def read(
        context: typer.Context,
        path: Annotated[str, typer.Argument()],
        profile: Annotated[str, typer.Option("--profile")] = "",
        vault_id: Annotated[str | None, typer.Option("--vault")] = None,
        heading: Annotated[str | None, typer.Option("--heading")] = None,
        start_line: Annotated[int | None, typer.Option("--start-line", min=1)] = None,
        end_line: Annotated[int | None, typer.Option("--end-line", min=1)] = None,
        expected_source_hash: Annotated[str | None, typer.Option("--expected-source-hash")] = None,
        json_output: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Hash-verify and read a bounded source range from one profile."""
        environ = os.environ
        try:
            if (start_line is None) != (end_line is None):
                raise ValueError("start-line and end-line must be supplied together")
            lines = (
                None if start_line is None or end_line is None else LineRange(start_line, end_line)
            )
            with factory(context) as services:
                resolved = services.resolve_profile(profile)
                response = services.retrieval(resolved).read(
                    ReadRequest(
                        path=path,
                        vault_id=vault_id,
                        heading=heading,
                        lines=lines,
                        expected_source_hash=expected_source_hash,
                    )
                )
        except Exception as exc:
            _emit_error(exc, json_output, environ)
        if json_output:
            _emit_json(read_payload(resolved.name, response), environ)
        else:
            _echo_read_text(response, environ)

    @root.command()
    def evaluate(
        context: typer.Context,
        evaluation_file: Annotated[Path, typer.Option("--file")],
        profile: Annotated[str, typer.Option("--profile")] = "",
        mode: Annotated[SearchMode, typer.Option("--mode", case_sensitive=False)] = (
            SearchMode.HYBRID
        ),
        json_output: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Run a bounded retrieval evaluation for one explicit profile."""
        environ = os.environ
        try:
            evaluation = load_evaluation(evaluation_file)
            with factory(context) as services:
                resolved = services.resolve_profile(profile)
                report = run_evaluation(
                    services.retrieval(resolved),
                    profile=resolved.name,
                    cases=evaluation.cases,
                    thresholds=evaluation.thresholds,
                    mode=mode,
                )
        except Exception as exc:
            _emit_error(exc, json_output, environ)
        if json_output:
            _emit_json(_evaluation_payload(report), environ)
        else:
            _echo_evaluation_text(report)
        if not report.passed:
            raise typer.Exit(5)

    @root.command("postgres-scale")
    def postgres_scale(
        vectors: Annotated[Literal[10_000, 50_000, 100_000], typer.Option("--vectors")],
        concurrency: Annotated[Literal[1, 5, 10, 25], typer.Option("--concurrency")],
        query_count: Annotated[int, typer.Option("--queries", min=1, max=500)] = 100,
        seed: Annotated[int, typer.Option("--seed", min=0)] = 0,
        service_config: Annotated[Path, typer.Option("--service-config")] = Path(
            "/config/service.yaml"
        ),
        json_output: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Run one bounded exact-dense PostgreSQL scale workload."""
        from .service.config import StorageBackend, load_service_config, materialize_secret_files
        from .storage.postgres import PostgresMigrator, PostgresPool

        environ = os.environ
        try:
            config = load_service_config(service_config)
            if config.storage.backend is not StorageBackend.POSTGRESQL:
                raise ValueError("PostgreSQL scale requires PostgreSQL storage")
            resolved_environ = materialize_secret_files(config, environ)
            database_url_env = config.storage.database_url_env
            dsn = None if database_url_env is None else resolved_environ.get(database_url_env)
            if not dsn:
                raise ValueError("PostgreSQL scale DSN is unavailable")
            case = ScaleCase(
                vectors=vectors,
                concurrency=concurrency,
                query_count=query_count,
                seed=seed,
            )
            pool = PostgresPool(dsn, min_size=1, max_size=concurrency, timeout=5.0)
            pool.open(wait=True)
            try:
                PostgresMigrator(pool).check()
                report = run_postgres_scale(pool, case)
            finally:
                pool.close()
        except Exception as exc:
            _emit_error(exc, json_output, environ)
        if json_output:
            _emit_json(_scale_payload(report), environ)
        else:
            _echo_scale_text(report)
        if report.errors or not report.no_hnsw:
            raise typer.Exit(5)

    @root.command()
    def status(
        context: typer.Context,
        profile: Annotated[str, typer.Option("--profile")] = "",
        json_output: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Show index, model, policy, and reconciliation state for one profile."""
        environ = os.environ
        try:
            with factory(context) as services:
                resolved = services.resolve_profile(profile)
                payload = services.status(resolved)
        except Exception as exc:
            _emit_error(exc, json_output, environ)
        if json_output:
            _emit_json(payload, environ)
        else:
            _echo_status_text(payload, environ)

    @root.command()
    def doctor(
        context: typer.Context,
        profile: Annotated[str, typer.Option("--profile")] = "",
        json_output: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Check local security, index features, and policy-permitted semantics."""
        environ = os.environ
        try:
            with factory(context) as services:
                doctor_operation = cast(Callable[[str], dict[str, JsonValue]], services.doctor)
                payload = doctor_operation(profile)
        except Exception as exc:
            _emit_error(exc, json_output, environ)
        if json_output:
            _emit_json(payload, environ)
        else:
            _echo_doctor_text(payload, environ)

    @root.command()
    def serve(
        service_config: Annotated[Path, typer.Option("--service-config")] = Path(
            "/config/service.yaml"
        ),
        data_root: Annotated[Path, typer.Option("--data-root")] = Path("/data"),
        host: Annotated[str, typer.Option("--host")] = "0.0.0.0",
        port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8080,
    ) -> None:
        """Serve the shared Git-synced retrieval HTTP API."""
        configure_structured_event_logging("vault_rag.service")
        runtime = build_runtime(service_config, data_root, os.environ)
        uvicorn.run(
            create_http_app(runtime),
            host=host,
            port=port,
            workers=1,
            access_log=False,
        )

    @root.command()
    def api(
        service_config: Annotated[Path, typer.Option("--service-config")] = Path(
            "/config/service.yaml"
        ),
        data_root: Annotated[Path, typer.Option("--data-root")] = Path("/data"),
        host: Annotated[str, typer.Option("--host")] = "0.0.0.0",
        port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8080,
    ) -> None:
        """Serve the PostgreSQL retrieval API without running synchronization."""
        configure_structured_event_logging("vault_rag.service")
        runtime = build_api_runtime(service_config, data_root, os.environ)
        uvicorn.run(
            create_http_app(runtime),
            host=host,
            port=port,
            workers=1,
            access_log=False,
        )

    @root.command()
    def worker(
        service_config: Annotated[Path, typer.Option("--service-config")] = Path(
            "/config/service.yaml"
        ),
        data_root: Annotated[Path, typer.Option("--data-root")] = Path("/data"),
    ) -> None:
        """Run the session-lock-coordinated PostgreSQL synchronization worker."""
        configure_structured_event_logging("vault_rag.worker")
        runtime = build_worker_runtime(service_config, data_root, os.environ)

        def request_stop(_signum: int, _frame: object) -> None:
            runtime.request_stop()

        previous_interrupt = signal.signal(signal.SIGINT, request_stop)
        previous_termination = signal.signal(signal.SIGTERM, request_stop)
        try:
            runtime.run_forever()
        finally:
            signal.signal(signal.SIGTERM, previous_termination)
            signal.signal(signal.SIGINT, previous_interrupt)
            runtime.close()

    @database_app.command("migrate")
    def db_migrate(
        service_config: Annotated[Path, typer.Option("--service-config")] = Path(
            "/config/service.yaml"
        ),
        json_output: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Apply application-owned PostgreSQL schema migrations."""
        environ = os.environ
        try:
            applied = migrate_database(service_config, environ)
        except Exception as exc:
            _emit_error(exc, json_output, environ)
        if json_output:
            _emit_json({"applied": list(applied)}, environ)
        else:
            rendered = ", ".join(str(version) for version in applied) or "none"
            typer.echo(f"Applied PostgreSQL migrations: {rendered}")

    @database_app.command("check")
    def db_check(
        service_config: Annotated[Path, typer.Option("--service-config")] = Path(
            "/config/service.yaml"
        ),
        json_output: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Verify PostgreSQL schema compatibility without changing it."""
        environ = os.environ
        try:
            applied = check_database(service_config, environ)
        except Exception as exc:
            _emit_error(exc, json_output, environ)
        if json_output:
            _emit_json({"applied": list(applied)}, environ)
        else:
            rendered = ", ".join(str(version) for version in applied)
            typer.echo(f"Compatible PostgreSQL migrations: {rendered}")

    @database_app.command("cleanup")
    def db_cleanup(
        vault_id: Annotated[str, typer.Argument()],
        service_config: Annotated[Path, typer.Option("--service-config")] = Path(
            "/config/service.yaml"
        ),
        json_output: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Explicitly run one bounded PostgreSQL revision cleanup pass."""
        environ = os.environ
        try:
            result = cleanup_database(service_config, environ, vault_id)
        except Exception as exc:
            _emit_error(exc, json_output, environ)
        payload: dict[str, JsonValue] = {
            "revisions_deleted": result.revisions_deleted,
            "blobs_deleted": result.blobs_deleted,
            "embeddings_deleted": result.embeddings_deleted,
            "elapsed_ms": result.elapsed_ms,
        }
        if json_output:
            _emit_json(payload, environ)
        else:
            typer.echo(
                "Deleted PostgreSQL data: "
                f"{result.revisions_deleted} revisions, {result.blobs_deleted} blobs, "
                f"{result.embeddings_deleted} embeddings in {result.elapsed_ms}ms"
            )

    return root


app = create_app()
