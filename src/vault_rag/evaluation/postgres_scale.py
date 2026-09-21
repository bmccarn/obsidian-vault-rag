"""Deterministic, content-free PostgreSQL exact-dense scale measurements."""

from __future__ import annotations

import hashlib
import math
import platform
import resource
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np  # pyright: ignore[reportMissingImports]
from pydantic import (  # pyright: ignore[reportMissingImports]
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from vault_rag.domain import LineRange, SourceKind
from vault_rag.embedding.fingerprint import observed_fingerprint
from vault_rag.errors import StorageError
from vault_rag.ingest.chunker import ChunkRecord
from vault_rag.ingest.models import DiscoveredSource
from vault_rag.storage import (
    DenseRequest,
    SourceRevision,
    StorageFilters,
    VaultFingerprint,
    VectorRecord,
)
from vault_rag.storage.postgres import PostgresPool, PostgresStore, PostgresWorkerLease

_MAX_VECTORS = 100_000
_MAX_CONCURRENCY = 25
_MAX_QUERIES = 500
_VECTOR_DIMENSIONS = 16
_REPORT_SCHEMA_VERSION = 1


def _stable_id(namespace: str, *parts: object, length: int = 32) -> str:
    payload = "\0".join((namespace, *(str(part) for part in parts))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a linear-interpolated percentile for an ordered sample."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class ScaleCase(BaseModel):
    """Bounded deterministic scale workload; small values are reserved for tests."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    vectors: int = Field(ge=1, le=_MAX_VECTORS)
    concurrency: int = Field(ge=1, le=_MAX_CONCURRENCY)
    query_count: int = Field(ge=1, le=_MAX_QUERIES)
    seed: int = Field(ge=0, le=2**63 - 1)


class ScaleResultSummary(BaseModel):
    """One content-free exact dense query outcome in submission order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-f0-9]+$")
    target_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-f0-9]+$")
    result_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    scores: tuple[float, ...] = Field(min_length=1, max_length=5)
    latency_ms: float = Field(ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_exact_order(self) -> ScaleResultSummary:
        if len(self.result_ids) != len(self.scores):
            raise ValueError("result identities and scores must have equal lengths")
        if any(not math.isfinite(score) for score in self.scores):
            raise ValueError("result scores must be finite")
        if any(left < right for left, right in zip(self.scores, self.scores[1:], strict=False)):
            raise ValueError("exact result scores must be descending")
        return self


class ScaleReport(BaseModel):
    """Frozen, secret-free report for one bounded exact-dense scale run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=_REPORT_SCHEMA_VERSION, frozen=True)
    case: ScaleCase
    dataset_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-f0-9]+$")
    revision_identity: str = Field(min_length=1, max_length=64, pattern=r"^[a-f0-9]+$")
    results: tuple[ScaleResultSummary, ...] = Field(max_length=_MAX_QUERIES)
    completed_queries: int = Field(ge=0, le=_MAX_QUERIES)
    errors: int = Field(ge=0, le=_MAX_QUERIES)
    p50_latency_ms: float = Field(ge=0.0, allow_inf_nan=False)
    p95_latency_ms: float = Field(ge=0.0, allow_inf_nan=False)
    p99_latency_ms: float = Field(ge=0.0, allow_inf_nan=False)
    throughput_per_second: float = Field(ge=0.0, allow_inf_nan=False)
    cpu_user_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    cpu_system_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    rss_bytes: int = Field(ge=0)
    postgres_operation_latency_ms: float = Field(ge=0.0, allow_inf_nan=False)
    postgres_operation_rows: int = Field(ge=0)
    pool_waits: int = Field(ge=0)
    pool_connections: int = Field(ge=0)
    index_size_bytes: int = Field(ge=0)
    promotion_duration_ms: float = Field(ge=0.0, allow_inf_nan=False)
    no_hnsw: bool

    @model_validator(mode="after")
    def validate_completed_queries(self) -> ScaleReport:
        if self.completed_queries != len(self.results):
            raise ValueError("completed query count must match result summaries")
        if self.completed_queries + self.errors != self.case.query_count:
            raise ValueError("completed queries and errors must account for the schedule")
        return self


def _dataset_identity(case: ScaleCase) -> tuple[str, str, str]:
    dataset_id = _stable_id("vault-rag-postgres-scale-dataset-v1", case.vectors, case.seed)
    return (
        dataset_id,
        _stable_id("vault-rag-postgres-scale-revision-v1", dataset_id),
        f"scale-{dataset_id[:24]}",
    )


def _vectors(case: ScaleCase) -> tuple[np.ndarray, ...]:
    generator = np.random.default_rng(case.seed)
    raw = generator.standard_normal((case.vectors, _VECTOR_DIMENSIONS), dtype=np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return tuple(
        np.asarray(vector / norm, dtype=np.float32) for vector, norm in zip(raw, norms, strict=True)
    )


def _revision_source(
    *,
    vault_id: str,
    dataset_id: str,
    ordinal: int,
    vector: np.ndarray,
    configuration_fingerprint: str,
    observed: str,
) -> SourceRevision:
    source_id = _stable_id("vault-rag-postgres-scale-source-v1", dataset_id, ordinal)
    chunk_id = _stable_id("vault-rag-postgres-scale-chunk-v1", dataset_id, ordinal)
    text = f"scale-source {source_id}"
    payload = text.encode("utf-8")
    path = f"synthetic/{ordinal:06d}.md"
    source = DiscoveredSource(
        vault_id=vault_id,
        root=Path("/scale"),
        relative_path=path,
        folded_path=path,
        kind=SourceKind.MARKDOWN,
        text=text,
        content_hash=f"sha256:{hashlib.sha256(payload).hexdigest()}",
        size_bytes=len(payload),
        mtime_ns=ordinal,
    )
    chunk = ChunkRecord(
        id=chunk_id,
        vault_id=vault_id,
        relative_path=path,
        ordinal=0,
        title="Synthetic scale record",
        heading=("Scale",),
        lines=LineRange(1, 1),
        text=text,
        embedding_text=text,
        token_count=1,
        metadata={},
        content_hash=f"sha256:{hashlib.sha256((source_id + ':chunk').encode()).hexdigest()}",
    )
    return SourceRevision(
        source=source,
        chunks=(chunk,),
        vectors=(
            VectorRecord(
                chunk_id=chunk_id,
                dimensions=_VECTOR_DIMENSIONS,
                config_fingerprint=configuration_fingerprint,
                observed_fingerprint=observed,
                vector=vector,
            ),
        ),
        pending=(),
        manifest_fingerprint=f"scale-manifest-{dataset_id}",
        parser_fingerprint="scale-parser-v1",
        chunker_fingerprint="scale-chunker-v1",
        embedding_config_fingerprint=configuration_fingerprint,
    )


def _seed_and_promote(
    pool: PostgresPool, case: ScaleCase, vectors: tuple[np.ndarray, ...]
) -> tuple[str, str, float]:
    dataset_id, revision_identity, vault_id = _dataset_identity(case)
    configuration_fingerprint = f"scale-config-{dataset_id}"
    observed = observed_fingerprint(configuration_fingerprint, _VECTOR_DIMENSIONS)
    commit_sha = hashlib.sha256(f"scale:{revision_identity}".encode()).hexdigest()[:40]
    store = PostgresStore(pool)
    revision = store.begin_revision(
        vault_id,
        f"scale/{dataset_id}",
        commit_sha,
        manifest={"scale_dataset": dataset_id},
    )
    revision.initialize()
    for ordinal, vector in enumerate(vectors):
        revision.replace_source(
            _revision_source(
                vault_id=vault_id,
                dataset_id=dataset_id,
                ordinal=ordinal,
                vector=vector,
                configuration_fingerprint=configuration_fingerprint,
                observed=observed,
            )
        )
    revision.record_vault_fingerprint(
        VaultFingerprint(
            vault_id=vault_id,
            manifest_fingerprint=f"scale-manifest-{dataset_id}",
            parser_fingerprint="scale-parser-v1",
            chunker_fingerprint="scale-chunker-v1",
            embedding_config_fingerprint=configuration_fingerprint,
        )
    )
    lease = PostgresWorkerLease(pool, f"scale:{vault_id}")
    if not lease.acquire():
        raise RuntimeError("scale workload vault is busy")
    started = perf_counter()
    try:
        lease.promote(revision, lexical_complete=True, fully_reconciled=True)
    finally:
        lease.release()
    return dataset_id, revision_identity, (perf_counter() - started) * 1000


def _index_size_bytes(pool: PostgresPool) -> int:
    with pool.connection() as connection:
        row = connection.execute(
            """
            SELECT COALESCE(sum(pg_relation_size(indexrelid)), 0) AS index_size_bytes
            FROM pg_stat_user_indexes
            WHERE schemaname = 'vault_rag'
            """
        ).fetchone()
    if row is None:
        raise RuntimeError("PostgreSQL index measurement did not return a row")
    return int(row["index_size_bytes"])


def _database_measurements(pool: PostgresPool, vault_id: str) -> tuple[float, int, bool]:
    started = perf_counter()
    try:
        with pool.connection() as connection:
            row = connection.execute(
                """
                SELECT
                    (SELECT count(*) FROM vault_rag.revision_chunks AS rc
                     JOIN vault_rag.vaults AS v ON v.active_revision_id = rc.revision_id
                     WHERE v.vault_id = %s) AS ready_rows,
                    NOT EXISTS (
                        SELECT 1 FROM pg_indexes
                        WHERE schemaname = 'vault_rag'
                          AND indexdef ILIKE '%%USING hnsw%%'
                    ) AS no_hnsw
                """,
                (vault_id,),
            ).fetchone()
    except Exception:
        pool.observe_operation("snapshot", "failure", (perf_counter() - started) * 1000)
        raise
    elapsed_ms = (perf_counter() - started) * 1000
    pool.observe_operation("snapshot", "success", elapsed_ms)
    if row is None:
        raise RuntimeError("PostgreSQL scale measurement did not return a row")
    return elapsed_ms, int(row["ready_rows"]), bool(row["no_hnsw"])


def _rss_bytes() -> int:
    value = max(0, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    return value if platform.system() == "Darwin" else value * 1024


def _queued_requests(stats: Mapping[str, int]) -> int:
    """Read Psycopg's cumulative queued-request counter or reject unsupported pools."""
    if "requests_queued" in stats:
        return stats["requests_queued"]
    if "requests_num" in stats:
        return 0
    raise RuntimeError("PostgreSQL pool does not expose requests_queued statistics")


def _pool_size(stats: Mapping[str, int]) -> int:
    if "pool_size" not in stats:
        raise RuntimeError("PostgreSQL pool does not expose pool_size statistics")
    return stats["pool_size"]


def run_postgres_scale(pool: PostgresPool, case: ScaleCase) -> ScaleReport:
    """Seed and benchmark a deterministic immutable exact-dense PostgreSQL revision.

    Result summaries contain only stable hashes. Percentiles use linear interpolation
    between adjacent sorted samples: ``(n - 1) * p``.
    """
    if not isinstance(case, ScaleCase):
        raise TypeError("case must be a ScaleCase")
    before_usage = resource.getrusage(resource.RUSAGE_SELF)
    index_size_before = _index_size_bytes(pool)
    vectors = _vectors(case)
    dataset_id, revision_identity, vault_id = _dataset_identity(case)
    seeded_dataset, seeded_revision, promotion_duration_ms = _seed_and_promote(pool, case, vectors)
    if (seeded_dataset, seeded_revision) != (dataset_id, revision_identity):
        raise RuntimeError("scale dataset identity changed during seeding")
    configuration_fingerprint = f"scale-config-{dataset_id}"
    observed = observed_fingerprint(configuration_fingerprint, _VECTOR_DIMENSIONS)
    request = DenseRequest(
        vault_ids=(vault_id,),
        observed_fingerprint=observed,
        filters=StorageFilters(),
        limit=5,
    )
    store = PostgresStore(pool)
    pool_before_queries = pool.stats()
    queued_before_queries = _queued_requests(pool_before_queries)

    def execute_query(index: int) -> ScaleResultSummary:
        target = index % case.vectors
        started = perf_counter()
        candidates = store.dense_search(request, vectors[target])
        elapsed_ms = (perf_counter() - started) * 1000
        return ScaleResultSummary(
            query_id=_stable_id("vault-rag-postgres-scale-query-v1", dataset_id, index),
            target_id=_stable_id("vault-rag-postgres-scale-chunk-v1", dataset_id, target),
            result_ids=tuple(candidate.chunk_id for candidate in candidates),
            scores=tuple(candidate.score for candidate in candidates),
            latency_ms=elapsed_ms,
        )

    started = perf_counter()
    outcomes: dict[int, ScaleResultSummary] = {}
    errors = 0
    with ThreadPoolExecutor(
        max_workers=case.concurrency, thread_name_prefix="postgres-scale"
    ) as executor:
        submitted = {
            index: executor.submit(execute_query, index) for index in range(case.query_count)
        }
        for index in range(case.query_count):
            try:
                outcomes[index] = submitted[index].result()
            except (RuntimeError, StorageError, ValueError):
                errors += 1
    pool_after_queries = pool.stats()
    queued_after_queries = _queued_requests(pool_after_queries)
    elapsed_seconds = max(perf_counter() - started, 1e-9)
    results = tuple(outcomes[index] for index in sorted(outcomes))
    index_size_after = _index_size_bytes(pool)
    database_latency_ms, database_rows, no_hnsw = _database_measurements(pool, vault_id)
    after_usage = resource.getrusage(resource.RUSAGE_SELF)
    latencies = tuple(result.latency_ms for result in results)
    return ScaleReport(
        case=case,
        dataset_id=dataset_id,
        revision_identity=revision_identity,
        results=results,
        completed_queries=len(results),
        errors=errors,
        p50_latency_ms=_percentile(latencies, 0.50) if latencies else 0.0,
        p95_latency_ms=_percentile(latencies, 0.95) if latencies else 0.0,
        p99_latency_ms=_percentile(latencies, 0.99) if latencies else 0.0,
        throughput_per_second=len(results) / elapsed_seconds,
        cpu_user_seconds=max(0.0, after_usage.ru_utime - before_usage.ru_utime),
        cpu_system_seconds=max(0.0, after_usage.ru_stime - before_usage.ru_stime),
        rss_bytes=_rss_bytes(),
        postgres_operation_latency_ms=database_latency_ms,
        postgres_operation_rows=database_rows,
        pool_waits=max(0, queued_after_queries - queued_before_queries),
        pool_connections=max(_pool_size(pool_before_queries), _pool_size(pool_after_queries)),
        index_size_bytes=max(0, index_size_after - index_size_before),
        promotion_duration_ms=promotion_duration_ms,
        no_hnsw=no_hnsw,
    )
