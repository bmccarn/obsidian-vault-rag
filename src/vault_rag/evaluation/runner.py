"""Profile-bound retrieval evaluation with current-source citation validation."""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError  # pyright: ignore[reportMissingImports]

from vault_rag.domain import LineRange
from vault_rag.errors import ConfigError
from vault_rag.retrieval import (
    ReadRequest,
    ReadResponse,
    SearchMode,
    SearchRequest,
    SearchResponse,
)

from .models import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationCitation,
    EvaluationConfig,
    EvaluationKind,
    EvaluationReport,
    EvaluationThresholds,
    ExpectedSource,
)

_MAX_VALIDATION_ERRORS = 20
_MAX_VALIDATION_MESSAGE = 200


class EvaluationService(Protocol):
    """Retrieval surface required by the public evaluation runner."""

    @property
    def vault_ids(self) -> tuple[str, ...]: ...

    def search(self, request: SearchRequest) -> SearchResponse: ...

    def read(self, request: ReadRequest) -> ReadResponse: ...


def _config_error(error: ValidationError) -> ConfigError:
    errors = error.errors(include_input=False, include_url=False)
    rendered: list[str] = []
    for item in errors[:_MAX_VALIDATION_ERRORS]:
        location = ".".join(str(part) for part in item.get("loc", ())) or "evaluation"
        message = " ".join(str(item.get("msg", "is invalid")).split())
        rendered.append(f"{location}: {message[:_MAX_VALIDATION_MESSAGE]}")
    if len(errors) > len(rendered):
        rendered.append(f"{len(errors) - len(rendered)} additional validation errors omitted")
    return ConfigError("invalid evaluation configuration: " + "; ".join(rendered))


def load_evaluation(path: Path) -> EvaluationConfig:
    """Load the exact version-1 TOML schema using the existing config error contract."""
    try:
        with path.open("rb") as evaluation_file:
            raw: Any = tomllib.load(evaluation_file)
    except FileNotFoundError as exc:
        raise ConfigError(f"evaluation file does not exist: {path}") from exc
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"could not load evaluation file: {path}") from exc
    try:
        config = EvaluationConfig.model_validate(raw)
    except ValidationError as exc:
        raise _config_error(exc) from exc
    if config.thresholds.minimum_cases < 25:
        raise ConfigError("evaluation minimum_cases must be at least 25")
    if config.thresholds.p95_latency_ms > 500.0:
        raise ConfigError("evaluation p95_latency_ms must not exceed 500")
    return config


def _validate_direct_cases(cases: Sequence[EvaluationCase]) -> tuple[EvaluationCase, ...]:
    copied = tuple(cases)
    if not copied:
        raise ConfigError("evaluation must contain at least one case")
    try:
        validated = EvaluationConfig(
            schema_version=1,
            thresholds=EvaluationThresholds(),
            cases=copied,
        )
    except ValidationError as exc:
        raise _config_error(exc) from exc
    return validated.cases


def _source_is_current(service: EvaluationService, source: ExpectedSource) -> bool:
    if source.vault_id is None:
        raise ValueError("resolved expected source must include vault_id")
    try:
        response = service.read(
            ReadRequest(path=source.path, vault_id=source.vault_id, lines=LineRange(1, 1))
        )
    except Exception:  # all read failures mean this exact profile source is not current
        return False
    return (
        response.ref.vault_id == source.vault_id
        and response.ref.path == source.path
        and response.ref.lines.start == 1
        and response.ref.lines.end == 1
        and bool(response.ref.source_hash)
    )


def _resolved_expected_sources(
    service: EvaluationService, case: EvaluationCase
) -> tuple[ExpectedSource, ...]:
    resolved: list[ExpectedSource] = []
    for expected in case.expected_sources:
        vault_id = expected.vault_id
        if vault_id is None:
            if len(service.vault_ids) != 1:
                raise ConfigError(
                    "vault_id is required for expected sources in a multi-vault profile",
                    details={"case_id": case.id, "path": expected.path},
                )
            vault_id = service.vault_ids[0]
        elif vault_id not in service.vault_ids:
            raise ConfigError(
                "expected source vault is outside the bound profile",
                details={"case_id": case.id, "vault_id": vault_id},
            )
        source = ExpectedSource(vault_id=vault_id, path=expected.path)
        if not _source_is_current(service, source):
            raise ConfigError(
                "expected source is not current in profile",
                details={"case_id": case.id, "vault_id": vault_id, "path": expected.path},
            )
        resolved.append(source)
    return tuple(resolved)


def _validated_citation(
    service: EvaluationService,
    *,
    response: SearchResponse,
    index: int,
) -> EvaluationCitation:
    hit = response.hits[index]
    ref = hit.ref
    valid = ref.lines.start >= 1 and ref.lines.end >= ref.lines.start
    if valid:
        try:
            current = service.read(
                ReadRequest(
                    path=ref.path,
                    vault_id=ref.vault_id,
                    lines=ref.lines,
                )
            )
        except Exception:  # stale, missing, out-of-profile, and bad lines are invalid citations
            valid = False
        else:
            valid = (
                current.ref.vault_id == ref.vault_id
                and current.ref.path == ref.path
                and current.ref.lines == ref.lines
                and current.ref.source_hash == ref.source_hash
            )
    return EvaluationCitation(
        vault_id=ref.vault_id,
        path=ref.path,
        start_line=ref.lines.start,
        end_line=ref.lines.end,
        source_hash=ref.source_hash,
        citation=ref.citation,
        valid=valid,
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def run_evaluation(
    service: EvaluationService,
    *,
    cases: Sequence[EvaluationCase],
    thresholds: EvaluationThresholds | None = None,
    profile: str = "",
    mode: SearchMode = SearchMode.HYBRID,
) -> EvaluationReport:
    """Evaluate top-five retrieval without retaining query or source content.

    Every expected source is hash/current-source validated before the first query.
    Every returned citation is then re-read through the bound retrieval service so
    stale hashes, foreign vault IDs, missing sources, and invalid line ranges count
    as invalid rather than contributing to recall or rank.
    """
    validated_cases = _validate_direct_cases(cases)
    selected_thresholds = thresholds or EvaluationThresholds()
    selected_mode = SearchMode(mode)
    resolved_by_case = tuple(_resolved_expected_sources(service, case) for case in validated_cases)

    results: list[EvaluationCaseResult] = []
    for case, expected_sources in zip(validated_cases, resolved_by_case, strict=True):
        response = service.search(SearchRequest(query=case.query, limit=5, mode=selected_mode))
        citations = tuple(
            _validated_citation(service, response=response, index=index)
            for index in range(len(response.hits))
        )
        expected_identities = {(source.vault_id, source.path) for source in expected_sources}
        recalled = any(
            citation.valid and (citation.vault_id, citation.path) in expected_identities
            for citation in citations
        )
        primary = expected_sources[0]
        rank = next(
            (
                position
                for position, citation in enumerate(citations, start=1)
                if citation.valid
                and (citation.vault_id, citation.path) == (primary.vault_id, primary.path)
            ),
            None,
        )
        reciprocal_rank = 0.0 if rank is None else 1.0 / rank
        identifier_rank_1 = rank == 1 if case.kind is EvaluationKind.IDENTIFIER else None
        invalid_citations = sum(not citation.valid for citation in citations)
        results.append(
            EvaluationCaseResult(
                id=case.id,
                kind=case.kind,
                expected_paths=tuple(source.path for source in expected_sources),
                expected_sources=expected_sources,
                mode=selected_mode,
                degraded=response.degraded.semantic_search,
                degradation_reason=(
                    None if response.degraded.reason is None else response.degraded.reason[:200]
                ),
                rank=rank,
                recall_at_5=recalled,
                reciprocal_rank=reciprocal_rank,
                identifier_rank_1=identifier_rank_1,
                invalid_citations=invalid_citations,
                latency_ms=response.elapsed_ms,
                citations=citations,
            )
        )

    case_count = len(results)
    identifier_results = tuple(
        result for result in results if result.kind is EvaluationKind.IDENTIFIER
    )
    recall_at_5 = sum(result.recall_at_5 for result in results) / case_count
    mean_reciprocal_rank = sum(result.reciprocal_rank for result in results) / case_count
    aggregate_identifier_rank_1 = (
        sum(result.identifier_rank_1 is True for result in identifier_results)
        / len(identifier_results)
        if identifier_results
        else 0.0
    )
    invalid_citations = sum(result.invalid_citations for result in results)
    degraded_cases = sum(result.degraded for result in results)
    degradation_reasons = tuple(
        sorted(
            {
                result.degradation_reason
                for result in results
                if result.degradation_reason is not None
            }
        )
    )
    latencies = tuple(result.latency_ms for result in results)
    p50_latency_ms = _percentile(latencies, 0.50)
    p95_latency_ms = _percentile(latencies, 0.95)
    passed = (
        case_count >= selected_thresholds.minimum_cases
        and recall_at_5 >= selected_thresholds.recall_at_5
        and aggregate_identifier_rank_1 >= selected_thresholds.identifier_rank_1
        and invalid_citations == 0
        and p95_latency_ms <= selected_thresholds.p95_latency_ms
        and (not selected_thresholds.require_no_degradation or degraded_cases == 0)
    )
    return EvaluationReport(
        profile=profile,
        mode=selected_mode,
        case_count=case_count,
        recall_at_5=recall_at_5,
        mean_reciprocal_rank=mean_reciprocal_rank,
        identifier_rank_1=aggregate_identifier_rank_1,
        invalid_citations=invalid_citations,
        degraded_cases=degraded_cases,
        degradation_reasons=degradation_reasons,
        p50_latency_ms=p50_latency_ms,
        p95_latency_ms=p95_latency_ms,
        thresholds=selected_thresholds,
        passed=passed,
        cases=tuple(results),
    )
