"""Validated, bounded evaluation schema and secret-free report models."""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import (  # pyright: ignore[reportMissingImports]
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from vault_rag.retrieval import SearchMode

_MAX_CASES = 1_000
_MAX_EXPECTED_PATHS = 20
_MAX_ID_LENGTH = 128
_MAX_PATH_LENGTH = 1_000
_MAX_QUERY_LENGTH = 10_000


class EvaluationKind(StrEnum):
    """Supported retrieval evaluation categories."""

    SEMANTIC = "semantic"
    IDENTIFIER = "identifier"


def _validated_path(value: str) -> str:
    if not value or len(value) > _MAX_PATH_LENGTH or value.startswith("/") or "\\" in value:
        raise ValueError("expected path must be a bounded POSIX-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("expected path must be a bounded POSIX-relative path")
    return value


class ExpectedSource(BaseModel):
    """One expected source identity without source content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    vault_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=63,
        pattern=r"^[a-z0-9][a-z0-9-]{0,62}$",
    )

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        _validated_path(self.path)
        return self


class EvaluationThresholds(BaseModel):
    """Version-1 acceptance thresholds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    recall_at_5: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    identifier_rank_1: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    minimum_cases: int = Field(default=25, ge=1, le=_MAX_CASES)
    p95_latency_ms: float = Field(default=500.0, gt=0.0, allow_inf_nan=False)
    require_no_degradation: Literal[True] = True


class EvaluationCase(BaseModel):
    """One bounded query and its expected profile-relative source paths."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(
        min_length=1,
        max_length=_MAX_ID_LENGTH,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    kind: EvaluationKind
    query: str = Field(min_length=1, max_length=_MAX_QUERY_LENGTH)
    expected_paths: tuple[str, ...] = Field(default=(), max_length=_MAX_EXPECTED_PATHS)
    expected_sources: tuple[ExpectedSource, ...] = Field(default=(), max_length=_MAX_EXPECTED_PATHS)

    @model_validator(mode="after")
    def validate_expected_sources(self) -> Self:
        paths = tuple(self.expected_paths)
        sources = tuple(self.expected_sources)
        paths_supplied = "expected_paths" in self.model_fields_set
        sources_supplied = "expected_sources" in self.model_fields_set
        if paths_supplied and sources_supplied:
            raise ValueError("use expected_sources or expected_paths, not both")
        if not paths_supplied and not sources_supplied:
            raise ValueError("at least one expected source is required")
        if paths_supplied:
            for path in paths:
                _validated_path(path)
            if len(set(paths)) != len(paths):
                raise ValueError("expected paths must not contain duplicates")
            sources = tuple(ExpectedSource(path=path) for path in paths)
        else:
            identities = tuple((source.vault_id, source.path) for source in sources)
            if len(set(identities)) != len(identities):
                raise ValueError("expected sources must not contain duplicates")
            paths = tuple(source.path for source in sources)
        object.__setattr__(self, "expected_paths", paths)
        object.__setattr__(self, "expected_sources", sources)
        return self


class EvaluationConfig(BaseModel):
    """The exact version-1 evaluation TOML document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    thresholds: EvaluationThresholds
    cases: tuple[EvaluationCase, ...] = Field(min_length=1, max_length=_MAX_CASES)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> Self:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for case in self.cases:
            if case.id in seen:
                duplicates.add(case.id)
            seen.add(case.id)
        if duplicates:
            raise ValueError("duplicate evaluation case IDs: " + ", ".join(sorted(duplicates)))
        return self


class EvaluationCitation(BaseModel):
    """One bounded citation identity and its current-source validation result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vault_id: str = Field(min_length=1, max_length=128)
    path: str = Field(min_length=1, max_length=_MAX_PATH_LENGTH)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    source_hash: str = Field(min_length=1, max_length=128)
    citation: str = Field(min_length=1, max_length=2_000)
    valid: bool

    @model_validator(mode="after")
    def validate_lines_and_path(self) -> Self:
        _validated_path(self.path)
        if self.end_line < self.start_line:
            raise ValueError("citation end line must not precede its start line")
        return self


class EvaluationCaseResult(BaseModel):
    """Deterministic, content-free result for one evaluation case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=_MAX_ID_LENGTH)
    kind: EvaluationKind
    expected_paths: tuple[str, ...] = Field(min_length=1, max_length=_MAX_EXPECTED_PATHS)
    expected_sources: tuple[ExpectedSource, ...] = Field(
        min_length=1, max_length=_MAX_EXPECTED_PATHS
    )
    mode: SearchMode
    degraded: bool
    degradation_reason: str | None = Field(default=None, max_length=200)
    rank: int | None = Field(default=None, ge=1, le=5)
    recall_at_5: bool
    reciprocal_rank: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    identifier_rank_1: bool | None
    invalid_citations: int = Field(ge=0, le=5)
    latency_ms: float = Field(ge=0.0, allow_inf_nan=False)
    citations: tuple[EvaluationCitation, ...] = Field(max_length=5)


class EvaluationReport(BaseModel):
    """Complete bounded aggregate report; it never stores queries or source bodies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    profile: str = Field(default="", max_length=200)
    mode: SearchMode
    case_count: int = Field(ge=1, le=_MAX_CASES)
    recall_at_5: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    identifier_rank_1: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    invalid_citations: int = Field(ge=0, le=_MAX_CASES * 5)
    degraded_cases: int = Field(ge=0, le=_MAX_CASES)
    degradation_reasons: tuple[str, ...] = Field(max_length=20)
    p50_latency_ms: float = Field(ge=0.0, allow_inf_nan=False)
    p95_latency_ms: float = Field(ge=0.0, allow_inf_nan=False)
    thresholds: EvaluationThresholds
    passed: bool
    cases: tuple[EvaluationCaseResult, ...] = Field(min_length=1, max_length=_MAX_CASES)
