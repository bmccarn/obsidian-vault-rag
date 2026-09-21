from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest  # pyright: ignore[reportMissingImports]
from pydantic import ValidationError  # pyright: ignore[reportMissingImports]

from vault_rag.domain import DegradedState, LineRange, SourceRef
from vault_rag.errors import ConfigError
from vault_rag.evaluation import (
    EvaluationCase,
    EvaluationKind,
    EvaluationThresholds,
    ExpectedSource,
    load_evaluation,
    run_evaluation,
)
from vault_rag.retrieval import (
    ReadRequest,
    ReadResponse,
    ScoreComponents,
    SearchHit,
    SearchMode,
    SearchRequest,
    SearchResponse,
)


def hit(
    path: str,
    *,
    vault_id: str = "vault-a",
    start: int = 1,
    end: int = 3,
    text: str = "private body",
) -> SearchHit:
    return SearchHit(
        chunk_id=f"chunk:{vault_id}:{path}",
        ref=SourceRef(vault_id, path, (), LineRange(start, end), "sha256:current"),
        text=text,
        metadata=MappingProxyType({"secret": "must-not-appear"}),
        scores=ScoreComponents(None, None, None, None, 0.0, False),
        continuation=None,
    )


class FakeService:
    def __init__(
        self,
        responses: dict[str, tuple[list[SearchHit], float]],
        *,
        current_paths: set[str] | None = None,
        invalid_paths: set[str] | None = None,
        vault_ids: tuple[str, ...] = ("vault-a",),
        current_sources: set[tuple[str, str]] | None = None,
        degraded_queries: set[str] | None = None,
    ) -> None:
        self.responses = responses
        self._vault_ids = vault_ids
        self.current_paths = current_paths or {
            item.ref.path for hits, _latency in responses.values() for item in hits
        }
        self.current_sources = current_sources or {
            (item.ref.vault_id, item.ref.path)
            for hits, _latency in responses.values()
            for item in hits
        }
        self.invalid_paths = invalid_paths or set()
        self.degraded_queries = degraded_queries or set()
        self.search_requests: list[SearchRequest] = []
        self.read_requests: list[ReadRequest] = []

    @property
    def vault_ids(self) -> tuple[str, ...]:
        return self._vault_ids

    def search(self, request: SearchRequest) -> SearchResponse:
        self.search_requests.append(request)
        hits, latency = self.responses.get(request.query, ([], 0.0))
        degraded = (
            DegradedState(True, "query_embedding_failed")
            if request.query in self.degraded_queries
            else DegradedState()
        )
        return SearchResponse(tuple(hits), degraded, latency, 0.0)

    def read(self, request: ReadRequest) -> ReadResponse:
        self.read_requests.append(request)
        vault_id = request.vault_id or "vault-a"
        if (
            request.path not in self.current_paths
            or (vault_id, request.path) not in self.current_sources
            or request.path in self.invalid_paths
        ):
            raise ValueError("source is not current")
        lines = request.lines or LineRange(1, 1)
        return ReadResponse(
            SourceRef(vault_id, request.path, (), lines, "sha256:current"),
            "body that must be discarded",
            None,
        )


def evaluation_case(case_id: str, kind: str, query: str, *expected_paths: str) -> EvaluationCase:
    return EvaluationCase(
        id=case_id,
        kind=kind,
        query=query,
        expected_paths=tuple(expected_paths),
    )


def test_evaluation_computes_recall_mrr_identifier_rank_and_percentiles() -> None:
    service = FakeService(
        {
            "semantic question": ([hit("notes/wrong.md"), hit("notes/right.md")], 10.0),
            "dzte-infra #298": ([hit("prs/dzte-infra-298.md")], 30.0),
        },
        current_paths={"notes/right.md", "notes/wrong.md", "prs/dzte-infra-298.md"},
    )

    report = run_evaluation(
        service,
        cases=(
            evaluation_case("semantic", "semantic", "semantic question", "notes/right.md"),
            evaluation_case("identifier", "identifier", "dzte-infra #298", "prs/dzte-infra-298.md"),
        ),
        thresholds=EvaluationThresholds(
            recall_at_5=0.9,
            identifier_rank_1=1.0,
            minimum_cases=2,
            p95_latency_ms=500.0,
        ),
    )

    assert report.case_count == 2
    assert report.recall_at_5 == 1.0
    assert report.mean_reciprocal_rank == 0.75
    assert report.identifier_rank_1 == 1.0
    assert report.invalid_citations == 0
    assert report.p50_latency_ms == 20.0
    assert report.p95_latency_ms == 29.0
    assert [case.rank for case in report.cases] == [2, 1]
    assert all(request.limit == 5 for request in service.search_requests)
    assert report.passed is True


def test_report_is_frozen_bounded_and_never_contains_source_bodies_or_metadata() -> None:
    service = FakeService({"query": ([hit("notes/right.md", text="TOP SECRET BODY")], 1.0)})
    report = run_evaluation(
        service,
        cases=(evaluation_case("case", "semantic", "query", "notes/right.md"),),
    )

    with pytest.raises(ValidationError):
        report.case_count = 10  # type: ignore[misc]
    rendered = repr(report)
    assert "TOP SECRET BODY" not in rendered
    assert "must-not-appear" not in rendered
    assert report.cases[0].citations[0].path == "notes/right.md"
    assert report.cases[0].citations[0].start_line == 1


def test_invalid_citation_does_not_compress_a_later_expected_source_rank() -> None:
    service = FakeService(
        {"query": ([hit("notes/stale.md"), hit("notes/right.md")], 4.0)},
        current_paths={"notes/stale.md", "notes/right.md"},
        invalid_paths={"notes/stale.md"},
    )

    report = run_evaluation(
        service,
        cases=(evaluation_case("case", "semantic", "query", "notes/right.md"),),
    )

    assert report.cases[0].rank == 2
    assert report.mean_reciprocal_rank == 0.5


def test_invalid_returned_citations_are_counted_without_aborting_complete_report() -> None:
    service = FakeService(
        {"query": ([hit("notes/right.md"), hit("notes/stale.md", start=8, end=10)], 4.0)},
        current_paths={"notes/right.md", "notes/stale.md"},
        invalid_paths={"notes/stale.md"},
    )

    report = run_evaluation(
        service,
        cases=(evaluation_case("case", "semantic", "query", "notes/right.md"),),
    )

    assert report.case_count == 1
    assert report.invalid_citations == 1
    assert report.cases[0].invalid_citations == 1
    assert report.cases[0].citations[1].valid is False


def test_expected_source_must_be_current_and_inside_bound_profile() -> None:
    service = FakeService({}, current_paths=set())

    with pytest.raises(ConfigError, match="expected source is not current in profile"):
        run_evaluation(
            service,
            cases=(evaluation_case("case", "semantic", "query", "notes/missing.md"),),
        )


def test_duplicate_case_ids_are_rejected_even_for_direct_model_construction() -> None:
    service = FakeService({"one": ([], 1.0), "two": ([], 2.0)})

    with pytest.raises(ConfigError, match="duplicate evaluation case ID"):
        run_evaluation(
            service,
            cases=(
                evaluation_case("same", "semantic", "one", "one.md"),
                evaluation_case("same", "semantic", "two", "two.md"),
            ),
        )


def test_multi_vault_expected_sources_require_and_match_vault_identity() -> None:
    service = FakeService(
        {"query": ([hit("same.md", vault_id="vault-b")], 1.0)},
        vault_ids=("vault-a", "vault-b"),
        current_paths={"same.md"},
        current_sources={("vault-a", "same.md"), ("vault-b", "same.md")},
    )
    ambiguous = EvaluationCase(
        id="ambiguous",
        kind="semantic",
        query="query",
        expected_paths=("same.md",),
    )

    with pytest.raises(ConfigError, match="vault_id is required"):
        run_evaluation(
            service,
            cases=(ambiguous,),
            thresholds=EvaluationThresholds(minimum_cases=1),
        )

    explicit = EvaluationCase(
        id="explicit",
        kind="semantic",
        query="query",
        expected_sources=(ExpectedSource(vault_id="vault-a", path="same.md"),),
    )
    report = run_evaluation(
        service,
        cases=(explicit,),
        thresholds=EvaluationThresholds(minimum_cases=1),
    )

    assert report.cases[0].recall_at_5 is False
    assert report.cases[0].rank is None
    assert report.cases[0].expected_sources == (ExpectedSource(vault_id="vault-a", path="same.md"),)


def test_evaluation_gates_case_count_latency_and_degradation() -> None:
    cases = tuple(
        evaluation_case(f"case-{index}", "semantic", f"query-{index}", "right.md")
        for index in range(24)
    )
    responses = {
        f"query-{index}": ([hit("right.md")], 600.0 if index >= 22 else 10.0) for index in range(24)
    }
    service = FakeService(
        responses,
        current_paths={"right.md"},
        degraded_queries={"query-0"},
    )

    report = run_evaluation(
        service,
        cases=cases,
        thresholds=EvaluationThresholds(
            recall_at_5=1.0,
            minimum_cases=25,
            p95_latency_ms=500.0,
        ),
        mode=SearchMode.DENSE,
    )

    assert report.mode is SearchMode.DENSE
    assert report.case_count == 24
    assert report.degraded_cases == 1
    assert report.degradation_reasons == ("query_embedding_failed",)
    assert report.p95_latency_ms > 10.0
    assert report.passed is False
    assert report.cases[0].degraded is True
    assert all(request.mode is SearchMode.DENSE for request in service.search_requests)


def test_lexical_evaluation_is_not_degraded_when_semantics_are_intentionally_skipped() -> None:
    service = FakeService(
        {"query": ([hit("right.md")], 1.0)},
        current_paths={"right.md"},
    )

    report = run_evaluation(
        service,
        cases=(evaluation_case("case", "semantic", "query", "right.md"),),
        thresholds=EvaluationThresholds(
            recall_at_5=1.0,
            minimum_cases=1,
            p95_latency_ms=500.0,
        ),
        mode=SearchMode.LEXICAL,
    )

    assert report.mode is SearchMode.LEXICAL
    assert report.degraded_cases == 0
    assert report.passed is True


@pytest.mark.parametrize(
    "toml_text",
    [
        "schema_version = 2\n",
        "schema_version = 1\n[thresholds]\nrecall_at_5 = 1.1\nidentifier_rank_1 = 1.0\n",
        "schema_version = 1\n[thresholds]\nrecall_at_5 = 0.9\nidentifier_rank_1 = 1.0\n"
        '[[cases]]\nid = "bad"\nkind = "other"\nquery = "q"\nexpected_paths = ["a.md"]\n',
        "schema_version = 1\n[thresholds]\nrecall_at_5 = 0.9\nidentifier_rank_1 = 1.0\n"
        '[[cases]]\nid = "bad"\nkind = "semantic"\nquery = "q"\nexpected_paths = ["../a.md"]\n',
    ],
)
def test_load_evaluation_rejects_malformed_schema(tmp_path: Path, toml_text: str) -> None:
    path = tmp_path / "eval.toml"
    path.write_text(toml_text, encoding="utf-8")

    with pytest.raises(ConfigError, match="invalid evaluation configuration"):
        load_evaluation(path)


@pytest.mark.parametrize(
    ("threshold", "message"),
    [
        ("minimum_cases = 24", "minimum_cases must be at least 25"),
        ("p95_latency_ms = 500.1", "p95_latency_ms must not exceed 500"),
        ("require_no_degradation = false", "invalid evaluation configuration"),
    ],
)
def test_load_evaluation_rejects_weakened_acceptance_gates(
    tmp_path: Path, threshold: str, message: str
) -> None:
    path = tmp_path / "eval.toml"
    cases = "\n".join(
        "[[cases]]\n"
        f'id = "case-{index:02}"\n'
        'kind = "semantic"\n'
        'query = "q"\n'
        'expected_paths = ["one.md"]\n'
        for index in range(25)
    )
    path.write_text(
        "schema_version = 1\n[thresholds]\nrecall_at_5 = 0.9\n"
        f"identifier_rank_1 = 1.0\n{threshold}\n{cases}",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match=message):
        load_evaluation(path)


def test_load_evaluation_rejects_duplicate_ids_and_paths(tmp_path: Path) -> None:
    path = tmp_path / "eval.toml"
    path.write_text(
        """
schema_version = 1

[thresholds]
recall_at_5 = 0.9
identifier_rank_1 = 1.0

[[cases]]
id = "same"
kind = "semantic"
query = "one"
expected_paths = ["one.md", "one.md"]

[[cases]]
id = "same"
kind = "identifier"
query = "two"
expected_paths = ["two.md"]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="duplicate"):
        load_evaluation(path)


def test_expected_source_and_case_models_validate_bounds() -> None:
    source = ExpectedSource(path="notes/right.md")
    qualified = ExpectedSource(vault_id="vault-a", path="notes/right.md")
    assert source.path == "notes/right.md"
    assert source.vault_id is None
    assert qualified.vault_id == "vault-a"
    assert EvaluationKind("semantic") is EvaluationKind.SEMANTIC
    with pytest.raises(ValueError):
        EvaluationCase(id="", kind="semantic", query="q", expected_paths=("right.md",))
    with pytest.raises(ValueError):
        EvaluationCase(id="case", kind="semantic", query="", expected_paths=("right.md",))
