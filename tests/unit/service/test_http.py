from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from prometheus_client.exposition import CONTENT_TYPE_LATEST

from vault_rag.domain import DegradedState, LineRange, SourceRef
from vault_rag.errors import ServiceBusyError, StaleSourceError, StorageError
from vault_rag.retrieval import (
    ReadRequest,
    ReadResponse,
    ScoreComponents,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from vault_rag.service.facade import CommitView, ReadEnvelope, Readiness, SearchEnvelope
from vault_rag.service.http import RuntimePort, create_app
from vault_rag.service.observability import ServiceMetrics

SHA = "a" * 40


@dataclass
class FakeService:
    error: Exception | None = None
    calls: list[tuple[str, SearchRequest | ReadRequest]] = field(default_factory=list)
    readiness: Readiness = field(default_factory=lambda: Readiness(True, {"homelab": True}))
    started: int = 0
    closed: int = 0

    def start(self) -> None:
        self.started += 1

    def close(self) -> None:
        self.closed += 1

    def search(self, profile: str, request: SearchRequest) -> SearchEnvelope:
        self.calls.append((profile, request))
        if self.error is not None:
            raise self.error
        ref = SourceRef(
            "homelab-ops",
            "04 - Runbooks/reboot.md",
            ("Reboot",),
            LineRange(4, 7),
            "hash",
        )
        hit = SearchHit(
            "chunk-1",
            ref,
            "safe reboot procedure",
            {"status": "active"},
            ScoreComponents(1, 1.0, 1, 1.0, 0.5, False),
            None,
        )
        return SearchEnvelope(
            profile,
            SearchResponse((hit,), DegradedState(), 1.0, 2.0),
            {"homelab-ops": CommitView("homelab-ops", SHA, SHA, None)},
        )

    def read(self, profile: str, request: ReadRequest) -> ReadEnvelope:
        self.calls.append((profile, request))
        if self.error is not None:
            raise self.error
        return ReadEnvelope(
            profile,
            ReadResponse(
                SourceRef("homelab-ops", request.path, (), LineRange(1, 2), "hash"),
                "reboot safely",
                None,
            ),
        )

    def profiles(self) -> dict[str, object]:
        return {
            "profiles": [{"name": "homelab", "vaults": ["homelab-ops"], "vaults_truncated": 0}],
            "profiles_truncated": 0,
        }

    def status(self, profile: str) -> dict[str, object]:
        return {
            "profile": profile,
            "vaults": {
                "homelab-ops": {
                    "index": {"counts": {"ready": 1}},
                    "commit": {
                        "configured_ref": "refs/heads/main",
                        "checkout_sha": SHA,
                        "reconciled_sha": SHA,
                        "actual_checkout_sha": SHA,
                        "commit_safe": True,
                    },
                }
            },
            "vaults_truncated": 0,
        }

    def ready(self) -> Readiness:
        return self.readiness

    def request_sync(self) -> None:
        return None


@dataclass
class EventSink:
    records: list[tuple[str, dict[str, object]]] = field(default_factory=list)

    def emit(self, event: str, **fields: object) -> None:
        self.records.append((event, fields))


@dataclass
class FakeRuntime:
    service: FakeService
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)
    events: EventSink = field(default_factory=EventSink)

    def start(self) -> None:
        self.service.start()

    def close(self) -> None:
        self.service.close()


def _app(runtime: FakeRuntime):
    """Contain the deliberately small transport double at its protocol boundary."""
    return create_app(cast(RuntimePort, runtime))


@pytest.fixture
def runtime() -> FakeRuntime:
    return FakeRuntime(FakeService())


def test_routes_map_strict_requests_to_the_service_and_render_shared_payloads(
    runtime: FakeRuntime,
) -> None:
    expected_source_hash = "sha256:4755db938696de059a6beeff63961a2c738d436281161ce6aa8f7e095e9401b1"

    with TestClient(_app(runtime)) as client:
        response = client.post(
            "/v1/search",
            json={
                "profile": "homelab",
                "query": "safe reboot",
                "mode": "hybrid",
                "limit": 10,
                "filters": {
                    "vault_ids": ["homelab-ops"],
                    "path_prefix": "04 - Runbooks",
                    "source_kind": "markdown",
                    "frontmatter": {"status": "active"},
                },
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["hits"][0]["vault_id"] == "homelab-ops"
        assert payload["vault_commits"]["homelab-ops"]["fully_reconciled"] is True

        read_response = client.post(
            "/v1/read",
            json={
                "profile": "homelab",
                "path": "04 - Runbooks/reboot.md",
                "start_line": 1,
                "end_line": 2,
                "expected_source_hash": expected_source_hash,
            },
        )
        assert read_response.json()["text"] == "reboot safely"
        assert client.get("/v1/profiles").json()["profiles"][0]["name"] == "homelab"
        status_response = client.get("/v1/status", params={"profile": "homelab"})
        assert status_response.json()["vaults"]["homelab-ops"]["index"]["counts"]["ready"] == 1
        assert (
            status_response.json()["vaults"]["homelab-ops"]["commit"]["actual_checkout_sha"] == SHA
        )
        assert status_response.json()["vaults"]["homelab-ops"]["commit"]["commit_safe"] is True
        assert client.post("/v1/admin/sync").status_code == 202
        assert client.get("/health/live").json() == {"live": True}
        assert client.get("/health/ready").json() == {"ready": True, "profiles": {"homelab": True}}
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert metrics.headers["content-type"] == CONTENT_TYPE_LATEST

    assert runtime.service.started == 1
    assert runtime.service.closed == 1
    profile, request = runtime.service.calls[0]
    assert profile == "homelab"
    assert isinstance(request, SearchRequest)
    assert request.filters.vault_ids == ("homelab-ops",)
    assert request.filters.path_prefix == "04 - Runbooks"
    read_profile, read_request = runtime.service.calls[1]
    assert read_profile == "homelab"
    assert isinstance(read_request, ReadRequest)
    assert read_request.expected_source_hash == (
        "sha256:4755db938696de059a6beeff63961a2c738d436281161ce6aa8f7e095e9401b1"
    )


def test_read_stale_source_returns_conflict_without_content() -> None:

    runtime = FakeRuntime(FakeService(error=StaleSourceError("caller hash mismatched")))

    with TestClient(_app(runtime)) as client:
        response = client.post(
            "/v1/read",
            json={
                "profile": "homelab",
                "path": "04 - Runbooks/reboot.md",
                "expected_source_hash": "sha256:" + "0" * 64,
            },
        )

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "stale_source"}}


def test_readiness_returns_service_unavailable_with_its_bounded_payload() -> None:

    runtime = FakeRuntime(FakeService(readiness=Readiness(False, {"homelab": False})))

    with TestClient(_app(runtime)) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"ready": False, "profiles": {"homelab": False}}


def test_adapter_rejects_invalid_bodies_and_one_sided_line_ranges(runtime: FakeRuntime) -> None:

    with TestClient(_app(runtime)) as client:
        assert client.post("/v1/search", json={"profile": "homelab"}).status_code == 422
        invalid_search = client.post(
            "/v1/search",
            json={"profile": "homelab", "query": "x", "unknown": True},
        )
        assert invalid_search.status_code == 422
        response = client.post(
            "/v1/read",
            json={"profile": "homelab", "path": "a.md", "start_line": 1},
        )
        overlong_profile = client.get("/v1/status", params={"profile": "x" * 201})

    assert response.status_code == 422
    assert overlong_profile.status_code == 422


def test_adapter_preserves_independent_heading_text_and_breadcrumb_bounds(
    runtime: FakeRuntime,
) -> None:

    with TestClient(_app(runtime)) as client:
        heading_text = "h" * 2_000
        accepted_text = client.post(
            "/v1/read",
            json={"profile": "homelab", "path": "a.md", "heading": heading_text},
        )
        accepted_parts = client.post(
            "/v1/read",
            json={"profile": "homelab", "path": "a.md", "heading": ["h"] * 50},
        )
        rejected_text = client.post(
            "/v1/read",
            json={"profile": "homelab", "path": "a.md", "heading": "h" * 2_001},
        )
        rejected_parts = client.post(
            "/v1/read",
            json={"profile": "homelab", "path": "a.md", "heading": ["h"] * 51},
        )

    assert accepted_text.status_code == 200
    assert accepted_parts.status_code == 200
    assert rejected_text.status_code == 422
    assert rejected_parts.status_code == 422
    assert isinstance(runtime.service.calls[-2][1], ReadRequest)
    assert isinstance(runtime.service.calls[-1][1], ReadRequest)
    assert runtime.service.calls[-2][1].heading == heading_text
    assert runtime.service.calls[-1][1].heading == ("h",) * 50


def test_adapter_rejects_oversized_bodies_before_validation(runtime: FakeRuntime) -> None:

    with TestClient(_app(runtime)) as client:
        response = client.post(
            "/v1/search",
            content=b" " * (1024 * 1024 + 1),
            headers={"Content-Type": "application/json"},
        )
        streamed_response = client.post(
            "/v1/search",
            content=(chunk for chunk in (b" " * (1024 * 1024), b"  ")),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 413
    assert response.json() == {"error": {"code": "request_too_large"}}
    assert streamed_response.status_code == 413
    assert streamed_response.json() == {"error": {"code": "request_too_large"}}


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (ValueError("bad request"), 400, "invalid_request"),
        (ServiceBusyError("busy"), 503, "service_busy"),
        (StorageError("database unavailable"), 503, "storage_error"),
    ],
)
def test_adapter_maps_expected_errors_without_leaking_details(
    runtime: FakeRuntime, error: Exception, status: int, code: str
) -> None:

    runtime.service.error = error
    with TestClient(_app(runtime)) as client:
        response = client.post("/v1/search", json={"profile": "homelab", "query": "safe reboot"})

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    if status == 503:
        assert response.headers["retry-after"] == "1"


def test_adapter_hides_unexpected_provider_and_git_failures(runtime: FakeRuntime) -> None:

    runtime.service.error = RuntimeError("https://user:supersecret@example.test/repo.git exploded")
    with TestClient(_app(runtime)) as client:
        response = client.post("/v1/search", json={"profile": "homelab", "query": "safe reboot"})

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["code"] == "internal_error"
    UUID(error["correlation_id"])
    assert "supersecret" not in response.text
    assert "example.test" not in response.text
    assert runtime.events.records[0] == (
        "http_internal_error",
        {"correlation_id": error["correlation_id"], "outcome": "failure"},
    )
    event, fields = runtime.events.records[1]
    assert event == "http_request"
    assert fields["endpoint"] == "/v1/search"
    assert fields["mode"] == "hybrid"
    assert fields["outcome"] == "failure"
    assert isinstance(fields["elapsed_ms"], float)


def test_http_metrics_change_on_success_and_error_finally_paths(runtime: FakeRuntime) -> None:

    with TestClient(_app(runtime)) as client:
        assert (
            client.post(
                "/v1/search",
                json={"profile": "homelab", "query": "sentinel-query", "mode": "hybrid"},
            ).status_code
            == 200
        )
        runtime.service.error = StorageError("postgresql://user:credential@db/sentinel")
        assert (
            client.post(
                "/v1/search",
                json={"profile": "homelab", "query": "sentinel-query", "mode": "hybrid"},
            ).status_code
            == 503
        )

    rendered = runtime.metrics.render().decode()
    assert (
        'vault_rag_http_requests_total{endpoint="/v1/search",mode="hybrid",outcome="success"} 1.0'
        in rendered
    )
    assert (
        'vault_rag_http_requests_total{endpoint="/v1/search",mode="hybrid",outcome="failure"} 1.0'
        in rendered
    )
    for sentinel in ("sentinel-query", "postgresql://", "credential"):
        assert sentinel not in rendered
        assert sentinel not in repr(runtime.events.records)
