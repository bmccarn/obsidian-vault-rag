from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from vault_rag.config import EgressPolicy, ResolvedProfile, ResolvedVault, VaultManifest
from vault_rag.domain import DegradedState, LineRange, SourceRef
from vault_rag.errors import ServiceBusyError
from vault_rag.retrieval import (
    ReadRequest,
    ReadResponse,
    ScoreComponents,
    SearchFilters,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from vault_rag.service.config import RepositoryConfig, ServiceConfig, ServiceEmbeddingConfig
from vault_rag.service.facade import VaultService
from vault_rag.service.locking import LifecycleLock
from vault_rag.service.observability import ServiceMetrics
from vault_rag.service.state import (
    IndexDiagnosticState,
    IndexReportState,
    StateRead,
    VaultSyncState,
)

SHA_A = "a" * 40
SHA_B = "b" * 40


def _config() -> ServiceConfig:
    return ServiceConfig(
        schema_version=1,
        sync_interval="1h",
        repositories={
            "alpha": RepositoryConfig(url="https://example.test/alpha.git", ref="refs/heads/main"),
            "beta": RepositoryConfig(url="https://example.test/beta.git", ref="refs/heads/main"),
        },
        profiles={
            "public": {"vaults": ("alpha", "beta")},
            "alpha-only": {"vaults": ("alpha",)},
        },
        embedding=ServiceEmbeddingConfig(
            base_url="http://127.0.0.1:9000/v1",
            model_env="MODEL",
            endpoint_class="local",
        ),
    )


def _profile(name: str, vault_ids: tuple[str, ...]) -> ResolvedProfile:
    return ResolvedProfile(
        name=name,
        vaults=tuple(
            ResolvedVault(
                root=Path("/vaults") / vault_id,
                manifest=VaultManifest(
                    schema_version=1,
                    id=vault_id,
                    egress_policy=EgressPolicy.LOCAL_ONLY,
                    include=("**/*.md",),
                ),
            )
            for vault_id in vault_ids
        ),
        embedding_model="model",
        api_key_env=None,
        effective_policy=EgressPolicy.LOCAL_ONLY,
        semantic_enabled=True,
    )


def _search_response(vault_id: str = "alpha") -> SearchResponse:
    ref = SourceRef(vault_id, "notes/a.md", ("A",), LineRange(1, 2), "hash")
    hit = SearchHit(
        "chunk",
        ref,
        "text",
        {},
        ScoreComponents(1, 1.0, None, None, 1.0, False),
        None,
    )
    return SearchResponse((hit,), DegradedState(), 1.0, 2.0)


@dataclass
class FakeRetrieval:
    response: SearchResponse = field(default_factory=_search_response)
    search_error: Exception | None = None
    read_error: Exception | None = None
    requests: list[SearchRequest] = field(default_factory=list)

    def search(self, request: SearchRequest) -> SearchResponse:
        self.requests.append(request)
        request.filters.to_storage(("alpha", "beta"), frozenset({"owner"}))
        if self.search_error is not None:
            raise self.search_error
        return self.response

    def read(self, request: ReadRequest) -> ReadResponse:
        if self.read_error is not None:
            raise self.read_error
        return ReadResponse(
            SourceRef(
                request.vault_id or "alpha",
                request.path,
                (),
                LineRange(1, 1),
                "hash",
            ),
            "text",
            None,
        )


@dataclass
class FakeFactory:
    retrieval_service: FakeRetrieval
    statuses: dict[str, dict[str, Any]]
    closed: int = 0

    def resolve_profile(self, name: str) -> ResolvedProfile:
        return _profile(name, ("alpha", "beta") if name == "public" else ("alpha",))

    def retrieval(self, _profile: ResolvedProfile) -> FakeRetrieval:
        return self.retrieval_service

    def status(self, profile: ResolvedProfile) -> dict[str, Any]:
        return self.statuses.get(profile.name, self.statuses["alpha-only"])

    def close(self) -> None:
        self.closed += 1


@dataclass
class FakeStateStore:
    states: dict[str, VaultSyncState]
    warnings: dict[str, str] = field(default_factory=dict)
    checkout_independent: bool = False

    def read(self, vault_id: str, configured_ref: str) -> StateRead:
        initial = VaultSyncState(vault_id=vault_id, configured_ref=configured_ref)
        return StateRead(
            state=self.states.get(vault_id, initial),
            warning=self.warnings.get(vault_id),
        )


class ObservingLock:
    """A lock double that proves every service operation enters shared access."""

    def __init__(self) -> None:
        self.active = False
        self.timeouts: list[float] = []

    @contextmanager
    def read(self, timeout: float) -> Iterator[None]:
        self.timeouts.append(timeout)
        self.active = True
        try:
            yield
        finally:
            self.active = False


class LockCheckingFactory(FakeFactory):
    """Factory double that rejects profile and index work outside shared access."""

    def __init__(self, lock: ObservingLock, statuses: dict[str, dict[str, Any]]) -> None:
        super().__init__(FakeRetrieval(), statuses)
        self._lock = lock

    def resolve_profile(self, name: str) -> ResolvedProfile:
        assert self._lock.active
        return super().resolve_profile(name)

    def status(self, profile: ResolvedProfile) -> dict[str, Any]:
        assert self._lock.active
        return super().status(profile)


class LockCheckingStateStore(FakeStateStore):
    """State double that rejects durable state reads outside shared access."""

    def __init__(self, lock: ObservingLock, states: dict[str, VaultSyncState]) -> None:
        super().__init__(states)
        self._lock = lock

    def read(self, vault_id: str, configured_ref: str) -> StateRead:
        assert self._lock.active
        return super().read(vault_id, configured_ref)


@dataclass
class FakeCoordinator:
    calls: list[str] = field(default_factory=list)

    def start(self) -> None:
        self.calls.append("start")

    def request_sync(self) -> None:
        self.calls.append("sync")

    def close(self) -> None:
        self.calls.append("close")


@dataclass
class FakeCheckoutInspector:
    heads: dict[str, str | None]

    def head(self, vault_id: str) -> str | None:
        return self.heads.get(vault_id)


def _checkout_inspector() -> FakeCheckoutInspector:
    return FakeCheckoutInspector({"alpha": SHA_A, "beta": SHA_A})


def test_facade_rejects_requests_when_checkout_head_mismatches_durable_state(
    service: VaultService,
) -> None:
    """Serving after activation before state persistence can return wrong index content."""
    service._checkout_inspector.heads["alpha"] = SHA_B

    assert service.ready().ready is False
    with pytest.raises(ServiceBusyError, match="repository reconciliation is in progress"):
        service.search("alpha-only", SearchRequest("query"))
    with pytest.raises(ServiceBusyError, match="repository reconciliation is in progress"):
        service.read("alpha-only", ReadRequest("notes/a.md", vault_id="alpha"))
    assert service._app_factory.retrieval_service.requests == []
    assert service.status("alpha-only")["profile"] == "alpha-only"


def test_facade_rejects_sqlite_checkout_that_differs_from_reconciled_identity(
    service: VaultService,
) -> None:
    """A moved checkout must not serve an older SQLite index as current."""
    service._state_store.states["alpha"] = VaultSyncState(
        vault_id="alpha",
        configured_ref="refs/heads/main",
        checkout_sha=SHA_B,
        reconciled_sha=SHA_A,
    )
    service._checkout_inspector.heads["alpha"] = SHA_B

    assert service.ready().profiles["alpha-only"] is False
    with pytest.raises(ServiceBusyError, match="repository reconciliation is in progress"):
        service.search("alpha-only", SearchRequest("query"))


def test_facade_rejects_requests_when_checkout_cannot_be_inspected(
    service: VaultService,
) -> None:
    """A missing local checkout cannot establish that indexed content remains current."""
    service._checkout_inspector.heads["alpha"] = None

    assert service.ready().ready is False
    with pytest.raises(ServiceBusyError, match="repository reconciliation is in progress"):
        service.search("alpha-only", SearchRequest("query"))


def test_facade_allows_lexical_search_and_read_when_semantic_indexing_is_pending(
    service: VaultService,
) -> None:
    """Provider failure must not hide compatible, reconciled lexical content."""
    search = service.search("alpha-only", SearchRequest("query"))
    read = service.read("alpha-only", ReadRequest("notes/a.md", vault_id="alpha"))

    assert service.ready().profiles["alpha-only"] is True
    assert search.response.hits[0].ref.vault_id == "alpha"
    assert read.response.text == "text"
    assert service._app_factory.retrieval_service.requests == [SearchRequest("query")]


def test_facade_serves_active_postgresql_revision_after_failed_attempt(
    service: VaultService,
) -> None:
    """A failed newer worker attempt must not hide the active database revision."""
    service._state_store.checkout_independent = True
    service._state_store.states["alpha"] = VaultSyncState(
        vault_id="alpha",
        configured_ref="refs/heads/main",
        fetched_sha=SHA_B,
        checkout_sha=SHA_A,
        attempted_sha=SHA_B,
        reconciled_sha=SHA_A,
        sync_degraded_reason="reconciliation_failed",
    )
    service._checkout_inspector.heads["alpha"] = None

    search = service.search("alpha-only", SearchRequest("query"))
    read = service.read("alpha-only", ReadRequest("notes/a.md", vault_id="alpha"))

    assert service.ready().profiles["alpha-only"] is True
    assert search.response.hits[0].ref.vault_id == "alpha"
    assert read.response.text == "text"
    assert service.status("alpha-only")["vaults"]["alpha"]["commit"]["reconciled_sha"] == SHA_A


@pytest.mark.parametrize(
    ("boundary", "value"),
    (
        ("zero sources", ("counts", "sources", 0)),
        ("zero chunks", ("counts", "chunks", 0)),
        ("boolean sources", ("counts", "sources", True)),
        ("boolean chunks", ("counts", "chunks", False)),
        ("noninteger sources", ("counts", "sources", "3")),
        ("noninteger chunks", ("counts", "chunks", "5")),
        ("incompatible schema", ("schema", "compatible", False)),
        ("missing reconciliation timestamp", ("last_successful_reconciliation", None, None)),
        ("state warning", ("warning", None, None)),
        ("mismatched actual checkout", ("checkout", None, None)),
    ),
)
def test_facade_rejects_lexically_unusable_profiles_before_retrieval(
    service: VaultService,
    boundary: str,
    value: tuple[str, str | None, object],
) -> None:
    """Each invalid lexical or durable boundary must block both retrieval surfaces."""
    section, field_name, replacement = value
    index = service._app_factory.statuses["alpha-only"]
    if section == "counts":
        index["counts"][field_name] = replacement
    elif section == "schema":
        index["schema"][field_name] = replacement
    elif section == "last_successful_reconciliation":
        index[section] = None
    elif section == "warning":
        service._state_store.warnings["alpha"] = "safe state warning"
    else:
        service._checkout_inspector.heads["alpha"] = SHA_B

    assert service.ready().profiles["alpha-only"] is False, boundary
    with pytest.raises(ServiceBusyError, match="repository reconciliation is in progress"):
        service.search("alpha-only", SearchRequest("query"))
    with pytest.raises(ServiceBusyError, match="repository reconciliation is in progress"):
        service.read("alpha-only", ReadRequest("notes/a.md", vault_id="alpha"))
    assert service._app_factory.retrieval_service.requests == []


@pytest.fixture
def service() -> VaultService:
    states = {
        vault_id: VaultSyncState(
            vault_id=vault_id,
            configured_ref="refs/heads/main",
            checkout_sha=SHA_A,
            reconciled_sha=SHA_A,
        )
        for vault_id in ("alpha", "beta")
    }
    index_status = {
        "schema": {"compatible": True},
        "counts": {"sources": 3, "chunks": 5, "ready": 0, "pending": 5},
        "last_successful_reconciliation": "2026-01-01T00:00:00+00:00",
    }
    factory = FakeFactory(
        FakeRetrieval(),
        {name: dict(index_status) for name in ("public", "alpha-only")},
    )
    return VaultService(
        _config(),
        factory,
        FakeCoordinator(),
        _checkout_inspector(),
        FakeStateStore(states),
        LifecycleLock(),
    )


def test_facade_records_only_search_mode_and_status_aggregates() -> None:
    """Passing a request or raw status payload to metrics could expose user/source data."""
    metrics = ServiceMetrics()
    states = {
        vault_id: VaultSyncState(
            vault_id=vault_id,
            configured_ref="refs/heads/main",
            checkout_sha=SHA_A,
            reconciled_sha=SHA_A,
        )
        for vault_id in ("alpha", "beta")
    }
    status = {
        "schema": {"compatible": True},
        "counts": {"sources": 3, "chunks": 5, "ready": 4, "pending": 1},
        "semantic_degradation": {"semantic_search": True},
        "last_successful_reconciliation": "2026-01-01T00:00:00+00:00",
    }
    ticks = iter((1.0, 1.125))
    service = VaultService(
        _config(),
        FakeFactory(FakeRetrieval(), {"alpha-only": status}),
        FakeCoordinator(),
        _checkout_inspector(),
        FakeStateStore(states),
        LifecycleLock(),
        metrics=metrics,
        timer=lambda: next(ticks),
    )

    service.search("public", SearchRequest("private query"))
    service.status("public")

    rendered = metrics.render().decode()
    assert 'vault_rag_search_total{mode="hybrid"} 1.0' in rendered
    assert 'vault_rag_search_duration_seconds_sum{mode="hybrid"} 0.125' in rendered
    assert 'vault_rag_sources{vault_id="alpha"} 3.0' in rendered
    assert 'vault_rag_semantic_degraded{vault_id="alpha"} 1.0' in rendered
    assert "private query" not in rendered
    assert "profile" not in rendered


@pytest.mark.parametrize("failure", ("retrieval", "contention"))
def test_facade_records_every_failed_search_attempt(failure: str) -> None:
    """Omitting failed attempts hides retrieval outages and lifecycle contention."""
    metrics = ServiceMetrics()
    retrieval = FakeRetrieval()
    if failure == "retrieval":
        retrieval.search_error = RuntimeError("synthetic failure")
    ticks = iter((1.0, 1.125))
    service = VaultService(
        _config(),
        FakeFactory(
            retrieval,
            {
                "alpha-only": {
                    "schema": {"compatible": True},
                    "counts": {"sources": 3, "chunks": 5, "ready": 0, "pending": 5},
                    "last_successful_reconciliation": "2026-01-01T00:00:00+00:00",
                }
            },
        ),
        FakeCoordinator(),
        _checkout_inspector(),
        FakeStateStore(
            {
                vault_id: VaultSyncState(
                    vault_id=vault_id,
                    configured_ref="refs/heads/main",
                    checkout_sha=SHA_A,
                    reconciled_sha=SHA_A,
                )
                for vault_id in ("alpha", "beta")
            }
        ),
        LifecycleLock(),
        metrics=metrics,
        timer=lambda: next(ticks),
    )

    if failure == "retrieval":
        with pytest.raises(RuntimeError, match="synthetic failure"):
            service.search("public", SearchRequest("private query"))
    else:
        with service._lock.write(), pytest.raises(ServiceBusyError):
            service.search("public", SearchRequest("private query"))

    rendered = metrics.render().decode()
    assert 'vault_rag_search_total{mode="hybrid"} 1.0' in rendered
    assert 'vault_rag_search_duration_seconds_sum{mode="hybrid"} 0.125' in rendered
    assert "private query" not in rendered


@pytest.mark.parametrize("operation", ("search", "read", "status", "ready"))
def test_public_operations_resolve_and_read_state_inside_shared_lock(operation: str) -> None:
    """Resolving or reading local index state before shared access races reconciliation."""
    lock = ObservingLock()
    states = {
        vault_id: VaultSyncState(
            vault_id=vault_id,
            configured_ref="refs/heads/main",
            checkout_sha=SHA_A,
            reconciled_sha=SHA_A,
        )
        for vault_id in ("alpha", "beta")
    }
    statuses = {
        "alpha-only": {
            "schema": {"compatible": True},
            "counts": {"sources": 3, "chunks": 5, "ready": 0, "pending": 5},
            "last_successful_reconciliation": "2026-01-01T00:00:00+00:00",
        }
    }
    service = VaultService(
        _config(),
        LockCheckingFactory(lock, statuses),
        FakeCoordinator(),
        _checkout_inspector(),
        LockCheckingStateStore(lock, states),
        lock,  # type: ignore[arg-type]
    )

    if operation == "search":
        service.search("public", SearchRequest("query"))
    elif operation == "read":
        service.read("public", ReadRequest("notes/a.md", vault_id="alpha"))
    elif operation == "status":
        service.status("public")
    else:
        service.ready()

    assert lock.timeouts
    assert set(lock.timeouts) == {0.05}


def test_status_curates_only_approved_per_vault_index_fields(service: VaultService) -> None:
    """Forwarding factory diagnostics would disclose infrastructure-only configuration."""
    approved = {
        "schema": {"compatible": True},
        "fingerprints": {"state": "current"},
        "counts": {"ready": 1},
        "model": "model",
        "embedding_revision": "rev",
        "requested_dimensions": 3,
        "last_successful_reconciliation": "2026-01-01T00:00:00+00:00",
        "index_age_seconds": 1.0,
        "semantic_degradation": {"semantic_search": False, "reason": None},
        "effective_egress_policy": "local-only",
    }
    service._app_factory.statuses["alpha-only"] = {
        **approved,
        "database_path": "/private/index.sqlite3",
        "endpoint_class": "remote",
        "max_batch_tokens": 999,
        "other_internal_value": "secret",
    }

    status = service.status("public")

    assert status["vaults"]["alpha"]["index"] == approved


def test_status_exposes_exact_bounded_durable_commit_state(service: VaultService) -> None:
    """Omitting durable progress or leaking repository configuration misleads operators."""
    report = IndexReportState(
        total_sources=3,
        added_sources=1,
        changed_sources=1,
        unchanged_sources=1,
        deleted_sources=0,
        ready_chunks=0,
        pending_chunks=5,
        parse_failures=0,
        embedding_failures=5,
        embedding_requests=1,
        blocking_failures=0,
        diagnostics=(
            IndexDiagnosticState(
                vault_id="alpha",
                path="notes/a.md",
                category="embedding",
                message="provider unavailable",
            ),
        ),
        elapsed_ms=42,
    )
    service._state_store.states["alpha"] = VaultSyncState(
        vault_id="alpha",
        configured_ref="refs/heads/main",
        configured_sha=SHA_A,
        fetched_sha=SHA_B,
        checkout_sha=SHA_A,
        attempted_sha=SHA_B,
        reconciled_sha=SHA_A,
        configured_at=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        fetched_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        checkout_at=datetime(2026, 1, 1, 0, 2, tzinfo=UTC),
        attempted_at=datetime(2026, 1, 1, 0, 3, tzinfo=UTC),
        reconciled_at=datetime(2026, 1, 1, 0, 4, tzinfo=UTC),
        report=report,
        sync_degraded_reason="embedding provider unavailable",
        sync_degraded_at=datetime(2026, 1, 1, 0, 5, tzinfo=UTC),
    )
    status = service.status("alpha-only")

    assert status == {
        "profile": "alpha-only",
        "vaults": {
            "alpha": {
                "index": {
                    "schema": {"compatible": True},
                    "counts": {"sources": 3, "chunks": 5, "ready": 0, "pending": 5},
                    "last_successful_reconciliation": "2026-01-01T00:00:00+00:00",
                },
                "commit": {
                    "configured_ref": "refs/heads/main",
                    "configured_sha": SHA_A,
                    "fetched_sha": SHA_B,
                    "checkout_sha": SHA_A,
                    "attempted_sha": SHA_B,
                    "reconciled_sha": SHA_A,
                    "configured_at": "2026-01-01T00:00:00Z",
                    "fetched_at": "2026-01-01T00:01:00Z",
                    "checkout_at": "2026-01-01T00:02:00Z",
                    "attempted_at": "2026-01-01T00:03:00Z",
                    "reconciled_at": "2026-01-01T00:04:00Z",
                    "report": {
                        "total_sources": 3,
                        "added_sources": 1,
                        "changed_sources": 1,
                        "unchanged_sources": 1,
                        "deleted_sources": 0,
                        "ready_chunks": 0,
                        "pending_chunks": 5,
                        "parse_failures": 0,
                        "embedding_failures": 5,
                        "embedding_requests": 1,
                        "blocking_failures": 0,
                        "diagnostics": [
                            {
                                "vault_id": "alpha",
                                "path": "notes/a.md",
                                "category": "embedding",
                                "message": "provider unavailable",
                            }
                        ],
                        "elapsed_ms": 42,
                    },
                    "sync_degraded_reason": "embedding provider unavailable",
                    "sync_degraded_at": "2026-01-01T00:05:00Z",
                    "warning": None,
                    "actual_checkout_sha": SHA_A,
                    "commit_safe": True,
                },
            }
        },
        "vaults_truncated": 0,
    }
    rendered = json.dumps(status)
    assert "https://example.test" not in rendered
    assert "api_key" not in rendered


def test_profiles_expose_only_service_configured_public_profiles(service: VaultService) -> None:
    """Using factory profile enumeration would leak private reconciliation profiles."""
    assert service.profiles() == {
        "profiles": [
            {"name": "alpha-only", "vaults": ["alpha"], "vaults_truncated": 0},
            {"name": "public", "vaults": ["alpha", "beta"], "vaults_truncated": 0},
        ],
        "profiles_truncated": 0,
    }
    with pytest.raises(ValueError, match="profile is not configured"):
        service.search("_reconcile_alpha", SearchRequest("query"))


def test_search_preserves_profile_filter_allowlists_and_hit_only_commit_views(
    service: VaultService,
) -> None:
    """Widening filters or reporting non-hit commits would expose unrelated vault state."""
    request = SearchRequest(
        "query",
        SearchFilters(vault_ids=("alpha",), frontmatter={"owner": "core"}),
    )
    envelope = service.search("public", request)
    assert envelope.response.hits[0].ref.vault_id == "alpha"
    assert tuple(envelope.commits) == ("alpha",)
    with pytest.raises(ValueError, match="outside the bound profile"):
        service.search("public", SearchRequest("query", SearchFilters(vault_ids=("other",))))
    with pytest.raises(ValueError, match="not configured"):
        service.search("missing", SearchRequest("query"))


def test_shared_locks_release_after_failures_and_reject_writer_contention(
    service: VaultService,
) -> None:
    """Failing to release reads would block reconciliation after a retrieval failure."""
    factory = service._app_factory
    factory.retrieval_service.search_error = RuntimeError("search failed")
    with pytest.raises(RuntimeError, match="search failed"):
        service.search("public", SearchRequest("query"))
    factory.retrieval_service.search_error = None
    factory.retrieval_service.read_error = RuntimeError("read failed")
    with pytest.raises(RuntimeError, match="read failed"):
        service.read("public", ReadRequest("notes/a.md", vault_id="alpha"))
    with service._lock.write(timeout=0.01):
        pass
    factory.retrieval_service.read_error = None
    with service._lock.write(), pytest.raises(ServiceBusyError):
        service.search("public", SearchRequest("query"))


def test_status_is_bounded_and_ready_requires_complete_matching_local_state(
    service: VaultService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignoring either vault state or usable index evidence would report false readiness."""
    status = service.status("public")
    assert status["vaults"]["alpha"]["commit"]["reconciled_sha"] == SHA_A
    import vault_rag.service.facade as facade

    monkeypatch.setattr(facade, "_MAX_STATUS_VAULTS", 1)
    bounded = service.status("public")
    assert tuple(bounded["vaults"]) == ("alpha",)
    assert bounded["vaults_truncated"] == 1
    assert service.ready().ready is True
    service._state_store.states["beta"] = VaultSyncState(
        vault_id="beta",
        configured_ref="refs/heads/main",
        checkout_sha=SHA_A,
        reconciled_sha=SHA_B,
    )
    assert service.ready().ready is True  # alpha-only remains complete
    service._app_factory.statuses["alpha-only"]["counts"] = {
        "sources": 0,
        "chunks": 5,
        "ready": 0,
        "pending": 5,
    }
    assert service.ready().ready is False


def test_lifecycle_delegates_once_and_closes_coordinator_before_factory(
    service: VaultService,
) -> None:
    """Repeated transport lifecycle hooks must not duplicate polling or resource cleanup."""
    coordinator = service._coordinator
    factory = service._app_factory
    service.start()
    service.start()
    service.request_sync()
    service.close()
    service.close()
    assert coordinator.calls == ["start", "sync", "close"]
    assert factory.closed == 1
