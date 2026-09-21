from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from vault_rag.config import (
    EgressPolicy,
    EmbeddingConfig,
    ResolvedProfile,
    ResolvedVault,
    VaultManifest,
)
from vault_rag.errors import RebuildRequiredError, StorageError
from vault_rag.indexing import IndexDiagnostic, IndexReport
from vault_rag.indexing.service import expected_vault_fingerprints
from vault_rag.service.config import RepositoryConfig, ServiceConfig, ServiceEmbeddingConfig
from vault_rag.service.coordinator import RepositorySyncCoordinator
from vault_rag.service.git import FetchedCommit
from vault_rag.service.locking import LifecycleLock
from vault_rag.service.state import IndexReportState, SyncStateStore, VaultSyncState
from vault_rag.storage import SCHEMA_VERSION, IndexSnapshot
from vault_rag.storage.records import ActiveRevision

SHA_A = "a" * 40
SHA_B = "b" * 40


def _manifest(vault_id: str) -> bytes:
    return (
        "schema_version = 1\n"
        f'id = "{vault_id}"\n'
        'egress_policy = "local-only"\n'
        'include = ["**/*.md"]\n'
    ).encode()


def _report(
    *,
    parse_failures: int = 0,
    embedding_failures: int = 0,
    blocking_failures: int = 0,
    diagnostics: tuple[IndexDiagnostic, ...] = (),
) -> IndexReport:
    return IndexReport(
        1,
        1,
        0,
        0,
        0,
        1,
        embedding_failures,
        parse_failures,
        embedding_failures,
        1,
        diagnostics,
        1,
        blocking_failures=blocking_failures,
    )


def _snapshot(*, usable: bool, pending_count: int = 0) -> IndexSnapshot:
    return IndexSnapshot(
        source_count=1 if usable else 0,
        chunk_count=1 if usable else 0,
        ready_count=1 if usable else 0,
        pending_count=pending_count,
        vector_bytes=0,
        indexed_at=datetime.now(UTC) if usable else None,
        schema_version=SCHEMA_VERSION if usable else 0,
        manifest_fingerprint="sha256:manifest" if usable else None,
        parser_fingerprint="sha256:parser" if usable else None,
        chunker_fingerprint="sha256:chunker" if usable else None,
        embedding_config_fingerprint="sha256:embedding" if usable else None,
        observed_fingerprints=(),
        vector_dimensions=(),
        vector_config_fingerprints=(),
    )


class FakeGit:
    def __init__(self, sha: str = SHA_B) -> None:
        self.fetched_sha = sha
        self.active_sha: str | None = None
        self.manifests: dict[str, bytes] = {"alpha": _manifest("alpha"), "beta": _manifest("beta")}
        self.fetch_error: Exception | None = None
        self.activations: list[str] = []
        self.discard_calls: list[tuple[FetchedCommit, Path]] = []

    def fetch(self, _remote: object, _checkout: Path) -> FetchedCommit:
        if self.fetch_error is not None:
            raise self.fetch_error
        return FetchedCommit(self.fetched_sha, None)

    def read_blob(
        self, _fetched: FetchedCommit, checkout: Path, path: str, *, max_bytes: int
    ) -> bytes:
        assert path == ".vault-rag.toml"
        return self.manifests[checkout.name][:max_bytes]

    def activate(self, fetched: FetchedCommit, _checkout: Path) -> None:
        self.activations.append(fetched.sha)
        self.active_sha = fetched.sha

    def head(self, _checkout: Path) -> str | None:
        return self.active_sha

    def discard(self, fetched: FetchedCommit, checkout: Path) -> None:
        self.discard_calls.append((fetched, checkout))


class FakeIndexer:
    def __init__(self, report: IndexReport, error: Exception | None = None) -> None:
        self.report = report
        self.error = error
        self.calls: list[bool] = []

    def run(self, *, rebuild: bool = False) -> IndexReport:
        self.calls.append(rebuild)
        if self.error is not None:
            raise self.error
        return self.report


class RecordingObserver:
    def __init__(self) -> None:
        self.fetch_observations: list[tuple[str, str, float]] = []
        self.reconciliation_observations: list[
            tuple[str, str, float, IndexSnapshot | None, IndexReportState | None, bool, bool]
        ] = []

    def fetch_finished(self, vault_id: str, outcome: str, elapsed_ms: float) -> None:
        self.fetch_observations.append((vault_id, outcome, elapsed_ms))

    def reconciliation_finished(
        self,
        vault_id: str,
        outcome: str,
        elapsed_ms: float,
        snapshot: IndexSnapshot | None,
        report: IndexReportState | None,
        sync_degraded: bool,
        semantic_degraded: bool,
    ) -> None:
        self.reconciliation_observations.append(
            (vault_id, outcome, elapsed_ms, snapshot, report, sync_degraded, semantic_degraded)
        )


class FakeActiveStore:
    def __init__(self, snapshot: IndexSnapshot, revisions: tuple[ActiveRevision, ...]) -> None:
        self._snapshot = snapshot
        self.revisions = revisions

    def snapshot(self, _vault_ids: tuple[str, ...]) -> IndexSnapshot:
        return self._snapshot

    def active_revisions(self, _vault_ids: tuple[str, ...]) -> tuple[ActiveRevision, ...]:
        return self.revisions


class CheckoutIndependentStateStore(SyncStateStore):
    checkout_independent = True


@dataclass
class FakeApp:
    root: Path
    snapshot: IndexSnapshot
    indexers: dict[str, FakeIndexer]
    active_store: FakeActiveStore | None = None
    rebuild_indexers: dict[str, FakeIndexer] | None = None

    def __post_init__(self) -> None:
        self.registry = SimpleNamespace(
            vaults={
                vault_id: SimpleNamespace(path=self.root / "repos" / vault_id)
                for vault_id in self.indexers
            },
            embedding=EmbeddingConfig(
                base_url="http://127.0.0.1:9000/v1",
                model_env="EMBED_MODEL",
                endpoint_class="local",
            ),
        )

    def resolve_profile(self, name: str) -> ResolvedProfile:
        vault_id = name.removeprefix("_reconcile_")
        manifest = VaultManifest(
            schema_version=1,
            id=vault_id,
            egress_policy=EgressPolicy.LOCAL_ONLY,
            include=("**/*.md",),
        )
        return ResolvedProfile(
            name=name,
            vaults=(ResolvedVault(root=self.root / "repos" / vault_id, manifest=manifest),),
            embedding_model="test-model",
            api_key_env=None,
            effective_policy=EgressPolicy.LOCAL_ONLY,
            semantic_enabled=True,
        )

    def store(self, _profile: ResolvedProfile) -> object:
        if self.active_store is not None:
            return self.active_store
        return SimpleNamespace(snapshot=lambda _vault_ids: self.snapshot)

    def indexer(self, profile: ResolvedProfile) -> FakeIndexer:
        return self.indexers[profile.vaults[0].manifest.id]

    def rebuild_indexer(self, profile: ResolvedProfile) -> FakeIndexer:
        vault_id = profile.vaults[0].manifest.id
        if self.rebuild_indexers is None:
            return self.indexers[vault_id]
        return self.rebuild_indexers[vault_id]


def _config(*vault_ids: str) -> ServiceConfig:
    return ServiceConfig(
        schema_version=1,
        sync_interval="1h",
        repositories={
            vault_id: RepositoryConfig(url="https://example.test/repo.git", ref="refs/heads/main")
            for vault_id in vault_ids
        },
        profiles={"public": {"vaults": vault_ids}},
        embedding=ServiceEmbeddingConfig(
            base_url="http://127.0.0.1:9000/v1", model_env="EMBED_MODEL", endpoint_class="local"
        ),
    )


@dataclass
class CoordinatorFixture:
    coordinator: RepositorySyncCoordinator
    state_store: SyncStateStore
    git: FakeGit
    app: FakeApp

    def state(
        self,
        *,
        reconciled_sha: str | None,
        attempted_sha: str | None,
        checkout_sha: str | None,
    ) -> None:
        self.state_store.write(
            VaultSyncState(
                vault_id="alpha",
                configured_ref="refs/heads/main",
                reconciled_sha=reconciled_sha,
                attempted_sha=attempted_sha,
                checkout_sha=checkout_sha,
            )
        )
        self.git.active_sha = checkout_sha


@pytest.fixture
def coordinator_fixture(tmp_path: Path) -> CoordinatorFixture:
    git = FakeGit()
    app = FakeApp(tmp_path, _snapshot(usable=True), {"alpha": FakeIndexer(_report())})
    state_store = SyncStateStore(tmp_path / "state")
    return CoordinatorFixture(
        RepositorySyncCoordinator(_config("alpha"), app, state_store, LifecycleLock(), git),
        state_store,
        git,
        app,
    )


def _active_revision(app: FakeApp, *, fingerprint_suffix: str = "") -> ActiveRevision:
    profile = app.resolve_profile("_reconcile_alpha")
    fingerprint = expected_vault_fingerprints(profile, app.registry.embedding)["alpha"]
    return ActiveRevision(
        vault_id="alpha",
        revision_id="00000000-0000-0000-0000-000000000001",
        commit_sha=SHA_A,
        manifest={},
        state="active",
        source_count=1,
        chunk_count=1,
        ready_count=1,
        pending_count=0,
        vector_bytes=0,
        manifest_fingerprint=fingerprint.manifest_fingerprint + fingerprint_suffix,
        parser_fingerprint=fingerprint.parser_fingerprint,
        chunker_fingerprint=fingerprint.chunker_fingerprint,
        embedding_config_fingerprint=fingerprint.embedding_config_fingerprint,
        promoted_at=datetime.now(UTC),
        configured_ref="refs/heads/main",
        fetched_sha=SHA_A,
        attempted_sha=SHA_A,
        reconciled_sha=SHA_A,
        sync_degradation_category=None,
        sync_degraded_at=None,
    )


def test_checkout_independent_store_skips_compatible_active_revision_without_checkout(
    tmp_path: Path,
) -> None:
    """A fresh PostgreSQL worker should not recreate a matching active revision."""
    app = FakeApp(tmp_path, _snapshot(usable=True), {"alpha": FakeIndexer(_report())})
    active = _active_revision(app)
    app.active_store = FakeActiveStore(_snapshot(usable=True), (active,))
    state_store = CheckoutIndependentStateStore(tmp_path / "state")
    state_store.write(
        VaultSyncState(
            vault_id="alpha",
            configured_ref="refs/heads/main",
            fetched_sha=SHA_A,
            attempted_sha=SHA_A,
            checkout_sha=SHA_A,
            reconciled_sha=SHA_A,
        )
    )
    git = FakeGit(SHA_A)
    coordinator = RepositorySyncCoordinator(
        _config("alpha"),
        app,
        state_store,
        LifecycleLock(),
        git,
        environ={"EMBED_MODEL": "test-model"},
    )

    coordinator.run_once()

    assert app.indexers["alpha"].calls == []
    assert git.activations == []
    assert app.active_store.revisions == (active,)


@pytest.mark.parametrize(
    "field",
    (
        "manifest_fingerprint",
        "parser_fingerprint",
        "chunker_fingerprint",
        "embedding_config_fingerprint",
    ),
)
def test_checkout_independent_fingerprint_mismatch_rebuilds(tmp_path: Path, field: str) -> None:
    """Every persisted fingerprint participates in PostgreSQL unchanged identity."""
    app = FakeApp(tmp_path, _snapshot(usable=True), {"alpha": FakeIndexer(_report())})
    app.active_store = FakeActiveStore(
        _snapshot(usable=True),
        (replace(_active_revision(app), **{field: "sha256:mismatch"}),),
    )
    coordinator = RepositorySyncCoordinator(
        _config("alpha"),
        app,
        CheckoutIndependentStateStore(tmp_path / "state"),
        LifecycleLock(),
        FakeGit(SHA_A),
        environ={"EMBED_MODEL": "test-model"},
    )

    coordinator.run_once()

    assert app.indexers["alpha"].calls == [True]


def test_unchanged_healthy_sha_skips_indexer_without_replaying_a_report(
    coordinator_fixture: CoordinatorFixture,
) -> None:
    """Reusing the last report for a no-op would publish stale run aggregates."""
    observer = RecordingObserver()
    coordinator = RepositorySyncCoordinator(
        _config("alpha"),
        coordinator_fixture.app,
        coordinator_fixture.state_store,
        LifecycleLock(),
        coordinator_fixture.git,
        observer=observer,
    )

    coordinator.run_once()
    coordinator.run_once()

    assert coordinator_fixture.app.indexers["alpha"].calls == [False]
    assert observer.reconciliation_observations[0][4] is not None
    assert observer.reconciliation_observations[1][4] is None


def test_unchanged_state_with_drifted_checkout_reconciles_forward(
    coordinator_fixture: CoordinatorFixture,
) -> None:
    """Skipping a drifted checkout would leave readiness permanently false."""
    coordinator_fixture.state(reconciled_sha=SHA_A, attempted_sha=SHA_A, checkout_sha=SHA_A)
    coordinator_fixture.git.fetched_sha = SHA_A
    coordinator_fixture.git.active_sha = SHA_B

    coordinator_fixture.coordinator.run_once()

    state = coordinator_fixture.state_store.read("alpha", "refs/heads/main").state
    assert coordinator_fixture.git.activations == [SHA_A]
    assert coordinator_fixture.app.indexers["alpha"].calls == [False]
    assert coordinator_fixture.git.active_sha == SHA_A
    assert (state.checkout_sha, state.attempted_sha, state.reconciled_sha) == (
        SHA_A,
        SHA_A,
        SHA_A,
    )


def test_forced_run_retries_pending_embeddings_for_unchanged_sha(
    coordinator_fixture: CoordinatorFixture,
) -> None:
    """Ignoring a forced pending retry would leave semantic work stuck forever."""
    coordinator_fixture.app.snapshot = _snapshot(usable=True, pending_count=1)
    coordinator_fixture.state(reconciled_sha=SHA_A, attempted_sha=SHA_A, checkout_sha=SHA_A)
    coordinator_fixture.git.fetched_sha = SHA_A

    coordinator_fixture.coordinator.run_once()
    coordinator_fixture.coordinator.run_once(retry_pending_embeddings=True)

    assert coordinator_fixture.app.indexers["alpha"].calls == [False]


def test_reconciliation_observer_receives_only_final_aggregate_state(
    coordinator_fixture: CoordinatorFixture,
) -> None:
    """Publishing checkout or attempted state as completion misleads observers."""
    observer = RecordingObserver()
    coordinator = RepositorySyncCoordinator(
        _config("alpha"),
        coordinator_fixture.app,
        coordinator_fixture.state_store,
        LifecycleLock(),
        coordinator_fixture.git,
        observer=observer,
    )

    coordinator.run_once()

    assert observer.fetch_observations[0][:2] == ("alpha", "success")
    assert observer.fetch_observations[0][2] >= 0
    assert len(observer.reconciliation_observations) == 1
    vault_id, outcome, elapsed_ms, snapshot, report, sync_degraded, semantic_degraded = (
        observer.reconciliation_observations[0]
    )
    assert (vault_id, outcome, sync_degraded, semantic_degraded) == (
        "alpha",
        "success",
        False,
        False,
    )
    assert elapsed_ms >= 0
    assert snapshot is not None and snapshot.source_count == 1
    assert report is not None and report.total_sources == 1


def test_changed_commit_activates_indexes_and_advances_reconciled_sha(
    coordinator_fixture: CoordinatorFixture,
) -> None:
    """Removing lexical completion state advancement would repeat successful work."""
    coordinator_fixture.state(reconciled_sha=SHA_A, attempted_sha=SHA_A, checkout_sha=SHA_A)

    coordinator_fixture.coordinator.run_once()

    state = coordinator_fixture.state_store.read("alpha", "refs/heads/main").state
    assert coordinator_fixture.app.indexers["alpha"].calls == [False]
    assert coordinator_fixture.git.activations == [SHA_B]
    assert state.checkout_sha == SHA_B
    assert state.attempted_sha == SHA_B
    assert state.reconciled_sha == SHA_B
    assert state.sync_degraded_reason is None


def test_fetch_failure_preserves_active_commit_state(
    coordinator_fixture: CoordinatorFixture,
) -> None:
    """Writing a fetch failure into checkout state would hide a usable active index."""
    coordinator_fixture.state(reconciled_sha=SHA_A, attempted_sha=SHA_A, checkout_sha=SHA_A)
    coordinator_fixture.git.fetch_error = RuntimeError("credential=never-persist-this")

    coordinator_fixture.coordinator.run_once()

    state = coordinator_fixture.state_store.read("alpha", "refs/heads/main").state
    assert (state.checkout_sha, state.reconciled_sha) == (SHA_A, SHA_A)
    assert state.sync_degraded_reason == "fetch_failed"
    assert "credential" not in state.model_dump_json()


def test_informational_diagnostic_does_not_block_lexical_reconciliation(
    coordinator_fixture: CoordinatorFixture,
) -> None:
    """Counting informational diagnostics would retry a complete commit forever."""
    coordinator_fixture.app.indexers["alpha"].report = _report(
        diagnostics=(IndexDiagnostic("alpha", "<policy>", "policy_transition", "changed"),)
    )

    coordinator_fixture.coordinator.run_once()

    state = coordinator_fixture.state_store.read("alpha", "refs/heads/main").state
    assert state.reconciled_sha == SHA_B


def test_blocking_failure_prevents_lexical_reconciliation(
    coordinator_fixture: CoordinatorFixture,
) -> None:
    """Ignoring an indexer's pre-truncation blocking signal would mark a failed commit done."""
    coordinator_fixture.app.indexers["alpha"].report = _report(blocking_failures=1)

    coordinator_fixture.coordinator.run_once()

    state = coordinator_fixture.state_store.read("alpha", "refs/heads/main").state
    assert state.reconciled_sha is None
    assert state.sync_degraded_reason == "lexical_incomplete"


def test_invalid_fetched_manifest_keeps_active_checkout_and_index(
    coordinator_fixture: CoordinatorFixture,
) -> None:
    """Activating before manifest validation would replace a working vault with invalid content."""
    coordinator_fixture.state(reconciled_sha=SHA_A, attempted_sha=SHA_A, checkout_sha=SHA_A)
    coordinator_fixture.git.manifests["alpha"] = b'id = "wrong"'

    coordinator_fixture.coordinator.run_once()

    state = coordinator_fixture.state_store.read("alpha", "refs/heads/main").state
    assert coordinator_fixture.git.activations == []
    assert coordinator_fixture.app.indexers["alpha"].calls == []
    assert (state.checkout_sha, state.reconciled_sha) == (SHA_A, SHA_A)
    assert state.sync_degraded_reason == "manifest_invalid"
    assert len(coordinator_fixture.git.discard_calls) == 1


def test_parse_failure_records_attempt_without_advancing_reconciled_sha(
    coordinator_fixture: CoordinatorFixture,
) -> None:
    """Treating parser failures as success would permanently skip broken sources."""
    coordinator_fixture.app.indexers["alpha"].report = _report(parse_failures=1)

    coordinator_fixture.coordinator.run_once()

    state = coordinator_fixture.state_store.read("alpha", "refs/heads/main").state
    assert state.attempted_sha == SHA_B
    assert state.reconciled_sha is None
    assert state.sync_degraded_reason == "lexical_incomplete"


def test_same_failed_commit_is_not_rebuilt_by_periodic_poll(
    coordinator_fixture: CoordinatorFixture,
) -> None:
    """A stable malformed commit must not create an unbounded failed-revision loop."""
    coordinator_fixture.app.indexers["alpha"].report = _report(parse_failures=1)

    coordinator_fixture.coordinator.run_once()
    coordinator_fixture.coordinator.run_once()

    assert coordinator_fixture.app.indexers["alpha"].calls == [False]
    state = coordinator_fixture.state_store.read("alpha", "refs/heads/main").state
    assert state.attempted_sha == SHA_B
    assert state.reconciled_sha is None
    assert state.sync_degraded_reason == "lexical_incomplete"


def test_explicit_request_can_retry_same_failed_commit(
    coordinator_fixture: CoordinatorFixture,
) -> None:
    """Operators must be able to retry a failed SHA after fixing runtime behavior."""
    coordinator_fixture.app.indexers["alpha"].report = _report(parse_failures=1)
    coordinator_fixture.coordinator.run_once()
    coordinator_fixture.app.indexers["alpha"].report = _report()

    coordinator_fixture.coordinator.run_once(force_reconcile=True)

    assert coordinator_fixture.app.indexers["alpha"].calls == [False, False]
    state = coordinator_fixture.state_store.read("alpha", "refs/heads/main").state
    assert state.reconciled_sha == SHA_B
    assert state.sync_degraded_reason is None


def test_embedding_failure_advances_lexical_reconciliation(
    coordinator_fixture: CoordinatorFixture,
) -> None:
    """Treating semantic degradation as lexical failure would retry identical indexing forever."""
    diagnostic = IndexDiagnostic("alpha", "note.md", "http", "endpoint unavailable")
    coordinator_fixture.app.indexers["alpha"].report = _report(
        embedding_failures=1,
        diagnostics=(diagnostic,),
    )

    coordinator_fixture.coordinator.run_once()

    state = coordinator_fixture.state_store.read("alpha", "refs/heads/main").state
    assert state.reconciled_sha == SHA_B
    assert state.report is not None and state.report.embedding_failures == 1


def test_missing_index_forces_rebuild_even_when_commit_is_unchanged(
    coordinator_fixture: CoordinatorFixture,
) -> None:
    """Skipping solely by SHA would leave a deleted database unrecoverable."""
    coordinator_fixture.app.snapshot = _snapshot(usable=False)
    coordinator_fixture.state(reconciled_sha=SHA_A, attempted_sha=SHA_A, checkout_sha=SHA_A)
    coordinator_fixture.git.fetched_sha = SHA_A

    coordinator_fixture.coordinator.run_once()
    assert coordinator_fixture.app.indexers["alpha"].calls == [True]


def test_rebuild_required_retries_with_explicit_rebuild_indexer(
    coordinator_fixture: CoordinatorFixture,
) -> None:
    fallback = FakeIndexer(_report())
    coordinator_fixture.app.indexers["alpha"].error = RebuildRequiredError("incompatible")
    coordinator_fixture.app.rebuild_indexers = {"alpha": fallback}

    coordinator_fixture.coordinator.run_once()

    assert coordinator_fixture.app.indexers["alpha"].calls == [False]
    assert fallback.calls == [True]


def test_unusable_store_error_retries_with_explicit_rebuild_indexer(
    coordinator_fixture: CoordinatorFixture,
) -> None:
    fallback = FakeIndexer(_report())
    coordinator_fixture.app.snapshot = _snapshot(usable=False)
    coordinator_fixture.app.indexers["alpha"].error = StorageError("corrupt")
    coordinator_fixture.app.rebuild_indexers = {"alpha": fallback}

    coordinator_fixture.coordinator.run_once()

    assert coordinator_fixture.app.indexers["alpha"].calls == [True]
    assert fallback.calls == [True]


def test_usable_store_error_does_not_rebuild(
    coordinator_fixture: CoordinatorFixture,
) -> None:
    fallback = FakeIndexer(_report())
    coordinator_fixture.app.indexers["alpha"].error = StorageError("write failure")
    coordinator_fixture.app.rebuild_indexers = {"alpha": fallback}

    coordinator_fixture.coordinator.run_once()

    assert coordinator_fixture.app.indexers["alpha"].calls == [False]
    assert fallback.calls == []


def test_repository_failure_does_not_stop_later_sorted_repository(tmp_path: Path) -> None:
    """Returning from one repository failure would starve independent vaults."""
    git = FakeGit()
    app = FakeApp(
        tmp_path,
        _snapshot(usable=True),
        {"alpha": FakeIndexer(_report()), "beta": FakeIndexer(_report())},
    )
    git.manifests["alpha"] = b"not toml = ["
    state_store = SyncStateStore(tmp_path / "state")
    coordinator = RepositorySyncCoordinator(
        _config("beta", "alpha"),
        app,
        state_store,
        LifecycleLock(),
        git,
    )

    coordinator.run_once()

    assert app.indexers["alpha"].calls == []
    assert app.indexers["beta"].calls == [False]


def test_vault_leases_skip_only_contended_repository(tmp_path: Path) -> None:
    git = FakeGit()
    app = FakeApp(
        tmp_path,
        _snapshot(usable=True),
        {"alpha": FakeIndexer(_report()), "beta": FakeIndexer(_report())},
    )
    released: list[str] = []
    guarded: list[str] = []

    class Lease:
        def __init__(self, vault_id: str) -> None:
            self.vault_id = vault_id

        def acquire(self) -> bool:
            return self.vault_id == "beta"

        def ensure_held(self) -> None:
            guarded.append(self.vault_id)

        def promote(
            self, revision: object, *, lexical_complete: bool, fully_reconciled: bool
        ) -> None:
            raise AssertionError("the SQLite test backend must not promote a revision")

        def release(self) -> None:
            released.append(self.vault_id)

    coordinator = RepositorySyncCoordinator(
        _config("beta", "alpha"),
        app,
        SyncStateStore(tmp_path / "state"),
        LifecycleLock(),
        git,
        lease_factory=Lease,
    )

    coordinator.run_once()

    assert app.indexers["alpha"].calls == []
    assert app.indexers["beta"].calls == [False]
    assert guarded and set(guarded) == {"beta"}
    assert released == ["alpha", "beta"]


def test_polling_runs_immediately_coalesces_wakes_and_closes_once(
    coordinator_fixture: CoordinatorFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A polling loop that waits before its first run or starts twice misses requested work."""
    calls: list[bool] = []
    first = Event()
    second = Event()

    def run_once(*, retry_pending_embeddings: bool = False) -> None:
        calls.append(retry_pending_embeddings)
        (first if len(calls) == 1 else second).set()

    monkeypatch.setattr(coordinator_fixture.coordinator, "run_once", run_once)
    coordinator_fixture.coordinator.start()
    coordinator_fixture.coordinator.start()
    assert first.wait(1)
    coordinator_fixture.coordinator.request_sync()
    coordinator_fixture.coordinator.request_sync()
    assert second.wait(1)
    coordinator_fixture.coordinator.close()
    coordinator_fixture.coordinator.close()

    assert calls == [False, True]
