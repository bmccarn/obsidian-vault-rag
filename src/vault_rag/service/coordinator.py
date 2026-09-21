"""Durable, independently-failing reconciliation of managed Git vaults."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, RLock, Thread, current_thread
from time import perf_counter
from typing import Protocol, cast

from vault_rag.config import (
    EgressPolicy,
    RegistryConfig,
    ResolvedProfile,
    ResolvedVault,
    VaultManifest,
)
from vault_rag.config.loader import load_manifest_bytes
from vault_rag.errors import ConfigError, RebuildRequiredError, StorageError
from vault_rag.indexing import Indexer, IndexReport
from vault_rag.indexing.service import expected_vault_fingerprints
from vault_rag.storage import SCHEMA_VERSION, IndexSnapshot
from vault_rag.storage.ports import (
    IndexStore,
    QueryStore,
    RevisionBuildStore,
    RevisionStore,
    SnapshotQueryStore,
)
from vault_rag.storage.records import ActiveRevision, VaultFingerprint

from .config import RepositoryConfig, ServiceConfig
from .git import GitRemote, ManagedGit
from .locking import LifecycleLock
from .state import IndexReportState, SyncStateStorePort, VaultSyncState

_MAX_MANIFEST_BYTES = 1024 * 1024


class SyncObserver(Protocol):
    """Receive bounded aggregate observations after repository transitions."""

    def fetch_finished(self, vault_id: str, outcome: str, elapsed_ms: float) -> None: ...

    def reconciliation_finished(
        self,
        vault_id: str,
        outcome: str,
        elapsed_ms: float,
        snapshot: IndexSnapshot | None,
        report: IndexReportState | None,
        sync_degraded: bool,
        semantic_degraded: bool,
    ) -> None: ...


class ReconciliationLease(Protocol):
    """One session-scoped, vault-scoped reconciliation lease."""

    def acquire(self) -> bool: ...

    def ensure_held(self) -> None: ...

    def promote(
        self,
        revision: RevisionBuildStore,
        *,
        lexical_complete: bool,
        fully_reconciled: bool,
    ) -> None: ...

    def release(self) -> None: ...


class _AppFactory(Protocol):
    """The coordinator's intentionally small application-factory boundary."""

    registry: RegistryConfig

    def resolve_profile(self, name: str) -> ResolvedProfile: ...

    def store(self, profile: ResolvedProfile) -> QueryStore: ...

    def rebuild_indexer(self, profile: ResolvedProfile) -> Indexer: ...
    def indexer(
        self,
        profile: ResolvedProfile,
        *,
        store: IndexStore | None = None,
    ) -> Indexer: ...


class RepositorySyncCoordinator:
    """Synchronize every configured vault in deterministic, isolated passes."""

    def __init__(
        self,
        config: ServiceConfig,
        app_factory: _AppFactory,
        state_store: SyncStateStorePort,
        lifecycle_lock: LifecycleLock,
        git: ManagedGit,
        *,
        environ: Mapping[str, str] | None = None,
        observer: SyncObserver | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        timer: Callable[[], float] = perf_counter,
        lease_factory: Callable[[str], ReconciliationLease] | None = None,
    ) -> None:
        self._config = config
        self._app_factory = app_factory
        self._state_store = state_store
        self._lifecycle_lock = lifecycle_lock
        self._git = git
        self._environ = dict(environ or {})
        self._observer = observer
        self._clock = clock
        self._timer = timer
        self._lease_factory = lease_factory
        self._sync_event = Event()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._closed = False
        self._joining = False
        self._thread_lock = RLock()
        self._retry_pending_embeddings = False

    def head(self, vault_id: str) -> str | None:
        if vault_id not in self._config.repositories:
            return None
        return self._git.head(self._checkout_path(vault_id))

    def run_once(
        self,
        *,
        retry_pending_embeddings: bool = False,
        force_reconcile: bool = False,
    ) -> None:
        """Attempt one independently leased reconciliation pass per vault."""
        for vault_id in sorted(self._config.repositories):
            lease = self._lease_factory(vault_id) if self._lease_factory is not None else None
            ensure_lease = lease.ensure_held if lease is not None else lambda: None
            try:
                if lease is not None and not lease.acquire():
                    continue
                self._sync_repository(
                    vault_id,
                    self._config.repositories[vault_id],
                    retry_pending_embeddings=retry_pending_embeddings,
                    force_reconcile=force_reconcile,
                    ensure_lease=ensure_lease,
                    lease=lease,
                )
            except Exception:
                self._record_unexpected_failure(
                    vault_id, self._config.repositories[vault_id], ensure_lease=ensure_lease
                )
            finally:
                if lease is not None:
                    with suppress(StorageError):
                        lease.release()

    def start(self) -> None:
        """Start exactly one daemon polling thread, whose first pass is immediate."""
        with self._thread_lock:
            if self._closed or (self._thread is not None and self._thread.is_alive()):
                return
            self._thread = Thread(
                target=self._poll,
                name="vault-rag-repository-sync",
                daemon=True,
            )
            self._thread.start()

    def request_sync(self, *, retry_pending_embeddings: bool = True) -> None:
        """Coalesce a caller-triggered wake with the next polling pass."""
        with self._thread_lock:
            if self._closed:
                return
            self._retry_pending_embeddings |= retry_pending_embeddings
            self._sync_event.set()

    def close(self) -> None:
        """Stop and join the polling thread once; repeated calls are harmless."""
        thread: Thread | None = None
        with self._thread_lock:
            self._closed = True
            self._stop_event.set()
            self._sync_event.set()
            if (
                self._thread is not None
                and not self._joining
                and self._thread is not current_thread()
            ):
                self._joining = True
                thread = self._thread
        if thread is not None:
            thread.join()

    def _poll(self) -> None:
        while not self._stop_event.is_set():
            self.run_once(retry_pending_embeddings=self._consume_pending_retry())
            if self._stop_event.is_set():
                return
            self._sync_event.wait(self._config.sync_interval.total_seconds())
            self._sync_event.clear()

    def _consume_pending_retry(self) -> bool:
        with self._thread_lock:
            retry_pending_embeddings = self._retry_pending_embeddings
            self._retry_pending_embeddings = False
            return retry_pending_embeddings

    def _sync_repository(
        self,
        vault_id: str,
        repository: RepositoryConfig,
        *,
        retry_pending_embeddings: bool,
        force_reconcile: bool,
        ensure_lease: Callable[[], None],
        lease: ReconciliationLease | None,
    ) -> None:
        ensure_lease()
        fetch_started = self._timer()
        state = self._state_store.read(vault_id, repository.ref).state
        checkout = self._checkout_path(vault_id)
        remote = GitRemote(
            url=str(repository.url),
            ref=repository.ref,
            credential=(
                self._environ.get(repository.credential_env)
                if repository.credential_env is not None
                else None
            ),
            ca_cert_path=(
                str(repository.ca_cert_path) if repository.ca_cert_path is not None else None
            ),
        )
        try:
            fetched = self._git.fetch(remote, checkout)
        except Exception:
            state = self._write_fetch(state, None, "fetch_failed", ensure_lease=ensure_lease)
            self._notify_fetch(vault_id, "failure", self._elapsed_ms(fetch_started))
            self._finish_reconciliation(
                state,
                "failure",
                self._elapsed_ms(fetch_started),
                None,
                None,
            )
            return

        try:
            state = self._write_fetch(state, fetched.sha, None, ensure_lease=ensure_lease)
            self._notify_fetch(vault_id, "success", self._elapsed_ms(fetch_started))
            reconciliation_started = self._timer()
            try:
                manifest_checkout = fetched.prepared_checkout or checkout
                manifest = load_manifest_bytes(
                    self._git.read_blob(
                        fetched,
                        manifest_checkout,
                        ".vault-rag.toml",
                        max_bytes=_MAX_MANIFEST_BYTES,
                    )
                )
                if manifest.id != vault_id:
                    raise ConfigError("fetched vault manifest identity does not match repository")
            except Exception:
                state = self._write_reconciliation(
                    state, reason="manifest_invalid", ensure_lease=ensure_lease
                )
                self._finish_reconciliation(
                    state, "failure", self._elapsed_ms(reconciliation_started), None, None
                )
                return

            snapshot: IndexSnapshot | None
            if self._state_store.checkout_independent:
                profile = self._fetched_profile(vault_id, manifest_checkout, manifest)
                backend = self._app_factory.store(profile)
                active_store = cast(SnapshotQueryStore, backend)
                if not callable(getattr(active_store, "active_revisions", None)):
                    raise StorageError("checkout-independent storage lacks active revision queries")
                active_revisions = active_store.active_revisions((vault_id,))
                snapshot = active_store.snapshot((vault_id,))
                index_usable = self._active_revision_matches(
                    active_revisions,
                    fetched.sha,
                    expected_vault_fingerprints(profile, self._app_factory.registry.embedding)[
                        vault_id
                    ],
                )
                unchanged = index_usable
            else:
                snapshot = self._index_snapshot(vault_id)
                index_usable = snapshot is not None and self._snapshot_is_usable(snapshot)
                unchanged = (
                    state.fetched_sha
                    == state.reconciled_sha
                    == state.attempted_sha
                    == state.checkout_sha
                    == self._git.head(checkout)
                    and index_usable
                )
            failed_same_commit = (
                not force_reconcile
                and state.attempted_sha == fetched.sha
                and state.sync_degraded_reason == "lexical_incomplete"
            )
            if failed_same_commit:
                self._finish_reconciliation(
                    state,
                    "failure",
                    self._elapsed_ms(reconciliation_started),
                    snapshot,
                    None,
                )
                return
            retry_pending = (
                retry_pending_embeddings and snapshot is not None and snapshot.pending_count > 0
            )
            if unchanged and not retry_pending:
                state = self._write_reconciliation(state, reason=None, ensure_lease=ensure_lease)
                self._finish_reconciliation(
                    state, "success", self._elapsed_ms(reconciliation_started), snapshot, None
                )
                return

            try:
                with self._lifecycle_lock.write():
                    ensure_lease()
                    self._git.activate(fetched, checkout)
                    state = self._write_reconciliation(
                        state.model_copy(
                            update={"checkout_sha": fetched.sha, "checkout_at": self._clock()}
                        ),
                        reason=None,
                        ensure_lease=ensure_lease,
                    )
                    profile = self._app_factory.resolve_profile(f"_reconcile_{vault_id}")
                    state = self._write_reconciliation(
                        state.model_copy(
                            update={"attempted_sha": fetched.sha, "attempted_at": self._clock()}
                        ),
                        reason=None,
                        ensure_lease=ensure_lease,
                    )
                    report = self._run_indexing(
                        profile,
                        repository,
                        fetched.sha,
                        index_usable=index_usable,
                        ensure_lease=ensure_lease,
                        lease=lease,
                    )
            except Exception:
                state = self._write_reconciliation(
                    state, reason="reconciliation_failed", ensure_lease=ensure_lease
                )
                self._finish_reconciliation(
                    state, "failure", self._elapsed_ms(reconciliation_started), snapshot, None
                )
                return

            state = self._persist_report(state, report, ensure_lease=ensure_lease)
            self._finish_reconciliation(
                state,
                "success" if state.sync_degraded_reason != "lexical_incomplete" else "failure",
                self._elapsed_ms(reconciliation_started),
                self._index_snapshot(vault_id),
                state.report,
            )
        finally:
            self._git.discard(fetched, checkout)

    def _run_indexing(
        self,
        profile: ResolvedProfile,
        repository: RepositoryConfig,
        commit_sha: str,
        *,
        index_usable: bool,
        ensure_lease: Callable[[], None],
        lease: ReconciliationLease | None,
    ) -> IndexReport:
        backend = self._app_factory.store(profile)
        if not isinstance(backend, RevisionStore):
            try:
                return self._app_factory.indexer(profile).run(rebuild=not index_usable)
            except RebuildRequiredError:
                return self._app_factory.rebuild_indexer(profile).run(rebuild=True)
            except StorageError:
                if index_usable:
                    raise
                return self._app_factory.rebuild_indexer(profile).run(rebuild=True)

        vault_id = profile.vaults[0].manifest.id
        started = self._timer()
        revision = backend.begin_revision(
            vault_id,
            repository.ref,
            commit_sha,
            manifest=profile.vaults[0].manifest.model_dump(mode="json"),
        )
        try:
            if not index_usable:
                revision.reset_vaults((vault_id,))
            try:
                report = self._app_factory.indexer(profile, store=revision).run(rebuild=False)
            except RebuildRequiredError:
                revision.reset_vaults((vault_id,))
                report = self._app_factory.indexer(profile, store=revision).run(rebuild=False)
            lexical_complete = report.parse_failures == 0 and report.blocking_failures == 0
            if lexical_complete:
                if lease is None:
                    raise StorageError("PostgreSQL revision promotion requires a lease")
                promotion_started = self._timer()
                try:
                    lease.promote(revision, lexical_complete=True, fully_reconciled=True)
                except Exception:
                    self._notify_phase("promotion", "failure", self._elapsed_ms(promotion_started))
                    raise
                self._notify_phase("promotion", "success", self._elapsed_ms(promotion_started))
            else:
                ensure_lease()
                revision.fail(
                    "lexical_incomplete",
                    "revision was not promoted because lexical reconciliation was incomplete",
                )
            self._notify_phase("build", "success", self._elapsed_ms(started))
            return report
        except Exception:
            with suppress(StorageError):
                ensure_lease()
                revision.fail("reconciliation_failed", "revision build failed")
            self._notify_phase("build", "failure", self._elapsed_ms(started))
            raise

    def _write_fetch(
        self,
        state: VaultSyncState,
        sha: str | None,
        reason: str | None,
        *,
        ensure_lease: Callable[[], None],
    ) -> VaultSyncState:
        ensure_lease()
        now = self._clock()
        updated = state.model_copy(
            update={
                "configured_sha": sha if sha is not None else state.configured_sha,
                "configured_at": now if sha is not None else state.configured_at,
                "fetched_sha": sha if sha is not None else state.fetched_sha,
                "fetched_at": now if sha is not None else state.fetched_at,
                "sync_degraded_reason": (
                    reason if reason is not None else state.sync_degraded_reason
                ),
                "sync_degraded_at": now if reason is not None else state.sync_degraded_at,
            }
        )
        self._state_store.write(updated)
        return updated

    def _write_reconciliation(
        self,
        state: VaultSyncState,
        *,
        reason: str | None,
        ensure_lease: Callable[[], None],
    ) -> VaultSyncState:
        ensure_lease()
        now = self._clock()
        updated = state.model_copy(
            update={
                "sync_degraded_reason": reason,
                "sync_degraded_at": now if reason is not None else None,
            }
        )
        self._state_store.write(updated)
        return updated

    def _finish_reconciliation(
        self,
        state: VaultSyncState,
        outcome: str,
        elapsed_ms: float,
        snapshot: IndexSnapshot | None,
        report: IndexReportState | None,
    ) -> None:
        self._notify_reconciliation(
            state.vault_id,
            outcome,
            elapsed_ms,
            snapshot,
            report,
            state.sync_degraded_reason is not None,
            self._semantic_degraded(state),
        )

    def _persist_report(
        self,
        state: VaultSyncState,
        report: IndexReport,
        *,
        ensure_lease: Callable[[], None],
    ) -> VaultSyncState:
        ensure_lease()
        lexical_complete = report.parse_failures == 0 and report.blocking_failures == 0
        now = self._clock()
        if lexical_complete:
            reason = "semantic_pending" if report.embedding_failures else None
            updated = state.model_copy(
                update={
                    "reconciled_sha": state.attempted_sha,
                    "reconciled_at": now,
                    "report": IndexReportState.from_report(report),
                    "sync_degraded_reason": reason,
                    "sync_degraded_at": now if reason is not None else None,
                }
            )
        else:
            updated = state.model_copy(
                update={
                    "report": IndexReportState.from_report(report),
                    "sync_degraded_reason": "lexical_incomplete",
                    "sync_degraded_at": now,
                }
            )
        self._state_store.write(updated)
        return updated

    def _fetched_profile(
        self, vault_id: str, root: Path, manifest: VaultManifest
    ) -> ResolvedProfile:
        """Bind a fetched manifest without requiring an active local checkout."""
        embedding = self._app_factory.registry.embedding
        model = self._environ.get(embedding.model_env)
        if not model:
            raise ConfigError(
                f"embedding model environment variable is required: {embedding.model_env}"
            )
        local_only = manifest.egress_policy is EgressPolicy.LOCAL_ONLY
        semantic_disabled = local_only and embedding.endpoint_class == "remote"
        return ResolvedProfile(
            name=f"_reconcile_{vault_id}",
            vaults=(ResolvedVault(root=root, manifest=manifest),),
            embedding_model=model,
            api_key_env=embedding.api_key_env,
            effective_policy=EgressPolicy.LOCAL_ONLY if local_only else EgressPolicy.REMOTE_ALLOWED,
            semantic_enabled=not semantic_disabled,
            semantic_disabled_reason=(
                "remote endpoint prohibited by local-only vault" if semantic_disabled else None
            ),
        )

    @staticmethod
    def _active_revision_matches(
        revisions: tuple[ActiveRevision, ...], commit_sha: str, expected: VaultFingerprint
    ) -> bool:
        if len(revisions) != 1:
            return False
        revision = revisions[0]
        return (
            revision.commit_sha == commit_sha
            and revision.vault_id == expected.vault_id
            and revision.manifest_fingerprint == expected.manifest_fingerprint
            and revision.parser_fingerprint == expected.parser_fingerprint
            and revision.chunker_fingerprint == expected.chunker_fingerprint
            and revision.embedding_config_fingerprint == expected.embedding_config_fingerprint
        )

    def _index_snapshot(self, vault_id: str) -> IndexSnapshot | None:
        try:
            profile = self._app_factory.resolve_profile(f"_reconcile_{vault_id}")
            return self._app_factory.store(profile).snapshot((vault_id,))
        except Exception:
            return None

    @staticmethod
    def _snapshot_is_usable(snapshot: IndexSnapshot) -> bool:
        return (
            snapshot.schema_version == SCHEMA_VERSION
            and snapshot.indexed_at is not None
            and snapshot.manifest_fingerprint is not None
            and snapshot.parser_fingerprint is not None
            and snapshot.chunker_fingerprint is not None
            and snapshot.embedding_config_fingerprint is not None
        )

    def _checkout_path(self, vault_id: str) -> Path:
        return self._app_factory.registry.vaults[vault_id].path

    def _record_unexpected_failure(
        self,
        vault_id: str,
        repository: RepositoryConfig,
        *,
        ensure_lease: Callable[[], None],
    ) -> None:
        try:
            state = self._state_store.read(vault_id, repository.ref).state
            state = self._write_reconciliation(
                state, reason="reconciliation_failed", ensure_lease=ensure_lease
            )
            self._finish_reconciliation(state, "failure", 0.0, None, None)
            self._notify_phase("failure", "failure", 0.0)
        except Exception:
            # State persistence must not prevent later independent repositories.
            return

    def _elapsed_ms(self, started: float) -> float:
        return max(0.0, (self._timer() - started) * 1000.0)

    @staticmethod
    def _semantic_degraded(state: VaultSyncState) -> bool:
        return state.report is not None and (
            state.report.embedding_failures > 0 or state.report.pending_chunks > 0
        )

    def _notify_phase(self, phase: str, outcome: str, elapsed_ms: float) -> None:
        observer = getattr(self._observer, "reconciliation_phase", None)
        if callable(observer):
            try:
                observer(phase, outcome, elapsed_ms)
            except Exception:
                return

    def _notify_fetch(self, vault_id: str, outcome: str, elapsed_ms: float) -> None:
        if self._observer is not None:
            try:
                self._observer.fetch_finished(vault_id, outcome, elapsed_ms)
            except Exception:
                return

    def _notify_reconciliation(
        self,
        vault_id: str,
        outcome: str,
        elapsed_ms: float,
        snapshot: IndexSnapshot | None,
        report: IndexReportState | None,
        sync_degraded: bool,
        semantic_degraded: bool,
    ) -> None:
        if self._observer is not None:
            try:
                self._observer.reconciliation_finished(
                    vault_id,
                    outcome,
                    elapsed_ms,
                    snapshot,
                    report,
                    sync_degraded,
                    semantic_degraded,
                )
            except Exception:
                return
