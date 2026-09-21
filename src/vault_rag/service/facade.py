"""Transport-neutral public operations over managed vault retrieval."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import RLock
from time import perf_counter
from types import MappingProxyType
from typing import Protocol, cast

from vault_rag.config import ResolvedProfile
from vault_rag.domain import JsonValue
from vault_rag.errors import ServiceBusyError, StorageError
from vault_rag.retrieval import ReadRequest, ReadResponse, SearchRequest, SearchResponse
from vault_rag.storage.postgres.query_store import PostgresStore
from vault_rag.storage.records import ActiveRevision

from .config import ServiceConfig
from .locking import LifecycleLock
from .observability import ServiceMetrics
from .state import StateRead, SyncStateStorePort

_MAX_STATUS_VAULTS = 100
_MAX_IDENTITY_CHARACTERS = 200
_PUBLIC_INDEX_FIELDS = (
    "schema",
    "fingerprints",
    "counts",
    "model",
    "embedding_revision",
    "requested_dimensions",
    "last_successful_reconciliation",
    "index_age_seconds",
    "semantic_degradation",
    "effective_egress_policy",
)


class RetrievalPort(Protocol):
    """Retrieval surface used by public read and search operations."""

    def search(self, request: SearchRequest) -> SearchResponse: ...

    def read(self, request: ReadRequest) -> ReadResponse: ...


class RequestSessionPort(Protocol):
    """One request's immutable profile, retrieval data, and revision identity."""

    profile: ResolvedProfile
    retrieval: RetrievalPort
    revisions: Mapping[str, ActiveRevision]
    store: object


class AppFactoryPort(Protocol):
    """Minimal application-factory surface needed by the public service."""

    def resolve_profile(self, name: str) -> ResolvedProfile: ...

    def retrieval(self, profile: ResolvedProfile) -> RetrievalPort: ...

    def status(self, profile: ResolvedProfile) -> dict[str, JsonValue]: ...
    def close(self) -> None: ...


class SyncCoordinatorPort(Protocol):
    """Lifecycle surface exposed to HTTP and worker transports."""

    def start(self) -> None: ...

    def request_sync(self) -> None: ...

    def close(self) -> None: ...


class CheckoutInspectorPort(Protocol):
    """Read the active managed-checkout commit for one configured vault."""

    def head(self, vault_id: str) -> str | None: ...


@dataclass(frozen=True, slots=True)
class CommitView:
    """Public reconciliation identifiers for one vault."""

    vault_id: str
    checkout_sha: str | None
    reconciled_sha: str | None
    sync_degraded_reason: str | None
    revision_id: str | None = None
    commit_sha: str | None = None


@dataclass(frozen=True, slots=True)
class SearchEnvelope:
    """A profile-scoped search response with hit-vault commit state."""

    profile: str
    response: SearchResponse
    commits: Mapping[str, CommitView]

    def __post_init__(self) -> None:
        object.__setattr__(self, "commits", MappingProxyType(dict(self.commits)))


@dataclass(frozen=True, slots=True)
class ReadEnvelope:
    """A profile-scoped source read response and its represented revisions."""

    profile: str
    response: ReadResponse
    commits: Mapping[str, CommitView] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "commits", MappingProxyType(dict(self.commits)))


@dataclass(frozen=True, slots=True)
class Readiness:
    """Local index readiness across caller-selectable profiles."""

    ready: bool
    profiles: Mapping[str, bool]

    def __post_init__(self) -> None:
        object.__setattr__(self, "profiles", MappingProxyType(dict(self.profiles)))


class VaultService:
    """Public service facade that keeps configured profiles and lifecycle coherent."""

    def __init__(
        self,
        config: ServiceConfig,
        app_factory: AppFactoryPort,
        coordinator: SyncCoordinatorPort,
        checkout_inspector: CheckoutInspectorPort,
        state_store: SyncStateStorePort,
        lock: LifecycleLock,
        *,
        metrics: ServiceMetrics | None = None,
        timer: Callable[[], float] = perf_counter,
    ) -> None:
        self._config = config
        self._app_factory = app_factory
        self._coordinator = coordinator
        self._checkout_inspector = checkout_inspector
        self._state_store = state_store
        self._lock = lock
        self._metrics = metrics
        self._timer = timer
        self._lifecycle_lock = RLock()
        self._started = False
        self._closed = False

    def search(self, profile_name: str, request: SearchRequest) -> SearchEnvelope:
        """Search with response identity derived from the request's own data view."""
        mode = request.mode.value
        started = self._timer()
        try:
            with self._lock.read(timeout=0.05):
                request_session = getattr(self._app_factory, "request_session", None)
                if callable(request_session):
                    session_factory = cast(
                        Callable[[str], AbstractContextManager[RequestSessionPort]],
                        request_session,
                    )
                    with session_factory(profile_name) as session:
                        if session.revisions:
                            response = session.retrieval.search(request)
                            return SearchEnvelope(
                                session.profile.name,
                                response,
                                self._revision_views(session.revisions),
                            )
                        self._require_profile_usable(profile_name)
                        response = session.retrieval.search(request)
                        hit_vaults = sorted({hit.ref.vault_id for hit in response.hits})
                        return SearchEnvelope(
                            session.profile.name,
                            response,
                            {vault_id: self._commit_view(vault_id) for vault_id in hit_vaults},
                        )
                profile = self._public_profile(profile_name)
                self._require_profile_usable(profile_name)
                response = self._app_factory.retrieval(profile).search(request)
                hit_vaults = sorted({hit.ref.vault_id for hit in response.hits})
                return SearchEnvelope(
                    profile.name,
                    response,
                    {vault_id: self._commit_view(vault_id) for vault_id in hit_vaults},
                )
        finally:
            self._observe_search(mode, (self._timer() - started) * 1000.0)

    def read(self, profile_name: str, request: ReadRequest) -> ReadEnvelope:
        """Read source bytes with response identity from the same data view."""
        with self._lock.read(timeout=0.05):
            request_session = getattr(self._app_factory, "request_session", None)
            if callable(request_session):
                session_factory = cast(
                    Callable[[str], AbstractContextManager[RequestSessionPort]],
                    request_session,
                )
                with session_factory(profile_name) as session:
                    if session.revisions:
                        return ReadEnvelope(
                            session.profile.name,
                            session.retrieval.read(request),
                            self._revision_views(session.revisions),
                        )
                    self._require_profile_usable(profile_name)
                    return ReadEnvelope(session.profile.name, session.retrieval.read(request))
            profile = self._public_profile(profile_name)
            self._require_profile_usable(profile_name)
            return ReadEnvelope(profile.name, self._app_factory.retrieval(profile).read(request))

    def profiles(self) -> dict[str, JsonValue]:
        """List only explicitly caller-selectable service profiles."""
        configured = sorted(self._config.profiles.items())
        visible = configured[:_MAX_STATUS_VAULTS]
        return {
            "profiles": [
                {
                    "name": name[:_MAX_IDENTITY_CHARACTERS],
                    "vaults": [
                        vault_id[:_MAX_IDENTITY_CHARACTERS]
                        for vault_id in profile.vaults[:_MAX_STATUS_VAULTS]
                    ],
                    "vaults_truncated": max(0, len(profile.vaults) - _MAX_STATUS_VAULTS),
                }
                for name, profile in visible
            ],
            "profiles_truncated": len(configured) - len(visible),
        }

    def status(self, profile_name: str) -> dict[str, JsonValue]:
        """Report index and commit identity from one request-owned backend snapshot."""
        with self._lock.read(timeout=0.05):
            request_session = getattr(self._app_factory, "request_session", None)
            session_statuses = getattr(self._app_factory, "status_by_vault_from_session", None)
            if callable(request_session) and callable(session_statuses):
                session_factory = cast(
                    Callable[[str], AbstractContextManager[RequestSessionPort]],
                    request_session,
                )
                render_vault_statuses = cast(
                    Callable[
                        [RequestSessionPort, tuple[str, ...]],
                        Mapping[str, Mapping[str, JsonValue]],
                    ],
                    session_statuses,
                )
                with session_factory(profile_name) as session:
                    if session.revisions:
                        configured_vaults = self._config.profiles[profile_name].vaults
                        visible_vaults = configured_vaults[:_MAX_STATUS_VAULTS]
                        indices = render_vault_statuses(session, visible_vaults)
                        revision_vaults: dict[str, JsonValue] = {
                            vault_id: {
                                "index": self._public_index_view(indices[vault_id]),
                                "commit": self._revision_payload(session.revisions[vault_id]),
                            }
                            for vault_id in visible_vaults
                        }
                        for vault_id in visible_vaults:
                            self._observe_status(
                                vault_id,
                                indices[vault_id],
                                sync_degraded=(
                                    session.revisions[vault_id].sync_degradation_category
                                    is not None
                                ),
                            )
                            self._observe_postgres_database_state(session.store, vault_id)
                        return {
                            "profile": session.profile.name[:_MAX_IDENTITY_CHARACTERS],
                            "vaults": revision_vaults,
                            "vaults_truncated": len(configured_vaults) - len(visible_vaults),
                        }
            profile = self._public_profile(profile_name)
            configured_vaults = self._config.profiles[profile_name].vaults
            visible_vaults = configured_vaults[:_MAX_STATUS_VAULTS]
            vaults: dict[str, JsonValue] = {}
            for vault_id in visible_vaults:
                index = self._app_factory.status(self._reconciliation_profile(vault_id))
                state_read = self._state_for(vault_id)
                actual_checkout_sha = (
                    None
                    if self._state_store.checkout_independent
                    else self._actual_checkout_sha(vault_id)
                )
                vaults[vault_id] = {
                    "index": self._public_index_view(index),
                    "commit": self._commit_payload(
                        state_read,
                        actual_checkout_sha,
                        checkout_independent=self._state_store.checkout_independent,
                    ),
                }
                self._observe_status(
                    vault_id,
                    index,
                    sync_degraded=state_read.state.sync_degraded_reason is not None,
                )
            return {
                "profile": profile.name[:_MAX_IDENTITY_CHARACTERS],
                "vaults": vaults,
                "vaults_truncated": len(configured_vaults) - len(visible_vaults),
            }

    def ready(self) -> Readiness:
        """Check local state only; storage loss is unready but never process-fatal."""
        try:
            with self._lock.read(timeout=0.05):
                profile_ready = {
                    profile_name: all(self._vault_ready(vault_id) for vault_id in configured.vaults)
                    for profile_name, configured in self._config.profiles.items()
                }
        except StorageError:
            profile_ready = {profile_name: False for profile_name in self._config.profiles}
        return Readiness(any(profile_ready.values()), profile_ready)

    def request_sync(self) -> None:
        """Request a coalesced background reconciliation pass."""
        with self._lifecycle_lock:
            if not self._closed:
                self._coordinator.request_sync()

    def start(self) -> None:
        """Start background synchronization once."""
        with self._lifecycle_lock:
            if self._closed or self._started:
                return
            self._started = True
            self._coordinator.start()

    def close(self) -> None:
        """Stop synchronization before releasing factory-owned resources once."""
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
        self._coordinator.close()
        self._app_factory.close()

    def _public_profile(self, profile_name: str) -> ResolvedProfile:
        if profile_name not in self._config.profiles:
            raise ValueError(f"profile is not configured: {profile_name}")
        return self._app_factory.resolve_profile(profile_name)

    def _reconciliation_profile(self, vault_id: str) -> ResolvedProfile:
        return self._app_factory.resolve_profile(f"_reconcile_{vault_id}")

    def _state_for(self, vault_id: str) -> StateRead:
        repository = self._config.repositories[vault_id]
        return self._state_store.read(vault_id, repository.ref)

    def _actual_checkout_sha(self, vault_id: str) -> str | None:
        return self._checkout_inspector.head(vault_id)

    @staticmethod
    def _commit_safe(
        state_read: StateRead,
        actual_checkout_sha: str | None,
        *,
        checkout_independent: bool,
    ) -> bool:
        state = state_read.state
        if state_read.warning is not None or state.reconciled_sha is None:
            return False
        if checkout_independent:
            return True
        return state.checkout_sha == state.reconciled_sha == actual_checkout_sha

    def _require_profile_usable(self, profile_name: str) -> None:
        for vault_id in self._config.profiles[profile_name].vaults:
            if not self._vault_ready(vault_id):
                raise ServiceBusyError("repository reconciliation is in progress")

    def _commit_view(self, vault_id: str) -> CommitView:
        state = self._state_for(vault_id).state
        return CommitView(
            vault_id=vault_id,
            checkout_sha=state.checkout_sha,
            reconciled_sha=state.reconciled_sha,
            sync_degraded_reason=state.sync_degraded_reason,
        )

    @staticmethod
    def _revision_views(
        revisions: Mapping[str, ActiveRevision],
    ) -> Mapping[str, CommitView]:
        return MappingProxyType(
            {
                vault_id: CommitView(
                    vault_id=vault_id,
                    checkout_sha=revision.commit_sha,
                    reconciled_sha=revision.commit_sha,
                    sync_degraded_reason=revision.sync_degradation_category,
                    revision_id=revision.revision_id,
                    commit_sha=revision.commit_sha,
                )
                for vault_id, revision in revisions.items()
            }
        )

    @staticmethod
    def _revision_payload(revision: ActiveRevision) -> dict[str, JsonValue]:
        """Render only PostgreSQL identity selected by this request's active row."""

        def serialized_time(value: datetime | None) -> str | None:
            return None if value is None else value.astimezone(UTC).isoformat()

        return {
            "configured_ref": revision.configured_ref,
            "configured_sha": revision.configured_sha,
            "fetched_sha": revision.fetched_sha,
            "checkout_sha": revision.checkout_sha,
            "attempted_sha": revision.attempted_sha,
            "reconciled_sha": revision.reconciled_sha,
            "configured_at": serialized_time(revision.configured_at),
            "fetched_at": serialized_time(revision.fetched_at),
            "checkout_at": serialized_time(revision.checkout_at),
            "attempted_at": serialized_time(revision.attempted_at),
            "reconciled_at": serialized_time(revision.reconciled_at),
            "report": None,
            "sync_degraded_reason": revision.sync_degradation_category,
            "sync_degraded_at": serialized_time(revision.sync_degraded_at),
            "warning": None,
            "actual_checkout_sha": revision.checkout_sha,
            "commit_safe": revision.fully_reconciled,
            "fully_reconciled": revision.fully_reconciled,
            "state": revision.state,
            "revision_id": revision.revision_id,
            "commit_sha": revision.commit_sha,
        }

    @staticmethod
    def _commit_payload(
        state_read: StateRead,
        actual_checkout_sha: str | None,
        *,
        checkout_independent: bool,
    ) -> dict[str, JsonValue]:
        state = state_read.state
        serialized_state = state.model_dump(mode="json")
        return {
            "configured_ref": state.configured_ref,
            "configured_sha": serialized_state["configured_sha"],
            "fetched_sha": serialized_state["fetched_sha"],
            "checkout_sha": serialized_state["checkout_sha"],
            "attempted_sha": serialized_state["attempted_sha"],
            "reconciled_sha": serialized_state["reconciled_sha"],
            "configured_at": serialized_state["configured_at"],
            "fetched_at": serialized_state["fetched_at"],
            "checkout_at": serialized_state["checkout_at"],
            "attempted_at": serialized_state["attempted_at"],
            "reconciled_at": serialized_state["reconciled_at"],
            "report": (state.report.model_dump(mode="json") if state.report is not None else None),
            "sync_degraded_reason": state.sync_degraded_reason,
            "sync_degraded_at": serialized_state["sync_degraded_at"],
            "warning": state_read.warning,
            "actual_checkout_sha": actual_checkout_sha,
            "commit_safe": VaultService._commit_safe(
                state_read,
                actual_checkout_sha,
                checkout_independent=checkout_independent,
            ),
        }

    def _vault_ready(self, vault_id: str) -> bool:
        state_read = self._state_for(vault_id)
        checkout_independent = self._state_store.checkout_independent
        actual_checkout_sha = None if checkout_independent else self._actual_checkout_sha(vault_id)
        if not self._commit_safe(
            state_read,
            actual_checkout_sha,
            checkout_independent=checkout_independent,
        ):
            return False
        index = self._app_factory.status(self._reconciliation_profile(vault_id))
        schema = index.get("schema")
        counts = index.get("counts")
        source_count = counts.get("sources") if isinstance(counts, Mapping) else None
        chunk_count = counts.get("chunks") if isinstance(counts, Mapping) else None
        actual_schema_version = schema.get("actual") if isinstance(schema, Mapping) else None
        schema_usable = isinstance(schema, Mapping) and (
            schema.get("compatible") is True
            or (
                checkout_independent
                and isinstance(actual_schema_version, int)
                and not isinstance(actual_schema_version, bool)
                and actual_schema_version > 0
            )
        )
        return (
            schema_usable
            and isinstance(source_count, int)
            and not isinstance(source_count, bool)
            and source_count > 0
            and isinstance(chunk_count, int)
            and not isinstance(chunk_count, bool)
            and chunk_count > 0
            and index.get("last_successful_reconciliation") is not None
        )

    @staticmethod
    def _public_index_view(index: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        return {field: index[field] for field in _PUBLIC_INDEX_FIELDS if field in index}

    def _observe_search(self, mode: str, elapsed_ms: float) -> None:
        if self._metrics is not None:
            try:
                self._metrics.observe_search(mode, elapsed_ms)
            except Exception:
                return

    def _observe_status(
        self,
        vault_id: str,
        index: Mapping[str, JsonValue],
        *,
        sync_degraded: bool,
    ) -> None:
        if self._metrics is None:
            return
        counts = index.get("counts")
        semantic = index.get("semantic_degradation")
        semantic_degraded = (
            isinstance(semantic, Mapping) and semantic.get("semantic_search") is True
        )
        try:
            self._metrics.observe_status(
                vault_id,
                counts if isinstance(counts, Mapping) else {},
                sync_degraded=sync_degraded,
                semantic_degraded=semantic_degraded,
            )
        except Exception:
            return

    def _observe_postgres_database_state(self, store: object, vault_id: str) -> None:
        """Refresh state gauges only when a PostgreSQL status session owns the read view."""
        if self._metrics is None or not isinstance(store, PostgresStore):
            return
        try:
            state = store.database_state(vault_id)
            self._metrics.observe_database_state(
                vault_id,
                revision_count=state.revision_count,
                pending_count=state.pending_count,
                pending_age_seconds=state.pending_age_seconds,
                sync_queue_depth=state.sync_queue_depth,
                sync_queue_age_seconds=state.sync_queue_age_seconds,
                dense_mode=state.dense_mode,
            )
        except Exception:
            return
