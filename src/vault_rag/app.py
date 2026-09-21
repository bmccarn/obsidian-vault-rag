"""Application wiring and bounded phase-1 diagnostics."""

from __future__ import annotations

import os
import stat
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from types import MappingProxyType, TracebackType
from typing import Protocol, cast, runtime_checkable

import httpx  # pyright: ignore[reportMissingImports]

from vault_rag.config import (
    EgressPolicy,
    RegistryConfig,
    ResolvedProfile,
    ResolvedVault,
    load_registry,
    register_vault_outcome,
    resolve_profile,
)
from vault_rag.config.loader import load_manifest
from vault_rag.config.paths import config_home, data_home, data_home_path
from vault_rag.domain import JsonValue
from vault_rag.embedding import EmbeddingClient, configuration_fingerprint
from vault_rag.errors import (
    ConfigError,
    RebuildRequiredError,
    SemanticUnavailableError,
    StorageError,
    VaultRagError,
)
from vault_rag.indexing import Indexer
from vault_rag.ingest import discover_sources
from vault_rag.retrieval import RetrievalService
from vault_rag.storage import SCHEMA_VERSION, IndexSnapshot, SQLiteStore
from vault_rag.storage.ports import IndexStore, QueryStore, SnapshotQueryStore
from vault_rag.storage.records import ActiveRevision

_DEFAULT_CONFIG_NAME = "config.toml"
_DEFAULT_DATABASE_NAME = "index.sqlite3"
_MAX_CHECKS = 100
_MAX_CHECK_MESSAGE = 500
_MAX_IDENTITY_LENGTH = 200
_MAX_MODEL_LENGTH = 500
_MAX_PATH_LENGTH = 2_000


def _check(
    name: str,
    ok: bool,
    status_value: str,
    message: str,
    **details: JsonValue,
) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {
        "name": name,
        "ok": ok,
        "status": status_value,
        "message": " ".join(message.split())[:_MAX_CHECK_MESSAGE],
    }
    result.update(details)
    return result


@dataclass(frozen=True, slots=True)
class RequestSession:
    """All request-visible data derived from one backend read snapshot."""

    profile: ResolvedProfile
    store: QueryStore
    retrieval: RetrievalService
    snapshot: IndexSnapshot
    revisions: Mapping[str, ActiveRevision]

    def __post_init__(self) -> None:
        object.__setattr__(self, "revisions", MappingProxyType(dict(self.revisions)))


@runtime_checkable
class _RevisionProfileResolver(Protocol):
    """Resolve a profile from active records already loaded by a pinned store."""

    def resolve_from_revisions(
        self, name: str, revisions: tuple[ActiveRevision, ...]
    ) -> ResolvedProfile: ...


class AppFactory:
    """Construct profile-scoped services from one immutable environment snapshot.

    Optional transports and clocks are explicit instance dependencies so tests do
    not need global mutation or live endpoints.
    """

    def __init__(
        self,
        *,
        config_path: Path,
        database_path: Path,
        registry: RegistryConfig,
        environ: Mapping[str, str],
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        timer: Callable[[], float] = time.perf_counter,
        store_factory: Callable[[], QueryStore] | None = None,
        profile_resolver: Callable[[str], ResolvedProfile] | None = None,
    ) -> None:
        self.config_path = config_path
        self.database_path = database_path
        self.registry = registry
        self._environ = MappingProxyType(dict(environ))
        self._http_client = http_client
        self._sleep = sleep
        self._clock = clock
        self._timer = timer
        self._store_factory = store_factory
        self._profile_resolver = profile_resolver
        self._embedding_client: EmbeddingClient | None = None
        self._closed = False
        self._lifecycle_lock = RLock()

    def __repr__(self) -> str:
        """Return wiring information without environment values or endpoint URLs."""
        return f"AppFactory(endpoint_class={self.registry.embedding.endpoint_class!r})"

    @classmethod
    def from_environment(
        cls,
        config_path: Path | None,
        environ: Mapping[str, str],
        *,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        timer: Callable[[], float] = time.perf_counter,
    ) -> AppFactory:
        """Load configuration once and capture all process inputs for one invocation."""
        resolved_config = (
            config_home(environ) / _DEFAULT_CONFIG_NAME
            if config_path is None
            else Path(config_path).expanduser()
        )
        registry = load_registry(resolved_config, environ)
        database_path = data_home_path(environ) / _DEFAULT_DATABASE_NAME
        return cls(
            config_path=resolved_config,
            database_path=database_path,
            registry=registry,
            environ=environ,
            http_client=http_client,
            sleep=sleep,
            clock=clock,
            timer=timer,
        )

    def __enter__(self) -> AppFactory:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def close(self) -> None:
        """Close the factory-owned embedding client once."""
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            embedding_client = self._embedding_client
            self._embedding_client = None
        if embedding_client is not None:
            embedding_client.close()

    def resolve_profile(self, name: str) -> ResolvedProfile:
        """Resolve exactly one named or environment-selected profile."""
        self._ensure_open()
        if self._profile_resolver is not None:
            return self._profile_resolver(name)
        return resolve_profile(name, self.registry, self._environ)

    def _public_model(self, model: str) -> str:
        api_key = (
            self._environ.get(self.registry.embedding.api_key_env, "")
            if self.registry.embedding.api_key_env
            else ""
        )
        return "<redacted-model>" if api_key and model == api_key else model

    def _embedding(self) -> EmbeddingClient:
        with self._lifecycle_lock:
            self._ensure_open()
            if self._embedding_client is None:
                self._embedding_client = EmbeddingClient(
                    self.registry.embedding,
                    self._environ,
                    http_client=self._http_client,
                    sleep=self._sleep,
                )
            return self._embedding_client

    def _ensure_open(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("app factory is closed")

    def store(self, profile: ResolvedProfile) -> QueryStore:
        """Construct and initialize the configured store for a bound profile."""
        self._ensure_open()
        if not profile.vaults:
            raise ValueError("profile must bind at least one vault")
        if self._store_factory is not None:
            store = self._store_factory()
            store.initialize()
            return store
        if self.database_path == data_home_path(self._environ) / _DEFAULT_DATABASE_NAME:
            data_home(self._environ)
        sqlite_store = SQLiteStore(self.database_path)
        sqlite_store.initialize()
        # Permission diagnostics report failure without masking a usable database.
        with suppress(OSError):
            os.chmod(self.database_path, 0o600)
        return sqlite_store

    def indexer(
        self,
        profile: ResolvedProfile,
        *,
        store: IndexStore | None = None,
    ) -> Indexer:
        """Construct the reviewed indexing service for one resolved profile."""
        index_store = self._index_store(profile) if store is None else store
        return Indexer(
            profile,
            self.registry.embedding,
            self._embedding(),
            index_store,
        )

    def rebuild_indexer(self, profile: ResolvedProfile) -> Indexer:
        """Construct an indexer that can recover an unusable local database."""
        self._ensure_open()
        if self._store_factory is not None:
            raise StorageError("configured backend requires an immutable revision rebuild")
        return Indexer(
            profile,
            self.registry.embedding,
            self._embedding(),
            SQLiteStore(self.database_path),
        )

    def _index_store(self, profile: ResolvedProfile) -> IndexStore:
        store = self.store(profile)
        if not isinstance(store, IndexStore):
            raise StorageError("configured backend requires an explicit revision build")
        return store

    def _semantic_state(
        self, profile: ResolvedProfile, snapshot: IndexSnapshot
    ) -> tuple[str, str | None, str]:
        expected_config = configuration_fingerprint(
            self.registry.embedding, profile.embedding_model
        )
        if snapshot.embedding_config_fingerprint is None:
            fingerprint_state = "missing-or-mixed"
        elif snapshot.embedding_config_fingerprint != expected_config:
            fingerprint_state = "rebuild-required"
        else:
            fingerprint_state = "current"

        reason: str | None = None
        if not profile.semantic_enabled:
            reason = "semantic_disabled_by_policy"
        elif snapshot.chunk_count == 0:
            reason = "no_compatible_vectors"
        elif snapshot.indexed_at is None or fingerprint_state != "current":
            reason = "rebuild_required"
        elif snapshot.pending_count:
            reason = "pending_embeddings"
        elif len(snapshot.observed_fingerprints) > 1:
            reason = "ambiguous_vectors"
        elif not snapshot.observed_fingerprints or not snapshot.vector_dimensions:
            reason = "no_compatible_vectors"
        elif snapshot.vector_config_fingerprints != (expected_config,):
            reason = "rebuild_required"
        observed = (
            snapshot.observed_fingerprints[0]
            if reason in {None, "pending_embeddings"} and len(snapshot.observed_fingerprints) == 1
            else "unavailable-observed-fingerprint"
        )
        return observed, reason, fingerprint_state

    def _retrieval_from_snapshot(
        self,
        profile: ResolvedProfile,
        store: QueryStore,
        snapshot: IndexSnapshot,
    ) -> RetrievalService:
        observed, reason, _fingerprint_state = self._semantic_state(profile, snapshot)
        retrieval_reason = reason if reason not in {None, "pending_embeddings"} else None
        return RetrievalService(
            profile,
            store,
            self._embedding(),
            observed,
            observed_vector_dimensions=snapshot.vector_dimensions,
            semantic_unavailable_reason=retrieval_reason,
            coverage_degraded_reason=(
                "pending_embeddings" if reason == "pending_embeddings" else None
            ),
            clock=self._clock,
            timer=self._timer,
        )

    def retrieval(self, profile: ResolvedProfile) -> RetrievalService:
        """Construct the reviewed retrieval service for one resolved profile."""
        store = self.store(profile)
        snapshot = store.snapshot(tuple(vault.manifest.id for vault in profile.vaults))
        return self._retrieval_from_snapshot(profile, store, snapshot)

    def _configured_profile_vault_ids(self, profile_name: str) -> tuple[str, tuple[str, ...]]:
        selected_name = profile_name or self._environ.get("VAULT_RAG_PROFILE", "")
        if not selected_name:
            raise ConfigError("profile is required")
        try:
            configured = self.registry.profiles[selected_name]
        except KeyError as exc:
            raise ConfigError(f"profile is not configured: {selected_name}") from exc
        if len(set(configured.vaults)) != len(configured.vaults):
            raise StorageError("configured PostgreSQL profile contains duplicate vault IDs")
        return selected_name, configured.vaults

    @staticmethod
    def _active_revision_map(
        vault_ids: tuple[str, ...], revisions: tuple[ActiveRevision, ...]
    ) -> Mapping[str, ActiveRevision]:
        expected = set(vault_ids)
        resolved: dict[str, ActiveRevision] = {}
        for revision in revisions:
            if (
                revision.vault_id not in expected
                or revision.vault_id in resolved
                or revision.manifest.get("id") != revision.vault_id
                or revision.state not in {"active", "active_degraded"}
            ):
                raise StorageError("active PostgreSQL revision set is invalid")
            resolved[revision.vault_id] = revision
        if set(resolved) != expected:
            raise StorageError("configured PostgreSQL profile has no complete active revision set")
        return MappingProxyType(resolved)

    @contextmanager
    def request_session(self, profile_name: str) -> Generator[RequestSession]:
        """Yield one request's profile, data, and identity from one stable read."""
        self._ensure_open()
        selected_name, configured_vault_ids = self._configured_profile_vault_ids(profile_name)
        if self._store_factory is None:
            profile = self.resolve_profile(selected_name)
            store = self.store(profile)
            snapshot = store.snapshot(tuple(vault.manifest.id for vault in profile.vaults))
            yield RequestSession(
                profile,
                store,
                self._retrieval_from_snapshot(profile, store, snapshot),
                snapshot,
                {},
            )
            return

        candidate = self._store_factory()
        if not isinstance(candidate, SnapshotQueryStore):
            profile = self.resolve_profile(selected_name)
            store = self.store(profile)
            snapshot = store.snapshot(tuple(vault.manifest.id for vault in profile.vaults))
            yield RequestSession(
                profile,
                store,
                self._retrieval_from_snapshot(profile, store, snapshot),
                snapshot,
                {},
            )
            return
        resolver = self._profile_resolver
        if not isinstance(resolver, _RevisionProfileResolver):
            raise StorageError("PostgreSQL request resolution requires active revision manifests")
        with candidate.consistent_read() as pinned:
            if not isinstance(pinned, SnapshotQueryStore):
                raise StorageError("PostgreSQL consistent read did not retain revision access")
            revisions = self._active_revision_map(
                configured_vault_ids, pinned.active_revisions(configured_vault_ids)
            )
            profile = resolver.resolve_from_revisions(selected_name, tuple(revisions.values()))
            if tuple(vault.manifest.id for vault in profile.vaults) != configured_vault_ids:
                raise StorageError(
                    "PostgreSQL profile resolution did not preserve configured vaults"
                )
            snapshot = pinned.snapshot(configured_vault_ids)
            yield RequestSession(
                profile,
                pinned,
                self._retrieval_from_snapshot(profile, pinned, snapshot),
                snapshot,
                revisions,
            )

    def register(self, root: Path, *, replace: bool = False) -> dict[str, JsonValue]:
        """Invoke atomic registry registration and return its bounded public identity."""
        outcome = register_vault_outcome(root, self.config_path, replace=replace)
        return {"vault_id": outcome.vault_id, "path": str(outcome.path)}

    def profile_list(self) -> dict[str, JsonValue]:
        """List bounded registry names without resolving models or credentials."""
        ordered = sorted(self.registry.profiles.items())
        visible = ordered[:_MAX_CHECKS]
        profiles: list[JsonValue] = []
        for name, profile in visible:
            vaults: list[JsonValue] = [
                vault_id[:_MAX_IDENTITY_LENGTH] for vault_id in profile.vaults[:_MAX_CHECKS]
            ]
            profiles.append(
                {
                    "name": name[:_MAX_IDENTITY_LENGTH],
                    "vaults": vaults,
                    "vaults_truncated": len(profile.vaults) - len(vaults),
                }
            )
        return {
            "profiles": profiles,
            "profiles_truncated": len(ordered) - len(visible),
        }

    def status(
        self,
        profile: ResolvedProfile,
        *,
        store: QueryStore | None = None,
        snapshot: IndexSnapshot | None = None,
    ) -> dict[str, JsonValue]:
        """Return a bounded, secret-free snapshot for one resolved profile."""
        active_store = self.store(profile) if store is None else store
        vault_ids = tuple(vault.manifest.id for vault in profile.vaults)
        active_snapshot = active_store.snapshot(vault_ids) if snapshot is None else snapshot
        snapshot = active_snapshot
        _observed, semantic_reason, fingerprint_state = self._semantic_state(profile, snapshot)
        indexed_at = snapshot.indexed_at
        index_age = (
            None if indexed_at is None else max(0.0, (self._clock() - indexed_at).total_seconds())
        )
        observed: list[JsonValue] = list(snapshot.observed_fingerprints[:10])
        schema: dict[str, JsonValue] = {
            "actual": snapshot.schema_version,
            "expected": SCHEMA_VERSION,
            "compatible": snapshot.schema_version == SCHEMA_VERSION,
        }
        fingerprints: dict[str, JsonValue] = {
            "state": fingerprint_state,
            "manifest": snapshot.manifest_fingerprint,
            "parser": snapshot.parser_fingerprint,
            "chunker": snapshot.chunker_fingerprint,
            "embedding_config": snapshot.embedding_config_fingerprint,
            "observed": observed,
            "observed_truncated": len(snapshot.observed_fingerprints) - len(observed),
        }
        counts: dict[str, JsonValue] = {
            "sources": snapshot.source_count,
            "chunks": snapshot.chunk_count,
            "ready": snapshot.ready_count,
            "pending": snapshot.pending_count,
        }
        semantic_degradation: dict[str, JsonValue] = {
            "semantic_search": semantic_reason is not None,
            "reason": semantic_reason,
        }
        visible_vault_ids: list[JsonValue] = [
            vault_id[:_MAX_IDENTITY_LENGTH] for vault_id in vault_ids[:_MAX_CHECKS]
        ]
        return {
            "profile": profile.name[:_MAX_IDENTITY_LENGTH],
            "vault_ids": visible_vault_ids,
            "vault_ids_truncated": len(vault_ids) - len(visible_vault_ids),
            "effective_egress_policy": profile.effective_policy.value,
            "model": self._public_model(profile.embedding_model)[:_MAX_MODEL_LENGTH],
            "endpoint_class": self.registry.embedding.endpoint_class,
            "embedding_revision": self.registry.embedding.revision,
            "requested_dimensions": self.registry.embedding.dimensions,
            "max_batch_tokens": self.registry.embedding.max_batch_tokens,
            "database_path": active_store.database_location[:_MAX_PATH_LENGTH],
            "schema": schema,
            "fingerprints": fingerprints,
            "counts": counts,
            "pending": snapshot.pending_count > 0,
            "last_successful_reconciliation": (
                None if indexed_at is None else indexed_at.astimezone(UTC).isoformat()
            ),
            "index_age_seconds": index_age,
            "semantic_degradation": semantic_degradation,
        }

    def status_from_session(self, session: RequestSession) -> dict[str, JsonValue]:
        """Render status from an already pinned request session."""
        return self.status(session.profile, store=session.store, snapshot=session.snapshot)

    def _single_vault_profile(self, profile: ResolvedProfile, vault_id: str) -> ResolvedProfile:
        vault = next(vault for vault in profile.vaults if vault.manifest.id == vault_id)
        policy = vault.manifest.egress_policy
        semantic_enabled = not (
            policy is EgressPolicy.LOCAL_ONLY and self.registry.embedding.endpoint_class == "remote"
        )
        return profile.model_copy(
            update={
                "vaults": (vault,),
                "effective_policy": policy,
                "semantic_enabled": semantic_enabled,
                "semantic_disabled_reason": (
                    None if semantic_enabled else "remote endpoint prohibited by local-only vault"
                ),
            }
        )

    def status_by_vault_from_session(
        self, session: RequestSession, vault_ids: tuple[str, ...] | None = None
    ) -> Mapping[str, dict[str, JsonValue]]:
        """Render requested vault status through the pinned store in a batch."""
        requested = tuple(session.revisions) if vault_ids is None else vault_ids
        if not requested or any(vault_id not in session.revisions for vault_id in requested):
            raise StorageError("requested PostgreSQL status vault is outside the request session")
        if not isinstance(session.store, SnapshotQueryStore):
            raise StorageError("PostgreSQL session lost batched snapshot access")
        snapshots = session.store.snapshots(requested)
        return MappingProxyType(
            {
                vault_id: self.status(
                    self._single_vault_profile(session.profile, vault_id),
                    store=session.store,
                    snapshot=snapshots[vault_id],
                )
                for vault_id in requested
            }
        )

    def doctor(self, profile_name: str) -> dict[str, JsonValue]:
        """Diagnose one configured profile without pre-emptive full resolution."""
        selected_name = profile_name or self._environ.get("VAULT_RAG_PROFILE", "")
        if not selected_name:
            raise ConfigError("profile is required")
        try:
            configured_profile = self.registry.profiles[selected_name]
        except KeyError as exc:
            raise ConfigError(f"profile is not configured: {selected_name}") from exc

        checks: list[dict[str, JsonValue]] = []
        try:
            config_stat = self.config_path.stat()
            config_mode = stat.S_IMODE(config_stat.st_mode)
            config_secure = config_mode & 0o077 == 0 and config_stat.st_uid == os.getuid()
            checks.append(
                _check(
                    "config_permissions",
                    config_secure,
                    "ok" if config_secure else "insecure",
                    "configuration is user-only"
                    if config_secure
                    else "configuration is accessible by group or other users",
                    mode=f"{config_mode:04o}",
                )
            )
        except OSError:
            checks.append(
                _check(
                    "config_permissions",
                    False,
                    "unavailable",
                    "configuration permissions could not be inspected",
                )
            )

        resolved_vaults: list[ResolvedVault] = []
        roots_ok = True
        manifests_ok = True
        for vault_id in configured_profile.vaults:
            registration = self.registry.vaults.get(vault_id)
            if registration is None:
                roots_ok = False
                manifests_ok = False
                continue
            try:
                root = registration.path.expanduser().resolve(strict=True)
            except (OSError, RuntimeError):
                roots_ok = False
                manifests_ok = False
                continue
            if not root.is_dir() or not os.access(root, os.R_OK | os.X_OK):
                roots_ok = False
                manifests_ok = False
                continue
            try:
                manifest = load_manifest(root / ".vault-rag.toml")
            except ConfigError:
                manifests_ok = False
                continue
            if manifest.id != vault_id:
                manifests_ok = False
                continue
            resolved_vaults.append(ResolvedVault(root=root, manifest=manifest))

        all_manifests_valid = manifests_ok and len(resolved_vaults) == len(
            configured_profile.vaults
        )
        checks.append(
            _check(
                "vault_roots",
                roots_ok,
                "ok" if roots_ok else "unreadable-or-missing",
                "all bound vault roots are readable"
                if roots_ok
                else "one or more bound vault roots are unreadable, missing, or unregistered",
                vault_count=len(configured_profile.vaults),
            )
        )
        checks.append(
            _check(
                "manifests",
                all_manifests_valid,
                "ok" if all_manifests_valid else "invalid",
                "all bound vault manifests are valid and match their registered IDs"
                if all_manifests_valid
                else (
                    "one or more bound vault manifests are invalid, unavailable, "
                    "or have an ID mismatch"
                ),
                manifest_count=len(resolved_vaults),
            )
        )

        collisions_ok = all_manifests_valid
        collision_message = (
            "no manifest-selected path collisions were found"
            if collisions_ok
            else "path collision checks are blocked by invalid roots or manifests"
        )
        if collisions_ok:
            try:
                for vault in resolved_vaults:
                    folded_paths: set[str] = set()
                    for source in discover_sources(vault).sources:
                        if source.folded_path in folded_paths:
                            collisions_ok = False
                            collision_message = "manifest-selected paths collide after case folding"
                            break
                        folded_paths.add(source.folded_path)
            except VaultRagError:
                collisions_ok = False
                collision_message = "manifest-selected paths could not be safely inspected"
            except ValueError:
                collisions_ok = False
                collision_message = "manifest path rules could not be evaluated"
        checks.append(
            _check(
                "path_collisions",
                collisions_ok,
                "ok" if collisions_ok else "failed",
                collision_message,
            )
        )

        store: SQLiteStore | None = None
        snapshot: IndexSnapshot | None = None
        fts_ok = False
        fts_status = "unavailable"
        fts_message = "SQLite index setup is unavailable"
        data_ready = False
        try:
            data_home(self._environ)
            data_ready = True
            existed = self.database_path.exists()
            store = SQLiteStore(self.database_path)
            store.initialize()
            if not existed:
                os.chmod(self.database_path, 0o600)
            fts_ok = store.fts5_available()
            fts_status = "ok" if fts_ok else "unavailable"
            fts_message = "SQLite FTS5 is available" if fts_ok else "SQLite FTS5 is unavailable"
            if fts_ok:
                snapshot = store.snapshot(tuple(configured_profile.vaults))
        except RebuildRequiredError:
            fts_status = "rebuild-required"
            fts_message = "SQLite index schema requires a rebuild"
        except (ConfigError, StorageError, OSError):
            fts_status = "unavailable"
            fts_message = "SQLite index or data-directory setup is unavailable"
        checks.append(_check("fts5", fts_ok, fts_status, fts_message))

        database_ok = False
        if data_ready:
            try:
                database_stat = self.database_path.stat()
                database_ok = (
                    os.access(self.database_path, os.R_OK | os.W_OK)
                    and stat.S_IMODE(database_stat.st_mode) & 0o077 == 0
                    and database_stat.st_uid == os.getuid()
                )
            except OSError:
                database_ok = False
        checks.append(
            _check(
                "database_permissions",
                database_ok,
                "ok" if database_ok else "insecure-or-unavailable",
                "database is readable, writable, and user-only"
                if database_ok
                else "database or data directory is unavailable or not user-only read/write",
                path=str(self.database_path)[:_MAX_PATH_LENGTH],
            )
        )

        all_vaults_valid = len(resolved_vaults) == len(configured_profile.vaults)
        effective_policy = (
            EgressPolicy.LOCAL_ONLY
            if any(
                vault.manifest.egress_policy is EgressPolicy.LOCAL_ONLY for vault in resolved_vaults
            )
            else EgressPolicy.REMOTE_ALLOWED
        )
        model = self._environ.get(self.registry.embedding.model_env, "")
        api_key_missing = bool(
            self.registry.embedding.api_key_env
            and not self._environ.get(self.registry.embedding.api_key_env)
        )
        policy_disabled = (
            all_vaults_valid
            and effective_policy is EgressPolicy.LOCAL_ONLY
            and self.registry.embedding.endpoint_class == "remote"
        )
        if not all_vaults_valid:
            checks.append(
                _check(
                    "model",
                    False,
                    "blocked",
                    "semantic diagnostics are blocked by invalid roots or manifests",
                    semantic_available=False,
                    dimensions=None,
                    model=self._public_model(model)[:_MAX_MODEL_LENGTH],
                    category="configuration",
                )
            )
        elif policy_disabled:
            checks.append(
                _check(
                    "model",
                    True,
                    "policy-disabled",
                    "semantic connectivity is disabled by effective egress policy",
                    semantic_available=False,
                    dimensions=None,
                    model=self._public_model(model)[:_MAX_MODEL_LENGTH],
                    category="policy",
                )
            )
        elif not model or api_key_missing:
            checks.append(
                _check(
                    "model",
                    False,
                    "semantic-unavailable",
                    "embedding model or API-key environment configuration is unavailable",
                    semantic_available=False,
                    dimensions=None,
                    model=self._public_model(model)[:_MAX_MODEL_LENGTH],
                    category="configuration",
                )
            )
        elif snapshot is None or store is None:
            checks.append(
                _check(
                    "model",
                    False,
                    "index-unavailable",
                    "active index state is unavailable for semantic diagnostics",
                    semantic_available=False,
                    dimensions=None,
                    model=self._public_model(model)[:_MAX_MODEL_LENGTH],
                    category="storage",
                )
            )
        else:
            client = self._embedding()
            try:
                probe = client.probe()
            except SemanticUnavailableError as exc:
                category = str(exc.details.get("category", "unavailable"))[:64]
                checks.append(
                    _check(
                        "model",
                        False,
                        "semantic-unavailable",
                        "embedding model connectivity or dimensions are unavailable",
                        semantic_available=False,
                        dimensions=None,
                        model=self._public_model(model)[:_MAX_MODEL_LENGTH],
                        category=category,
                    )
                )
            else:
                expected_config = configuration_fingerprint(self.registry.embedding, model)
                if snapshot.chunk_count > 0 and (
                    snapshot.indexed_at is None
                    or snapshot.embedding_config_fingerprint != expected_config
                    or (
                        snapshot.vector_config_fingerprints
                        and snapshot.vector_config_fingerprints != (expected_config,)
                    )
                ):
                    status_value = "rebuild-required"
                    message = "active vectors were built with a different embedding configuration"
                    ok = False
                elif snapshot.pending_count:
                    status_value = "pending-embeddings"
                    message = "semantic coverage is incomplete because embeddings are pending"
                    ok = False
                elif snapshot.chunk_count == 0 or not snapshot.vector_dimensions:
                    status_value = "no-compatible-vectors"
                    message = "no compatible active vectors are available"
                    ok = False
                elif len(snapshot.observed_fingerprints) != 1:
                    status_value = "ambiguous-vectors"
                    message = "active vectors have missing or ambiguous observed fingerprints"
                    ok = False
                elif snapshot.vector_dimensions != (probe.dimensions,):
                    status_value = "dimension-mismatch"
                    message = "embedding model dimensions do not match active stored vectors"
                    ok = False
                else:
                    status_value = "ok"
                    message = "embedding model and active vector dimensions are compatible"
                    ok = True
                checks.append(
                    _check(
                        "model",
                        ok,
                        status_value,
                        message,
                        semantic_available=ok,
                        dimensions=probe.dimensions,
                        stored_dimensions=list(snapshot.vector_dimensions),
                        model=self._public_model(probe.model)[:_MAX_MODEL_LENGTH],
                        category=None,
                    )
                )

        for check in checks:
            if check["name"] == "model":
                check["revision"] = self.registry.embedding.revision
                check["requested_dimensions"] = self.registry.embedding.dimensions

        visible = checks[:_MAX_CHECKS]
        return {
            "profile": selected_name[:_MAX_IDENTITY_LENGTH],
            "healthy": all(cast(bool, item["ok"]) for item in visible),
            "checks": cast(list[JsonValue], visible),
            "checks_truncated": len(checks) - len(visible),
        }
