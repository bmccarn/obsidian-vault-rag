"""Resolve retrieval profiles from manifests in active PostgreSQL revisions."""

from __future__ import annotations

from collections.abc import Mapping

from psycopg import Error  # pyright: ignore[reportMissingImports]
from pydantic import ValidationError

from vault_rag.config import EgressPolicy, RegistryConfig, ResolvedProfile, ResolvedVault
from vault_rag.config.models import VaultManifest
from vault_rag.errors import ConfigError, StorageError
from vault_rag.storage.postgres.pool import PostgresPool
from vault_rag.storage.records import ActiveRevision


class PostgresProfileResolver:
    """Build immutable profiles without requiring API-local Git checkouts."""

    def __init__(
        self,
        pool: PostgresPool,
        registry: RegistryConfig,
        environ: Mapping[str, str],
    ) -> None:
        self._pool = pool
        self._registry = registry
        self._environ = dict(environ)

    def __call__(self, name: str) -> ResolvedProfile:
        profile_name, configured_vault_ids = self._configured_profile(name)
        return self._resolve(
            profile_name, configured_vault_ids, self._load_manifests(configured_vault_ids)
        )

    def resolve_from_revisions(
        self, name: str, revisions: tuple[ActiveRevision, ...]
    ) -> ResolvedProfile:
        """Resolve a profile using manifests already read by its pinned store."""
        profile_name, configured_vault_ids = self._configured_profile(name)
        manifests: dict[str, VaultManifest] = {}
        try:
            for revision in revisions:
                if revision.vault_id in manifests:
                    raise StorageError(
                        "active PostgreSQL revision set contains duplicate vault IDs"
                    )
                manifest = VaultManifest.model_validate(revision.manifest)
                if manifest.id != revision.vault_id:
                    raise StorageError(
                        "active PostgreSQL manifest identity does not match its revision vault"
                    )
                manifests[revision.vault_id] = manifest
        except ValidationError as exc:
            raise StorageError("active PostgreSQL manifest is invalid") from exc
        if set(manifests) != set(configured_vault_ids):
            raise StorageError("configured PostgreSQL profile has no complete active revision set")
        return self._resolve(profile_name, configured_vault_ids, manifests)

    def _configured_profile(self, name: str) -> tuple[str, tuple[str, ...]]:
        profile_name = name or self._environ.get("VAULT_RAG_PROFILE", "")
        if not profile_name:
            raise ConfigError("profile is required")
        try:
            configured = self._registry.profiles[profile_name]
        except KeyError as exc:
            raise ConfigError(f"profile is not configured: {profile_name}") from exc
        return profile_name, configured.vaults

    def _resolve(
        self,
        profile_name: str,
        vault_ids: tuple[str, ...],
        manifests: Mapping[str, VaultManifest],
    ) -> ResolvedProfile:
        resolved_vaults: list[ResolvedVault] = []
        for vault_id in vault_ids:
            manifest = manifests.get(vault_id)
            if manifest is None:
                raise ConfigError(f"vault has no active database manifest: {vault_id}")
            if manifest.id != vault_id:
                raise ConfigError(f"active database manifest identity does not match {vault_id}")
            registration = self._registry.vaults[vault_id]
            resolved_vaults.append(
                ResolvedVault(root=registration.path.expanduser().absolute(), manifest=manifest)
            )

        model = self._environ.get(self._registry.embedding.model_env)
        if not model:
            raise ConfigError(
                "embedding model environment variable is required: "
                f"{self._registry.embedding.model_env}"
            )
        effective_policy = (
            EgressPolicy.LOCAL_ONLY
            if any(
                vault.manifest.egress_policy is EgressPolicy.LOCAL_ONLY for vault in resolved_vaults
            )
            else EgressPolicy.REMOTE_ALLOWED
        )
        disabled_reason = None
        if (
            effective_policy is EgressPolicy.LOCAL_ONLY
            and self._registry.embedding.endpoint_class == "remote"
        ):
            disabled_reason = "remote endpoint prohibited by local-only vault"

        return ResolvedProfile(
            name=profile_name,
            vaults=tuple(resolved_vaults),
            embedding_model=model,
            api_key_env=self._registry.embedding.api_key_env,
            effective_policy=effective_policy,
            semantic_enabled=disabled_reason is None,
            semantic_disabled_reason=disabled_reason,
        )

    def _load_manifests(self, vault_ids: tuple[str, ...]) -> dict[str, VaultManifest]:
        try:
            with self._pool.connection() as connection:
                rows = connection.execute(
                    """
                    SELECT vault.vault_id, revision.manifest_json
                    FROM vault_rag.vaults AS vault
                    JOIN vault_rag.vault_revisions AS revision
                      ON revision.revision_id = vault.active_revision_id
                    WHERE vault.vault_id = ANY(%s)
                    """,
                    (list(vault_ids),),
                ).fetchall()
        except Error as exc:
            raise StorageError("could not load active PostgreSQL manifests") from exc

        manifests: dict[str, VaultManifest] = {}
        try:
            for row in rows:
                if row["manifest_json"] is not None:
                    manifests[str(row["vault_id"])] = VaultManifest.model_validate(
                        row["manifest_json"]
                    )
        except ValidationError as exc:
            raise StorageError("active PostgreSQL manifest is invalid") from exc
        return manifests
