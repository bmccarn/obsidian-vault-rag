from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from vault_rag.config import EgressPolicy, ResolvedProfile, ResolvedVault, VaultManifest
from vault_rag.service.config import ServiceConfig
from vault_rag.service.facade import VaultService
from vault_rag.service.locking import LifecycleLock
from vault_rag.storage.records import ActiveRevision


@dataclass
class _Coordinator:
    def start(self) -> None:
        pass

    def request_sync(self) -> None:
        pass

    def close(self) -> None:
        pass

    def head(self, vault_id: str) -> str | None:
        del vault_id
        return None


@dataclass
class _State:
    checkout_independent: bool = True


@dataclass
class _Session:
    profile: ResolvedProfile
    revisions: Mapping[str, ActiveRevision]
    store: object


class _Factory:
    def __init__(self, session: _Session) -> None:
        self.session = session
        self.calls: list[tuple[str, ...]] = []

    @contextmanager
    def request_session(self, profile_name: str) -> Generator[_Session, None, None]:
        assert profile_name == "public"
        yield self.session

    def status_by_vault_from_session(
        self, session: _Session, vault_ids: tuple[str, ...]
    ) -> Mapping[str, Mapping[str, object]]:
        assert session is self.session
        self.calls.append(vault_ids)
        return {
            vault_id: {
                "schema": {"compatible": True},
                "counts": {"sources": 1, "chunks": 1, "ready": 1, "pending": 0},
                "semantic_degradation": {"semantic_search": False},
            }
            for vault_id in vault_ids
        }

    def close(self) -> None:
        pass


def test_session_status_limits_batched_snapshot_request_to_visible_vaults() -> None:
    """Status must not ask a pinned store for unrendered configured vaults."""
    vault_ids = tuple(f"vault-{index}" for index in range(101))

    def manifest(vault_id: str) -> VaultManifest:
        return VaultManifest(
            schema_version=1,
            id=vault_id,
            egress_policy=EgressPolicy.LOCAL_ONLY,
            include=("**/*.md",),
        )

    profile = ResolvedProfile(
        name="public",
        vaults=tuple(
            ResolvedVault(root=Path("/unused") / vault_id, manifest=manifest(vault_id))
            for vault_id in vault_ids
        ),
        embedding_model="model",
        api_key_env=None,
        effective_policy=EgressPolicy.LOCAL_ONLY,
        semantic_enabled=True,
    )
    revisions = {
        vault_id: ActiveRevision(
            vault_id=vault_id,
            revision_id=str(index),
            commit_sha="a" * 40,
            manifest=manifest(vault_id).model_dump(mode="json"),
            state="active",
            source_count=1,
            chunk_count=1,
            ready_count=1,
            pending_count=0,
            vector_bytes=1,
            manifest_fingerprint="m",
            parser_fingerprint="p",
            chunker_fingerprint="c",
            embedding_config_fingerprint="e",
            promoted_at=datetime.now(UTC),
            configured_ref="main",
        )
        for index, vault_id in enumerate(vault_ids)
    }
    factory = _Factory(_Session(profile, revisions, object()))
    config = ServiceConfig.model_validate(
        {
            "schemaVersion": 1,
            "syncInterval": "1h",
            "repositories": {
                vault_id: {"url": f"https://example.test/{vault_id}.git", "ref": "refs/heads/main"}
                for vault_id in vault_ids
            },
            "profiles": {"public": {"vaults": vault_ids}},
            "embedding": {
                "baseUrl": "http://127.0.0.1:9000/v1",
                "modelEnv": "MODEL",
                "endpointClass": "local",
            },
        }
    )
    service = VaultService(
        config,
        cast(Any, factory),
        _Coordinator(),
        _Coordinator(),
        cast(Any, _State()),
        LifecycleLock(),
    )

    response = service.status("public")

    assert factory.calls == [vault_ids[:100]]
    assert response["vaults_truncated"] == 1
    assert len(cast(Mapping[str, Any], response["vaults"])) == 100
