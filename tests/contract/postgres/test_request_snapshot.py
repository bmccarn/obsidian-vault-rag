from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from pydantic import HttpUrl

from vault_rag.app import AppFactory, RequestSession
from vault_rag.config import EgressPolicy, EmbeddingConfig, RegistryConfig, VaultManifest
from vault_rag.config.models import ProfileConfig, RegistryVault
from vault_rag.domain import JsonValue
from vault_rag.retrieval import ReadRequest, SearchMode, SearchRequest
from vault_rag.service.config import ServiceConfig
from vault_rag.service.facade import VaultService
from vault_rag.service.http import create_app
from vault_rag.service.locking import LifecycleLock
from vault_rag.service.observability import ServiceMetrics
from vault_rag.service.postgres_profile import PostgresProfileResolver
from vault_rag.service.state import StateRead, VaultSyncState
from vault_rag.storage.ports import RevisionBuildStore
from vault_rag.storage.postgres import (
    PostgresMigrator,
    PostgresPool,
    PostgresStore,
    PostgresWorkerLease,
)

from .helpers import revision


def _promote(pool: PostgresPool, build: RevisionBuildStore) -> None:
    lease = PostgresWorkerLease(pool, "sync:vault-a")
    assert lease.acquire() is True
    try:
        lease.promote(build, lexical_complete=True, fully_reconciled=True)
    finally:
        lease.release()


class ReadOnlyQueryStoreSpy(PostgresStore):
    """Keep PostgreSQL API requests on query-only methods at every read boundary."""

    _WRITERS = frozenset(
        {
            "begin_revision",
            "replace_source",
            "record_vault_fingerprint",
            "update_pending_embeddings",
            "reset_vaults",
            "requeue_disabled_chunks",
            "suppress_source",
            "delete_sources",
            "promote",
            "fail",
        }
    )

    def __getattribute__(self, name: str) -> Any:
        if name in type(self)._WRITERS:
            raise AssertionError(f"PostgreSQL API attempted writer-shaped method {name}")
        return super().__getattribute__(name)


def test_request_session_pins_active_revision_profile_and_snapshot(
    tmp_path: Path, postgres_pool: PostgresPool
) -> None:
    """A promotion after resolution must not alter any request-owned identity."""
    PostgresMigrator(postgres_pool).apply()
    manifest = VaultManifest(
        schema_version=1,
        id="vault-a",
        egress_policy=EgressPolicy.LOCAL_ONLY,
        include=("**/*.md",),
    )
    store = PostgresStore(postgres_pool)
    build = store.begin_revision(
        "vault-a", "refs/heads/main", "a" * 40, manifest=manifest.model_dump(mode="json")
    )
    build.replace_source(revision("old snapshot bytes"))
    _promote(postgres_pool, build)
    registry = RegistryConfig(
        vaults={"vault-a": RegistryVault(path=tmp_path / "never-read")},
        profiles={"public": ProfileConfig(vaults=("vault-a",))},
        embedding=EmbeddingConfig(
            base_url=HttpUrl("http://127.0.0.1:9000/v1"),
            model_env="MODEL",
            endpoint_class="local",
        ),
    )
    resolver = PostgresProfileResolver(postgres_pool, registry, {"MODEL": "model"})
    factory = AppFactory(
        config_path=tmp_path / "config.toml",
        database_path=tmp_path / "must-not-exist.sqlite3",
        registry=registry,
        environ={"MODEL": "model"},
        store_factory=lambda: PostgresStore(postgres_pool),
        profile_resolver=resolver,
    )

    with factory.request_session("public") as session:
        assert session.profile.vaults[0].manifest.id == "vault-a"
        assert session.profile.effective_policy is EgressPolicy.LOCAL_ONLY
        assert session.revisions["vault-a"].commit_sha == "a" * 40
        assert session.snapshot.source_count == 1
        assert (
            session.retrieval.search(SearchRequest("old", mode=SearchMode.LEXICAL)).hits[0].text
            == "old snapshot bytes"
        )
        assert session.retrieval.read(ReadRequest("notes/example.md")).text == "old snapshot bytes"

        replacement_manifest = manifest.model_copy(
            update={"egress_policy": EgressPolicy.REMOTE_ALLOWED}
        )
        replacement = store.begin_revision(
            "vault-a",
            "refs/heads/main",
            "b" * 40,
            manifest=replacement_manifest.model_dump(mode="json"),
        )
        replacement.replace_source(revision("new snapshot bytes"))
        _promote(postgres_pool, replacement)

        assert session.store.snapshot(("vault-a",)).source_count == 1
        status = factory.status_from_session(session)
        counts = status["counts"]
        assert isinstance(counts, dict)
        assert counts["sources"] == 1
        assert (
            session.retrieval.search(SearchRequest("old", mode=SearchMode.LEXICAL)).hits[0].text
            == "old snapshot bytes"
        )
        assert session.retrieval.read(ReadRequest("notes/example.md")).text == "old snapshot bytes"
        assert session.revisions["vault-a"].commit_sha == "a" * 40

    with factory.request_session("public") as next_session:
        assert next_session.profile.effective_policy is EgressPolicy.REMOTE_ALLOWED
        assert next_session.revisions["vault-a"].commit_sha == "b" * 40
        assert (
            next_session.retrieval.search(SearchRequest("new", mode=SearchMode.LEXICAL))
            .hits[0]
            .text
            == "new snapshot bytes"
        )

    factory.close()


@dataclass
class _NoopCoordinator:
    def start(self) -> None:
        return None

    def request_sync(self) -> None:
        return None

    def close(self) -> None:
        return None

    def head(self, vault_id: str) -> str | None:
        del vault_id
        return None


class _SessionOnlyState:
    checkout_independent = True

    def read(self, vault_id: str, configured_ref: str) -> StateRead:
        return StateRead(state=VaultSyncState(vault_id=vault_id, configured_ref=configured_ref))

    def write(self, state: VaultSyncState) -> None:
        del state


class _Events:
    def emit(self, event: str, **bounded_fields: JsonValue) -> None:
        del event, bounded_fields


@dataclass
class _HttpRuntime:
    service: VaultService
    metrics: ServiceMetrics
    events: _Events

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_two_independent_postgres_api_pools_render_per_vault_identity(
    tmp_path: Path, postgres_pool: PostgresPool
) -> None:
    """Independent API pools must expose one pinned, per-vault active view."""
    PostgresMigrator(postgres_pool).apply()
    manifests = {
        vault_id: VaultManifest(
            schema_version=1,
            id=vault_id,
            egress_policy=EgressPolicy.LOCAL_ONLY,
            include=("**/*.md",),
        )
        for vault_id in ("vault-a", "vault-b")
    }
    writer = PostgresStore(postgres_pool)
    for vault_id, text, commit in (
        ("vault-a", "alpha only", "a" * 40),
        ("vault-b", "bravo one\nbravo two", "b" * 40),
    ):
        build = writer.begin_revision(
            vault_id,
            "refs/heads/main",
            commit,
            manifest=manifests[vault_id].model_dump(mode="json"),
        )
        if vault_id == "vault-b":
            pending_source = revision(
                "bravo pending",
                vault_id=vault_id,
                path="notes/pending.md",
                vector_values=None,
                pending=True,
            )
            build.replace_source(
                replace(
                    pending_source,
                    chunks=(replace(pending_source.chunks[0], id="chunk-b"),),
                    pending=(replace(pending_source.pending[0], chunk_id="chunk-b"),),
                )
            )
        build.replace_source(revision(text, vault_id=vault_id))
        _promote(postgres_pool, build)
    registry = RegistryConfig(
        vaults={
            vault_id: RegistryVault(path=tmp_path / vault_id) for vault_id in ("vault-a", "vault-b")
        },
        profiles={"public": ProfileConfig(vaults=("vault-a", "vault-b"))},
        embedding=EmbeddingConfig(
            base_url=HttpUrl("http://127.0.0.1:9000/v1"),
            model_env="MODEL",
            endpoint_class="local",
        ),
    )
    dsn = os.environ["VAULT_RAG_TEST_POSTGRES_DSN"]
    pool_one = PostgresPool(dsn, min_size=1, max_size=2, timeout=2)
    pool_two = PostgresPool(dsn, min_size=1, max_size=2, timeout=2)
    pool_one.open(wait=True)
    pool_two.open(wait=True)
    try:
        factory = AppFactory(
            config_path=tmp_path / "config.toml",
            database_path=tmp_path / "not-used.sqlite3",
            registry=registry,
            environ={"MODEL": "model"},
            store_factory=lambda: PostgresStore(pool_one),
            profile_resolver=PostgresProfileResolver(pool_one, registry, {"MODEL": "model"}),
        )
        original_status_by_vault = factory.status_by_vault_from_session
        promoted = False

        def promote_during_status(
            session: RequestSession, vault_ids: tuple[str, ...]
        ) -> Mapping[str, dict[str, JsonValue]]:
            nonlocal promoted
            if not promoted:
                promoted = True
                replacement_manifest = manifests["vault-a"].model_copy(
                    update={"egress_policy": EgressPolicy.REMOTE_ALLOWED}
                )
                replacement = PostgresStore(pool_two).begin_revision(
                    "vault-a",
                    "refs/heads/main",
                    "c" * 40,
                    manifest=replacement_manifest.model_dump(mode="json"),
                )
                replacement.replace_source(revision("alpha replacement", vault_id="vault-a"))
                _promote(pool_two, replacement)
            return original_status_by_vault(session, vault_ids)

        factory.status_by_vault_from_session = promote_during_status
        config = ServiceConfig.model_validate(
            {
                "schemaVersion": 1,
                "syncInterval": "1h",
                "repositories": {
                    vault_id: {
                        "url": f"https://example.test/{vault_id}.git",
                        "ref": "refs/heads/main",
                    }
                    for vault_id in ("vault-a", "vault-b")
                },
                "profiles": {"public": {"vaults": ("vault-a", "vault-b")}},
                "embedding": {
                    "baseUrl": "http://127.0.0.1:9000/v1",
                    "modelEnv": "MODEL",
                    "endpointClass": "local",
                },
            }
        )
        metrics = ServiceMetrics()
        service = VaultService(
            config,
            factory,
            _NoopCoordinator(),
            _NoopCoordinator(),
            _SessionOnlyState(),
            LifecycleLock(),
            metrics=metrics,
        )
        runtime = _HttpRuntime(service, metrics, _Events())
        with TestClient(create_app(runtime)) as client:
            status = client.get("/v1/status", params={"profile": "public"})
            assert status.status_code == 200
            vaults = status.json()["vaults"]
            assert vaults["vault-a"]["index"]["effective_egress_policy"] == "local-only"
            next_status = client.get("/v1/status", params={"profile": "public"})
            assert next_status.status_code == 200
            assert next_status.json()["vaults"]["vault-a"]["commit"]["commit_sha"] == "c" * 40
            assert (
                next_status.json()["vaults"]["vault-a"]["index"]["effective_egress_policy"]
                == "remote-allowed"
            )
            rendered = metrics.render().decode()
            assert 'vault_rag_sources{vault_id="vault-a"} 1.0' in rendered
            assert 'vault_rag_sources{vault_id="vault-b"} 2.0' in rendered
            assert 'vault_rag_pending_chunks{vault_id="vault-b"} 1.0' in rendered
            assert 'vault_rag_semantic_degraded{vault_id="vault-b"} 1.0' in rendered
            assert 'vault_rag_revision_count{vault_id="vault-a"} 2.0' in rendered
            assert 'vault_rag_revision_count{vault_id="vault-b"} 1.0' in rendered
            assert 'vault_rag_pending_embedding_age_seconds{vault_id="vault-b"}' in rendered
            assert "vault_rag_sync_queue_depth 0.0" in rendered
            assert 'vault_rag_dense_mode{vault_id="vault-a"} 1.0' in rendered
            assert 'vault_rag_dense_mode{vault_id="vault-b"} 1.0' in rendered
            assert vaults["vault-a"]["index"]["counts"]["sources"] == 1
            assert vaults["vault-b"]["index"]["counts"]["sources"] == 2
            assert vaults["vault-a"]["commit"]["commit_sha"] == "a" * 40
            assert vaults["vault-b"]["commit"]["commit_sha"] == "b" * 40
            search = client.post(
                "/v1/search",
                json={"profile": "public", "query": "alpha", "mode": "lexical"},
            )
            assert search.status_code == 200
            assert search.json()["vault_commits"]["vault-a"]["commit_sha"] == "c" * 40
            for mode in ("dense", "hybrid"):
                assert (
                    client.post(
                        "/v1/search",
                        json={"profile": "public", "query": "alpha", "mode": mode},
                    ).status_code
                    == 200
                )
            rendered = metrics.render().decode()
            for mode in ("lexical", "dense", "hybrid"):
                assert f'vault_rag_search_total{{mode="{mode}"}} 1.0' in rendered
            read_only_store = ReadOnlyQueryStoreSpy(pool_one)
            factory._store_factory = lambda: read_only_store
            assert (
                client.post(
                    "/v1/search",
                    json={"profile": "public", "query": "alpha", "mode": "lexical"},
                ).status_code
                == 200
            )
            assert (
                client.post(
                    "/v1/read",
                    json={"profile": "public", "vault_id": "vault-a", "path": "notes/example.md"},
                ).status_code
                == 200
            )
            assert client.get("/v1/status", params={"profile": "public"}).status_code == 200
            assert client.get("/health/ready").status_code == 503
    finally:
        pool_one.close()
        pool_two.close()
