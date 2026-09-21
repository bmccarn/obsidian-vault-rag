from __future__ import annotations

from pathlib import Path
from threading import Event, Thread

import httpx
import pytest
from pydantic import HttpUrl

import vault_rag.app as app_module
from vault_rag.app import AppFactory
from vault_rag.config import (
    EgressPolicy,
    EmbeddingConfig,
    RegistryConfig,
    ResolvedProfile,
    ResolvedVault,
    VaultManifest,
)


def _profile(root: Path) -> ResolvedProfile:
    manifest = VaultManifest(
        schema_version=1,
        id="vault",
        egress_policy=EgressPolicy.LOCAL_ONLY,
        include=("**/*.md",),
    )
    return ResolvedProfile(
        name="_reconcile_vault",
        vaults=(ResolvedVault(root=root, manifest=manifest),),
        embedding_model="test-model",
        api_key_env=None,
        effective_policy=EgressPolicy.LOCAL_ONLY,
        semantic_enabled=True,
    )


def _factory(tmp_path: Path, http_client: httpx.Client) -> AppFactory:
    return AppFactory(
        config_path=tmp_path / "config.toml",
        database_path=tmp_path / "index.sqlite3",
        registry=RegistryConfig(
            embedding=EmbeddingConfig(
                base_url=HttpUrl("http://127.0.0.1:9000/v1"),
                model_env="EMBED_MODEL",
                endpoint_class="local",
            )
        ),
        environ={"EMBED_MODEL": "test-model"},
        http_client=http_client,
    )


def test_explicit_database_path_does_not_initialize_environment_data_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A service-owned data path must work under a read-only XDG home."""
    http_client = httpx.Client()
    factory = _factory(tmp_path, http_client)
    monkeypatch.setattr(
        app_module,
        "data_home",
        lambda _environ: pytest.fail("explicit database path must not use data_home"),
    )

    try:
        factory.store(_profile(tmp_path))
    finally:
        factory.close()
        http_client.close()


def test_status_uses_the_store_created_when_no_store_is_injected(tmp_path: Path) -> None:
    """Dereferencing the optional argument would break ordinary status callers."""
    http_client = httpx.Client()
    factory = _factory(tmp_path, http_client)
    try:
        assert factory.status(_profile(tmp_path))["database_path"] == str(
            tmp_path / "index.sqlite3"
        )
    finally:
        factory.close()
        http_client.close()


def test_factory_reuses_one_embedding_client_and_closes_it_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing client reuse would create separate clients for each service."""
    created: list[object] = []
    closed: list[object] = []

    class FakeEmbeddingClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            created.append(self)

        def close(self) -> None:
            closed.append(self)

    monkeypatch.setattr(app_module, "EmbeddingClient", FakeEmbeddingClient)
    http_client = httpx.Client()
    factory = _factory(tmp_path, http_client)
    profile = _profile(tmp_path)

    factory.indexer(profile)
    factory.retrieval(profile)
    factory.close()
    factory.close()

    assert len(created) == 1
    assert closed == created
    assert not http_client.is_closed
    with pytest.raises(RuntimeError, match="closed"):
        factory.indexer(profile)
    http_client.close()


def test_factory_closes_the_client_created_during_concurrent_first_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A close racing lazy creation must not leak the newly created client."""
    created: list[object] = []
    closed: list[object] = []
    creating = Event()
    release_creation = Event()
    close_finished = Event()

    class FakeEmbeddingClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            creating.set()
            release_creation.wait()
            created.append(self)

        def close(self) -> None:
            closed.append(self)

    def close_factory() -> None:
        factory.close()
        close_finished.set()

    monkeypatch.setattr(app_module, "EmbeddingClient", FakeEmbeddingClient)
    http_client = httpx.Client()
    factory = _factory(tmp_path, http_client)
    creator = Thread(target=factory._embedding)
    creator.start()
    assert creating.wait(1)
    closer = Thread(target=close_factory)
    closer.start()
    close_finished.wait(0.1)
    release_creation.set()
    creator.join(1)
    closer.join(1)
    assert not creator.is_alive()
    assert not closer.is_alive()

    assert created == closed
    http_client.close()


def test_factory_creates_one_embedding_client_for_concurrent_first_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unsynchronized lazy creation would hand concurrent callers distinct clients."""
    created: list[object] = []
    creating = Event()
    release_creation = Event()

    class FakeEmbeddingClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            created.append(self)
            creating.set()
            release_creation.wait()

        def close(self) -> None:
            return

    monkeypatch.setattr(app_module, "EmbeddingClient", FakeEmbeddingClient)
    http_client = httpx.Client()
    factory = _factory(tmp_path, http_client)
    first = Thread(target=factory._embedding)
    second = Thread(target=factory._embedding)
    first.start()
    assert creating.wait(1)
    second.start()
    release_creation.set()
    first.join(1)
    second.join(1)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(created) == 1
    factory.close()
    http_client.close()
