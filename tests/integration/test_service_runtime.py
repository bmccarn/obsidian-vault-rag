from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import cast

import httpx
from fastapi.testclient import TestClient

from vault_rag.app import AppFactory
from vault_rag.service.config import load_service_config
from vault_rag.service.coordinator import RepositorySyncCoordinator
from vault_rag.service.observability import ServiceObservability
from vault_rag.service.state import SyncStateStore, VaultSyncState


def _config(path: Path) -> None:
    path.write_text(
        """
schema_version: 1
sync_interval: 1h
repositories:
  homelab-ops:
    url: https://example.test/homelab-ops.git
    ref: refs/heads/main
    credential_env: GIT_TOKEN
profiles:
  homelab:
    vaults: [homelab-ops]
embedding:
  base_url: http://127.0.0.1:9000/v1
  model_env: EMBED_MODEL
  endpoint_class: local
""".lstrip(),
        encoding="utf-8",
    )


@dataclass
class FakeService:
    calls: list[str]

    def start(self) -> None:
        self.calls.append("start")

    def close(self) -> None:
        self.calls.append("service-close")


@dataclass
class FakeHttpClient:
    calls: list[str]

    def close(self) -> None:
        self.calls.append("http-close")


def test_runtime_lifecycle_is_idempotent_and_closes_service_before_shared_client() -> None:
    from vault_rag.service.observability import StructuredEvents
    from vault_rag.service.runtime import ServiceRuntime

    calls: list[str] = []
    runtime = ServiceRuntime(
        service=FakeService(calls),
        metrics=object(),
        http_client=FakeHttpClient(calls),
    )

    runtime.start()
    runtime.close()
    runtime.close()
    runtime.start()
    assert isinstance(runtime.events, StructuredEvents)

    assert calls == ["start", "service-close", "http-close"]


def test_build_runtime_materializes_inputs_and_uses_service_data_paths(tmp_path: Path) -> None:
    from vault_rag.service.runtime import build_runtime

    config_path = tmp_path / "service.yaml"
    data_root = tmp_path / "data"
    _config(config_path)

    runtime = build_runtime(
        config_path,
        data_root,
        {"GIT_TOKEN": "token", "EMBED_MODEL": "synthetic-model"},
    )
    try:
        assert type(runtime.service).__name__ == "VaultService"
        assert runtime.service._metrics is runtime.metrics
        assert isinstance(
            observer := cast(
                ServiceObservability,
                cast(RepositorySyncCoordinator, runtime.service._coordinator)._observer,
            ),
            ServiceObservability,
        )
        assert observer._metrics is runtime.metrics
        assert observer._events is runtime.events
        assert (data_root / "repos").exists() is False
    finally:
        runtime.close()


def test_http_runtime_preserves_ready_lexical_retrieval_when_sync_and_embeddings_fail(
    monkeypatch: object, tmp_path: Path
) -> None:
    import vault_rag.service.runtime as runtime_module
    from vault_rag.service.http import create_app

    config_path = tmp_path / "service.yaml"
    data_root = tmp_path / "data"
    _config(config_path)
    config = load_service_config(config_path)
    checkout = data_root / "repos" / "homelab-ops"
    checkout.mkdir(parents=True)
    (checkout / ".vault-rag.toml").write_text(
        (
            'schema_version = 1\nid = "homelab-ops"\negress_policy = "local-only"\n'
            'include = ["**/*.md"]\n'
        ),
        encoding="utf-8",
    )
    (checkout / "reboot.md").write_text(
        "# Reboot\nsafe lexical reboot procedure\n", encoding="utf-8"
    )

    def embed_ok(request: httpx.Request) -> httpx.Response:
        count = len(json.loads(request.content)["input"])
        return httpx.Response(
            200,
            json={"data": [{"index": index, "embedding": [1.0, 0.0]} for index in range(count)]},
            request=request,
        )

    prep_client = httpx.Client(transport=httpx.MockTransport(embed_ok))
    prep_factory = AppFactory(
        config_path=config_path,
        database_path=data_root / "index.sqlite3",
        registry=config.registry(data_root),
        environ={"EMBED_MODEL": "synthetic"},
        http_client=prep_client,
    )
    try:
        report = prep_factory.indexer(prep_factory.resolve_profile("_reconcile_homelab-ops")).run()
        assert report.ready_chunks > 0
    finally:
        prep_factory.close()
        prep_client.close()

    sha = "a" * 40
    SyncStateStore(data_root / "sync-state").write(
        VaultSyncState(
            vault_id="homelab-ops",
            configured_ref="refs/heads/main",
            checkout_sha=sha,
            reconciled_sha=sha,
        )
    )

    fetch_started = Event()

    class FailingGit:
        def fetch(self, *_args: object, **_kwargs: object) -> object:
            fetch_started.set()
            raise RuntimeError("remote fetch failed")

        def head(self, _checkout: Path) -> str | None:
            return sha

    failed_embedding_calls: list[httpx.Request] = []
    real_client = httpx.Client

    def embed_failure_client() -> httpx.Client:
        return real_client(
            transport=httpx.MockTransport(
                lambda request: (
                    failed_embedding_calls.append(request) or httpx.Response(503, request=request)
                )
            )
        )

    monkeypatch.setattr(runtime_module, "ManagedGit", FailingGit)
    monkeypatch.setattr(runtime_module.httpx, "Client", embed_failure_client)
    runtime = runtime_module.build_runtime(
        config_path, data_root, {"GIT_TOKEN": "token", "EMBED_MODEL": "synthetic"}
    )

    with TestClient(create_app(runtime)) as client:
        assert fetch_started.wait(1)
        assert client.get("/health/ready").json()["ready"] is True
        assert client.get("/v1/profiles").json()["profiles"][0]["name"] == "homelab"
        assert client.get("/v1/status", params={"profile": "homelab"}).status_code == 200
        search = client.post(
            "/v1/search",
            json={"profile": "homelab", "query": "safe reboot", "mode": "hybrid"},
        )
        assert search.status_code == 200
        assert search.json()["hits"][0]["path"] == "reboot.md"
        assert search.json()["degraded"]["semantic_search"] is True
        assert (
            client.post("/v1/read", json={"profile": "homelab", "path": "reboot.md"})
            .json()["text"]
            .endswith("safe lexical reboot procedure\n")
        )

    assert failed_embedding_calls


def test_http_runtime_rejects_requests_when_active_checkout_head_mismatches_state(
    monkeypatch: object, tmp_path: Path
) -> None:
    import vault_rag.service.runtime as runtime_module
    from vault_rag.service.http import create_app

    config_path = tmp_path / "service.yaml"
    data_root = tmp_path / "data"
    _config(config_path)
    config = load_service_config(config_path)
    checkout = data_root / "repos" / "homelab-ops"
    checkout.mkdir(parents=True)
    (checkout / ".vault-rag.toml").write_text(
        (
            'schema_version = 1\nid = "homelab-ops"\negress_policy = "local-only"\n'
            'include = ["**/*.md"]\n'
        ),
        encoding="utf-8",
    )
    (checkout / "reboot.md").write_text(
        "# Reboot\nsafe lexical reboot procedure\n", encoding="utf-8"
    )

    def embed_ok(request: httpx.Request) -> httpx.Response:
        count = len(json.loads(request.content)["input"])
        return httpx.Response(
            200,
            json={"data": [{"index": index, "embedding": [1.0, 0.0]} for index in range(count)]},
            request=request,
        )

    prep_client = httpx.Client(transport=httpx.MockTransport(embed_ok))
    prep_factory = AppFactory(
        config_path=config_path,
        database_path=data_root / "index.sqlite3",
        registry=config.registry(data_root),
        environ={"EMBED_MODEL": "synthetic"},
        http_client=prep_client,
    )
    try:
        report = prep_factory.indexer(prep_factory.resolve_profile("_reconcile_homelab-ops")).run()
        assert report.ready_chunks > 0
    finally:
        prep_factory.close()
        prep_client.close()

    sha = "a" * 40
    SyncStateStore(data_root / "sync-state").write(
        VaultSyncState(
            vault_id="homelab-ops",
            configured_ref="refs/heads/main",
            checkout_sha=sha,
            reconciled_sha=sha,
        )
    )

    fetch_started = Event()

    class MismatchedGit:
        def fetch(self, *_args: object, **_kwargs: object) -> object:
            fetch_started.set()
            raise RuntimeError("remote fetch failed")

        def head(self, _checkout: Path) -> str | None:
            return "b" * 40

    real_client = httpx.Client

    def embed_failure_client() -> httpx.Client:
        return real_client(
            transport=httpx.MockTransport(lambda request: httpx.Response(503, request=request))
        )

    retrieval_calls: list[str] = []

    def retrieval_must_not_run(_factory: AppFactory, profile: object) -> object:
        retrieval_calls.append(str(profile))
        raise AssertionError("unsafe checkout must be rejected before retrieval construction")

    monkeypatch.setattr(runtime_module, "ManagedGit", MismatchedGit)
    monkeypatch.setattr(runtime_module.httpx, "Client", embed_failure_client)
    monkeypatch.setattr(AppFactory, "retrieval", retrieval_must_not_run)
    runtime = runtime_module.build_runtime(
        config_path, data_root, {"GIT_TOKEN": "token", "EMBED_MODEL": "synthetic"}
    )

    with TestClient(create_app(runtime)) as client:
        assert fetch_started.wait(1)
        assert client.get("/health/ready").status_code == 503
        assert (
            client.post(
                "/v1/search",
                json={"profile": "homelab", "query": "safe reboot", "mode": "hybrid"},
            ).status_code
            == 503
        )
        assert (
            client.post("/v1/read", json={"profile": "homelab", "path": "reboot.md"}).status_code
            == 503
        )
        assert client.get("/v1/status", params={"profile": "homelab"}).status_code == 200

    assert retrieval_calls == []
