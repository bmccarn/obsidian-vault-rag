from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, cast

from fastapi.testclient import TestClient
from mcp.client import Client
from prometheus_client.parser import text_string_to_metric_families

from vault_rag.domain import DegradedState, LineRange, SourceRef
from vault_rag.errors import StaleSourceError
from vault_rag.retrieval import (
    ReadRequest,
    ReadResponse,
    ScoreComponents,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from vault_rag.service.config import MCPTransportConfig
from vault_rag.service.facade import CommitView, ReadEnvelope, Readiness, SearchEnvelope
from vault_rag.service.http import RuntimePort, create_app
from vault_rag.service.mcp import MCPRuntimePort, build_mcp_server
from vault_rag.service.observability import ServiceMetrics

SHA = "a" * 40
SOURCE_HASH = "sha256:" + "b" * 64
_MODERN_META = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientInfo": {"name": "vault-rag-test", "version": "1"},
    "io.modelcontextprotocol/clientCapabilities": {},
}


@dataclass
class FakeService:
    error: Exception | None = None
    calls: list[tuple[str, SearchRequest | ReadRequest]] = field(default_factory=list)
    started: int = 0
    closed: int = 0

    def start(self) -> None:
        self.started += 1

    def close(self) -> None:
        self.closed += 1

    def search(self, profile: str, request: SearchRequest) -> SearchEnvelope:
        self.calls.append((profile, request))
        if self.error is not None:
            raise self.error
        ref = SourceRef(
            "homelab-ops",
            "04 - Runbooks/reboot.md",
            ("Reboot",),
            LineRange(4, 7),
            SOURCE_HASH,
        )
        hit = SearchHit(
            "chunk-1",
            ref,
            "safe reboot procedure",
            {"status": "active"},
            ScoreComponents(1, 1.0, 1, 1.0, 0.5, False),
            None,
        )
        return SearchEnvelope(
            profile,
            SearchResponse((hit,), DegradedState(), 1.0, 2.0),
            {
                "homelab-ops": CommitView(
                    "homelab-ops",
                    SHA,
                    SHA,
                    None,
                    revision_id="revision-1",
                    commit_sha=SHA,
                )
            },
        )

    def read(self, profile: str, request: ReadRequest) -> ReadEnvelope:
        self.calls.append((profile, request))
        if self.error is not None:
            raise self.error
        return ReadEnvelope(
            profile,
            ReadResponse(
                SourceRef(
                    "homelab-ops",
                    request.path,
                    ("Reboot",),
                    request.lines or LineRange(1, 2),
                    request.expected_source_hash or SOURCE_HASH,
                ),
                "reboot safely",
                None,
            ),
            {
                "homelab-ops": CommitView(
                    "homelab-ops",
                    SHA,
                    SHA,
                    None,
                    revision_id="revision-1",
                    commit_sha=SHA,
                )
            },
        )

    def profiles(self) -> dict[str, object]:
        return {
            "profiles": [{"name": "homelab", "vaults": ["homelab-ops"], "vaults_truncated": 0}],
            "profiles_truncated": 0,
        }

    def status(self, profile: str) -> dict[str, object]:
        return {
            "profile": profile,
            "vaults": {
                "homelab-ops": {
                    "index": {"counts": {"ready": 1}},
                    "commit": {"commit_safe": True, "commit_sha": SHA},
                }
            },
            "vaults_truncated": 0,
        }

    def ready(self) -> Readiness:
        return Readiness(True, {"homelab": True})

    def request_sync(self) -> None:
        raise AssertionError("MCP must not expose synchronization")


@dataclass
class EventSink:
    records: list[tuple[str, dict[str, object]]] = field(default_factory=list)

    def emit(self, event: str, **fields: object) -> None:
        self.records.append((event, fields))


@dataclass
class FakeRuntime:
    service: FakeService
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)
    events: EventSink = field(default_factory=EventSink)

    def start(self) -> None:
        self.service.start()

    def close(self) -> None:
        self.service.close()


def _runtime(error: Exception | None = None) -> FakeRuntime:
    return FakeRuntime(FakeService(error=error))


def _mcp(runtime: FakeRuntime):
    return build_mcp_server(cast(MCPRuntimePort, runtime))


def _run[ResultT](operation: Any) -> ResultT:
    return asyncio.run(operation)


def test_discovery_lists_rich_read_only_tools_resources_prompt_and_cache_hints() -> None:
    runtime = _runtime()

    async def inspect_server() -> None:
        async with Client(_mcp(runtime)) as client:
            assert client.protocol_version == "2026-07-28"
            listed = await client.list_tools()
            assert listed.ttl_ms == 3_600_000
            assert listed.cache_scope == "private"
            tools = {tool.name: tool for tool in listed.tools}
            assert set(tools) == {
                "vault_search",
                "vault_read",
                "vault_profiles",
                "vault_status",
            }
            assert "recommended_read" in str(tools["vault_search"].output_schema)
            assert "expected_source_hash" in str(tools["vault_read"].input_schema)
            for tool in tools.values():
                assert tool.annotations is not None
                assert tool.annotations.read_only_hint is True
                assert tool.annotations.destructive_hint is False
                assert tool.annotations.idempotent_hint is True
                assert tool.annotations.open_world_hint is False

            resources = await client.list_resources()
            assert {str(resource.uri) for resource in resources.resources} == {
                "vault-rag://guide",
                "vault-rag://profiles",
            }
            guide = await client.read_resource("vault-rag://guide")
            assert "expected_source_hash" in guide.contents[0].text
            assert "stateless JSON mode" in guide.contents[0].text
            assert "`markdown`, `text`, `log`, or `json`" in guide.contents[0].text
            assert "canvas" not in guide.contents[0].text
            assert guide.ttl_ms == 30_000
            prompts = await client.list_prompts()
            assert [prompt.name for prompt in prompts.prompts] == ["grounded_vault_research"]
            prompt = await client.get_prompt(
                "grounded_vault_research",
                {"profile": "homelab", "question": "How do I reboot safely?"},
            )
            assert "recommended_read" in prompt.messages[0].content.text

    _run(inspect_server())


def test_search_and_read_publish_structured_copy_safe_revision_fenced_results() -> None:
    runtime = _runtime()

    async def use_tools() -> None:
        async with Client(_mcp(runtime)) as client:
            search = await client.call_tool(
                "vault_search",
                {
                    "profile": "homelab",
                    "query": "safe reboot",
                    "limit": 5,
                    "mode": "hybrid",
                    "vault_ids": ["homelab-ops"],
                    "path_prefix": "04 - Runbooks",
                    "source_kind": "markdown",
                    "frontmatter": {"status": "active"},
                },
            )
            assert search.is_error is False
            assert search.structured_content is not None
            recommended = search.structured_content["hits"][0]["recommended_read"]
            assert recommended == {
                "profile": "homelab",
                "vault_id": "homelab-ops",
                "path": "04 - Runbooks/reboot.md",
                "start_line": 4,
                "end_line": 7,
                "expected_source_hash": SOURCE_HASH,
            }
            assert search.structured_content["vault_commits"]["homelab-ops"]["revision_id"] == (
                "revision-1"
            )

            read = await client.call_tool("vault_read", recommended)
            assert read.is_error is False
            assert read.structured_content is not None
            assert read.structured_content["text"] == "reboot safely"
            assert read.structured_content["citation"].endswith("#L4-L7")

    _run(use_tools())

    search_request = cast(SearchRequest, runtime.service.calls[0][1])
    assert search_request.filters.vault_ids == ("homelab-ops",)
    assert search_request.filters.frontmatter == {"status": "active"}
    read_request = cast(ReadRequest, runtime.service.calls[1][1])
    assert read_request.expected_source_hash == SOURCE_HASH
    assert read_request.vault_id == "homelab-ops"
    assert read_request.lines == LineRange(4, 7)


def test_expected_errors_are_actionable_and_unexpected_errors_are_correlated() -> None:
    stale_runtime = _runtime(StaleSourceError("sensitive stale detail"))

    async def stale_read() -> None:
        async with Client(_mcp(stale_runtime)) as client:
            result = await client.call_tool(
                "vault_read",
                {
                    "profile": "homelab",
                    "vault_id": "homelab-ops",
                    "path": "runbook.md",
                    "expected_source_hash": "sha256:" + "0" * 64,
                },
            )
            assert result.is_error is True
            assert "stale_source" in result.content[0].text
            assert "sensitive" not in result.content[0].text

    _run(stale_read())

    internal_runtime = _runtime(RuntimeError("postgresql://user:secret@example/db"))

    async def internal_search() -> None:
        async with Client(_mcp(internal_runtime)) as client:
            result = await client.call_tool(
                "vault_search", {"profile": "homelab", "query": "reboot"}
            )
            assert result.is_error is True
            assert "correlation_id=" in result.content[0].text
            assert "secret" not in result.content[0].text

    _run(internal_search())
    errors = [
        record for record in internal_runtime.events.records if record[0] == "mcp_internal_error"
    ]
    assert len(errors) == 1
    assert errors[0][1]["tool"] == "vault_search"


def test_stateless_json_http_transport_has_no_session_or_sse_and_rejects_rebinding() -> None:
    runtime = _runtime()
    config = MCPTransportConfig(
        enabled=True,
        allowed_hosts=("testserver",),
        allowed_origins=("https://vault-rag.example.test",),
    )
    app = create_app(cast(RuntimePort, runtime), mcp_config=config)
    discover = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "server/discover",
        "params": {"_meta": _MODERN_META},
    }
    headers = {
        "Mcp-Protocol-Version": "2026-07-28",
        "Mcp-Method": "server/discover",
        "Accept": "application/json",
    }

    with TestClient(app) as client:
        response = client.post("/mcp", json=discover, headers=headers)
        subscription = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "subscriptions/listen",
                "params": {"_meta": _MODERN_META, "notifications": {}},
            },
            headers={
                **headers,
                "Mcp-Method": "subscriptions/listen",
                "Accept": "application/json, text/event-stream",
            },
        )
        receive_stream = client.get("/mcp")
        receive_stream_head = client.head("/mcp")
        wrong_host = client.post(
            "http://wrong.example/mcp",
            json=discover,
            headers={**headers, "host": "wrong.example"},
        )
        wrong_origin = client.post(
            "/mcp",
            json=discover,
            headers={**headers, "origin": "https://wrong.example"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert "mcp-session-id" not in response.headers
    assert "text/event-stream" not in response.text
    assert receive_stream.status_code == 405
    assert receive_stream.headers["allow"] == "POST"
    assert receive_stream.json() == {"error": {"code": "method_not_allowed"}}
    assert "text/event-stream" not in receive_stream.headers.get("content-type", "")
    assert receive_stream_head.status_code == 405
    assert receive_stream_head.headers["allow"] == "POST"
    discovery = response.json()["result"]
    assert discovery["supportedVersions"] == ["2026-07-28"]
    assert discovery["ttlMs"] == 3_600_000
    assert discovery["capabilities"]["tools"]["listChanged"] is False
    assert discovery["capabilities"]["prompts"]["listChanged"] is False
    assert discovery["capabilities"]["resources"] == {
        "subscribe": False,
        "listChanged": False,
    }
    assert subscription.status_code == 404
    assert subscription.headers["content-type"] == "application/json"
    assert "mcp-session-id" not in subscription.headers
    assert "text/event-stream" not in subscription.text
    assert subscription.json()["error"]["code"] == -32601
    assert wrong_host.status_code == 421
    assert wrong_origin.status_code == 403
    assert runtime.service.started == 1
    assert runtime.service.closed == 1


def test_mcp_is_disabled_by_default_and_body_limit_is_shared_with_http() -> None:
    disabled_runtime = _runtime()
    with TestClient(create_app(cast(RuntimePort, disabled_runtime))) as client:
        assert client.post("/mcp", json={}).status_code == 404

    enabled_runtime = _runtime()
    app = create_app(
        cast(RuntimePort, enabled_runtime),
        mcp_config=MCPTransportConfig(enabled=True, allowed_hosts=("testserver",)),
    )
    with TestClient(app) as client:
        response = client.post(
            "/mcp",
            content=b"x" * (1024 * 1024 + 1),
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 413
    assert response.json() == {"error": {"code": "request_too_large"}}


def test_mcp_metrics_and_events_use_only_bounded_tool_outcome_labels() -> None:
    runtime = _runtime()

    async def call_profiles() -> None:
        async with Client(_mcp(runtime)) as client:
            result = await client.call_tool("vault_profiles")
            assert result.is_error is False

    _run(call_profiles())
    families = {
        family.name: family
        for family in text_string_to_metric_families(runtime.metrics.render().decode())
    }
    samples = families["vault_rag_mcp_tool_calls"].samples
    assert any(
        sample.labels == {"tool": "vault_profiles", "outcome": "success"} and sample.value == 1
        for sample in samples
    )
    events = [record for record in runtime.events.records if record[0] == "mcp_tool_call"]
    assert len(events) == 1
    assert events[0][1]["tool"] == "vault_profiles"
    assert set(events[0][1]) == {"tool", "outcome", "elapsed_ms"}
