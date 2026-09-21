"""Private-network verifier for the synthetic split-runtime Compose fixture."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import psycopg

_TIMEOUT_SECONDS = 90.0
_SOURCE_HASH = "sha256:8d3ef3b18d3da6a989c60c68bc37ae45d0e8c214b1a3a0af628019ab04a9176c"
_VAULT_ID = "fixture-vault"
_PROFILE = "fixture"
_MCP_META = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientInfo": {"name": "compose-smoke", "version": "1"},
    "io.modelcontextprotocol/clientCapabilities": {},
}


def _request(url: str, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            raw = response.read()
            return response.status, json.loads(raw)
    except urllib.error.HTTPError as error:
        return error.code, {}
    except (OSError, TimeoutError, json.JSONDecodeError):
        return 0, {}


def _identity(payload: dict[str, Any]) -> tuple[str, str] | None:
    commits = payload.get("vault_commits")
    if not isinstance(commits, dict):
        return None
    commit = commits.get(_VAULT_ID)
    if not isinstance(commit, dict):
        return None
    commit_sha = commit.get("commit_sha")
    revision_id = commit.get("revision_id")
    if not isinstance(commit_sha, str) or not isinstance(revision_id, str):
        return None
    return commit_sha, revision_id


def _mcp_request(
    host: str,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    name: str | None = None,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    request_params = dict(params or {})
    request_params["_meta"] = _MCP_META
    body = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": method, "params": request_params}
    ).encode()
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Mcp-Protocol-Version": "2026-07-28",
        "Mcp-Method": method,
    }
    if name is not None:
        headers["Mcp-Name"] = name
    request = urllib.request.Request(
        f"http://{host}:8080/mcp",
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            raw = response.read()
            headers = {name.lower(): value for name, value in response.headers.items()}
            return response.status, json.loads(raw), headers
    except urllib.error.HTTPError as error:
        return error.code, {}, {}
    except (OSError, TimeoutError, json.JSONDecodeError):
        return 0, {}, {}


def _verify_mcp(host: str) -> bool:
    status, discovery, headers = _mcp_request(host, "server/discover")
    if (
        status != 200
        or discovery.get("result", {}).get("supportedVersions") != ["2026-07-28"]
        or headers.get("content-type") != "application/json"
        or "mcp-session-id" in headers
    ):
        return False
    status, result, _ = _mcp_request(
        host,
        "tools/call",
        {
            "name": "vault_search",
            "arguments": {
                "profile": _PROFILE,
                "query": "synthetic fixture",
                "mode": "lexical",
                "limit": 5,
            },
        },
        name="vault_search",
    )
    structured = result.get("result", {}).get("structuredContent", {})
    hits = structured.get("hits")
    return status == 200 and isinstance(hits, list) and bool(hits)


def _verify_api(host: str) -> tuple[str, str] | None:
    base = f"http://{host}:8080"
    status, _ = _request(f"{base}/v1/status?profile={_PROFILE}")
    if status != 200:
        return None
    status, search = _request(
        f"{base}/v1/search",
        {"profile": _PROFILE, "query": "synthetic fixture", "mode": "lexical", "limit": 5},
    )
    if status != 200 or not search.get("hits"):
        return None
    identity = _identity(search)
    status, read = _request(
        f"{base}/v1/read",
        {
            "profile": _PROFILE,
            "path": "note.md",
            "expected_source_hash": _SOURCE_HASH,
        },
    )
    if status != 200 or read.get("source_hash") != _SOURCE_HASH or _identity(read) != identity:
        return None
    return identity


def _revision_count() -> int | None:
    secret_path = os.environ.get("VAULT_RAG_DATABASE_URL_FILE")
    if secret_path is None:
        return None
    try:
        database_url = Path(secret_path).read_text(encoding="utf-8").strip()
        with (
            psycopg.connect(database_url, connect_timeout=5) as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(
                "SELECT count(*) FROM vault_rag.vault_revisions WHERE vault_id = %s",
                (_VAULT_ID,),
            )
            result = cursor.fetchone()
    except (OSError, psycopg.Error):
        return None
    return int(result[0]) if result is not None else None


def verify() -> int:
    deadline = time.monotonic() + _TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        api_a = _verify_api("api-a")
        api_b = _verify_api("api-b")
        mcp_a = _verify_mcp("api-a")
        mcp_b = _verify_mcp("api-b")
        revision_count = _revision_count()
        if api_a is not None and api_a == api_b and mcp_a and mcp_b and revision_count == 1:
            return 0
        time.sleep(1)
    print("split-runtime verification did not converge before its deadline", file=sys.stderr)
    return 1


if __name__ == "__main__":
    if sys.argv[1:] != ["verify"]:
        raise SystemExit("usage: compose-smoke.py verify")
    raise SystemExit(verify())
