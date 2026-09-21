from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_mcp_smoke_treats_http_header_names_case_insensitively(monkeypatch) -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "deploy/docker/compose-smoke.py"))
    responses: list[tuple[int, dict[str, Any], dict[str, str]]] = [
        (
            200,
            {"result": {"supportedVersions": ["2026-07-28"]}},
            {"content-type": "application/json"},
        ),
        (
            200,
            {"result": {"structuredContent": {"hits": [{"path": "note.md"}]}}},
            {"content-type": "application/json"},
        ),
    ]

    monkeypatch.setitem(
        module["_verify_mcp"].__globals__,
        "_mcp_request",
        lambda *args, **kwargs: responses.pop(0),
    )

    assert module["_verify_mcp"]("api-a") is True


def test_mcp_smoke_rejects_any_stateless_session_header(monkeypatch) -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "deploy/docker/compose-smoke.py"))
    monkeypatch.setitem(
        module["_verify_mcp"].__globals__,
        "_mcp_request",
        lambda *args, **kwargs: (
            200,
            {"result": {"supportedVersions": ["2026-07-28"]}},
            {"content-type": "application/json", "mcp-session-id": "unexpected"},
        ),
    )

    assert module["_verify_mcp"]("api-a") is False
