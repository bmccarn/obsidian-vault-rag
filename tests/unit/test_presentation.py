from __future__ import annotations

import json
from typing import Any, cast

from vault_rag.domain import DegradedState, LineRange, SourceRef
from vault_rag.errors import ConfigError
from vault_rag.indexing import IndexDiagnostic, IndexReport
from vault_rag.presentation import (
    error_payload,
    index_payload,
    read_payload,
    search_payload,
    serialize_json,
)
from vault_rag.retrieval import (
    Continuation,
    ReadResponse,
    ScoreComponents,
    SearchHit,
    SearchResponse,
)
from vault_rag.service.facade import CommitView


def _response() -> SearchResponse:
    return SearchResponse(
        hits=(
            SearchHit(
                chunk_id="chunk-1",
                ref=SourceRef(
                    "vault-a",
                    "notes/a.md",
                    ("A",),
                    LineRange(2, 4),
                    "hash-a",
                ),
                text="retrieved text",
                metadata={"large": "x" * 2_100, "owner": "core"},
                scores=ScoreComponents(1, 0.5, None, None, 0.5, False),
                continuation=Continuation(12, 100, 5, 3),
            ),
        ),
        degraded=DegradedState(True, "embedding unavailable"),
        elapsed_ms=3.5,
        index_age=9.0,
    )


def test_search_payload_bounds_metadata_and_preserves_continuation() -> None:
    """Removing metadata bounds or continuation fields would expose oversized results."""
    payload = cast(
        dict[str, Any],
        search_payload(
            "public",
            _response(),
            {"vault-a": CommitView("vault-a", "a" * 40, "a" * 40, None)},
        ),
    )

    hit = payload["hits"][0]
    assert hit["metadata"] == {
        "truncated": True,
        "original_characters": 2_127,
        "canonical_json_preview": '{"large":"' + "x" * 1_790,
    }
    assert hit["metadata_truncated"] is True
    assert hit["continuation"] == {
        "remaining_characters": 12,
        "next_offset": 100,
        "next_line": 5,
        "next_character": 3,
    }
    assert payload["vault_commits"] == {
        "vault-a": {
            "checkout_sha": "a" * 40,
            "reconciled_sha": "a" * 40,
            "sync_degraded_reason": None,
            "fully_reconciled": True,
            "revision_id": None,
            "commit_sha": None,
        }
    }
    assert serialize_json(payload) == json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def test_read_and_index_payload_preserve_public_values_and_redact_diagnostics() -> None:
    """Dropping continuation or redaction would alter stable retrieval output or leak a secret."""
    response = _response().hits[0]
    read = read_payload(
        "public",
        ReadResponse(response.ref, response.text, response.continuation),
    )
    report = IndexReport(
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        (IndexDiagnostic("vault-a", "notes/a.md", "parse", "Bearer top-secret-key"),),
        7,
    )

    assert read["continuation"] == {
        "remaining_characters": 12,
        "next_offset": 100,
        "next_line": 5,
        "next_character": 3,
    }
    assert index_payload(report, {"TEST_API_KEY": "top-secret-key"})["diagnostics"] == [
        {
            "vault_id": "vault-a",
            "path": "notes/a.md",
            "category": "parse",
            "message": "<redacted-authorization>",
        }
    ]


def test_error_payload_redacts_bounds_and_includes_correlation_id() -> None:
    """Omitting redaction, detail limits, or correlation IDs would make failures unsafe."""
    error = ConfigError(
        "authorization: top-secret-key",
        details={"token": "top-secret-key", "long": "x" * 501},
    )

    payload = error_payload(
        error,
        {"TEST_API_KEY": "top-secret-key"},
        correlation_id="request-1",
    )

    assert payload == {
        "error": {
            "code": "config_error",
            "message": "<redacted-authorization>",
            "details": {"long": "x" * 500 + "…[truncated]", "token": "<redacted>"},
            "correlation_id": "request-1",
        }
    }
