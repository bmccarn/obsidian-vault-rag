"""Transport-neutral, bounded JSON presentation for retrieval operations."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING, cast

from vault_rag.domain import JsonValue
from vault_rag.errors import ConfigError, VaultRagError
from vault_rag.indexing import IndexReport
from vault_rag.retrieval import Continuation, ReadResponse, SearchResponse

if TYPE_CHECKING:
    from vault_rag.service.facade import CommitView


_MAX_METADATA_CHARACTERS = 2_000
_MAX_ERROR_CHARACTERS = 500
_MAX_IDENTITY_CHARACTERS = 200
_MAX_VAULT_IDENTITIES = 100
_CREDENTIAL_URL = re.compile(
    r"(?P<scheme>(?:https?|postgres(?:ql)?)://)[^\s/@:]+:[^\s/@]+@",
    re.IGNORECASE,
)
_AUTHORIZATION = re.compile(r"(?i)(authorization\s*[:=]\s*|bearer\s+)[A-Za-z0-9._~+/=-]{8,}")


def serialize_json(value: JsonValue) -> str:
    """Serialize a JSON-compatible value in the stable public format."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def secret_values(environ: Mapping[str, str]) -> tuple[str, ...]:
    """Return likely secret values longest first for deterministic replacement."""
    credential_names = (
        "API_KEY",
        "ACCESS_TOKEN",
        "AUTH_TOKEN",
        "SECRET_KEY",
        "PASSWORD",
        "CREDENTIAL",
        "AUTHORIZATION",
    )
    values = {
        value
        for key, value in environ.items()
        if value and len(value) >= 8 and any(name in key.upper() for name in credential_names)
    }
    return tuple(sorted(values, key=len, reverse=True))


def redact_text(text: str, environ: Mapping[str, str]) -> str:
    """Redact credentials from text that can be returned to a caller."""
    safe = _CREDENTIAL_URL.sub(r"\g<scheme><redacted>@", text)
    safe = _AUTHORIZATION.sub("<redacted-authorization>", safe)
    for value in secret_values(environ):
        safe = safe.replace(value, "<redacted>")
    return safe


def safe_text(text: str, environ: Mapping[str, str]) -> str:
    """Redact, compact, and bound an error message."""
    compact = " ".join(redact_text(text, environ).split())
    return compact[:_MAX_ERROR_CHARACTERS] or "operation failed"


def bounded_details(value: JsonValue, *, depth: int = 0) -> JsonValue:
    """Retain a deterministic bounded representation of structured details."""
    if depth >= 4:
        return "<truncated>"
    if isinstance(value, str):
        if len(value) <= _MAX_ERROR_CHARACTERS:
            return value
        return value[:_MAX_ERROR_CHARACTERS] + "…[truncated]"
    if isinstance(value, list):
        visible = [bounded_details(item, depth=depth + 1) for item in value[:20]]
        if len(value) > len(visible):
            visible.append({"truncated_items": len(value) - len(visible)})
        return visible
    if isinstance(value, dict):
        visible_items = sorted(value.items())[:20]
        bounded = {key[:100]: bounded_details(item, depth=depth + 1) for key, item in visible_items}
        if len(value) > len(visible_items):
            bounded["truncated_fields"] = len(value) - len(visible_items)
        return bounded
    return value


def redact_json(value: JsonValue, environ: Mapping[str, str]) -> JsonValue:
    """Redact every string leaf in a JSON-compatible value."""
    if isinstance(value, str):
        return redact_text(value, environ)
    if isinstance(value, list):
        return [redact_json(item, environ) for item in value]
    if isinstance(value, dict):
        return {key: redact_json(item, environ) for key, item in value.items()}
    return value


def error_exit_code(error: Exception) -> int:
    """Return the existing CLI process status associated with an error."""
    if isinstance(error, VaultRagError):
        return error.exit_code
    if isinstance(error, (TypeError, ValueError)):
        return ConfigError.exit_code
    return 1


def error_payload(
    error: Exception, environ: Mapping[str, str], *, correlation_id: str | None = None
) -> dict[str, JsonValue]:
    """Render an exception as a redacted bounded public error payload."""
    if isinstance(error, VaultRagError):
        code = error.code
        message = error.message
        details: JsonValue = cast(JsonValue, error.details)
    elif isinstance(error, (TypeError, ValueError)):
        code = ConfigError.code
        message = str(error)
        details = {}
    else:
        code = "internal_error"
        message = "operation failed"
        details = {}
    error_value: dict[str, JsonValue] = {
        "code": code,
        "message": safe_text(message, environ),
        "details": bounded_details(redact_json(details, environ)),
    }
    if correlation_id is not None:
        error_value["correlation_id"] = correlation_id[:_MAX_IDENTITY_CHARACTERS]
    return {"error": error_value}


def metadata_payload(metadata: Mapping[str, JsonValue]) -> tuple[JsonValue, bool]:
    """Bound hit metadata while retaining a deterministic canonical preview."""
    copied: JsonValue = dict(metadata)
    encoded = serialize_json(copied)
    if len(encoded) <= _MAX_METADATA_CHARACTERS:
        return copied, False
    return (
        {
            "truncated": True,
            "original_characters": len(encoded),
            "canonical_json_preview": encoded[: _MAX_METADATA_CHARACTERS - 200],
        },
        True,
    )


def continuation_payload(value: Continuation | None) -> JsonValue:
    """Return the public pagination continuation or JSON null."""
    if value is None:
        return None
    return {
        "remaining_characters": value.remaining_characters,
        "next_offset": value.next_offset,
        "next_line": value.next_line,
        "next_character": value.next_character,
    }


def index_payload(report: IndexReport, environ: Mapping[str, str]) -> dict[str, JsonValue]:
    """Render an indexing report without exposing diagnostic secrets."""
    diagnostics: list[JsonValue] = [
        {
            "vault_id": item.vault_id,
            "path": item.path,
            "category": item.category,
            "message": redact_text(item.message, environ),
        }
        for item in report.diagnostics
    ]
    return {
        "total_sources": report.total_sources,
        "added_sources": report.added_sources,
        "changed_sources": report.changed_sources,
        "unchanged_sources": report.unchanged_sources,
        "deleted_sources": report.deleted_sources,
        "ready_chunks": report.ready_chunks,
        "pending_chunks": report.pending_chunks,
        "parse_failures": report.parse_failures,
        "embedding_failures": report.embedding_failures,
        "embedding_requests": report.embedding_requests,
        "diagnostics": diagnostics,
        "elapsed_ms": report.elapsed_ms,
    }


def _commit_payload(commits: Mapping[str, CommitView]) -> tuple[dict[str, JsonValue], int]:
    visible = sorted(commits.items())[:_MAX_VAULT_IDENTITIES]
    return (
        {
            vault_id[:_MAX_IDENTITY_CHARACTERS]: {
                "checkout_sha": view.checkout_sha,
                "reconciled_sha": view.reconciled_sha,
                "sync_degraded_reason": view.sync_degraded_reason,
                "fully_reconciled": (
                    view.checkout_sha is not None and view.checkout_sha == view.reconciled_sha
                ),
                "revision_id": view.revision_id,
                "commit_sha": view.commit_sha,
            }
            for vault_id, view in visible
        },
        len(commits) - len(visible),
    )


def search_payload(
    profile: str,
    response: SearchResponse,
    commits: Mapping[str, CommitView] | None = None,
) -> dict[str, JsonValue]:
    """Render stable retrieval results and optional hit-vault commit views."""
    hits: list[JsonValue] = []
    for hit in response.hits:
        metadata, metadata_truncated = metadata_payload(hit.metadata)
        hits.append(
            {
                "chunk_id": hit.chunk_id,
                "vault_id": hit.ref.vault_id,
                "path": hit.ref.path,
                "heading": list(hit.ref.heading),
                "start_line": hit.ref.lines.start,
                "end_line": hit.ref.lines.end,
                "source_hash": hit.ref.source_hash,
                "citation": hit.ref.citation,
                "text": hit.text,
                "continuation": continuation_payload(hit.continuation),
                "metadata": metadata,
                "metadata_truncated": metadata_truncated,
                "scores": cast(JsonValue, asdict(hit.scores)),
            }
        )
    payload: dict[str, JsonValue] = {
        "profile": profile[:_MAX_IDENTITY_CHARACTERS],
        "hits": hits,
        "degraded": {
            "semantic_search": response.degraded.semantic_search,
            "reason": response.degraded.reason,
        },
        "elapsed_ms": response.elapsed_ms,
        "index_age_seconds": response.index_age,
    }
    if commits is not None:
        commit_payload, commits_truncated = _commit_payload(commits)
        payload["vault_commits"] = commit_payload
        payload["vault_commits_truncated"] = commits_truncated
    return payload


def read_payload(
    profile: str,
    response: ReadResponse,
    commits: Mapping[str, CommitView] | None = None,
) -> dict[str, JsonValue]:
    """Render a stable source read response."""
    payload: dict[str, JsonValue] = {
        "profile": profile[:_MAX_IDENTITY_CHARACTERS],
        "vault_id": response.ref.vault_id,
        "path": response.ref.path,
        "heading": list(response.ref.heading),
        "start_line": response.ref.lines.start,
        "end_line": response.ref.lines.end,
        "source_hash": response.ref.source_hash,
        "citation": response.ref.citation,
        "text": response.text,
        "continuation": continuation_payload(response.continuation),
    }
    if commits is not None:
        commit_payload, commits_truncated = _commit_payload(commits)
        payload["vault_commits"] = commit_payload
        payload["vault_commits_truncated"] = commits_truncated
    return payload
