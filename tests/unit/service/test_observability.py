from __future__ import annotations

import io
import json
import logging
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from vault_rag.service.observability import (
    ServiceMetrics,
    ServiceObservability,
    StructuredEvents,
    bounded_http_endpoint,
    configure_structured_event_logging,
)
from vault_rag.service.state import IndexDiagnosticState, IndexReportState
from vault_rag.storage import IndexSnapshot


def _snapshot() -> IndexSnapshot:
    return IndexSnapshot(
        source_count=3,
        chunk_count=5,
        ready_count=4,
        pending_count=1,
        vector_bytes=96,
        indexed_at=datetime(2026, 8, 7, tzinfo=UTC),
        schema_version=1,
        manifest_fingerprint="manifest",
        parser_fingerprint="parser",
        chunker_fingerprint="chunker",
        embedding_config_fingerprint="embedding",
        observed_fingerprints=(),
        vector_dimensions=(),
        vector_config_fingerprints=(),
    )


def _report() -> IndexReportState:
    return IndexReportState(
        total_sources=11,
        added_sources=2,
        changed_sources=3,
        unchanged_sources=4,
        deleted_sources=5,
        ready_chunks=6,
        pending_chunks=7,
        parse_failures=8,
        embedding_failures=9,
        embedding_requests=10,
        blocking_failures=12,
        diagnostics=(
            IndexDiagnosticState(
                vault_id="vault-a",
                path="sensitive-source.md",
                category="parse",
                message="diagnostic-message",
            ),
        ),
        elapsed_ms=13,
    )


def _events(stream: io.StringIO) -> StructuredEvents:
    logger = logging.getLogger("vault-rag.test.structured-events")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(stream))
    return StructuredEvents(logger)


def test_runtime_structured_logging_is_idempotent_and_content_free(
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = logging.getLogger("vault-rag.test.runtime-events")
    logger.handlers.clear()
    logger.propagate = True

    configure_structured_event_logging(logger.name)
    configure_structured_event_logging(logger.name)
    StructuredEvents(logger).emit(
        "http_request",
        endpoint="/v1/search",
        mode="hybrid",
        outcome="success",
        elapsed_ms=12.5,
        query="must-not-appear",
    )

    assert len(logger.handlers) == 1
    assert logger.propagate is False
    payload = json.loads(capsys.readouterr().err)
    assert payload == {
        "elapsed_ms": 12.5,
        "endpoint": "/v1/search",
        "event": "http_request",
        "mode": "hybrid",
        "outcome": "success",
    }
    logger.handlers.clear()


def test_http_endpoint_bounding_never_logs_arbitrary_paths() -> None:
    assert bounded_http_endpoint("/v1/search") == "/v1/search"
    assert bounded_http_endpoint("/private/source-name.md") == "other"


def test_metrics_use_an_isolated_registry_with_only_stable_labels() -> None:
    """Using the global registry or unbounded operation data leaks unrelated/sensitive series."""
    metrics = ServiceMetrics()
    metrics.fetch_finished("vault-a", "success", 125)
    metrics.reconciliation_finished(
        "vault-a",
        "success",
        250,
        _snapshot(),
        _report(),
        False,
        True,
    )
    metrics.observe_search("hybrid", 375)

    rendered = metrics.render().decode()

    assert 'vault_rag_fetch_total{outcome="success",vault_id="vault-a"} 1.0' in rendered
    assert 'vault_rag_fetch_duration_seconds_sum{vault_id="vault-a"} 0.125' in rendered
    assert 'vault_rag_reconciliation_total{outcome="success",vault_id="vault-a"} 1.0' in rendered
    assert 'vault_rag_reconciliation_duration_seconds_sum{vault_id="vault-a"} 0.25' in rendered
    assert 'vault_rag_sources{vault_id="vault-a"} 3.0' in rendered
    assert 'vault_rag_chunks{vault_id="vault-a"} 5.0' in rendered
    assert 'vault_rag_ready_chunks{vault_id="vault-a"} 4.0' in rendered
    assert 'vault_rag_pending_chunks{vault_id="vault-a"} 1.0' in rendered
    assert 'vault_rag_vector_bytes{vault_id="vault-a"} 96.0' in rendered
    assert 'vault_rag_search_total{mode="hybrid"} 1.0' in rendered
    assert 'vault_rag_search_duration_seconds_sum{mode="hybrid"} 0.375' in rendered
    assert 'vault_rag_sync_degraded{vault_id="vault-a"} 0.0' in rendered
    assert 'vault_rag_semantic_degraded{vault_id="vault-a"} 1.0' in rendered
    assert 'vault_rag_reconciliation_total_sources{vault_id="vault-a"} 11.0' in rendered
    assert 'vault_rag_reconciliation_added_sources{vault_id="vault-a"} 2.0' in rendered
    assert 'vault_rag_reconciliation_changed_sources{vault_id="vault-a"} 3.0' in rendered
    assert 'vault_rag_reconciliation_unchanged_sources{vault_id="vault-a"} 4.0' in rendered
    assert 'vault_rag_reconciliation_deleted_sources{vault_id="vault-a"} 5.0' in rendered
    assert 'vault_rag_reconciliation_parse_failures{vault_id="vault-a"} 8.0' in rendered
    assert 'vault_rag_reconciliation_embedding_requests{vault_id="vault-a"} 10.0' in rendered
    assert 'vault_rag_reconciliation_embedding_failures{vault_id="vault-a"} 9.0' in rendered
    assert 'vault_rag_reconciliation_blocking_failures{vault_id="vault-a"} 12.0' in rendered
    assert "vault_rag_reconciliation_ready_chunks" not in rendered
    assert "vault_rag_reconciliation_pending_chunks" not in rendered
    metrics.reconciliation_finished(
        "vault-a",
        "success",
        25,
        replace(_snapshot(), pending_count=17),
        None,
        False,
        True,
    )
    rendered = metrics.render().decode()
    assert 'vault_rag_pending_chunks{vault_id="vault-a"} 17.0' in rendered
    assert 'vault_rag_reconciliation_total_sources{vault_id="vault-a"} 11.0' in rendered
    assert "python_gc" not in rendered
    for prohibited in ("commit", "profile", "query", "path", "url", "error"):
        assert prohibited not in rendered


def test_composed_observability_emits_bounded_repository_events() -> None:
    """Forwarding reports directly would expose diagnostics, commits, or remote content."""
    stream = io.StringIO()
    observability = ServiceObservability(ServiceMetrics(), _events(stream))
    report = _report()

    observability.fetch_finished("vault-a", "success", 125)
    observability.reconciliation_finished(
        "vault-a",
        "failure",
        250,
        _snapshot(),
        report,
        True,
        False,
    )

    fetch, reconciliation = map(json.loads, stream.getvalue().splitlines())
    assert fetch == {
        "elapsed_ms": 125,
        "event": "repository_fetch",
        "outcome": "success",
        "vault_id": "vault-a",
    }
    assert reconciliation == {
        "added_sources": 2,
        "blocking_failures": 12,
        "changed_sources": 3,
        "chunk_count": 5,
        "deleted_sources": 5,
        "elapsed_ms": 250,
        "embedding_failures": 9,
        "embedding_requests": 10,
        "event": "repository_reconciliation",
        "outcome": "failure",
        "parse_failures": 8,
        "pending_count": 1,
        "pending_chunks": 7,
        "ready_count": 4,
        "ready_chunks": 6,
        "semantic_degraded": False,
        "source_count": 3,
        "sync_degraded": True,
        "total_sources": 11,
        "unchanged_sources": 4,
        "vault_id": "vault-a",
        "vector_bytes": 96,
    }
    rendered = observability._metrics.render().decode()
    event_json = stream.getvalue()
    for sensitive in (
        "diagnostic-message",
        "sensitive-source.md",
        "a" * 40,
        "query",
        "url",
        "token",
    ):
        assert sensitive not in rendered
        assert sensitive not in event_json


def test_structured_events_drop_unknown_fields_truncate_strings_and_are_canonical() -> None:
    """Serializing arbitrary fields could expose request or remote content."""
    stream = io.StringIO()
    events = _events(stream)
    secret_query = "secret-query"
    source_body = "source-body"
    remote_url = "https://example.invalid/private"
    token = "token-value"
    remote_body = "remote-body"
    raw_object = object()

    events.emit(
        "search_completed",
        vault_id="vault-a",
        mode="hybrid",
        outcome="x" * 501,
        query=secret_query,
        source_body=source_body,
        url=remote_url,
        token=token,
        remote_body=remote_body,
        elapsed_ms=raw_object,  # type: ignore[arg-type]
    )

    line = stream.getvalue().strip()
    assert line == json.dumps(
        json.loads(line), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    payload = json.loads(line)
    assert payload == {
        "event": "search_completed",
        "mode": "hybrid",
        "outcome": "x" * 500,
        "vault_id": "vault-a",
    }
    for secret in (secret_query, source_body, remote_url, token, remote_body):
        assert secret not in line


def test_metrics_expose_only_fixed_cardinality_operational_families() -> None:
    """Operational paths must emit bounded categories rather than request or database content."""
    metrics = ServiceMetrics()

    metrics.observe_http("/v1/search", "hybrid", "success", 12)
    metrics.observe_postgres_operation("lexical_search", "success", 3)
    metrics.observe_pool(size=4, used=1, idle=3, wait=0)
    metrics.observe_pool_acquire(2)
    metrics.pool_reconnected()
    metrics.pool_connection_failed()
    metrics.observe_lock("wait", 1)
    metrics.observe_lock("busy", 0)
    metrics.observe_lock("loss", 1)
    metrics.observe_reconciliation("build", "success", 4)
    metrics.observe_reconciliation("promotion", "success", 5)
    metrics.observe_reconciliation("failure", "failure", 6)
    metrics.observe_database_state(
        "vault-a",
        revision_count=2,
        pending_count=3,
        pending_age_seconds=4,
        sync_queue_depth=5,
        sync_queue_age_seconds=6,
        dense_mode=True,
    )
    metrics.observe_cleanup("success", deleted=7, elapsed_ms=8)
    metrics.observe_migration("check", "success", version=2, elapsed_ms=9)

    rendered = metrics.render().decode()

    for family in (
        "vault_rag_http_requests_total",
        "vault_rag_postgres_operations_total",
        "vault_rag_pool_size",
        "vault_rag_pool_acquire_duration_seconds",
        "vault_rag_pool_reconnect_total",
        "vault_rag_pool_connection_failure_total",
        "vault_rag_worker_lock_total",
        "vault_rag_reconciliation_phase_total",
        "vault_rag_revision_count",
        "vault_rag_pending_embedding_age_seconds",
        "vault_rag_sync_queue_depth",
        "vault_rag_cleanup_total",
        "vault_rag_migration_total",
        "vault_rag_dense_mode",
    ):
        assert family in rendered
    for sentinel in (
        "select secret",
        "/absolute/secret-path",
        "postgresql://user:credential@db/secret",
        "provider-secret",
        "fingerprint-secret",
    ):
        assert sentinel not in rendered
