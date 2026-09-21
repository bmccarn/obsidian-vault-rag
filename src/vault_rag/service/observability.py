"""Bounded, transport-neutral service metrics and structured events."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from vault_rag.domain import JsonValue
from vault_rag.service.state import IndexReportState
from vault_rag.storage import IndexSnapshot

_MAX_STRING_LENGTH = 500
_ALLOWED_EVENT_FIELDS = frozenset(
    {
        "endpoint",
        "vault_id",
        "correlation_id",
        "outcome",
        "mode",
        "tool",
        "elapsed_ms",
        "source_count",
        "chunk_count",
        "ready_count",
        "pending_count",
        "vector_bytes",
        "sync_degraded",
        "semantic_degraded",
        "total_sources",
        "added_sources",
        "changed_sources",
        "unchanged_sources",
        "deleted_sources",
        "parse_failures",
        "embedding_requests",
        "embedding_failures",
        "blocking_failures",
        "ready_chunks",
        "pending_chunks",
    }
)
_METRIC_REPORT_FIELDS = (
    "total_sources",
    "added_sources",
    "changed_sources",
    "unchanged_sources",
    "deleted_sources",
    "parse_failures",
    "embedding_requests",
    "embedding_failures",
    "blocking_failures",
)
_EVENT_REPORT_FIELDS = (
    "total_sources",
    "added_sources",
    "changed_sources",
    "unchanged_sources",
    "deleted_sources",
    "parse_failures",
    "embedding_requests",
    "embedding_failures",
    "blocking_failures",
    "ready_chunks",
    "pending_chunks",
)
_OUTCOMES = frozenset({"success", "failure"})
_MODES = frozenset({"lexical", "dense", "hybrid", "none"})
_HTTP_ENDPOINTS = frozenset(
    {
        "/v1/search",
        "/v1/read",
        "/v1/profiles",
        "/v1/status",
        "/v1/admin/sync",
        "/health/live",
        "/health/ready",
        "/metrics",
        "/mcp",
        "other",
    }
)
_MCP_TOOLS = frozenset({"vault_search", "vault_read", "vault_profiles", "vault_status"})
_POSTGRES_OPERATIONS = frozenset(
    {
        "active_revisions",
        "identifier_candidates",
        "lexical_search",
        "dense_search",
        "vector_scope",
        "chunks_by_ids",
        "source_provenance",
        "pending_chunks",
        "snapshot",
        "snapshots",
        "request_sync",
        "claim_request",
        "complete_request",
        "migration_check",
        "migration_apply",
        "cleanup",
        "database_state",
    }
)
_LOCK_EVENTS = frozenset({"wait", "busy", "loss"})
_RECONCILIATION_PHASES = frozenset({"fetch", "build", "promotion", "failure"})
_MIGRATION_OPERATIONS = frozenset({"check", "apply"})
_STRUCTURED_HANDLER_NAME = "vault-rag-structured-events"


def configure_structured_event_logging(logger_name: str) -> None:
    """Send one named structured-event logger to stderr exactly once."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if any(handler.get_name() == _STRUCTURED_HANDLER_NAME for handler in logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.set_name(_STRUCTURED_HANDLER_NAME)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def bounded_http_endpoint(endpoint: str) -> str:
    """Collapse arbitrary request paths into the fixed HTTP operation set."""
    return endpoint if endpoint in _HTTP_ENDPOINTS else "other"


class StructuredEvents:
    """Emit canonical, explicitly allowlisted JSON records through one logger."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def emit(self, event: str, **bounded_fields: JsonValue) -> None:
        """Log one bounded event without coercing arbitrary objects to text."""
        payload: dict[str, JsonValue] = {"event": event[:_MAX_STRING_LENGTH]}
        for name, value in bounded_fields.items():
            if name in _ALLOWED_EVENT_FIELDS:
                included, bounded = self._bounded_value(value)
                if included:
                    payload[name] = bounded
        self._logger.info(
            "%s",
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )

    @staticmethod
    def _bounded_value(value: JsonValue) -> tuple[bool, JsonValue]:
        if value is None or isinstance(value, bool):
            return True, value
        if isinstance(value, int):
            return True, value
        if isinstance(value, float):
            if not math.isfinite(value):
                return False, None
            return True, value
        if isinstance(value, str):
            return True, value[:_MAX_STRING_LENGTH]
        return False, None


class ServiceMetrics:
    """Prometheus observations isolated from the process-global collector registry."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self._fetch_total = Counter(
            "vault_rag_fetch_total",
            "Completed managed repository fetches.",
            ("vault_id", "outcome"),
            registry=self.registry,
        )
        self._fetch_duration = Histogram(
            "vault_rag_fetch_duration_seconds",
            "Managed repository fetch duration in seconds.",
            ("vault_id",),
            registry=self.registry,
        )
        self._reconciliation_total = Counter(
            "vault_rag_reconciliation_total",
            "Completed repository reconciliations.",
            ("vault_id", "outcome"),
            registry=self.registry,
        )
        self._reconciliation_duration = Histogram(
            "vault_rag_reconciliation_duration_seconds",
            "Repository reconciliation duration in seconds.",
            ("vault_id",),
            registry=self.registry,
        )
        self._sources = Gauge(
            "vault_rag_sources",
            "Current active source count.",
            ("vault_id",),
            registry=self.registry,
        )
        self._chunks = Gauge(
            "vault_rag_chunks", "Current active chunk count.", ("vault_id",), registry=self.registry
        )
        self._ready_chunks = Gauge(
            "vault_rag_ready_chunks",
            "Current ready embedding chunk count.",
            ("vault_id",),
            registry=self.registry,
        )
        self._pending_chunks = Gauge(
            "vault_rag_pending_chunks",
            "Current pending embedding chunk count.",
            ("vault_id",),
            registry=self.registry,
        )
        self._vector_bytes = Gauge(
            "vault_rag_vector_bytes",
            "Current persisted vector bytes.",
            ("vault_id",),
            registry=self.registry,
        )
        self._search_total = Counter(
            "vault_rag_search_total",
            "Completed searches.",
            ("mode",),
            registry=self.registry,
        )
        self._search_duration = Histogram(
            "vault_rag_search_duration_seconds",
            "Search duration in seconds.",
            ("mode",),
            registry=self.registry,
        )
        self._sync_degraded = Gauge(
            "vault_rag_sync_degraded",
            "Whether synchronization is currently degraded.",
            ("vault_id",),
            registry=self.registry,
        )
        self._semantic_degraded = Gauge(
            "vault_rag_semantic_degraded",
            "Whether semantic retrieval is currently degraded.",
            ("vault_id",),
            registry=self.registry,
        )
        self._report_gauges = {
            field: Gauge(
                f"vault_rag_reconciliation_{field}",
                f"Latest reconciliation {field.replace('_', ' ')}.",
                ("vault_id",),
                registry=self.registry,
            )
            for field in _METRIC_REPORT_FIELDS
        }
        self._http_total = Counter(
            "vault_rag_http_requests_total",
            "Completed HTTP requests.",
            ("endpoint", "mode", "outcome"),
            registry=self.registry,
        )
        self._http_duration = Histogram(
            "vault_rag_http_request_duration_seconds",
            "HTTP request duration in seconds.",
            ("endpoint", "mode", "outcome"),
            registry=self.registry,
        )
        self._mcp_total = Counter(
            "vault_rag_mcp_tool_calls_total",
            "Completed MCP tool calls.",
            ("tool", "outcome"),
            registry=self.registry,
        )
        self._mcp_duration = Histogram(
            "vault_rag_mcp_tool_call_duration_seconds",
            "MCP tool call duration in seconds.",
            ("tool", "outcome"),
            registry=self.registry,
        )
        self._postgres_total = Counter(
            "vault_rag_postgres_operations_total",
            "Completed named PostgreSQL operations.",
            ("operation", "outcome"),
            registry=self.registry,
        )
        self._postgres_duration = Histogram(
            "vault_rag_postgres_operation_duration_seconds",
            "Named PostgreSQL operation duration in seconds.",
            ("operation", "outcome"),
            registry=self.registry,
        )
        self._pool_size = Gauge(
            "vault_rag_pool_size", "Configured pool size.", registry=self.registry
        )
        self._pool_used = Gauge(
            "vault_rag_pool_used", "Checked out pool connections.", registry=self.registry
        )
        self._pool_idle = Gauge(
            "vault_rag_pool_idle", "Idle pool connections.", registry=self.registry
        )
        self._pool_wait = Gauge("vault_rag_pool_wait", "Pool waiters.", registry=self.registry)
        self._pool_acquire = Histogram(
            "vault_rag_pool_acquire_duration_seconds",
            "Pool acquisition duration in seconds.",
            registry=self.registry,
        )
        self._pool_reconnect = Counter(
            "vault_rag_pool_reconnect_total", "Pool reconnects.", registry=self.registry
        )
        self._pool_connection_failure = Counter(
            "vault_rag_pool_connection_failure_total",
            "Pool connection failures.",
            registry=self.registry,
        )
        self._lock_total = Counter(
            "vault_rag_worker_lock_total",
            "Worker advisory-lock observations.",
            ("event",),
            registry=self.registry,
        )
        self._lock_wait = Histogram(
            "vault_rag_worker_lock_wait_duration_seconds",
            "Worker advisory-lock acquisition duration in seconds.",
            registry=self.registry,
        )
        self._reconciliation_phase = Counter(
            "vault_rag_reconciliation_phase_total",
            "Reconciliation phase completions.",
            ("phase", "outcome"),
            registry=self.registry,
        )
        self._reconciliation_phase_duration = Histogram(
            "vault_rag_reconciliation_phase_duration_seconds",
            "Reconciliation phase duration in seconds.",
            ("phase", "outcome"),
            registry=self.registry,
        )
        self._revision_count = Gauge(
            "vault_rag_revision_count",
            "Vault revision count.",
            ("vault_id",),
            registry=self.registry,
        )
        self._pending_embedding_age = Gauge(
            "vault_rag_pending_embedding_age_seconds",
            "Age of oldest pending embedding.",
            ("vault_id",),
            registry=self.registry,
        )
        self._sync_queue_depth = Gauge(
            "vault_rag_sync_queue_depth",
            "Pending synchronization queue depth.",
            registry=self.registry,
        )
        self._sync_queue_age = Gauge(
            "vault_rag_sync_queue_age_seconds",
            "Age of oldest pending synchronization request.",
            registry=self.registry,
        )
        self._dense_mode = Gauge(
            "vault_rag_dense_mode",
            "Whether exact dense retrieval is usable.",
            ("vault_id",),
            registry=self.registry,
        )
        self._cleanup_total = Counter(
            "vault_rag_cleanup_total",
            "Completed cleanup passes.",
            ("outcome",),
            registry=self.registry,
        )
        self._cleanup_deleted = Gauge(
            "vault_rag_cleanup_deleted",
            "Rows deleted by the latest cleanup.",
            registry=self.registry,
        )
        self._cleanup_duration = Histogram(
            "vault_rag_cleanup_duration_seconds",
            "Cleanup duration in seconds.",
            ("outcome",),
            registry=self.registry,
        )
        self._migration_total = Counter(
            "vault_rag_migration_total",
            "Completed migration operations.",
            ("operation", "outcome"),
            registry=self.registry,
        )
        self._migration_duration = Histogram(
            "vault_rag_migration_duration_seconds",
            "Migration operation duration in seconds.",
            ("operation", "outcome"),
            registry=self.registry,
        )
        self._migration_version = Gauge(
            "vault_rag_migration_version",
            "Observed PostgreSQL schema version.",
            registry=self.registry,
        )

    def render(self) -> bytes:
        """Render only this instance's metric families in Prometheus text format."""
        return generate_latest(self.registry)

    def observe_http(self, endpoint: str, mode: str, outcome: str, elapsed_ms: int | float) -> None:
        """Record one completed request using route-template and fixed outcome labels."""
        safe_endpoint = bounded_http_endpoint(endpoint)
        safe_mode = self._mode(mode)
        safe_outcome = self._outcome(outcome)
        self._http_total.labels(safe_endpoint, safe_mode, safe_outcome).inc()
        self._http_duration.labels(safe_endpoint, safe_mode, safe_outcome).observe(
            self._seconds(elapsed_ms)
        )

    def observe_mcp(self, tool: str, outcome: str, elapsed_ms: int | float) -> None:
        """Record one MCP tool call using only fixed low-cardinality labels."""
        if tool not in _MCP_TOOLS:
            raise ValueError("unsupported MCP tool metric")
        safe_outcome = self._outcome(outcome)
        self._mcp_total.labels(tool, safe_outcome).inc()
        self._mcp_duration.labels(tool, safe_outcome).observe(self._seconds(elapsed_ms))

    def observe_postgres_operation(
        self, operation: str, outcome: str, elapsed_ms: int | float
    ) -> None:
        """Record a member of the frozen PostgreSQL operation allowlist."""
        if operation not in _POSTGRES_OPERATIONS:
            raise ValueError("unsupported PostgreSQL metric operation")
        safe_outcome = self._outcome(outcome)
        self._postgres_total.labels(operation, safe_outcome).inc()
        self._postgres_duration.labels(operation, safe_outcome).observe(self._seconds(elapsed_ms))

    def observe_pool(self, *, size: int, used: int, idle: int, wait: int) -> None:
        """Set bounded pool gauges from pool-maintained aggregate counters."""
        self._pool_size.set(max(0, size))
        self._pool_used.set(max(0, used))
        self._pool_idle.set(max(0, idle))
        self._pool_wait.set(max(0, wait))

    def observe_pool_acquire(self, elapsed_ms: int | float) -> None:
        """Record one pool acquisition latency."""
        self._pool_acquire.observe(self._seconds(elapsed_ms))

    def pool_reconnected(self) -> None:
        """Record a successful reconnect after a pool outage."""
        self._pool_reconnect.inc()

    def pool_connection_failed(self) -> None:
        """Record a pool open or checkout failure without exception details."""
        self._pool_connection_failure.inc()

    def observe_lock(self, event: str, elapsed_ms: int | float = 0) -> None:
        """Record a worker-lock wait, busy result, or lease loss."""
        if event not in _LOCK_EVENTS:
            raise ValueError("unsupported worker lock metric event")
        self._lock_total.labels(event).inc()
        if event == "wait":
            self._lock_wait.observe(self._seconds(elapsed_ms))

    def observe_reconciliation(self, phase: str, outcome: str, elapsed_ms: int | float) -> None:
        """Record one bounded reconciliation phase."""
        if phase not in _RECONCILIATION_PHASES:
            raise ValueError("unsupported reconciliation metric phase")
        safe_outcome = self._outcome(outcome)
        self._reconciliation_phase.labels(phase, safe_outcome).inc()
        self._reconciliation_phase_duration.labels(phase, safe_outcome).observe(
            self._seconds(elapsed_ms)
        )

    def observe_database_state(
        self,
        vault_id: str,
        *,
        revision_count: int,
        pending_count: int,
        pending_age_seconds: int | float,
        sync_queue_depth: int,
        sync_queue_age_seconds: int | float,
        dense_mode: bool,
    ) -> None:
        """Set state gauges supplied by bounded aggregate database queries."""
        self._revision_count.labels(vault_id).set(max(0, revision_count))
        self._pending_chunks.labels(vault_id).set(max(0, pending_count))
        self._pending_embedding_age.labels(vault_id).set(max(0.0, float(pending_age_seconds)))
        self._sync_queue_depth.set(max(0, sync_queue_depth))
        self._sync_queue_age.set(max(0.0, float(sync_queue_age_seconds)))
        self._dense_mode.labels(vault_id).set(float(dense_mode))

    def observe_cleanup(self, outcome: str, *, deleted: int, elapsed_ms: int | float) -> None:
        """Record an explicit bounded cleanup pass."""
        safe_outcome = self._outcome(outcome)
        self._cleanup_total.labels(safe_outcome).inc()
        self._cleanup_deleted.set(max(0, deleted))
        self._cleanup_duration.labels(safe_outcome).observe(self._seconds(elapsed_ms))

    def observe_migration(
        self, operation: str, outcome: str, *, version: int, elapsed_ms: int | float
    ) -> None:
        """Record migration health and the numeric schema version."""
        if operation not in _MIGRATION_OPERATIONS:
            raise ValueError("unsupported migration metric operation")
        safe_outcome = self._outcome(outcome)
        self._migration_total.labels(operation, safe_outcome).inc()
        self._migration_duration.labels(operation, safe_outcome).observe(self._seconds(elapsed_ms))
        if safe_outcome == "success":
            self._migration_version.set(max(0, version))

    def fetch_finished(self, vault_id: str, outcome: str, elapsed_ms: int | float) -> None:
        """Record one fetch result with a stable outcome category."""
        self._fetch_total.labels(vault_id, self._outcome(outcome)).inc()
        self._fetch_duration.labels(vault_id).observe(self._seconds(elapsed_ms))

    def reconciliation_finished(
        self,
        vault_id: str,
        outcome: str,
        elapsed_ms: int | float,
        snapshot: IndexSnapshot | None,
        report: IndexReportState | None,
        sync_degraded: bool,
        semantic_degraded: bool,
    ) -> None:
        """Record a final reconciliation and refresh bounded per-vault state."""
        self._reconciliation_total.labels(vault_id, self._outcome(outcome)).inc()
        self._reconciliation_duration.labels(vault_id).observe(self._seconds(elapsed_ms))
        if snapshot is not None:
            self.observe_snapshot(vault_id, snapshot, sync_degraded, semantic_degraded)
        else:
            self._sync_degraded.labels(vault_id).set(float(sync_degraded))
            self._semantic_degraded.labels(vault_id).set(float(semantic_degraded))
        if report is not None:
            for field, gauge in self._report_gauges.items():
                gauge.labels(vault_id).set(getattr(report, field))

    def observe_search(self, mode: str, elapsed_ms: int | float) -> None:
        """Record a completed search without recording request content or identity."""
        if mode not in _MODES:
            raise ValueError("unsupported search metric mode")
        self._search_total.labels(mode).inc()
        self._search_duration.labels(mode).observe(self._seconds(elapsed_ms))

    def observe_snapshot(
        self,
        vault_id: str,
        snapshot: IndexSnapshot,
        sync_degraded: bool,
        semantic_degraded: bool,
    ) -> None:
        """Set gauges from one aggregate vault snapshot."""
        self._sources.labels(vault_id).set(snapshot.source_count)
        self._chunks.labels(vault_id).set(snapshot.chunk_count)
        self._ready_chunks.labels(vault_id).set(snapshot.ready_count)
        self._pending_chunks.labels(vault_id).set(snapshot.pending_count)
        self._vector_bytes.labels(vault_id).set(snapshot.vector_bytes)
        self._sync_degraded.labels(vault_id).set(float(sync_degraded))
        self._semantic_degraded.labels(vault_id).set(float(semantic_degraded))

    def observe_status(
        self,
        vault_id: str,
        counts: Mapping[str, JsonValue],
        *,
        vector_bytes: int | None = None,
        sync_degraded: bool,
        semantic_degraded: bool,
    ) -> None:
        """Set gauges from a curated public status aggregate."""
        self._sources.labels(vault_id).set(self._count(counts, "sources"))
        self._chunks.labels(vault_id).set(self._count(counts, "chunks"))
        self._ready_chunks.labels(vault_id).set(self._count(counts, "ready"))
        self._pending_chunks.labels(vault_id).set(self._count(counts, "pending"))
        if vector_bytes is not None:
            self._vector_bytes.labels(vault_id).set(vector_bytes)
        self._sync_degraded.labels(vault_id).set(float(sync_degraded))
        self._semantic_degraded.labels(vault_id).set(float(semantic_degraded))

    @staticmethod
    def _count(counts: Mapping[str, JsonValue], name: str) -> int:
        value = counts.get(name, 0)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    @staticmethod
    def _outcome(outcome: str) -> str:
        if outcome not in _OUTCOMES:
            raise ValueError("unsupported metric outcome")
        return outcome

    @staticmethod
    def _mode(mode: str) -> str:
        if mode not in _MODES:
            raise ValueError("unsupported metric mode")
        return mode

    @staticmethod
    def _seconds(elapsed_ms: int | float) -> float:
        return max(0.0, float(elapsed_ms)) / 1000.0


class ServiceObservability:
    """Compose bounded metrics and structured repository transition events."""

    def __init__(self, metrics: ServiceMetrics, events: StructuredEvents) -> None:
        self._metrics = metrics
        self._events = events

    def fetch_finished(self, vault_id: str, outcome: str, elapsed_ms: float) -> None:
        self._metrics.fetch_finished(vault_id, outcome, elapsed_ms)
        self._metrics.observe_reconciliation("fetch", outcome, elapsed_ms)
        self._events.emit(
            "repository_fetch",
            vault_id=vault_id,
            outcome=outcome,
            elapsed_ms=elapsed_ms,
        )

    def reconciliation_phase(self, phase: str, outcome: str, elapsed_ms: float) -> None:
        """Record a bounded reconciliation phase independently of event content."""
        self._metrics.observe_reconciliation(phase, outcome, elapsed_ms)

    def reconciliation_finished(
        self,
        vault_id: str,
        outcome: str,
        elapsed_ms: float,
        snapshot: IndexSnapshot | None,
        report: IndexReportState | None,
        sync_degraded: bool,
        semantic_degraded: bool,
    ) -> None:
        self._metrics.reconciliation_finished(
            vault_id,
            outcome,
            elapsed_ms,
            snapshot,
            report,
            sync_degraded,
            semantic_degraded,
        )
        fields: dict[str, JsonValue] = {
            "vault_id": vault_id,
            "outcome": outcome,
            "elapsed_ms": elapsed_ms,
            "sync_degraded": sync_degraded,
            "semantic_degraded": semantic_degraded,
        }
        if snapshot is not None:
            fields.update(
                source_count=snapshot.source_count,
                chunk_count=snapshot.chunk_count,
                ready_count=snapshot.ready_count,
                pending_count=snapshot.pending_count,
                vector_bytes=snapshot.vector_bytes,
            )
        if report is not None:
            for field in _EVENT_REPORT_FIELDS:
                fields[field] = getattr(report, field)
        self._events.emit("repository_reconciliation", **fields)
