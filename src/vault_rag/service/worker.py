"""Lease-coordinated synchronization worker runtime."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from threading import Event, RLock, Thread
from typing import Protocol
from uuid import UUID

from vault_rag.errors import StorageError
from vault_rag.storage.ports import RevisionBuildStore


class WorkerCoordinator(Protocol):
    """Synchronous reconciliation surface owned by the worker."""

    def run_once(
        self,
        *,
        retry_pending_embeddings: bool = False,
        force_reconcile: bool = False,
    ) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ClaimedSyncRequest:
    """Opaque ownership token for one durable synchronization request."""

    request_id: UUID
    claim_token: UUID


class WorkerLease(Protocol):
    """Session-scoped lease for one vault reconciliation pass."""

    def acquire(self) -> bool: ...

    def refresh(self) -> bool: ...

    def promote(
        self,
        revision: RevisionBuildStore,
        *,
        lexical_complete: bool,
        fully_reconciled: bool,
    ) -> None: ...

    def release(self) -> None: ...


class LeaseSession:
    """Heartbeat one vault lease and provide an explicit publication guard."""

    def __init__(self, lease: WorkerLease, heartbeat_interval: timedelta) -> None:
        if heartbeat_interval.total_seconds() <= 0:
            raise ValueError("worker heartbeat interval must be positive")
        self._lease = lease
        self._heartbeat_interval = heartbeat_interval.total_seconds()
        self._stop = Event()
        self._lost = Event()
        self._thread: Thread | None = None
        self._acquired = False

    def acquire(self) -> bool:
        if not self._lease.acquire():
            return False
        self._stop.clear()
        self._lost.clear()
        self._acquired = True
        self._thread = Thread(target=self._heartbeat, daemon=True)
        self._thread.start()
        return True

    def ensure_held(self) -> None:
        if self._lost.is_set():
            raise StorageError("PostgreSQL reconciliation lease was lost")
        self._refresh()

    def promote(
        self,
        revision: RevisionBuildStore,
        *,
        lexical_complete: bool,
        fully_reconciled: bool,
    ) -> None:
        """Publish through the lock-owning PostgreSQL session."""
        self.ensure_held()
        try:
            self._lease.promote(
                revision,
                lexical_complete=lexical_complete,
                fully_reconciled=fully_reconciled,
            )
        except StorageError:
            self._lost.set()
            raise

    def release(self) -> None:
        if not self._acquired:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._acquired = False
        self._lease.release()

    def _refresh(self) -> None:
        try:
            refreshed = self._lease.refresh()
        except StorageError as exc:
            self._lost.set()
            raise StorageError("PostgreSQL reconciliation lease was lost") from exc
        if not refreshed:
            self._lost.set()
            raise StorageError("PostgreSQL reconciliation lease was lost")

    def _heartbeat(self) -> None:
        while not self._stop.wait(self._heartbeat_interval):
            try:
                self._refresh()
            except StorageError:
                return


class WorkerRequestQueue(Protocol):
    """Durable admin-triggered synchronization requests."""

    def claim_request(self) -> ClaimedSyncRequest | None: ...

    def complete_request(self, request: ClaimedSyncRequest, *, succeeded: bool) -> None: ...


class WorkerRuntime:
    """Run bounded reconciliation passes; the coordinator leases each vault."""

    def __init__(
        self,
        coordinator: WorkerCoordinator,
        *,
        poll_interval: timedelta,
        request_queue: WorkerRequestQueue | None = None,
        cleanup: Callable[[], None] | None = None,
        close_resources: Callable[[], None] | None = None,
        health_check: Callable[[], bool] | None = None,
    ) -> None:
        if poll_interval.total_seconds() <= 0:
            raise ValueError("worker poll interval must be positive")
        self._coordinator = coordinator
        self._poll_interval = poll_interval.total_seconds()
        self._request_queue = request_queue
        self._close_resources = close_resources
        self._cleanup = cleanup
        self._health_check = health_check
        self._stop = Event()
        self._wake = Event()
        self._lifecycle_lock = RLock()
        self._closed = False

    def live(self) -> bool:
        """Report process liveness independently of transient database availability."""
        with self._lifecycle_lock:
            return not self._closed

    def ready(self) -> bool:
        """Check the owned worker database connection without changing its lifecycle."""
        if not self.live():
            return False
        if self._health_check is None:
            return True
        try:
            return self._health_check()
        except StorageError:
            return False

    def run_once(self) -> bool:
        """Run one pass; each repository independently acquires its own lease."""
        with self._lifecycle_lock:
            if self._closed:
                return False
        request: ClaimedSyncRequest | None = None
        succeeded = False
        try:
            if self._request_queue is not None:
                request = self._request_queue.claim_request()
            self._coordinator.run_once(
                retry_pending_embeddings=True,
                force_reconcile=request is not None,
            )
            succeeded = True
            if self._cleanup is not None:
                self._cleanup()
            return True
        finally:
            if request is not None and self._request_queue is not None:
                self._request_queue.complete_request(request, succeeded=succeeded)

    def run_forever(self) -> None:
        """Poll until closed, allowing explicit wakes between scheduled passes."""
        while not self._stop.is_set():
            with suppress(StorageError):
                self.run_once()
            self._wake.wait(self._poll_interval)
            self._wake.clear()

    def request_stop(self) -> None:
        """Stop polling after the active reconciliation pass completes."""
        self._stop.set()
        self._wake.set()

    def request_sync(self) -> None:
        """Wake the local worker loop for an early lease attempt."""
        self._wake.set()

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
            self._wake.set()
        try:
            self._coordinator.close()
        finally:
            if self._close_resources is not None:
                self._close_resources()
