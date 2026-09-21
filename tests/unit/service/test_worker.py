from __future__ import annotations

from datetime import timedelta
from threading import Event
from uuid import UUID, uuid4

import pytest

from vault_rag.errors import StorageError
from vault_rag.service.worker import ClaimedSyncRequest, LeaseSession, WorkerRuntime


class FakeCoordinator:
    def __init__(self) -> None:
        self.runs = 0
        self.closed = False
        self.force_reconciles: list[bool] = []

    def run_once(
        self,
        *,
        retry_pending_embeddings: bool = False,
        force_reconcile: bool = False,
    ) -> None:
        assert retry_pending_embeddings is True
        self.runs += 1
        self.force_reconciles.append(force_reconcile)

    def close(self) -> None:
        self.closed = True


class FakeLease:
    def __init__(self, *, acquired: bool) -> None:
        self.acquired = acquired
        self.refreshes = 0
        self.releases = 0

    def acquire(self) -> bool:
        return self.acquired

    def refresh(self) -> bool:
        self.refreshes += 1
        return self.acquired

    def promote(self, revision: object, *, lexical_complete: bool, fully_reconciled: bool) -> None:
        raise AssertionError("worker lease promotion is not exercised by this fake")

    def release(self) -> None:
        self.releases += 1


class FakeRequestQueue:
    def __init__(self, request_id: UUID | None) -> None:
        self.request = None if request_id is None else ClaimedSyncRequest(request_id, uuid4())
        self.completed: list[tuple[UUID, bool]] = []

    def claim_request(self) -> ClaimedSyncRequest | None:
        request = self.request
        self.request = None
        return request

    def complete_request(self, request: ClaimedSyncRequest, *, succeeded: bool) -> None:
        self.completed.append((request.request_id, succeeded))


def test_vault_lease_session_skips_when_not_acquired() -> None:
    lease = FakeLease(acquired=False)
    session = LeaseSession(lease, timedelta(seconds=10))

    assert session.acquire() is False
    session.release()
    assert lease.releases == 0


@pytest.mark.parametrize("interval", [timedelta(0), timedelta(seconds=-1)])
def test_worker_rejects_nonpositive_poll_interval(interval: timedelta) -> None:
    with pytest.raises(ValueError):
        WorkerRuntime(FakeCoordinator(), poll_interval=interval)


def test_vault_lease_session_rejects_nonpositive_heartbeat() -> None:
    with pytest.raises(ValueError):
        LeaseSession(FakeLease(acquired=True), timedelta(0))


def test_worker_run_forever_handles_explicit_wake_and_stop() -> None:
    holder: list[WorkerRuntime] = []

    class StoppingCoordinator(FakeCoordinator):
        def run_once(
            self,
            *,
            retry_pending_embeddings: bool = False,
            force_reconcile: bool = False,
        ) -> None:
            super().run_once(
                retry_pending_embeddings=retry_pending_embeddings,
                force_reconcile=force_reconcile,
            )
            holder[0].request_stop()

    coordinator = StoppingCoordinator()
    worker = WorkerRuntime(
        coordinator,
        poll_interval=timedelta(minutes=5),
    )
    holder.append(worker)
    worker.request_sync()

    worker.run_forever()
    worker.close()

    assert coordinator.runs == 1
    assert coordinator.force_reconciles == [False]
    assert worker.run_once() is False


def test_worker_runs_one_pass_and_releases_owned_lease() -> None:
    coordinator = FakeCoordinator()
    worker = WorkerRuntime(
        coordinator,
        poll_interval=timedelta(minutes=5),
    )

    assert worker.run_once() is True
    assert coordinator.runs == 1
    assert coordinator.force_reconciles == [False]

    worker.close()
    assert coordinator.closed is True


def test_worker_only_runs_cleanup_when_runtime_explicitly_wires_it() -> None:
    coordinator = FakeCoordinator()
    cleanups: list[str] = []
    worker = WorkerRuntime(
        coordinator,
        poll_interval=timedelta(minutes=5),
        cleanup=lambda: cleanups.append("cleanup"),
    )

    assert worker.run_once() is True

    assert coordinator.runs == 1
    assert cleanups == ["cleanup"]


def test_vault_lease_session_fails_publication_guard_after_heartbeat_loss() -> None:
    heartbeat_failed = Event()

    class LeaseLosingLease(FakeLease):
        def refresh(self) -> bool:
            self.refreshes += 1
            heartbeat_failed.set()
            raise StorageError("database connection lost")

    lease = LeaseLosingLease(acquired=True)
    session = LeaseSession(lease, timedelta(milliseconds=1))

    assert session.acquire() is True
    assert heartbeat_failed.wait(timeout=1)
    with pytest.raises(StorageError, match="lease was lost"):
        session.ensure_held()
    session.release()

    assert lease.refreshes == 1
    assert lease.releases == 1


def test_worker_releases_lease_when_sync_request_claim_fails() -> None:
    class FailingRequestQueue(FakeRequestQueue):
        def claim_request(self) -> ClaimedSyncRequest | None:
            raise StorageError("database unavailable")

    coordinator = FakeCoordinator()
    worker = WorkerRuntime(
        coordinator,
        poll_interval=timedelta(minutes=5),
        request_queue=FailingRequestQueue(None),
    )

    with pytest.raises(StorageError, match="database unavailable"):
        worker.run_once()
    assert coordinator.runs == 0


def test_worker_polling_recovers_after_request_queue_outage() -> None:
    holder: list[WorkerRuntime] = []

    class FailOnceRequestQueue(FakeRequestQueue):
        def __init__(self) -> None:
            super().__init__(None)
            self.claims = 0

        def claim_request(self) -> ClaimedSyncRequest | None:
            self.claims += 1
            if self.claims == 1:
                raise StorageError("database unavailable")
            return None

    class StoppingCoordinator(FakeCoordinator):
        def run_once(
            self,
            *,
            retry_pending_embeddings: bool = False,
            force_reconcile: bool = False,
        ) -> None:
            super().run_once(
                retry_pending_embeddings=retry_pending_embeddings,
                force_reconcile=force_reconcile,
            )
            holder[0].request_stop()

    requests = FailOnceRequestQueue()
    coordinator = StoppingCoordinator()
    worker = WorkerRuntime(
        coordinator,
        poll_interval=timedelta(milliseconds=1),
        request_queue=requests,
    )
    holder.append(worker)

    worker.run_forever()

    assert requests.claims == 2
    assert coordinator.runs == 1
    assert coordinator.force_reconciles == [False]


def test_worker_completes_claimed_sync_request_and_owned_resources() -> None:
    coordinator = FakeCoordinator()
    request_id = uuid4()
    requests = FakeRequestQueue(request_id)
    closed: list[str] = []
    worker = WorkerRuntime(
        coordinator,
        poll_interval=timedelta(minutes=5),
        request_queue=requests,
        close_resources=lambda: closed.append("resources"),
    )

    assert worker.run_once() is True
    assert requests.completed == [(request_id, True)]
    assert coordinator.force_reconciles == [True]

    worker.close()
    worker.close()
    assert closed == ["resources"]


def test_cleanup_failure_does_not_fence_successful_claimed_sync_request() -> None:
    coordinator = FakeCoordinator()
    request_id = uuid4()
    requests = FakeRequestQueue(request_id)
    worker = WorkerRuntime(
        coordinator,
        poll_interval=timedelta(minutes=5),
        request_queue=requests,
        cleanup=lambda: (_ for _ in ()).throw(StorageError("cleanup unavailable")),
    )

    with pytest.raises(StorageError, match="cleanup unavailable"):
        worker.run_once()

    assert coordinator.runs == 1
    assert requests.completed == [(request_id, True)]
