from threading import Event, Thread

import pytest

from vault_rag.errors import ServiceBusyError
from vault_rag.service.locking import LifecycleLock

THREAD_TIMEOUT = 1.0


def join(thread: Thread) -> None:
    thread.join(THREAD_TIMEOUT)
    assert not thread.is_alive()


def test_waiting_writer_blocks_new_reader() -> None:
    lock = LifecycleLock()
    first_reader_entered = Event()
    release_first_reader = Event()
    writer_entered = Event()
    release_writer = Event()
    second_reader_attempted = Event()
    second_reader_entered = Event()

    def hold_first_reader() -> None:
        with lock.read(timeout=THREAD_TIMEOUT):
            first_reader_entered.set()
            assert release_first_reader.wait(THREAD_TIMEOUT)

    def hold_writer() -> None:
        with lock.write():
            writer_entered.set()
            assert release_writer.wait(THREAD_TIMEOUT)

    def enter_second_reader() -> None:
        second_reader_attempted.set()
        with lock.read(timeout=THREAD_TIMEOUT):
            second_reader_entered.set()

    first_reader = Thread(target=hold_first_reader)
    first_reader.start()
    assert first_reader_entered.wait(THREAD_TIMEOUT)

    writer = Thread(target=hold_writer)
    writer.start()
    with lock._condition:
        assert lock._condition.wait_for(lambda: lock._waiting_writers == 1, timeout=THREAD_TIMEOUT)

    second_reader = Thread(target=enter_second_reader)
    second_reader.start()
    assert second_reader_attempted.wait(THREAD_TIMEOUT)
    assert not second_reader_entered.wait(0.05)

    release_first_reader.set()
    assert writer_entered.wait(THREAD_TIMEOUT)
    assert not second_reader_entered.is_set()

    release_writer.set()
    assert second_reader_entered.wait(THREAD_TIMEOUT)
    join(first_reader)
    join(writer)
    join(second_reader)


def test_readers_are_admitted_concurrently() -> None:
    lock = LifecycleLock()
    first_reader_entered = Event()
    second_reader_entered = Event()
    release_readers = Event()

    def hold_reader(entered: Event) -> None:
        with lock.read(timeout=THREAD_TIMEOUT):
            entered.set()
            assert release_readers.wait(THREAD_TIMEOUT)

    first_reader = Thread(target=hold_reader, args=(first_reader_entered,))
    first_reader.start()
    assert first_reader_entered.wait(THREAD_TIMEOUT)

    second_reader = Thread(target=hold_reader, args=(second_reader_entered,))
    second_reader.start()
    assert second_reader_entered.wait(THREAD_TIMEOUT)

    release_readers.set()
    join(first_reader)
    join(second_reader)


def test_writer_is_exclusive() -> None:
    lock = LifecycleLock()

    with lock.write():
        with pytest.raises(ServiceBusyError), lock.read(timeout=0):
            pass
        with pytest.raises(ServiceBusyError), lock.write(timeout=0):
            pass


def test_read_timeout_raises_service_busy_error() -> None:
    lock = LifecycleLock()

    with (
        lock.write(),
        pytest.raises(ServiceBusyError) as raised,
        lock.read(timeout=0),
    ):
        pass

    assert str(raised.value) == "repository reconciliation is in progress"
    assert raised.value.code == "service_busy"


def test_write_timeout_does_not_block_future_readers() -> None:
    lock = LifecycleLock()

    with lock.read(timeout=THREAD_TIMEOUT):
        with pytest.raises(ServiceBusyError), lock.write(timeout=0):
            pass
        with lock.read(timeout=THREAD_TIMEOUT):
            pass


def test_exceptions_release_reader_and_writer() -> None:
    lock = LifecycleLock()

    with (
        pytest.raises(RuntimeError, match="reader failure"),
        lock.read(timeout=THREAD_TIMEOUT),
    ):
        raise RuntimeError("reader failure")
    with lock.write(timeout=THREAD_TIMEOUT):
        pass

    with (
        pytest.raises(RuntimeError, match="writer failure"),
        lock.write(timeout=THREAD_TIMEOUT),
    ):
        raise RuntimeError("writer failure")
    with lock.read(timeout=THREAD_TIMEOUT):
        pass
