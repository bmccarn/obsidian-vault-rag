"""Synchronization for repository reconciliation and query access."""

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Condition
from time import monotonic

from vault_rag.errors import ServiceBusyError


class LifecycleLock:
    """A writer-fair reader-writer lock exposed through context managers."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @contextmanager
    def read(self, timeout: float) -> Iterator[None]:
        """Acquire shared access before the deadline."""
        deadline = monotonic() + timeout
        with self._condition:
            while self._writer or self._waiting_writers:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise ServiceBusyError("repository reconciliation is in progress")
                self._condition.wait(remaining)
            self._readers += 1

        try:
            yield
        finally:
            with self._condition:
                self._readers -= 1
                if self._readers == 0:
                    self._condition.notify_all()

    @contextmanager
    def write(self, timeout: float | None = None) -> Iterator[None]:
        """Acquire exclusive access, optionally before the deadline."""
        deadline = None if timeout is None else monotonic() + timeout
        with self._condition:
            self._waiting_writers += 1
            self._condition.notify_all()
            try:
                while self._writer or self._readers:
                    if deadline is None:
                        self._condition.wait()
                        continue
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise ServiceBusyError("repository reconciliation is in progress")
                    self._condition.wait(remaining)
                self._writer = True
            finally:
                self._waiting_writers -= 1
                self._condition.notify_all()

        try:
            yield
        finally:
            with self._condition:
                self._writer = False
                self._condition.notify_all()
