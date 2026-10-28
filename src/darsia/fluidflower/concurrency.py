"""Small concurrency controls shared by FluidFlower workflows."""

from __future__ import annotations

from threading import Lock


class MutableWorkerLimit:
    """Thread-safe live concurrency target for bounded work schedulers."""

    def __init__(self, value: int, *, maximum: int) -> None:
        self._lock = Lock()
        self._maximum = max(1, int(maximum))
        self._value = max(1, min(int(value), self._maximum))

    @property
    def maximum(self) -> int:
        return self._maximum

    def set(self, value: int) -> None:
        with self._lock:
            self._value = max(1, min(int(value), self._maximum))

    def get(self) -> int:
        with self._lock:
            return self._value
