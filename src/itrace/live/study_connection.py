from collections.abc import Callable
from threading import Lock
from time import monotonic


class BrowserConnection:
    def __init__(self, clock: Callable[[], float] = monotonic) -> None:
        self._clock = clock
        self._lock = Lock()
        self._last_seen: float | None = None

    def renew(self) -> None:
        with self._lock:
            self._last_seen = self._clock()

    def expired(self) -> bool:
        with self._lock:
            return self._last_seen is not None and self._clock() - self._last_seen > 10.0
