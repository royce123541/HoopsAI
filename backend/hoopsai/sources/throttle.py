import logging
import time
from collections.abc import Callable

log = logging.getLogger(__name__)


class Throttle:
    """Enforce a minimum interval between calls (stats.nba.com bans aggressive clients)."""

    def __init__(
        self,
        min_interval_s: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval_s = min_interval_s
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            remaining = self.min_interval_s - (self._clock() - self._last)
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()


def call_with_retry[T](
    fn: Callable[[], T],
    *,
    attempts: int,
    base_delay_s: float,
    what: str,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn`, retrying any exception with exponential backoff (base, 2x base, 4x base...)."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts:
                raise
            delay = base_delay_s * 2 ** (attempt - 1)
            log.warning(
                "%s failed (attempt %d/%d: %s); retrying in %.0fs",
                what,
                attempt,
                attempts,
                type(exc).__name__,
                delay,
            )
            sleep(delay)
    raise AssertionError("unreachable")
