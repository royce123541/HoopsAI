from datetime import date

import pytest

from hoopsai.sources.base import current_season, parse_season, request_key, season_label
from hoopsai.sources.throttle import Throttle, call_with_retry


@pytest.mark.parametrize(
    ("season", "label"), [(2004, "2004-05"), (1999, "1999-00"), (2025, "2025-26")]
)
def test_season_label(season: int, label: str) -> None:
    assert season_label(season) == label
    assert parse_season(label) == season


def test_parse_season_accepts_start_year_and_rejects_mismatch() -> None:
    assert parse_season("2024") == 2024
    with pytest.raises(ValueError):
        parse_season("2024-26")


@pytest.mark.parametrize(
    ("today", "season"),
    [(date(2026, 10, 6), 2026), (date(2026, 6, 15), 2025), (date(2026, 8, 1), 2026)],
)
def test_current_season(today: date, season: int) -> None:
    assert current_season(today) == season


def test_request_key_is_independent_of_param_order() -> None:
    a = request_key("nba_stats", "leaguegamelog", {"season": "2024-25", "season_type": "Playoffs"})
    b = request_key("nba_stats", "leaguegamelog", {"season_type": "Playoffs", "season": "2024-25"})
    assert a == b


def test_throttle_sleeps_only_for_remaining_interval() -> None:
    now = [100.0]
    sleeps: list[float] = []

    def sleep(s: float) -> None:
        sleeps.append(s)
        now[0] += s

    throttle = Throttle(0.6, clock=lambda: now[0], sleep=sleep)
    throttle.wait()  # first call: no wait
    now[0] += 0.2
    throttle.wait()
    now[0] += 1.0
    throttle.wait()  # interval already elapsed
    assert sleeps == [pytest.approx(0.4)]


def test_call_with_retry_backs_off_then_succeeds() -> None:
    sleeps: list[float] = []
    outcomes: list[Exception | str] = [ConnectionError(), TimeoutError(), "ok"]

    def flaky() -> str:
        result = outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    assert call_with_retry(flaky, attempts=5, base_delay_s=2, what="t", sleep=sleeps.append) == "ok"
    assert sleeps == [2, 4]


def test_call_with_retry_reraises_after_last_attempt() -> None:
    sleeps: list[float] = []

    def broken() -> None:
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        call_with_retry(broken, attempts=3, base_delay_s=1, what="t", sleep=sleeps.append)
    assert sleeps == [1, 2]
