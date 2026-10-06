"""Where live game data comes from.

The plan's cdn.nba.com live feed returns 403 (Akamai "Access Denied") from this deployment's
network, for browsers too, so live data comes from stats.nba.com: ScoreboardV3 for game
status and PlayByPlayV3 for events. PlayByPlayV3 is also the training data's source, so the
in-game model sees the same event vocabulary live as in training."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from nba_api.stats.endpoints import playbyplayv3, scoreboardv3

from hoopsai.sources.base import GameStatus, PbpEventRow
from hoopsai.sources.nba_stats.parse import SCHEDULE_STATUS, parse_play_by_play
from hoopsai.sources.throttle import Throttle, call_with_retry


@dataclass(frozen=True)
class LiveGameStatus:
    game_id: str
    status: GameStatus
    period: int
    home_score: int
    away_score: int


class LiveSource(Protocol):
    def scoreboard(self, day: date) -> list[LiveGameStatus]: ...
    def play_by_play(self, game_id: str) -> list[PbpEventRow]: ...


class StatsLiveSource:
    """stats.nba.com, throttled. Few retries: a live poll that fails is simply retried on the
    next cycle, seconds later."""

    def __init__(self, min_interval_s: float = 0.6, timeout_s: int = 10) -> None:
        self._throttle = Throttle(min_interval_s)
        self._timeout_s = timeout_s

    def _call[T](self, what: str, fn: Callable[[], T]) -> T:
        def attempt() -> T:
            self._throttle.wait()
            return fn()

        return call_with_retry(attempt, attempts=2, base_delay_s=1.0, what=what)

    def scoreboard(self, day: date) -> list[LiveGameStatus]:
        payload = self._call(
            f"scoreboard {day}",
            lambda: scoreboardv3.ScoreboardV3(
                game_date=day.isoformat(), league_id="00", timeout=self._timeout_s
            ).get_dict(),
        )
        return [
            LiveGameStatus(
                game_id=g["gameId"],
                status=SCHEDULE_STATUS.get(g["gameStatus"], GameStatus.SCHEDULED),
                period=int(g.get("period") or 0),
                home_score=int(g["homeTeam"].get("score") or 0),
                away_score=int(g["awayTeam"].get("score") or 0),
            )
            for g in payload["scoreboard"]["games"]
        ]

    def play_by_play(self, game_id: str) -> list[PbpEventRow]:
        payload = self._call(
            f"play-by-play {game_id}",
            lambda: playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=self._timeout_s).get_dict(),
        )
        return parse_play_by_play(payload)
