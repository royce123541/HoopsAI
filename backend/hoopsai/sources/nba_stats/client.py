from collections.abc import Callable, Mapping
from typing import Any

from nba_api.stats.endpoints import leaguegamelog, playbyplayv3, scheduleleaguev2
from nba_api.stats.static import teams as static_teams

from hoopsai.sources.base import (
    GameLogParse,
    PbpEventRow,
    RawResponse,
    ScheduledGameRow,
    SeasonType,
    TeamRow,
    season_label,
)
from hoopsai.sources.nba_stats import parse
from hoopsai.sources.throttle import Throttle, call_with_retry

NBA_API_SEASON_TYPE = {
    SeasonType.REGULAR: "Regular Season",
    SeasonType.PLAYOFFS: "Playoffs",
    SeasonType.PLAYIN: "PlayIn",
}


class NbaStatsSource:
    """stats.nba.com via `nba_api`, throttled and retried. Historical data plus schedules."""

    name = "nba_stats"

    def __init__(
        self,
        min_interval_s: float = 0.6,
        attempts: int = 5,
        base_delay_s: float = 2.0,
        timeout_s: int = 30,
    ) -> None:
        self._throttle = Throttle(min_interval_s)
        self._attempts = attempts
        self._base_delay_s = base_delay_s
        self._timeout_s = timeout_s

    def _get(self, endpoint: str, params: dict[str, Any], call: Callable[[], Any]) -> RawResponse:
        def attempt() -> dict[str, Any]:
            self._throttle.wait()
            payload: dict[str, Any] = call().get_dict()
            return payload

        payload = call_with_retry(
            attempt,
            attempts=self._attempts,
            base_delay_s=self._base_delay_s,
            what=f"{endpoint} {params}",
        )
        return RawResponse(source=self.name, endpoint=endpoint, params=params, payload=payload)

    def list_teams(self) -> list[TeamRow]:
        """The 30 franchises, bundled with nba_api (no request). Team ids are stable across
        relocations (SEA -> OKC keeps 1610612760), so historical games resolve to them."""
        return [
            TeamRow(
                team_id=t["id"],
                abbreviation=t["abbreviation"],
                city=t["city"],
                nickname=t["nickname"],
                full_name=t["full_name"],
            )
            for t in static_teams.get_teams()
        ]

    def fetch_team_game_logs(self, season: int, season_type: SeasonType) -> RawResponse:
        params = {"season": season_label(season), "season_type": NBA_API_SEASON_TYPE[season_type]}
        return self._get(
            "leaguegamelog",
            params,
            lambda: leaguegamelog.LeagueGameLog(
                season=params["season"],
                season_type_all_star=params["season_type"],
                player_or_team_abbreviation="T",
                timeout=self._timeout_s,
            ),
        )

    def fetch_schedule(self, season: int) -> RawResponse:
        params = {"season": season_label(season)}
        return self._get(
            "scheduleleaguev2",
            params,
            lambda: scheduleleaguev2.ScheduleLeagueV2(
                season=params["season"], timeout=self._timeout_s
            ),
        )

    def fetch_play_by_play(self, game_id: str) -> RawResponse:
        params = {"game_id": game_id}
        return self._get(
            "playbyplayv3",
            params,
            lambda: playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=self._timeout_s),
        )

    def parse_team_game_logs(
        self, payload: dict[str, Any], home_team_ids: Mapping[str, int] | None = None
    ) -> GameLogParse:
        return parse.parse_team_game_logs(payload, home_team_ids)

    def parse_schedule(self, payload: dict[str, Any]) -> list[ScheduledGameRow]:
        return parse.parse_schedule(payload)

    def parse_play_by_play(self, payload: dict[str, Any]) -> list[PbpEventRow]:
        return parse.parse_play_by_play(payload)
