"""A DataSource serving recorded stats.nba.com fixtures instead of making requests."""

from typing import Any

from hoopsai.sources.base import RawResponse, SeasonType, season_label
from hoopsai.sources.nba_stats.client import NBA_API_SEASON_TYPE, NbaStatsSource
from tests.conftest import load_fixture

GAMELOG_HEADERS = load_fixture("gamelog_2024_regular")["resultSets"][0]["headers"]


class FixtureSource(NbaStatsSource):
    """Serves fixtures for known requests and valid empty payloads otherwise."""

    def __init__(
        self, fail_game_ids: set[str] | None = None, neutral_game_ids: set[str] | None = None
    ) -> None:
        super().__init__()
        self.fail_game_ids = fail_game_ids or set()
        self.neutral_game_ids = neutral_game_ids or set()
        self.calls: list[str] = []

    def _resp(self, endpoint: str, params: dict[str, Any], payload: dict[str, Any]) -> RawResponse:
        self.calls.append(f"{endpoint}:{sorted(params.values())}")
        return RawResponse(source=self.name, endpoint=endpoint, params=params, payload=payload)

    def fetch_schedule(self, season: int) -> RawResponse:
        if season in (2025, 2026):
            payload = load_fixture(f"schedule_{season}")
        else:
            payload = {"leagueSchedule": {"seasonYear": season_label(season), "gameDates": []}}
        return self._resp("scheduleleaguev2", {"season": season_label(season)}, payload)

    def fetch_team_game_logs(self, season: int, season_type: SeasonType) -> RawResponse:
        name = f"gamelog_{season}_{'regular' if season_type is SeasonType.REGULAR else season_type}"
        if season == 2024 and season_type in (SeasonType.REGULAR, SeasonType.PLAYIN):
            payload = load_fixture(name)
            for row in payload["resultSets"][0]["rowSet"]:
                if row[4] in self.neutral_game_ids:  # neutral site: both teams listed with "@"
                    row[6] = row[6].replace(" vs. ", " @ ")
        else:
            payload = {"resultSets": [{"headers": GAMELOG_HEADERS, "rowSet": []}]}
        params = {"season": season_label(season), "season_type": NBA_API_SEASON_TYPE[season_type]}
        return self._resp("leaguegamelog", params, payload)

    def fetch_play_by_play(self, game_id: str) -> RawResponse:
        if game_id in self.fail_game_ids:
            self.calls.append(f"playbyplayv3:{game_id}")
            raise ConnectionError("simulated stats.nba.com timeout")
        payload = load_fixture("pbp_0022400001")
        payload["game"]["gameId"] = game_id
        return self._resp("playbyplayv3", {"game_id": game_id}, payload)
