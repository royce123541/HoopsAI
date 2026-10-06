"""Source-agnostic types for ingestion. A DataSource fetches raw payloads (stored verbatim in
raw.api_responses) and parses them into core-table rows, so a paid provider can replace
stats.nba.com by implementing this protocol."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Protocol, TypedDict


class SeasonType(StrEnum):
    REGULAR = "regular"
    PLAYOFFS = "playoffs"
    PLAYIN = "playin"


class GameStatus(StrEnum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    FINAL = "final"


@dataclass(frozen=True)
class RawResponse:
    source: str
    endpoint: str
    params: dict[str, Any]
    payload: dict[str, Any]
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def request_key(self) -> str:
        return request_key(self.source, self.endpoint, self.params)


def request_key(source: str, endpoint: str, params: dict[str, Any]) -> str:
    return f"{source}:{endpoint}:{json.dumps(params, sort_keys=True, separators=(',', ':'))}"


# Row shapes match the core.* table columns so they can be bulk-upserted directly.


class TeamRow(TypedDict):
    team_id: int
    abbreviation: str
    city: str
    nickname: str
    full_name: str


class GameRow(TypedDict):
    game_id: str
    season: int
    season_type: str
    game_date: date
    home_team_id: int
    away_team_id: int
    status: str
    home_score: int | None
    away_score: int | None


class ScheduledGameRow(GameRow):
    tip_time_utc: datetime | None
    is_neutral: bool


class TeamGameStatsRow(TypedDict):
    game_id: str
    team_id: int
    is_home: bool
    minutes: int
    pts: int
    fgm: int
    fga: int
    fg3m: int
    fg3a: int
    ftm: int
    fta: int
    oreb: int
    dreb: int
    reb: int
    ast: int
    stl: int
    blk: int
    tov: int
    pf: int
    plus_minus: int


class PbpEventRow(TypedDict):
    game_id: str
    action_id: int
    action_number: int
    period: int
    clock_seconds: float
    team_id: int | None
    location: str | None
    person_id: int | None
    action_type: str | None
    sub_type: str | None
    description: str | None
    is_field_goal: bool
    shot_value: int | None
    shot_result: str | None
    score_home: int
    score_away: int


@dataclass(frozen=True)
class GameLogParse:
    games: list[GameRow]
    team_stats: list[TeamGameStatsRow]


class DataSource(Protocol):
    name: str

    def list_teams(self) -> list[TeamRow]: ...
    def fetch_team_game_logs(self, season: int, season_type: SeasonType) -> RawResponse: ...
    def fetch_schedule(self, season: int) -> RawResponse: ...
    def fetch_play_by_play(self, game_id: str) -> RawResponse: ...

    def parse_team_game_logs(
        self, payload: dict[str, Any], home_team_ids: Mapping[str, int] | None = None
    ) -> GameLogParse: ...
    def parse_schedule(self, payload: dict[str, Any]) -> list[ScheduledGameRow]: ...
    def parse_play_by_play(self, payload: dict[str, Any]) -> list[PbpEventRow]: ...


# ---------------------------------------------------------------- seasons


def season_label(season: int) -> str:
    """2024 -> "2024-25"."""
    return f"{season}-{(season + 1) % 100:02d}"


def parse_season(value: str) -> int:
    """Accept "2024" or "2024-25" and return the start year."""
    start, _, end = value.partition("-")
    season = int(start)
    if end and season_label(season) != value:
        raise ValueError(f"invalid season {value!r}; expected e.g. 2024-25")
    return season


def current_season(today: date) -> int:
    """NBA seasons tip off in October; anything from August on belongs to the new season."""
    return today.year if today.month >= 8 else today.year - 1
