"""Ingestion pipeline against a real Postgres (compose `db`), with network fetches replaced by
recorded fixtures. Parsing uses the production NbaStatsSource parsers."""

from datetime import date
from typing import Any

import pytest
from sqlalchemy import Engine, Select, func, select
from sqlalchemy.orm import Session

from hoopsai.db.models import ApiResponse, Game, IngestionRun, PbpEvent, Team, TeamGameStats
from hoopsai.ingest import store
from hoopsai.ingest.pipeline import Ingestor
from hoopsai.sources.base import RawResponse, SeasonType, season_label
from hoopsai.sources.nba_stats.client import NBA_API_SEASON_TYPE, NbaStatsSource
from tests.conftest import load_fixture

pytestmark = pytest.mark.usefixtures("db")

TODAY = date(2026, 10, 6)  # 2026-27 preseason: 2024 and 2025 are completed seasons
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


def one[T](engine: Engine, stmt: Select[T]) -> T:
    with Session(engine) as session:
        return session.scalars(stmt).one()


def all_of[T](engine: Engine, stmt: Select[T]) -> list[T]:
    with Session(engine) as session:
        return list(session.scalars(stmt))


def count(engine: Engine, model: Any, *where: Any) -> int:
    with engine.connect() as conn:
        return conn.scalar(select(func.count()).select_from(model).where(*where)) or 0


def test_backfill_loads_games_box_scores_and_play_by_play(db: Engine) -> None:
    source = FixtureSource()

    summary = Ingestor(db, source).backfill(2024, 2025, pbp=True, force=False, today=TODAY)

    # 2024: 4 regular + 2 play-in games from game logs; 2025: 5 games from the schedule.
    assert count(db, Game) == 11
    assert count(db, TeamGameStats) == 12
    assert count(db, PbpEvent) == 11 * 40
    # 2 schedules + 6 game logs (3 types x 2 seasons) + 11 play-by-play
    assert summary.succeeded == 19
    assert summary.failed == []
    assert count(db, IngestionRun, IngestionRun.status == store.SUCCESS) == 19
    assert count(db, ApiResponse) == 19


def test_raw_payload_round_trips(db: Engine) -> None:
    ingestor = Ingestor(db, FixtureSource())
    ingestor.seed_teams()
    ingestor.schedule(2026)

    row = one(db, select(ApiResponse))
    assert row.endpoint == "scheduleleaguev2"
    assert row.params == {"season": "2026-27"}
    assert store.decompress_payload(row.payload_gz) == load_fixture("schedule_2026")


def test_rerun_skips_completed_seasons_and_force_is_idempotent(db: Engine) -> None:
    Ingestor(db, FixtureSource()).backfill(2024, 2025, pbp=True, force=False, today=TODAY)

    source = FixtureSource()
    summary = Ingestor(db, source).backfill(2024, 2025, pbp=True, force=False, today=TODAY)
    assert source.calls == []
    assert summary.skipped == 8
    assert summary.succeeded == 0

    forced = Ingestor(db, FixtureSource()).backfill(2024, 2025, pbp=True, force=True, today=TODAY)
    assert forced.succeeded == 19
    assert count(db, Game) == 11
    assert count(db, TeamGameStats) == 12
    assert count(db, PbpEvent) == 11 * 40


def test_current_season_is_always_refetched(db: Engine) -> None:
    Ingestor(db, FixtureSource()).backfill(2026, 2026, pbp=False, force=False, today=TODAY)
    source = FixtureSource()

    summary = Ingestor(db, source).backfill(2026, 2026, pbp=False, force=False, today=TODAY)

    assert summary.skipped == 0
    assert "scheduleleaguev2:['2026-27']" in source.calls


def test_failed_task_is_recorded_and_retried_on_next_run(db: Engine) -> None:
    bad = "0022400061"
    summary = Ingestor(db, FixtureSource(fail_game_ids={bad})).backfill(
        2024, 2024, pbp=True, force=False, today=TODAY
    )

    assert summary.failed == [f"pbp:{bad}"]
    assert summary.succeeded == 4 + 5  # schedule + 3 game logs, 5 of 6 play-by-play
    run = one(db, select(IngestionRun).where(IngestionRun.task_key == f"pbp:{bad}"))
    assert run.status == store.FAILED
    assert run.error is not None and "simulated" in run.error
    assert count(db, PbpEvent, PbpEvent.game_id == bad) == 0

    source = FixtureSource()
    retry = Ingestor(db, source).backfill(2024, 2024, pbp=True, force=False, today=TODAY)

    assert retry.failed == []
    assert source.calls == [f"playbyplayv3:['{bad}']"]
    run = one(db, select(IngestionRun).where(IngestionRun.task_key == f"pbp:{bad}"))
    assert (run.status, run.attempts) == (store.SUCCESS, 2)


def test_daily_loads_upcoming_schedule(db: Engine) -> None:
    summary = Ingestor(db, FixtureSource()).daily(TODAY)

    assert summary.failed == []
    games = all_of(db, select(Game).order_by(Game.game_id))
    assert [(g.game_id, g.status, g.home_score) for g in games] == [
        ("0022600001", "scheduled", None),
        ("0022600002", "scheduled", None),
    ]
    assert games[0].tip_time_utc is not None


def test_game_upserts_merge_sources_without_downgrading_finals(db: Engine) -> None:
    ingestor = Ingestor(db, FixtureSource())
    ingestor.seed_teams()
    schedule = load_fixture("schedule_2025")
    final_game = ingestor.source.parse_schedule(schedule)[0]  # 0022500001, OKC 125-124 HOU
    stale = {**final_game, "status": "scheduled", "home_score": None, "away_score": None}
    from_game_log = {k: v for k, v in final_game.items() if k not in ("tip_time_utc", "is_neutral")}

    with db.begin() as conn:
        store.upsert_games(conn, [final_game])
        store.upsert_games(conn, [stale])  # stale schedule must not reopen a final game
        store.upsert_games(conn, [from_game_log])  # game-log rows must not null the tip time

    game = one(db, select(Game).where(Game.game_id == "0022500001"))
    assert (game.status, game.home_score, game.away_score) == ("final", 125, 124)
    assert game.tip_time_utc == final_game["tip_time_utc"]
    assert count(db, Team) == 30


def test_game_logs_use_schedule_home_team_for_neutral_site_games(db: Engine) -> None:
    lal, minnesota = 1610612747, 1610612750
    ingestor = Ingestor(db, FixtureSource(neutral_game_ids={"0022400062"}))
    ingestor.seed_teams()
    with db.begin() as conn:  # as loaded from the schedule, which runs before the game logs
        store.upsert_games(
            conn,
            [
                {
                    "game_id": "0022400062",
                    "season": 2024,
                    "season_type": "regular",
                    "game_date": date(2024, 10, 22),
                    "home_team_id": lal,
                    "away_team_id": minnesota,
                    "status": "final",
                    "is_neutral": True,
                }
            ],
        )

    ingestor.game_logs(2024, SeasonType.REGULAR)

    assert ingestor.summary.failed == []
    game = one(db, select(Game).where(Game.game_id == "0022400062"))
    assert (game.home_team_id, game.home_score, game.away_score) == (lal, 110, 103)
    assert count(db, TeamGameStats, TeamGameStats.game_id == "0022400062") == 2
