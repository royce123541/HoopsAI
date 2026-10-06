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
from hoopsai.sources.base import SeasonType
from tests.conftest import load_fixture
from tests.fixture_source import FixtureSource

pytestmark = pytest.mark.usefixtures("db")

TODAY = date(2026, 10, 6)  # 2026-27 preseason: 2024 and 2025 are completed seasons


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
