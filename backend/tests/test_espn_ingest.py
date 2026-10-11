"""ESPN ingestion against a real Postgres (compose `db`), with requests replaced by the recorded
2026 Finals Game 5 (ESPN event 401859967, NBA game 0042500405: Knicks at Spurs, June 13)."""

import copy
from datetime import UTC, date, datetime
from typing import Any

import pytest
from sqlalchemy import Engine, func, select

from hoopsai.db.models import Game, IngestionRun, PbpEvent, TeamGameStats
from hoopsai.ingest import store
from hoopsai.ingest.espn import EspnIngestor
from hoopsai.live.source import EspnLiveSource
from hoopsai.sources.base import GameStatus, RawResponse
from hoopsai.sources.espn.client import EspnClient
from hoopsai.sources.nba_stats.client import NbaStatsSource
from tests.test_espn_parse import NYK, SAS, load

pytestmark = pytest.mark.usefixtures("db")

FINALS_DAY = date(2026, 6, 13)
ESPN_ID = "401859967"


class FakeEspn(EspnClient):
    """Serves the recorded scoreboard and summary; other days have no games."""

    def __init__(self, scoreboard: dict[str, Any] | None = None) -> None:
        super().__init__(min_interval_s=0)
        self.scoreboard = scoreboard or load("scoreboard_20260613")
        self.calls: list[str] = []

    def _get(self, endpoint: str, params: dict[str, Any]) -> RawResponse:
        self.calls.append(f"{endpoint}:{params}")
        if endpoint == "summary":
            payload = load("summary_401859967")
        elif params["dates"] in ("20260613", "202606"):
            payload = self.scoreboard
        else:
            payload = {"events": []}
        return RawResponse(source="espn", endpoint=endpoint, params=params, payload=payload)


def count(engine: Engine, model: Any, *where: Any) -> int:
    with engine.connect() as conn:
        return conn.scalar(select(func.count()).select_from(model).where(*where)) or 0


def game(engine: Engine) -> Any:
    with engine.connect() as conn:
        return conn.execute(select(Game)).one()


def seed_nba_game(engine: Engine, status: GameStatus = GameStatus.SCHEDULED) -> None:
    """The game as loaded earlier from stats.nba.com, under its NBA id."""
    with engine.begin() as conn:
        store.upsert_teams(conn, NbaStatsSource().list_teams())
        store.upsert_games(
            conn,
            [
                {
                    "game_id": "0042500405",
                    "season": 2025,
                    "season_type": "playoffs",
                    "game_date": FINALS_DAY,
                    "home_team_id": SAS,
                    "away_team_id": NYK,
                    "status": str(status),
                    "home_score": None,
                    "away_score": None,
                }
            ],
        )


def test_new_game_gets_an_espn_id_and_its_box_score(db: Engine) -> None:
    summary = EspnIngestor(db, FakeEspn()).daily(FINALS_DAY)

    g = game(db)
    assert (g.game_id, g.espn_id, g.status) == (f"espn-{ESPN_ID}", ESPN_ID, "final")
    assert (g.home_score, g.away_score, g.season) == (90, 94, 2025)
    assert g.tip_time_utc == datetime(2026, 6, 14, 0, 30, tzinfo=UTC)
    assert count(db, TeamGameStats) == 2
    assert count(db, PbpEvent) == 50  # the trimmed fixture's plays
    assert summary.failed == []


def test_existing_nba_game_keeps_its_id(db: Engine) -> None:
    seed_nba_game(db)

    EspnIngestor(db, FakeEspn()).daily(FINALS_DAY)

    g = game(db)
    assert (g.game_id, g.espn_id, g.status) == ("0042500405", ESPN_ID, "final")
    assert count(db, TeamGameStats, TeamGameStats.game_id == "0042500405") == 2
    assert count(db, PbpEvent, PbpEvent.game_id == "0042500405") == 50


def test_box_score_is_loaded_once(db: Engine) -> None:
    fake = FakeEspn()
    EspnIngestor(db, fake).daily(FINALS_DAY)
    EspnIngestor(db, fake).daily(FINALS_DAY)

    assert sum(c.startswith("summary") for c in fake.calls) == 1
    key = f"espn:game:espn-{ESPN_ID}"
    assert count(db, IngestionRun, IngestionRun.task_key == key) == 1


def test_unfinished_game_is_rechecked_later(db: Engine) -> None:
    pending = copy.deepcopy(load("scoreboard_20260613"))
    pending["events"][0]["status"]["type"]["state"] = "in"
    EspnIngestor(db, FakeEspn(pending)).daily(FINALS_DAY)
    assert (game(db).status, count(db, TeamGameStats)) == ("live", 0)

    # A week later the game is no longer among the recent days, but is still re-read.
    fake = FakeEspn()
    EspnIngestor(db, fake).daily(date(2026, 6, 20))

    assert "scoreboard:{'dates': '20260613'}" in fake.calls
    assert (game(db).status, count(db, TeamGameStats)) == ("final", 2)


def test_schedule_reads_the_season_month_by_month(db: Engine) -> None:
    fake = FakeEspn()
    ingestor = EspnIngestor(db, fake)
    ingestor.seed_teams()

    ingestor.schedule(2025)

    months = [c.split("'dates': '")[1][:6] for c in fake.calls]
    assert months == ["202510", "202511", "202512", "202601", "202602", "202603", "202604",
                      "202605", "202606"]  # fmt: skip
    assert count(db, Game) == 1


def test_live_source_uses_database_ids(db: Engine) -> None:
    seed_nba_game(db, GameStatus.LIVE)
    source = EspnLiveSource(db, FakeEspn())

    [status] = source.scoreboard(FINALS_DAY)
    events = source.play_by_play(status.game_id)

    assert (status.game_id, status.home_score, status.away_score) == ("0042500405", 90, 94)
    assert {e["game_id"] for e in events} == {"0042500405"}
