"""Model monitoring, job-run recording and the ops API."""

import json
import logging
from datetime import UTC, date, datetime

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, insert, text

from hoopsai.api.main import create_app
from hoopsai.db.models import ModelVersionSnapshot, Prediction
from hoopsai.ingest import jobs
from hoopsai.ingest.pipeline import Ingestor
from hoopsai.logs import JsonFormatter
from hoopsai.monitor import judge, run_monitor
from tests.fixture_source import FixtureSource

# The schedule_2025 fixture's five final 2025-26 games and their winners (home team won?).
GAMES = {
    "0022500001": True,  # OKC 125-124 HOU
    "0022500002": False,  # LAL 109-119 GSW
    "0022500003": True,  # NYK 119-111 CLE
    "0022500004": False,  # DAL 92-125 SAS
    "0042500121": True,  # NYK 113-102 ATL
}
JUNE = date(2026, 6, 20)  # still the 2025-26 season; the fixture games are > 30 days old


def frame(probs: list[float], outcomes: list[int], version: str = "7") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(len(probs))],
            "model_version": version,
            "home_win_prob": probs,
            "home_win": outcomes,
        }
    )


# ---------------------------------------------------------------- judge


def test_judge_needs_enough_games() -> None:
    result = judge("pregame", "season", frame([0.6] * 10, [1] * 10), lambda v: 0.62)
    assert result.status == "insufficient"
    assert result.metrics is None and "10 finished games" in result.message


def test_judge_ok_when_close_to_expectation() -> None:
    probs = [0.7] * 70 + [0.3] * 30
    outcomes = [1] * 49 + [0] * 21 + [0] * 21 + [1] * 9  # 70% / 30% observed
    result = judge("pregame", "season", frame(probs, outcomes), lambda v: 0.62, min_games=50)
    assert result.status == "ok"
    assert result.model_version == "7" and result.expected_logloss == 0.62
    assert result.metrics is not None and result.metrics["games"] == 100


def test_judge_warns_when_much_worse_than_expected() -> None:
    probs = [0.8] * 60
    outcomes = [0] * 40 + [1] * 20  # confident and mostly wrong
    result = judge("pregame", "season", frame(probs, outcomes), lambda v: 0.62)
    assert result.status == "warning"
    assert result.message.startswith("Worse than expected")


def test_judge_warns_on_poor_calibration_with_many_games() -> None:
    # Right side of 50% every time, but far too timid: low log loss gap, high ECE.
    probs = [0.55] * 250
    outcomes = [1] * 240 + [0] * 10
    result = judge("pregame", "season", frame(probs, outcomes), lambda v: 0.9)
    assert result.status == "warning"
    assert "Calibration error" in result.message


def test_judge_without_a_baseline_still_reports() -> None:
    result = judge("ingame", "season", frame([0.6] * 30, [1] * 30), lambda v: None)
    assert result.status == "ok"
    assert "no held-out log loss" in result.message


# ---------------------------------------------------------------- run_monitor


@pytest.fixture
def seeded(db: Engine) -> Engine:
    Ingestor(db, FixtureSource()).backfill(2025, 2025, pbp=False, force=False, today=JUNE)
    with db.begin() as conn:
        conn.execute(
            insert(ModelVersionSnapshot).values(
                version="7",
                run_id="r7",
                feature_version="v1",
                calibration="none",
                train_seasons="2005-2024",
                backtest={"pooled_model": {"logloss": 0.62}},
                importance={},
            )
        )
    return db


def predict(engine: Engine, probs: dict[str, float], before_tip: bool = True) -> None:
    with engine.begin() as conn:
        conn.execute(
            insert(Prediction),
            [
                {
                    "game_id": g,
                    "model_version": "7",
                    "feature_version": "v1",
                    "home_win_prob": p,
                    "factors": [],
                    "made_before_tip": before_tip,
                }
                for g, p in probs.items()
            ],
        )


def test_monitor_scores_pre_tip_predictions_only(seeded: Engine) -> None:
    good = {g: (0.8 if home_won else 0.2) for g, home_won in GAMES.items()}
    predict(seeded, good)
    predict(seeded, {g: 1 - p for g, p in good.items()}, before_tip=False)  # ignored

    results = run_monitor(seeded, JUNE, ingame_expected=lambda v: None, min_games={"pregame": 3})

    by_key = {(r.model, r.window): r for r in results}
    season = by_key[("pregame", "season")]
    assert (season.status, season.games, season.expected_logloss) == ("ok", 5, 0.62)
    assert season.metrics is not None and season.metrics["accuracy"] == 1.0
    assert by_key[("pregame", "last_30_days")].status == "insufficient"  # games are older
    assert by_key[("ingame", "season")].games == 0
    with seeded.connect() as conn:
        assert conn.scalar(text("SELECT count(*) FROM serving.monitoring_runs")) == 4


def test_monitor_warns_on_bad_predictions_and_api_reports_it(seeded: Engine) -> None:
    predict(seeded, {g: (0.1 if home_won else 0.9) for g, home_won in GAMES.items()})
    run_monitor(seeded, JUNE, ingame_expected=lambda v: None, min_games={"pregame": 3})

    with TestClient(create_app()) as client:
        monitoring = client.get("/api/model/monitoring").json()
        assert [(m["model"], m["window"]) for m in monitoring] == [
            ("pregame", "season"),
            ("pregame", "last_30_days"),
            ("ingame", "season"),
            ("ingame", "last_30_days"),
        ]
        assert monitoring[0]["status"] == "warning"
        status = client.get("/api/status").json()
        assert status["models"] == {"pregame": "7", "ingame": None}
        assert len(status["monitoring"]) == 4


# ---------------------------------------------------------------- jobs and logs


def test_recorded_jobs_store_success_and_failure(db: Engine) -> None:
    @jobs.recorded("demo")
    def works() -> str:
        return "3 games"

    @jobs.recorded("demo_broken")
    def breaks() -> None:
        raise RuntimeError("stats.nba.com timed out")

    assert works() == "3 games"
    with pytest.raises(RuntimeError):
        breaks()

    with TestClient(create_app()) as client:
        runs = {j["job"]: j for j in client.get("/api/status").json()["jobs"]}
    assert runs["demo"]["status"] == "success" and runs["demo"]["detail"] == "3 games"
    assert runs["demo_broken"]["status"] == "failed"
    assert runs["demo_broken"]["detail"] == "RuntimeError: stats.nba.com timed out"
    assert runs["demo"]["finished_at"] is not None


def test_json_log_lines() -> None:
    record = logging.LogRecord("hoopsai.x", logging.WARNING, __file__, 1, "%d games", (3,), None)
    record.game_id = "0022600001"  # passed via extra=
    entry = json.loads(JsonFormatter().format(record))
    assert entry["level"] == "WARNING" and entry["message"] == "3 games"
    assert entry["logger"] == "hoopsai.x" and entry["game_id"] == "0022600001"
    assert datetime.fromisoformat(entry["time"]).tzinfo == UTC
