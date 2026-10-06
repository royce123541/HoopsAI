"""Batch scoring and the read API against the test database: real fixture games, features
built by the production builder, and a small model trained on a synthetic league."""

from collections.abc import Iterator
from datetime import UTC, date, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, func, insert, select, text

from hoopsai.api.main import create_app
from hoopsai.db.models import ModelVersionSnapshot, Prediction
from hoopsai.features.build import build_features
from hoopsai.features.store import rebuild_features
from hoopsai.ingest.pipeline import Ingestor
from hoopsai.ml.backtest import labelled
from hoopsai.ml.pregame import DEFAULT_PARAMS, PregameModel, fit
from hoopsai.predict.pregame import PredictSummary, score
from tests.fixture_source import FixtureSource
from tests.synthetic import make_league

OPENING_NIGHT = date(2026, 10, 20)  # schedule_2026 fixture: DET-BOS 19:00Z, NYK-PHI 23:00Z
BEFORE_SEASON = datetime(2026, 10, 7, tzinfo=UTC)
NYK, PHI = 1610612752, 1610612755


@pytest.fixture(scope="module")
def model() -> PregameModel:
    data = labelled(build_features(*make_league(seasons=(2020, 2021, 2022), scheduled_days=0)))
    return fit(data, params={**DEFAULT_PARAMS, "n_estimators": 20, "min_child_samples": 5})


@pytest.fixture
def seeded(db: Engine) -> Engine:
    """2024-25 and 2025-26 results plus the 2026-27 opening-night schedule, with features."""
    ingestor = Ingestor(db, FixtureSource())
    ingestor.backfill(2024, 2026, pbp=False, force=False, today=date(2026, 10, 7))
    rebuild_features(db)
    return db


def _metrics(logloss: float) -> dict[str, float]:
    return {"logloss": logloss, "brier": 0.2, "accuracy": 0.68, "ece": 0.02, "n": 1321.0}


def add_version(engine: Engine, version: str) -> None:
    backtest = {
        "seasons": [{"season": 2025, "model": _metrics(0.60), "elo": _metrics(0.61)}],
        "pooled_model": _metrics(0.60),
        "pooled_elo": _metrics(0.61),
        "reliability": [
            {"bin_low": 0.5, "bin_high": 0.6, "mean_pred": 0.55, "observed": 0.56, "count": 10}
        ],
        "acceptance": {"beats_elo_logloss": True},
    }
    with engine.begin() as conn:
        conn.execute(
            insert(ModelVersionSnapshot).values(
                version=version,
                run_id=f"run-{version}",
                feature_version="v1",
                calibration="none",
                train_seasons="2005-2025",
                backtest=backtest,
                importance={"diff_elo_pre": 30.0, "home_std_net": 10.0},
            )
        )


def score_opening_night(
    engine: Engine, model: PregameModel, version: str, now: datetime = BEFORE_SEASON
) -> PredictSummary:
    return score(engine, model, version, OPENING_NIGHT, OPENING_NIGHT, only_scheduled=True, now=now)


def count_predictions(engine: Engine) -> int:
    with engine.connect() as conn:
        return conn.scalar(select(func.count()).select_from(Prediction)) or 0


# ---------------------------------------------------------------- scorer


def test_score_writes_only_changed_predictions(seeded: Engine, model: PregameModel) -> None:
    add_version(seeded, "7")
    first = score_opening_night(seeded, model, "7")
    assert (first.games, first.written) == (2, 2)

    again = score_opening_night(seeded, model, "7")
    assert (again.games, again.written) == (2, 0)

    add_version(seeded, "8")  # a new production model re-scores everything
    new_model = score_opening_night(seeded, model, "8")
    assert new_model.written == 2
    assert count_predictions(seeded) == 4


def test_made_before_tip_flag(seeded: Engine, model: PregameModel) -> None:
    add_version(seeded, "7")
    between_tips = datetime(2026, 10, 20, 21, 0, tzinfo=UTC)  # after DET-BOS tip, before NYK-PHI
    score_opening_night(seeded, model, "7", now=between_tips)
    with seeded.connect() as conn:
        rows = conn.execute(text("SELECT game_id, made_before_tip FROM serving.predictions"))
        flags: dict[str, bool] = {game_id: before for game_id, before in rows}
    assert flags == {"0022600001": False, "0022600002": True}


def test_score_rebuilds_missing_features(seeded: Engine, model: PregameModel) -> None:
    add_version(seeded, "7")
    with seeded.begin() as conn:
        conn.execute(text("TRUNCATE features.game_features, features.elo_ratings"))
    summary = score_opening_night(seeded, model, "7")
    assert summary.written == 2
    with seeded.connect() as conn:
        assert conn.scalar(text("SELECT count(*) FROM features.game_features")) > 0


# ---------------------------------------------------------------- API


@pytest.fixture
def client(seeded: Engine, model: PregameModel) -> Iterator[TestClient]:
    add_version(seeded, "7")
    score_opening_night(seeded, model, "7")
    with TestClient(create_app()) as c:
        yield c


def test_slate_lists_games_with_predictions(client: TestClient) -> None:
    body = client.get("/api/games", params={"date": "2026-10-20"}).json()

    assert body["date"] == "2026-10-20"
    assert body["prev_date"] == "2026-04-18"  # last fixture game before opening night
    assert body["next_date"] is None
    games = body["games"]
    assert [g["game_id"] for g in games] == ["0022600001", "0022600002"]  # by tip time
    nyk = games[1]
    assert (nyk["home"]["abbreviation"], nyk["away"]["abbreviation"]) == ("NYK", "PHI")
    assert 0 < nyk["prediction"]["home_win_prob"] < 1
    assert nyk["prediction"]["model_version"] == "7"
    assert nyk["prediction"]["made_before_tip"] is True


def test_slate_for_a_date_without_games(client: TestClient) -> None:
    body = client.get("/api/games", params={"date": "2026-08-01"}).json()
    assert body["games"] == []
    assert body["next_date"] == "2026-10-20"


def test_game_detail_has_labelled_factors_and_comparison(client: TestClient) -> None:
    body = client.get("/api/games/0022600002").json()

    assert 0 < len(body["factors"]) <= 5
    for f in body["factors"]:
        assert f["value"] is not None
        assert f["favors"] == ("home" if f["contribution"] > 0 else "away")
        assert f["label"][0].isupper()
    impacts = [abs(f["contribution"]) for f in body["factors"]]
    assert impacts == sorted(impacts, reverse=True)
    elo = next(r for r in body["comparison"] if r["key"] == "elo_pre")
    assert elo["home"] > 1000 and elo["away"] > 1000
    assert len(body["comparison"]) == 10


def test_unknown_ids_are_404(client: TestClient) -> None:
    assert client.get("/api/games/0000000000").status_code == 404
    assert client.get("/api/teams/1").status_code == 404


def test_teams_sorted_by_current_elo(client: TestClient) -> None:
    teams = client.get("/api/teams").json()
    assert len(teams) == 30
    rated = [t["elo"] for t in teams if t["elo"] is not None]
    assert rated == sorted(rated, reverse=True)


def test_team_detail_uses_team_perspective_probabilities(client: TestClient) -> None:
    nyk = client.get(f"/api/teams/{NYK}").json()
    phi = client.get(f"/api/teams/{PHI}").json()

    assert nyk["team"]["abbreviation"] == "NYK"
    assert nyk["recent"] and all(g["status"] == "final" for g in nyk["recent"])
    nyk_game = next(g for g in nyk["upcoming"] if g["game_id"] == "0022600002")
    phi_game = next(g for g in phi["upcoming"] if g["game_id"] == "0022600002")
    assert (nyk_game["is_home"], phi_game["is_home"]) == (True, False)
    assert nyk_game["win_prob"] + phi_game["win_prob"] == pytest.approx(1.0)
    # Current Elo is the next game's pre-game rating (after between-season regression).
    game = client.get("/api/games/0022600002").json()
    elo_row = next(r for r in game["comparison"] if r["key"] == "elo_pre")
    assert nyk["elo"] == pytest.approx(elo_row["home"])


def test_model_endpoint(client: TestClient) -> None:
    body = client.get("/api/model").json()
    assert body["version"] == "7"
    assert body["seasons"][0]["season"] == 2025
    assert body["top_features"][0] == {
        "feature": "diff_elo_pre",
        "label": "Elo rating difference",
        "share": pytest.approx(0.75),
    }


def test_model_endpoint_404_before_any_prediction(db: Engine) -> None:
    with TestClient(create_app()) as c:
        assert c.get("/api/model").status_code == 404
