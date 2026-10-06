"""Batch pre-game scoring (docs/ARCHITECTURE.md §3.4): predictions are precomputed and the API
only reads them, so no inference happens on the request path."""

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient
from sqlalchemy import Engine, bindparam, insert, select, text

from hoopsai.dates import eastern_today
from hoopsai.db.models import ModelVersionSnapshot, Prediction
from hoopsai.features.build import FEATURE_VERSION, MODEL_FEATURES
from hoopsai.features.store import rebuild_features
from hoopsai.ml import registry
from hoopsai.ml.pregame import PregameModel

log = logging.getLogger(__name__)

STORED_FACTORS = 10  # the API shows the top 5 that have a value
UNCHANGED_TOLERANCE = 1e-4


@dataclass
class PredictSummary:
    model_version: str
    games: int
    written: int

    def __str__(self) -> str:
        unchanged = self.games - self.written
        return (
            f"model v{self.model_version}: {self.games} games scored, "
            f"{self.written} predictions written, {unchanged} unchanged"
        )


def _target_games(engine: Engine, start: date, end: date, only_scheduled: bool) -> pd.DataFrame:
    status_filter = "AND g.status = 'scheduled'" if only_scheduled else ""
    rows = pd.read_sql(
        text(f"""
            SELECT g.game_id, g.game_date, g.tip_time_utc, g.status,
                   f.feature_version, f.features
            FROM core.games g
            LEFT JOIN features.game_features f USING (game_id)
            WHERE g.game_date BETWEEN :start AND :end {status_filter}
            ORDER BY g.game_date, g.game_id
        """),
        engine,
        params={"start": start, "end": end},
    )
    return rows


def _feature_frame(rows: pd.DataFrame) -> pd.DataFrame:
    values = pd.DataFrame(list(rows["features"]), index=rows.index)[MODEL_FEATURES]
    return pd.concat([rows.drop(columns=["features"]), values.astype(float)], axis=1)


def _latest_predictions(engine: Engine, game_ids: list[str]) -> dict[str, tuple[str, float]]:
    stmt = text("""
        SELECT DISTINCT ON (game_id) game_id, model_version, home_win_prob
        FROM serving.predictions WHERE game_id IN :ids
        ORDER BY game_id, created_at DESC, id DESC
    """).bindparams(bindparam("ids", expanding=True))
    with engine.connect() as conn:
        return {g: (v, p) for g, v, p in conn.execute(stmt, {"ids": game_ids})}


def snapshot_model_version(engine: Engine, version: str, tracking_uri: str) -> None:
    """Copy the version's backtest report and feature importance out of MLflow, once."""
    with engine.connect() as conn:
        if conn.scalar(
            select(ModelVersionSnapshot.version).where(ModelVersionSnapshot.version == version)
        ):
            return
    mlflow.set_tracking_uri(tracking_uri)
    mv = MlflowClient().get_model_version(registry.MODEL_NAME, version)
    run_id = str(mv.run_id)
    backtest: dict[str, Any] = mlflow.artifacts.load_dict(f"runs:/{run_id}/backtest.json")
    importance: dict[str, Any] = mlflow.artifacts.load_dict(f"runs:/{run_id}/importance.json")
    params = MlflowClient().get_run(run_id).data.params
    with engine.begin() as conn:
        conn.execute(
            insert(ModelVersionSnapshot).values(
                version=version,
                run_id=run_id,
                feature_version=params.get("feature_version", FEATURE_VERSION),
                calibration=params.get("calibration", "none"),
                train_seasons=params.get("train_seasons", ""),
                backtest=backtest,
                importance=importance,
            )
        )


def score(
    engine: Engine,
    model: PregameModel,
    version: str,
    start: date,
    end: date,
    *,
    only_scheduled: bool,
    now: datetime,
) -> PredictSummary:
    rows = _target_games(engine, start, end, only_scheduled)
    stale = rows["features"].isna() | (rows["feature_version"] != model.feature_version)
    if len(rows) and stale.any():
        log.info("features missing or outdated for %d games; rebuilding", int(stale.sum()))
        rebuild_features(engine)
        rows = _target_games(engine, start, end, only_scheduled)
    if rows.empty:
        return PredictSummary(model_version=version, games=0, written=0)

    frame = _feature_frame(rows)
    probs = model.predict_proba(frame)
    factors = model.explain(frame, top_k=STORED_FACTORS)
    latest = _latest_predictions(engine, list(frame["game_id"]))

    new_rows = []
    for game_id, game_date, tip, prob, game_factors in zip(
        frame["game_id"], frame["game_date"], frame["tip_time_utc"], probs, factors, strict=True
    ):
        previous = latest.get(game_id)
        if previous and previous[0] == version and abs(previous[1] - prob) < UNCHANGED_TOLERANCE:
            continue
        before_tip = now < tip if pd.notna(tip) else eastern_today(now) < game_date
        new_rows.append(
            {
                "game_id": game_id,
                "model_version": version,
                "feature_version": model.feature_version,
                "home_win_prob": float(prob),
                "factors": game_factors,
                "made_before_tip": bool(before_tip),
                "created_at": now,
            }
        )
    if new_rows:
        with engine.begin() as conn:
            conn.execute(insert(Prediction), new_rows)
    return PredictSummary(model_version=version, games=len(frame), written=len(new_rows))


def predict_pregame(
    engine: Engine,
    tracking_uri: str,
    *,
    start: date | None = None,
    days: int = 7,
    only_scheduled: bool = True,
    now: datetime | None = None,
) -> PredictSummary:
    """Score games from `start` (default: today, US Eastern) for `days` days."""
    now = now or datetime.now(UTC)
    start = start or eastern_today(now)
    model, version = registry.load_production(tracking_uri)
    snapshot_model_version(engine, version, tracking_uri)
    summary = score(
        engine,
        model,
        version,
        start,
        start + timedelta(days=days - 1),
        only_scheduled=only_scheduled,
        now=now,
    )
    log.info("predict_pregame: %s", summary)
    return summary
