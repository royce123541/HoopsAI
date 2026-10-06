"""`hoopsai train`: backtest, fit the production model on all completed seasons, register."""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import Engine

from hoopsai.features.build import build_features, load_core
from hoopsai.ml import registry
from hoopsai.ml.backtest import BacktestReport, labelled, walk_forward
from hoopsai.ml.pregame import DEFAULT_PARAMS, PregameModel, fit
from hoopsai.sources.base import current_season

log = logging.getLogger(__name__)

N_TEST_SEASONS = 4


@dataclass
class TrainResult:
    report: BacktestReport
    model: PregameModel
    registration: registry.Registration | None


def completed_seasons(features: pd.DataFrame, today: date) -> list[int]:
    data = labelled(features)
    return sorted(int(s) for s in data["season"].unique() if s < current_season(today))


def train(
    features: pd.DataFrame,
    *,
    today: date,
    params: dict[str, Any] | None = None,
    n_test_seasons: int = N_TEST_SEASONS,
    tracking_uri: str | None = None,
) -> TrainResult:
    """Backtest on the last `n_test_seasons` completed seasons, then fit on all of them.
    Registers in MLflow when `tracking_uri` is given."""
    params = params or DEFAULT_PARAMS
    seasons = completed_seasons(features, today)
    if len(seasons) < n_test_seasons + 2:
        raise ValueError(f"need at least {n_test_seasons + 2} completed seasons, have {seasons}")
    report = walk_forward(features, seasons[-n_test_seasons:], params)

    data = labelled(features)
    data = data[data["season"].isin(seasons)]
    last = seasons[-1]
    if report.calibration == "isotonic":
        model = fit(data[data["season"] < last], data[data["season"] == last], params)
    else:
        model = fit(data, params=params)

    registration = None
    if tracking_uri:
        registration = registry.log_and_register(model, report, params, tracking_uri)
    return TrainResult(report=report, model=model, registration=registration)


def train_from_db(
    engine: Engine, *, today: date, tracking_uri: str | None, **kwargs: Any
) -> TrainResult:
    return train(
        build_features(*load_core(engine)), today=today, tracking_uri=tracking_uri, **kwargs
    )


def format_report(report: BacktestReport) -> str:
    lines = [
        f"calibration: {report.calibration}   alternatives: "
        + ", ".join(f"{k}={v:.4f}" for k, v in report.alternatives.items()),
        "",
        "season   log loss (elo)      brier (elo)        accuracy (elo)   ece",
    ]
    for s in [*report.seasons, None]:
        m, e = (s.model, s.elo) if s else (report.pooled_model, report.pooled_elo)
        label = str(s.season) if s else "pooled"
        lines.append(
            f"{label:7s}  {m['logloss']:.4f} ({e['logloss']:.4f})   "
            f"{m['brier']:.4f} ({e['brier']:.4f})   "
            f"{m['accuracy']:.3f} ({e['accuracy']:.3f})    {m['ece']:.3f}"
        )
    lines.append("")
    lines.append(
        "acceptance: "
        + ", ".join(f"{k}={'PASS' if ok else 'FAIL'}" for k, ok in report.acceptance().items())
    )
    return "\n".join(lines)
