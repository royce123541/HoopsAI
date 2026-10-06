"""MLflow tracking and model registry for the pre-game and in-game models.

Each training run logs params, metrics and the model, and registers a new version. The
`production` alias moves to it only if it passes its acceptance checks and beats the current
production version's out-of-sample log loss on the same held-out season (pre-game: the most
recent walk-forward backtest season; in-game: the newest season, held out of training)."""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version as pkg_version
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.pyfunc.model import PythonModel

from hoopsai.ml.backtest import BacktestReport
from hoopsai.ml.ingame import InGameModel, InGameReport
from hoopsai.ml.pregame import PregameModel

log = logging.getLogger(__name__)

MODEL_NAME = "hoopsai-pregame"
EXPERIMENT = "hoopsai-pregame"
INGAME_MODEL_NAME = "hoopsai-ingame"
INGAME_EXPERIMENT = "hoopsai-ingame"
PRODUCTION_ALIAS = "production"

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

PIP_REQUIREMENTS = [
    f"{pkg}=={pkg_version(pkg)}" for pkg in ("lightgbm", "scikit-learn", "pandas", "numpy")
]

Decision = tuple[bool, str]


class PregamePyfunc(PythonModel):
    """MLflow wrapper. Serving code unwraps it to use `PregameModel` directly (for SHAP)."""

    def __init__(self, model: PregameModel) -> None:
        self.model = model

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> list[float]:
        return [float(p) for p in self.model.predict_proba(model_input)]


class InGamePyfunc(PythonModel):
    """MLflow wrapper; input rows are game states plus a `pregame_prob` column."""

    def __init__(self, model: InGameModel) -> None:
        self.model = model

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> list[float]:
        return [
            float(p) for p in self.model.predict_proba(model_input, model_input["pregame_prob"])
        ]


@dataclass
class Registration:
    run_id: str
    version: str
    promoted: bool
    reason: str


def _metric_key(season: int) -> str:
    return f"backtest_logloss_{season}"


def _production(client: MlflowClient, name: str = MODEL_NAME) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(name, PRODUCTION_ALIAS)
    except MlflowException:
        return None


def _compare(
    acceptance: dict[str, bool], season: int, ours: float, production: ModelVersion | None
) -> Decision:
    failed = [k for k, ok in acceptance.items() if not ok]
    if failed:
        return False, f"failed acceptance: {', '.join(failed)}"
    if production is None:
        return True, "no production model yet"
    theirs = production.tags.get(_metric_key(season))
    if theirs is None:
        return True, f"production v{production.version} was not evaluated on {season}"
    if ours < float(theirs):
        return True, f"log loss {ours:.4f} < production v{production.version} {float(theirs):.4f}"
    return False, f"log loss {ours:.4f} >= production v{production.version} {float(theirs):.4f}"


def promotion_decision(report: BacktestReport, production: ModelVersion | None) -> Decision:
    latest = report.seasons[-1]
    return _compare(report.acceptance(), latest.season, latest.model["logloss"], production)


def ingame_promotion_decision(report: InGameReport, production: ModelVersion | None) -> Decision:
    return _compare(report.acceptance(), report.test_season, report.model["logloss"], production)


def _register(
    *,
    name: str,
    experiment: str,
    tracking_uri: str,
    python_model: PythonModel,
    log_run: Callable[[], None],
    tags: dict[str, str],
    decide: Callable[[ModelVersion | None], Decision],
) -> Registration:
    """Log a run (params/metrics via `log_run`) and the model, register a version, tag it,
    and move the production alias if `decide` says so."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    client = MlflowClient()
    with mlflow.start_run() as run:
        log_run()
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=python_model,
            registered_model_name=name,
            pip_requirements=PIP_REQUIREMENTS,
        )
    version = str(info.registered_model_version)
    for key, value in tags.items():
        client.set_model_version_tag(name, version, key, value)
    promote, reason = decide(_production(client, name))
    if promote:
        client.set_registered_model_alias(name, PRODUCTION_ALIAS, version)
    client.set_model_version_tag(name, version, "promotion", reason)
    log.info("registered %s v%s (promoted=%s: %s)", name, version, promote, reason)
    return Registration(run_id=run.info.run_id, version=version, promoted=promote, reason=reason)


def _importance(features: list[str], gains: Any) -> dict[str, float]:
    pairs = zip(features, (float(g) for g in gains), strict=True)
    return dict(sorted(pairs, key=lambda kv: -kv[1]))


def log_and_register(
    model: PregameModel, report: BacktestReport, params: dict[str, Any], tracking_uri: str
) -> Registration:
    def log_run() -> None:
        mlflow.log_params({f"lgbm_{k}": v for k, v in params.items()})
        mlflow.log_params(
            {
                "calibration": model.calibration,
                "feature_version": model.feature_version,
                "n_features": len(model.features),
                "train_seasons": f"{model.train_seasons[0]}-{model.train_seasons[-1]}",
                "test_seasons": ",".join(str(s.season) for s in report.seasons),
            }
        )
        mlflow.log_metrics({f"pooled_model_{k}": v for k, v in report.pooled_model.items()})
        mlflow.log_metrics({f"pooled_elo_{k}": v for k, v in report.pooled_elo.items()})
        for s in report.seasons:
            mlflow.log_metrics({f"s{s.season}_model_{k}": v for k, v in s.model.items()})
            mlflow.log_metrics({f"s{s.season}_elo_{k}": v for k, v in s.elo.items()})
        mlflow.log_dict(report.to_dict(), "backtest.json")
        gains = model.estimator.booster_.feature_importance("gain")
        mlflow.log_dict(_importance(model.features, gains), "importance.json")

    # Full precision (str(float) round-trips): rounding could make an identical score "better".
    tags = {_metric_key(s.season): str(s.model["logloss"]) for s in report.seasons}
    tags |= {"feature_version": model.feature_version, "calibration": model.calibration}
    return _register(
        name=MODEL_NAME,
        experiment=EXPERIMENT,
        tracking_uri=tracking_uri,
        python_model=PregamePyfunc(model),
        log_run=log_run,
        tags=tags,
        decide=lambda production: promotion_decision(report, production),
    )


def log_and_register_ingame(
    model: InGameModel, report: InGameReport, params: dict[str, Any], tracking_uri: str
) -> Registration:
    def log_run() -> None:
        mlflow.log_params({f"lgbm_{k}": v for k, v in params.items()})
        mlflow.log_params(
            {
                "train_seasons": f"{model.train_seasons[0]}-{model.train_seasons[-1]}",
                "test_season": report.test_season,
                "features": ",".join(model.features),
            }
        )
        mlflow.log_metrics({f"test_model_{k}": v for k, v in report.model.items()})
        mlflow.log_metrics({f"test_pregame_only_{k}": v for k, v in report.pregame_only.items()})
        for quarter, m in report.by_quarter.items():
            mlflow.log_metrics({f"test_{quarter}_{k}": v for k, v in m.items()})
        mlflow.log_dict(report.to_dict(), "evaluation.json")
        gains = model.estimator.booster_.feature_importance("gain")
        mlflow.log_dict(_importance(model.features, gains), "importance.json")

    return _register(
        name=INGAME_MODEL_NAME,
        experiment=INGAME_EXPERIMENT,
        tracking_uri=tracking_uri,
        python_model=InGamePyfunc(model),
        log_run=log_run,
        tags={_metric_key(report.test_season): str(report.model["logloss"])},
        decide=lambda production: ingame_promotion_decision(report, production),
    )


def production_version(tracking_uri: str, name: str) -> str | None:
    """The version the production alias points at (a cheap registry lookup, no download)."""
    mlflow.set_tracking_uri(tracking_uri)
    production = _production(MlflowClient(), name)
    return None if production is None else str(production.version)


def _load(tracking_uri: str, name: str) -> tuple[Any, str]:
    version = production_version(tracking_uri, name)
    if version is None:
        raise LookupError(f"no {name}@{PRODUCTION_ALIAS} model registered")
    # Load that exact version, not the alias: a promotion in between must not pair one
    # version's number with another version's model.
    pyfunc = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
    return pyfunc.unwrap_python_model().model, version


def heldout_logloss(tracking_uri: str, name: str, version: str) -> float | None:
    """The log loss a version scored on its newest held-out season (its version tag)."""
    mlflow.set_tracking_uri(tracking_uri)
    try:
        tags = MlflowClient().get_model_version(name, version).tags
    except MlflowException:
        return None
    seasons = sorted(k for k in tags if k.startswith("backtest_logloss_"))
    return float(tags[seasons[-1]]) if seasons else None


def load_production(tracking_uri: str) -> tuple[PregameModel, str]:
    """The production pre-game model and its registry version."""
    model, version = _load(tracking_uri, MODEL_NAME)
    return model, version


def load_production_ingame(tracking_uri: str) -> tuple[InGameModel, str]:
    """The production in-game model and its registry version."""
    model, version = _load(tracking_uri, INGAME_MODEL_NAME)
    return model, version
