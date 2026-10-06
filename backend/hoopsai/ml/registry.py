"""MLflow tracking and model registry for the pre-game model.

Each training run logs params, backtest metrics and the model, and registers a new version
of `hoopsai-pregame`. The `production` alias moves to it only if it passes the acceptance
checks and beats the current production version's out-of-sample log loss on the same, most
recent backtest season (both numbers come from walk-forward runs that never saw it)."""

import logging
import os
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
from hoopsai.ml.pregame import PregameModel

log = logging.getLogger(__name__)

MODEL_NAME = "hoopsai-pregame"
EXPERIMENT = "hoopsai-pregame"
PRODUCTION_ALIAS = "production"

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


class PregamePyfunc(PythonModel):
    """MLflow wrapper. Serving code unwraps it to use `PregameModel` directly (for SHAP)."""

    def __init__(self, model: PregameModel) -> None:
        self.model = model

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> list[float]:
        return [float(p) for p in self.model.predict_proba(model_input)]


@dataclass
class Registration:
    run_id: str
    version: str
    promoted: bool
    reason: str


def _metric_key(season: int) -> str:
    return f"backtest_logloss_{season}"


def _production(client: MlflowClient) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
    except MlflowException:
        return None


def promotion_decision(report: BacktestReport, production: ModelVersion | None) -> tuple[bool, str]:
    failed = [k for k, ok in report.acceptance().items() if not ok]
    if failed:
        return False, f"failed acceptance: {', '.join(failed)}"
    if production is None:
        return True, "no production model yet"
    latest = report.seasons[-1]
    theirs = production.tags.get(_metric_key(latest.season))
    if theirs is None:
        return True, f"production v{production.version} was not evaluated on {latest.season}"
    ours = latest.model["logloss"]
    if ours < float(theirs):
        return True, f"log loss {ours:.4f} < production v{production.version} {float(theirs):.4f}"
    return False, f"log loss {ours:.4f} >= production v{production.version} {float(theirs):.4f}"


def log_and_register(
    model: PregameModel, report: BacktestReport, params: dict[str, Any], tracking_uri: str
) -> Registration:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    with mlflow.start_run() as run:
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
        importance = dict(
            zip(
                model.features,
                (float(g) for g in model.estimator.booster_.feature_importance("gain")),
                strict=True,
            )
        )
        mlflow.log_dict(dict(sorted(importance.items(), key=lambda kv: -kv[1])), "importance.json")
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=PregamePyfunc(model),
            registered_model_name=MODEL_NAME,
            pip_requirements=[
                f"{pkg}=={pkg_version(pkg)}"
                for pkg in ("lightgbm", "scikit-learn", "pandas", "numpy")
            ],
        )

    version = str(info.registered_model_version)
    # Full precision (str(float) round-trips): rounding could make an identical score "better".
    tags = {_metric_key(s.season): str(s.model["logloss"]) for s in report.seasons}
    tags |= {"feature_version": model.feature_version, "calibration": model.calibration}
    for key, value in tags.items():
        client.set_model_version_tag(MODEL_NAME, version, key, value)

    promote, reason = promotion_decision(report, _production(client))
    if promote:
        client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, version)
    client.set_model_version_tag(MODEL_NAME, version, "promotion", reason)
    log.info("registered %s v%s (promoted=%s: %s)", MODEL_NAME, version, promote, reason)
    return Registration(run_id=run.info.run_id, version=version, promoted=promote, reason=reason)


def load_production(tracking_uri: str) -> tuple[PregameModel, str]:
    """The current production model and its registry version (used by serving in M3)."""
    mlflow.set_tracking_uri(tracking_uri)
    production = _production(MlflowClient())
    if production is None:
        raise LookupError(f"no {MODEL_NAME}@{PRODUCTION_ALIAS} model registered")
    pyfunc = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{PRODUCTION_ALIAS}")
    wrapper: PregamePyfunc = pyfunc.unwrap_python_model()
    return wrapper.model, str(production.version)
