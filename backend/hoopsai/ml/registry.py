"""Model registry for the pre-game and in-game models, in MLflow or in Postgres.

Each training run logs params, metrics and the model, and registers a new version. The
`production` alias moves to it only if it passes its acceptance checks and beats the current
production version's out-of-sample log loss on the same held-out season (pre-game: the most
recent walk-forward backtest season; in-game: the newest season, held out of training).

The tracking URI "db" keeps the same registry in Postgres (serving.registered_models) instead
of an MLflow server, for hosts such as GitHub Actions with a free Neon database."""

import logging
import os
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import version as pkg_version
from typing import Any, Protocol

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.pyfunc.model import PythonModel
from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert

from hoopsai.db.models import RegisteredModel
from hoopsai.db.session import get_sync_engine
from hoopsai.ml.backtest import BacktestReport
from hoopsai.ml.ingame import InGameModel, InGameReport
from hoopsai.ml.pregame import PregameModel

log = logging.getLogger(__name__)

MODEL_NAME = "hoopsai-pregame"
EXPERIMENT = "hoopsai-pregame"
INGAME_MODEL_NAME = "hoopsai-ingame"
INGAME_EXPERIMENT = "hoopsai-ingame"
PRODUCTION_ALIAS = "production"
DB_TRACKING_URI = "db"  # registry in Postgres instead of an MLflow server

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


class VersionInfo(Protocol):
    """What promotion needs of a registered version: an MLflow ModelVersion or a DbVersion."""

    @property
    def version(self) -> Any: ...
    @property
    def tags(self) -> dict[str, str]: ...


@dataclass(frozen=True)
class DbVersion:
    version: str
    tags: dict[str, str]


@dataclass
class RunRecord:
    """Everything logged with a model version, for either registry."""

    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)  # "name.json" -> JSON content
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class Registration:
    run_id: str
    version: str
    promoted: bool
    reason: str


@dataclass(frozen=True)
class VersionDetails:
    run_id: str
    params: dict[str, Any]
    artifacts: dict[str, Any]


def _metric_key(season: int) -> str:
    return f"backtest_logloss_{season}"


def _production(client: MlflowClient, name: str = MODEL_NAME) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(name, PRODUCTION_ALIAS)
    except MlflowException:
        return None


def _compare(
    acceptance: dict[str, bool], season: int, ours: float, production: VersionInfo | None
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


def promotion_decision(report: BacktestReport, production: VersionInfo | None) -> Decision:
    latest = report.seasons[-1]
    return _compare(report.acceptance(), latest.season, latest.model["logloss"], production)


def ingame_promotion_decision(report: InGameReport, production: VersionInfo | None) -> Decision:
    return _compare(report.acceptance(), report.test_season, report.model["logloss"], production)


# ---------------------------------------------------------------- registering


def _register(
    *,
    name: str,
    experiment: str,
    tracking_uri: str,
    python_model: PregamePyfunc | InGamePyfunc,
    record: RunRecord,
    decide: Callable[[VersionInfo | None], Decision],
) -> Registration:
    """Log the run and the model, register a version, tag it, and move the production alias
    if `decide` says so."""
    if tracking_uri == DB_TRACKING_URI:
        return _register_db(name, python_model.model, record, decide)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    client = MlflowClient()
    with mlflow.start_run() as run:
        mlflow.log_params(record.params)
        mlflow.log_metrics(record.metrics)
        for artifact, content in record.artifacts.items():
            mlflow.log_dict(content, artifact)
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=python_model,
            registered_model_name=name,
            pip_requirements=PIP_REQUIREMENTS,
        )
    version = str(info.registered_model_version)
    for key, value in record.tags.items():
        client.set_model_version_tag(name, version, key, value)
    promote, reason = decide(_production(client, name))
    if promote:
        client.set_registered_model_alias(name, PRODUCTION_ALIAS, version)
    client.set_model_version_tag(name, version, "promotion", reason)
    log.info("registered %s v%s (promoted=%s: %s)", name, version, promote, reason)
    return Registration(run_id=run.info.run_id, version=version, promoted=promote, reason=reason)


def _db_run_id(name: str, version: int | str) -> str:
    return f"db:{name}:{version}"


def _db_production(name: str) -> DbVersion | None:
    with get_sync_engine().connect() as conn:
        row = conn.execute(
            select(RegisteredModel.version, RegisteredModel.tags).where(
                RegisteredModel.name == name, RegisteredModel.is_production
            )
        ).first()
    return None if row is None else DbVersion(str(row.version), dict(row.tags))


def _store_db_version(
    name: str,
    version: int,
    model: PregameModel | InGameModel,
    record: RunRecord,
    *,
    production: bool,
) -> None:
    """Insert (or replace) one version; making it production demotes the others."""
    with get_sync_engine().begin() as conn:
        if production:
            conn.execute(
                update(RegisteredModel)
                .where(RegisteredModel.name == name)
                .values(is_production=False)
            )
        values = {
            "model": pickle.dumps(model),
            "params": record.params,
            "metrics": record.metrics,
            "artifacts": record.artifacts,
            "tags": record.tags,
            "is_production": production,
        }
        stmt = insert(RegisteredModel).values(name=name, version=version, **values)
        conn.execute(
            stmt.on_conflict_do_update(
                index_elements=["name", "version"], set_={k: stmt.excluded[k] for k in values}
            )
        )


def _register_db(
    name: str,
    model: PregameModel | InGameModel,
    record: RunRecord,
    decide: Callable[[VersionInfo | None], Decision],
) -> Registration:
    promote, reason = decide(_db_production(name))
    with get_sync_engine().connect() as conn:
        latest = conn.scalar(
            select(func.max(RegisteredModel.version)).where(RegisteredModel.name == name)
        )
    version = (latest or 0) + 1
    record.tags = record.tags | {"promotion": reason}
    _store_db_version(name, version, model, record, production=promote)
    log.info("registered %s v%s in db (promoted=%s: %s)", name, version, promote, reason)
    return Registration(_db_run_id(name, version), str(version), promote, reason)


def copy_production_to_db(tracking_uri: str, name: str) -> str:
    """Copy MLflow's production version of `name` into the db registry as production,
    keeping its version number (so predictions keep pointing at the same version)."""
    version = production_version(tracking_uri, name)
    if version is None:
        raise LookupError(f"no {name}@{PRODUCTION_ALIAS} model registered in {tracking_uri}")
    model, _ = _load(tracking_uri, name)
    details = version_details(tracking_uri, name, version)
    record = RunRecord(
        params=details.params,
        metrics=_mlflow_metrics(tracking_uri, details.run_id),
        artifacts=details.artifacts,
        tags=_tags(tracking_uri, name, version) or {},
    )
    _store_db_version(name, int(version), model, record, production=True)
    return version


def _mlflow_metrics(tracking_uri: str, run_id: str) -> dict[str, float]:
    mlflow.set_tracking_uri(tracking_uri)
    return dict(MlflowClient().get_run(run_id).data.metrics)


def _importance(features: list[str], gains: Any) -> dict[str, float]:
    pairs = zip(features, (float(g) for g in gains), strict=True)
    return dict(sorted(pairs, key=lambda kv: -kv[1]))


def log_and_register(
    model: PregameModel, report: BacktestReport, params: dict[str, Any], tracking_uri: str
) -> Registration:
    record = RunRecord()
    record.params = {f"lgbm_{k}": v for k, v in params.items()} | {
        "calibration": model.calibration,
        "feature_version": model.feature_version,
        "n_features": len(model.features),
        "train_seasons": f"{model.train_seasons[0]}-{model.train_seasons[-1]}",
        "test_seasons": ",".join(str(s.season) for s in report.seasons),
    }
    record.metrics = {f"pooled_model_{k}": v for k, v in report.pooled_model.items()}
    record.metrics |= {f"pooled_elo_{k}": v for k, v in report.pooled_elo.items()}
    for s in report.seasons:
        record.metrics |= {f"s{s.season}_model_{k}": v for k, v in s.model.items()}
        record.metrics |= {f"s{s.season}_elo_{k}": v for k, v in s.elo.items()}
    gains = model.estimator.booster_.feature_importance("gain")
    record.artifacts = {
        "backtest.json": report.to_dict(),
        "importance.json": _importance(model.features, gains),
    }
    # Full precision (str(float) round-trips): rounding could make an identical score "better".
    record.tags = {_metric_key(s.season): str(s.model["logloss"]) for s in report.seasons}
    record.tags |= {"feature_version": model.feature_version, "calibration": model.calibration}
    return _register(
        name=MODEL_NAME,
        experiment=EXPERIMENT,
        tracking_uri=tracking_uri,
        python_model=PregamePyfunc(model),
        record=record,
        decide=lambda production: promotion_decision(report, production),
    )


def log_and_register_ingame(
    model: InGameModel, report: InGameReport, params: dict[str, Any], tracking_uri: str
) -> Registration:
    record = RunRecord()
    record.params = {f"lgbm_{k}": v for k, v in params.items()} | {
        "train_seasons": f"{model.train_seasons[0]}-{model.train_seasons[-1]}",
        "test_season": report.test_season,
        "features": ",".join(model.features),
    }
    record.metrics = {f"test_model_{k}": v for k, v in report.model.items()}
    record.metrics |= {f"test_pregame_only_{k}": v for k, v in report.pregame_only.items()}
    for quarter, m in report.by_quarter.items():
        record.metrics |= {f"test_{quarter}_{k}": v for k, v in m.items()}
    gains = model.estimator.booster_.feature_importance("gain")
    record.artifacts = {
        "evaluation.json": report.to_dict(),
        "importance.json": _importance(model.features, gains),
    }
    record.tags = {_metric_key(report.test_season): str(report.model["logloss"])}
    return _register(
        name=INGAME_MODEL_NAME,
        experiment=INGAME_EXPERIMENT,
        tracking_uri=tracking_uri,
        python_model=InGamePyfunc(model),
        record=record,
        decide=lambda production: ingame_promotion_decision(report, production),
    )


# ---------------------------------------------------------------- reading


def production_version(tracking_uri: str, name: str) -> str | None:
    """The version the production alias points at (a cheap registry lookup, no download)."""
    if tracking_uri == DB_TRACKING_URI:
        db_production = _db_production(name)
        return None if db_production is None else db_production.version
    mlflow.set_tracking_uri(tracking_uri)
    production = _production(MlflowClient(), name)
    return None if production is None else str(production.version)


def _load(tracking_uri: str, name: str) -> tuple[Any, str]:
    version = production_version(tracking_uri, name)
    if version is None:
        raise LookupError(f"no {name}@{PRODUCTION_ALIAS} model registered")
    # Load that exact version, not the alias: a promotion in between must not pair one
    # version's number with another version's model.
    if tracking_uri == DB_TRACKING_URI:
        with get_sync_engine().connect() as conn:
            blob = conn.execute(
                select(RegisteredModel.model).where(
                    RegisteredModel.name == name, RegisteredModel.version == int(version)
                )
            ).scalar_one()
        return pickle.loads(blob), version  # our own table, written only by this module
    pyfunc = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
    return pyfunc.unwrap_python_model().model, version


def _tags(tracking_uri: str, name: str, version: str) -> dict[str, str] | None:
    if tracking_uri == DB_TRACKING_URI:
        with get_sync_engine().connect() as conn:
            tags = conn.scalar(
                select(RegisteredModel.tags).where(
                    RegisteredModel.name == name, RegisteredModel.version == int(version)
                )
            )
        return None if tags is None else dict(tags)
    mlflow.set_tracking_uri(tracking_uri)
    try:
        return dict(MlflowClient().get_model_version(name, version).tags)
    except MlflowException:
        return None


def heldout_logloss(tracking_uri: str, name: str, version: str) -> float | None:
    """The log loss a version scored on its newest held-out season (its version tag)."""
    tags = _tags(tracking_uri, name, version) or {}
    seasons = sorted(k for k in tags if k.startswith("backtest_logloss_"))
    return float(tags[seasons[-1]]) if seasons else None


def version_details(tracking_uri: str, name: str, version: str) -> VersionDetails:
    """A version's run params and JSON artifacts (backtest report, feature importance)."""
    if tracking_uri == DB_TRACKING_URI:
        with get_sync_engine().connect() as conn:
            row = conn.execute(
                select(RegisteredModel.params, RegisteredModel.artifacts).where(
                    RegisteredModel.name == name, RegisteredModel.version == int(version)
                )
            ).one()
        return VersionDetails(_db_run_id(name, version), dict(row.params), dict(row.artifacts))
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    run_id = str(client.get_model_version(name, version).run_id)
    artifacts = {
        a.path: mlflow.artifacts.load_dict(f"runs:/{run_id}/{a.path}")
        for a in client.list_artifacts(run_id)
        if a.path.endswith(".json")
    }
    return VersionDetails(run_id, dict(client.get_run(run_id).data.params), artifacts)


def load_production(tracking_uri: str) -> tuple[PregameModel, str]:
    """The production pre-game model and its registry version."""
    model, version = _load(tracking_uri, MODEL_NAME)
    return model, version


def load_production_ingame(tracking_uri: str) -> tuple[InGameModel, str]:
    """The production in-game model and its registry version."""
    model, version = _load(tracking_uri, INGAME_MODEL_NAME)
    return model, version
