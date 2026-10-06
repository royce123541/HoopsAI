from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from mlflow.entities.model_registry import ModelVersion

from hoopsai.features.build import build_features
from hoopsai.ml import backtest, metrics, registry
from hoopsai.ml.backtest import BacktestReport, SeasonResult, walk_forward
from hoopsai.ml.pregame import DEFAULT_PARAMS, fit
from hoopsai.ml.pregame import fit as real_fit
from hoopsai.ml.train import completed_seasons, train
from tests.synthetic import make_league

FAST = {**DEFAULT_PARAMS, "n_estimators": 20, "min_child_samples": 5, "num_leaves": 4}
TODAY = date(2023, 9, 1)  # 2023-24 has not started: 2015-2022 are completed


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return build_features(
        *make_league(seasons=tuple(range(2015, 2023)), days_per_season=30, scheduled_days=0)
    )


# ---------------------------------------------------------------- metrics


def test_log_loss_and_brier_known_values() -> None:
    y, p = [1, 0], [0.8, 0.4]
    assert metrics.log_loss(y, p) == pytest.approx(-(np.log(0.8) + np.log(0.6)) / 2)
    assert metrics.brier(y, p) == pytest.approx((0.2**2 + 0.4**2) / 2)
    assert metrics.accuracy(y, p) == 1.0
    assert np.isfinite(metrics.log_loss([1], [0.0]))  # clipped, not infinite


def test_ece_is_zero_when_bins_match_outcomes() -> None:
    p = [0.25] * 4 + [0.75] * 4
    y = [1, 0, 0, 0] + [1, 1, 1, 0]
    assert metrics.ece(y, p) == pytest.approx(0.0)
    assert metrics.ece([1, 1], [0.1, 0.1]) == pytest.approx(0.9)


# ---------------------------------------------------------------- model


def test_walk_forward_never_trains_on_the_test_season(
    features: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[int, list[int], int | None]] = []

    def spy(train: pd.DataFrame, calib: pd.DataFrame | None = None, params: Any = None) -> Any:
        model = real_fit(train, calib, params)
        seen.append((0, model.train_seasons, model.calibration_season))
        return model

    monkeypatch.setattr(backtest, "fit", spy)
    report = walk_forward(features, [2021, 2022], FAST)

    assert [s.season for s in report.seasons] == [2021, 2022]
    assert len(seen) == 4  # 2 seasons x 2 calibration variants
    for test_season, (_, trained, calibrated) in zip([2021, 2022] * 2, seen, strict=True):
        assert max(trained) < test_season
        assert calibrated is None or calibrated == test_season - 1
    assert report.calibration in ("none", "isotonic")
    assert report.pooled_model["n"] == report.pooled_elo["n"] == 2 * 60


def test_predictions_are_probabilities_and_explanations_rank_by_impact(
    features: pd.DataFrame,
) -> None:
    data = backtest.labelled(features)
    model = fit(data[data["season"] < 2022], params=FAST)
    test = data[data["season"] == 2022]

    p = model.predict_proba(test)
    assert p.shape == (len(test),)
    assert ((p > 0) & (p < 1)).all()

    explained = model.explain(test.head(3), top_k=4)
    assert len(explained) == 3
    for factors in explained:
        assert len(factors) == 4
        impacts = [abs(f["contribution"]) for f in factors]
        assert impacts == sorted(impacts, reverse=True)
        assert {f["feature"] for f in factors} <= set(model.features)


def test_completed_seasons_excludes_the_current_one(features: pd.DataFrame) -> None:
    assert completed_seasons(features, TODAY) == list(range(2015, 2023))
    assert completed_seasons(features, date(2022, 12, 1)) == list(range(2015, 2022))


# ---------------------------------------------------------------- registry


def _report(model_ll: float, elo_ll: float = 0.70, season: int = 2022) -> BacktestReport:
    m = {"logloss": model_ll, "brier": 0.20, "accuracy": 0.6, "ece": 0.01, "n": 100.0}
    e = {"logloss": elo_ll, "brier": 0.25, "accuracy": 0.6, "ece": 0.01, "n": 100.0}
    seasons = [SeasonResult(season - 1, m, e), SeasonResult(season, m, e)]
    return BacktestReport("none", seasons, m, e, [])


def _version(tags: dict[str, str]) -> ModelVersion:
    return cast(ModelVersion, SimpleNamespace(version="3", tags=tags))


def test_promotion_rules() -> None:
    assert registry.promotion_decision(_report(0.62), None) == (True, "no production model yet")

    worse_than_elo = _report(0.72)
    promote, reason = registry.promotion_decision(worse_than_elo, None)
    assert not promote and "beats_elo_logloss" in reason

    prod = _version({"backtest_logloss_2022": "0.6100"})
    assert registry.promotion_decision(_report(0.6050), prod)[0] is True
    assert registry.promotion_decision(_report(0.6150), prod)[0] is False
    # A production model never evaluated on the newest season gets replaced.
    assert registry.promotion_decision(_report(0.65, season=2023), prod)[0] is True


def test_train_registers_promotes_and_loads_production(
    features: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Synthetic box scores are noise, so the model cannot beat Elo; acceptance is not the
    # subject here (see test_promotion_rules).
    monkeypatch.setattr(BacktestReport, "acceptance", lambda self, **_: {"ok": True})
    monkeypatch.chdir(tmp_path)  # MLflow writes artifacts under ./mlruns
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"

    first = train(features, today=TODAY, params=FAST, n_test_seasons=2, tracking_uri=tracking_uri)
    assert first.registration is not None
    assert (first.registration.version, first.registration.promoted) == ("1", True)

    loaded, version = registry.load_production(tracking_uri)
    assert version == "1"
    sample = backtest.labelled(features).tail(20)
    np.testing.assert_allclose(loaded.predict_proba(sample), first.model.predict_proba(sample))
    assert loaded.train_seasons == list(range(2015, 2023))

    # Same data and seed -> same backtest log loss -> not strictly better -> not promoted.
    second = train(features, today=TODAY, params=FAST, n_test_seasons=2, tracking_uri=tracking_uri)
    assert second.registration is not None
    assert (second.registration.version, second.registration.promoted) == ("2", False)
    assert registry.load_production(tracking_uri)[1] == "1"
