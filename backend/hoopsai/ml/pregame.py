"""Pre-game win-probability model: LightGBM on point-in-time features, optionally followed by
isotonic calibration fitted on a later, held-out season."""

from dataclasses import dataclass, field
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from hoopsai.features.build import FEATURE_VERSION, MODEL_FEATURES
from hoopsai.ml.metrics import FloatArray, clip

Calibration = Literal["none", "isotonic"]

# Shallow, heavily regularised trees: ~30k games of a noisy outcome. Walk-forward backtest
# on 2022-2025 (2026-10-06): pooled log loss 0.6173-0.6183 for 250-500 trees / 8-31 leaves
# vs Elo 0.6297; 100 trees underfits (0.6220). Isotonic calibration lost to none (0.627 vs
# 0.618): the raw model is already calibrated (ECE ~0.015).
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.02,
    "n_estimators": 300,
    "num_leaves": 8,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_lambda": 10.0,
    "random_state": 0,
    "verbose": -1,
}

LABEL = "home_win"


@dataclass
class PregameModel:
    estimator: lgb.LGBMClassifier
    calibrator: IsotonicRegression | None
    features: list[str]
    feature_version: str = FEATURE_VERSION
    train_seasons: list[int] = field(default_factory=list)
    calibration_season: int | None = None

    @property
    def calibration(self) -> Calibration:
        return "none" if self.calibrator is None else "isotonic"

    def _x(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame[self.features].astype(float)

    def raw_proba(self, frame: pd.DataFrame) -> FloatArray:
        proba = np.asarray(self.estimator.predict_proba(self._x(frame)), dtype=float)
        return proba[:, 1]

    def predict_proba(self, frame: pd.DataFrame) -> FloatArray:
        """Calibrated P(home team wins), clipped away from 0 and 1."""
        p = self.raw_proba(frame)
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        return clip(p)

    def explain(self, frame: pd.DataFrame, top_k: int = 5) -> list[list[dict[str, Any]]]:
        """Top contributing features per game (TreeSHAP values in log-odds, from the
        uncalibrated model; isotonic calibration is monotone, so the ranking still holds)."""
        x = self._x(frame)
        contrib = np.asarray(self.estimator.predict(x, pred_contrib=True))[:, :-1]  # drop bias
        out = []
        for row_values, row_contrib in zip(x.to_numpy(), contrib, strict=True):
            order = np.argsort(-np.abs(row_contrib))[:top_k]
            out.append(
                [
                    {
                        "feature": self.features[i],
                        "value": None if np.isnan(row_values[i]) else float(row_values[i]),
                        "contribution": float(row_contrib[i]),
                    }
                    for i in order
                ]
            )
        return out


def fit(
    train: pd.DataFrame,
    calib: pd.DataFrame | None = None,
    params: dict[str, Any] | None = None,
    features: list[str] | None = None,
) -> PregameModel:
    """Train on `train`; if `calib` is given, fit isotonic calibration on its predictions."""
    feats = list(features or MODEL_FEATURES)
    estimator = lgb.LGBMClassifier(**(params or DEFAULT_PARAMS))
    estimator.fit(train[feats].astype(float), train[LABEL].astype(int))
    model = PregameModel(
        estimator=estimator,
        calibrator=None,
        features=feats,
        train_seasons=sorted(int(s) for s in train["season"].unique()),
    )
    if calib is not None:
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(model.raw_proba(calib), calib[LABEL].astype(int))
        model.calibrator = calibrator
        model.calibration_season = int(calib["season"].max())
    return model
