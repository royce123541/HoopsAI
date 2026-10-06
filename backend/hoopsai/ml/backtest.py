"""Walk-forward evaluation by season: a test season is never seen in training or calibration."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from hoopsai.features.elo import elo_home_win_probability
from hoopsai.ml import metrics
from hoopsai.ml.pregame import LABEL, Calibration, fit

TRAIN_FROM_SEASON = 2005  # 2004-05 is Elo's burn-in season


def labelled(features: pd.DataFrame) -> pd.DataFrame:
    """Final games usable for training/evaluation."""
    return features[(features["status"] == "final") & (features["season"] >= TRAIN_FROM_SEASON)]


def elo_baseline(frame: pd.DataFrame) -> metrics.FloatArray:
    return np.array(
        [
            elo_home_win_probability(h, a, bool(n))
            for h, a, n in zip(
                frame["home_elo_pre"], frame["away_elo_pre"], frame["is_neutral"], strict=True
            )
        ]
    )


@dataclass
class SeasonResult:
    season: int
    model: dict[str, float]
    elo: dict[str, float]


@dataclass
class BacktestReport:
    calibration: Calibration
    seasons: list[SeasonResult]
    pooled_model: dict[str, float]
    pooled_elo: dict[str, float]
    reliability: list[dict[str, float]]
    alternatives: dict[str, float] = field(default_factory=dict)  # pooled logloss per variant

    def acceptance(self, recent: int = 2, max_ece: float = 0.03) -> dict[str, bool]:
        """Plan §8: beat Elo on log loss and Brier in the most recent seasons; ECE <= 0.03."""
        last = self.seasons[-recent:]
        return {
            "beats_elo_logloss": all(s.model["logloss"] < s.elo["logloss"] for s in last),
            "beats_elo_brier": all(s.model["brier"] < s.elo["brier"] for s in last),
            "calibrated": self.pooled_model["ece"] <= max_ece,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration": self.calibration,
            "seasons": [{"season": s.season, "model": s.model, "elo": s.elo} for s in self.seasons],
            "pooled_model": self.pooled_model,
            "pooled_elo": self.pooled_elo,
            "reliability": self.reliability,
            "alternatives": self.alternatives,
            "acceptance": self.acceptance(),
        }


def _predict_season(
    data: pd.DataFrame, season: int, calibration: Calibration, params: dict[str, Any] | None
) -> metrics.FloatArray:
    test = data[data["season"] == season]
    if calibration == "isotonic":
        model = fit(data[data["season"] < season - 1], data[data["season"] == season - 1], params)
    else:
        model = fit(data[data["season"] < season], params=params)
    assert season not in model.train_seasons and model.calibration_season != season
    return model.predict_proba(test)


def walk_forward(
    features: pd.DataFrame,
    test_seasons: list[int],
    params: dict[str, Any] | None = None,
) -> BacktestReport:
    """Backtest both calibration variants on `test_seasons` and keep the one with the lower
    pooled log loss."""
    data = labelled(features)
    preds: dict[Calibration, dict[int, metrics.FloatArray]] = {"none": {}, "isotonic": {}}
    for calibration in ("none", "isotonic"):
        for season in test_seasons:
            preds[calibration][season] = _predict_season(data, season, calibration, params)

    y_all = np.concatenate([data.loc[data["season"] == s, LABEL].to_numpy() for s in test_seasons])
    pooled = {
        c: metrics.log_loss(y_all, np.concatenate(list(p.values()))) for c, p in preds.items()
    }
    best: Calibration = min(pooled, key=lambda c: pooled[c])

    seasons, p_all, elo_all = [], [], []
    for season in test_seasons:
        test = data[data["season"] == season]
        y, p, e = test[LABEL].to_numpy(), preds[best][season], elo_baseline(test)
        seasons.append(SeasonResult(season, metrics.summary(y, p), metrics.summary(y, e)))
        p_all.append(p)
        elo_all.append(e)
    p_cat, e_cat = np.concatenate(p_all), np.concatenate(elo_all)
    return BacktestReport(
        calibration=best,
        seasons=seasons,
        pooled_model=metrics.summary(y_all, p_cat),
        pooled_elo=metrics.summary(y_all, e_cat),
        reliability=metrics.reliability(y_all, p_cat),
        alternatives={f"pooled_logloss_{c}": v for c, v in pooled.items()},
    )
