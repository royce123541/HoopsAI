"""In-game win probability (docs/ARCHITECTURE.md §3.3 B).

One training row per play-by-play state, labelled with the game's final result. The model
sees the score margin, time left, possession and the pre-game estimate. That estimate is
out-of-sample: each season's pre-game probabilities come from a pre-game model trained only
on earlier seasons. Otherwise the in-game model would learn to trust an overconfident,
in-sample number. Rows are split by season (never within a game) for evaluation."""

import logging
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from hoopsai.features.build import build_features, load_core
from hoopsai.live.state import GAME_SECONDS, GameState, game_states, tip_off_state
from hoopsai.ml import metrics
from hoopsai.ml.backtest import labelled
from hoopsai.ml.pregame import DEFAULT_PARAMS as PREGAME_PARAMS
from hoopsai.ml.pregame import fit as fit_pregame

log = logging.getLogger(__name__)

FEATURES = [
    "score_diff",
    "seconds_remaining",
    "diff_per_sqrt_time",
    "possession",
    "pregame_logit",
    "pregame_logit_x_time",
    "period",
]
# +1: the home team's chance can only rise with these (more points, the ball, more strength).
MONOTONE = [1, 0, 1, 1, 1, 1, 0]

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "n_estimators": 400,
    "num_leaves": 31,
    "min_child_samples": 500,
    "subsample": 0.5,
    "subsample_freq": 1,
    "monotone_constraints": MONOTONE,
    "monotone_constraints_method": "advanced",
    "random_state": 0,
    "verbose": -1,
}

# Play-by-play before this is older-era basketball (fewer threes, slower pace); default
# training window for `hoopsai train-ingame` and the weekly retrain.
FIRST_TRAINING_SEASON = 2015

# Events that never change score, clock-state or possession; they only duplicate rows.
_NOISE_EVENTS = ("Substitution", "Timeout", "Instant Replay")


def logit(p: float | np.ndarray) -> Any:
    p = np.clip(p, metrics.PROB_EPS, 1 - metrics.PROB_EPS)
    return np.log(p / (1 - p))


def state_features(states: pd.DataFrame, pregame_prob: pd.Series | float) -> pd.DataFrame:
    """Model inputs from state columns: period, score_home, score_away, possession,
    seconds_remaining."""
    diff = states["score_home"] - states["score_away"]
    remaining = states["seconds_remaining"].clip(lower=0)
    pre = logit(np.asarray(pregame_prob, dtype=float))
    return pd.DataFrame(
        {
            "score_diff": diff.astype(float),
            "seconds_remaining": remaining.astype(float),
            "diff_per_sqrt_time": diff / np.sqrt(remaining + 1),
            "possession": states["possession"].astype(float),
            "pregame_logit": pre,
            "pregame_logit_x_time": pre * remaining / GAME_SECONDS,
            "period": states["period"].astype(float),
        },
        index=states.index,
    )


@dataclass
class InGameModel:
    estimator: lgb.LGBMClassifier
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    train_seasons: list[int] = field(default_factory=list)

    def predict_proba(self, states: pd.DataFrame, pregame_prob: pd.Series | float) -> np.ndarray:
        """P(home win) per state. A game that is over (clock at zero in the 4th period or
        later with a lead) is certain: exactly 0 or 1."""
        x = state_features(states, pregame_prob)[self.features]
        p = np.asarray(self.estimator.predict_proba(x), dtype=float)[:, 1]
        p = metrics.clip(p)
        over = (x["seconds_remaining"] <= 0) & (x["score_diff"] != 0) & (x["period"] >= 4)
        return np.where(over, (x["score_diff"] > 0).astype(float), p)

    def predict_state(self, state: GameState, pregame_prob: float, final: bool = False) -> float:
        if final and state.score_home != state.score_away:
            return 1.0 if state.score_home > state.score_away else 0.0
        frame = pd.DataFrame([state.as_dict()])
        return float(self.predict_proba(frame, pregame_prob)[0])


# ---------------------------------------------------------------- training data


def oos_pregame_probs(features: pd.DataFrame, seasons: list[int]) -> pd.Series:
    """game_id -> pre-game P(home win), each season predicted by a model trained only on
    earlier seasons (the walk-forward scheme of hoopsai.ml.backtest)."""
    data = labelled(features)
    out = []
    for season in seasons:
        model = fit_pregame(data[data["season"] < season], params=PREGAME_PARAMS)
        target = features[(features["season"] == season) & (features["status"] == "final")]
        out.append(pd.Series(model.predict_proba(target), index=target["game_id"].to_numpy()))
    return pd.concat(out)


def load_season_states(engine: Engine, season: int) -> pd.DataFrame:
    """All states of a season's final games with play-by-play, plus a tip-off state each."""
    events = pd.read_sql(
        text("""
            SELECT e.game_id, e.action_id, e.period, e.clock_seconds, e.location,
                   e.action_type, e.sub_type, e.score_home, e.score_away,
                   CASE WHEN e.action_type = 'Free Throw' THEN left(e.description, 4) END
                     AS description
            FROM core.pbp_events e JOIN core.games g USING (game_id)
            WHERE g.season = :season AND g.status = 'final'
            ORDER BY e.game_id, e.action_id
        """),
        engine,
        params={"season": season},
    )
    # These never change score or possession (next_possession keeps it), so drop them first.
    events = events[~events["action_type"].isin(_NOISE_EVENTS)]
    frames = []
    for game_id, game_events in events.groupby("game_id", sort=False):
        rows = [tip_off_state(), *game_states(game_events.to_dict("records"))]
        frame = pd.DataFrame([s.as_dict() for s in rows])
        frame["game_id"] = game_id
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_training_frame(engine: Engine, seasons: list[int]) -> pd.DataFrame:
    features = build_features(*load_core(engine))
    pregame = oos_pregame_probs(features, seasons)
    outcome = labelled(features).set_index("game_id")["home_win"]
    frames = []
    for season in seasons:
        states = load_season_states(engine, season)
        if states.empty:
            log.warning("season %s has no play-by-play; skipping", season)
            continue
        states = states[states["game_id"].isin(pregame.index)]
        states["season"] = season
        states["pregame_prob"] = pregame.loc[states["game_id"]].to_numpy()
        states["home_win"] = outcome.loc[states["game_id"]].to_numpy()
        frames.append(states)
        log.info(
            "season %s: %d states from %d games", season, len(states), states["game_id"].nunique()
        )
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- fit and evaluate


def fit(train: pd.DataFrame, params: dict[str, Any] | None = None) -> InGameModel:
    x = state_features(train, train["pregame_prob"])[FEATURES]
    estimator = lgb.LGBMClassifier(**(params or DEFAULT_PARAMS))
    estimator.fit(x, train["home_win"].astype(int))
    return InGameModel(
        estimator=estimator, train_seasons=sorted(int(s) for s in train["season"].unique())
    )


QUARTER_BUCKETS = [("Q1", 2160, 2880), ("Q2", 1440, 2160), ("Q3", 720, 1440), ("Q4", 0, 720)]


@dataclass
class InGameReport:
    test_season: int
    model: dict[str, float]
    pregame_only: dict[str, float]
    by_quarter: dict[str, dict[str, float]]
    reliability: list[dict[str, float]]

    def acceptance(self, max_ece: float = 0.03) -> dict[str, bool]:
        return {
            "beats_pregame_only": self.model["logloss"] < self.pregame_only["logloss"],
            "calibrated": self.model["ece"] <= max_ece,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_season": self.test_season,
            "model": self.model,
            "pregame_only": self.pregame_only,
            "by_quarter": self.by_quarter,
            "reliability": self.reliability,
            "acceptance": self.acceptance(),
        }


def evaluate(model: InGameModel, test: pd.DataFrame) -> InGameReport:
    y = test["home_win"].to_numpy()
    p = model.predict_proba(test, test["pregame_prob"])
    by_quarter = {}
    regulation = test["period"] <= 4
    for name, lo, hi in QUARTER_BUCKETS:
        mask = (
            regulation & (test["seconds_remaining"] > lo) & (test["seconds_remaining"] <= hi)
        ).to_numpy()
        if mask.any():
            by_quarter[name] = metrics.summary(y[mask], p[mask])
    return InGameReport(
        test_season=int(test["season"].max()),
        model=metrics.summary(y, p),
        pregame_only=metrics.summary(y, test["pregame_prob"].to_numpy()),
        by_quarter=by_quarter,
        reliability=metrics.reliability(y, p),
    )


def train_and_evaluate(
    frame: pd.DataFrame, params: dict[str, Any] | None = None
) -> tuple[InGameModel, InGameReport]:
    """Evaluate on the newest season (model trained on the rest), then refit on everything."""
    test_season = int(frame["season"].max())
    held_out = fit(frame[frame["season"] < test_season], params)
    report = evaluate(held_out, frame[frame["season"] == test_season])
    return fit(frame, params), report


def format_report(report: InGameReport) -> str:
    lines = [f"test season {report.test_season}: log loss / Brier / accuracy / ECE"]
    rows = [("model", report.model), ("pregame only", report.pregame_only)]
    rows += [(f"  {q}", m) for q, m in report.by_quarter.items()]
    for name, m in rows:
        lines.append(
            f"{name:14s} {m['logloss']:.4f} / {m['brier']:.4f} / {m['accuracy']:.3f} / "
            f"{m['ece']:.3f}   (n={int(m['n']):,})"
        )
    checks = ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in report.acceptance().items())
    lines.append(f"acceptance: {checks}")
    return "\n".join(lines)


def seasons_with_pbp(engine: Engine, min_games: int = 1000) -> list[int]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT g.season FROM core.games g
                WHERE g.status = 'final' AND EXISTS (
                    SELECT 1 FROM core.pbp_events e WHERE e.game_id = g.game_id)
                GROUP BY g.season HAVING count(*) >= :n ORDER BY g.season
            """),
            {"n": min_games},
        )
        return [int(r[0]) for r in rows]
