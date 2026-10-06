"""Model monitoring (docs/ARCHITECTURE.md §3.5): how well the models do on real games, against
how well they were expected to do.

- Pre-game: each finished game's latest prediction made *before tip-off* (predictions
  scored after the fact are excluded: they are not forecasts).
- In-game: every live point of finished games (`source = 'live'`; replays are excluded).

Each is checked over the current season and the last 30 days. A window is a warning when its
log loss exceeds the model's held-out log loss by more than a tolerance, or when it is badly
calibrated; it is "insufficient" until enough games have finished."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal

import pandas as pd
from sqlalchemy import Engine, insert, text

from hoopsai.db.models import MonitoringRun
from hoopsai.ml import metrics
from hoopsai.sources.base import current_season

log = logging.getLogger(__name__)

ModelKind = Literal["pregame", "ingame"]
Status = Literal["ok", "warning", "insufficient"]

MIN_GAMES: dict[ModelKind, int] = {"pregame": 50, "ingame": 20}
# Log loss this far above the held-out expectation is a warning. Real seasons vary: the
# pre-game backtest ranged 0.601-0.644 across 2022-2025, so smaller gaps are noise.
TOLERANCE: dict[ModelKind, float] = {"pregame": 0.03, "ingame": 0.05}
MAX_ECE = 0.06  # calibration error that is a warning (needs >= 200 games to judge)
ECE_MIN_GAMES = 200

PREGAME_SQL = """
    SELECT DISTINCT ON (p.game_id) p.game_id, p.model_version, p.home_win_prob,
           (g.home_score > g.away_score)::int AS home_win
    FROM serving.predictions p JOIN core.games g USING (game_id)
    WHERE p.made_before_tip AND g.status = 'final' AND g.season = :season
      AND g.game_date >= :since
    ORDER BY p.game_id, p.created_at DESC, p.id DESC
"""

INGAME_SQL = """
    SELECT s.game_id, s.model_version, s.home_win_prob,
           (g.home_score > g.away_score)::int AS home_win
    FROM serving.live_wp_snapshots s JOIN core.games g USING (game_id)
    WHERE s.source = 'live' AND g.status = 'final' AND g.season = :season
      AND g.game_date >= :since
"""


@dataclass(frozen=True)
class MonitorResult:
    model: ModelKind
    window: str
    model_version: str | None
    games: int
    metrics: dict[str, float] | None
    expected_logloss: float | None
    status: Status
    message: str

    def as_row(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "window": self.window,
            "model_version": self.model_version,
            "games": self.games,
            "metrics": self.metrics,
            "expected_logloss": self.expected_logloss,
            "status": self.status,
            "message": self.message,
        }


def judge(
    model: ModelKind,
    window: str,
    rows: pd.DataFrame,
    expected: Callable[[str], float | None],
    min_games: int | None = None,
) -> MonitorResult:
    games = int(rows["game_id"].nunique()) if not rows.empty else 0
    needed = min_games if min_games is not None else MIN_GAMES[model]
    if games < needed:
        return MonitorResult(
            model,
            window,
            None,
            games,
            None,
            None,
            "insufficient",
            f"{games} finished games with predictions; at least {needed} are needed.",
        )
    # Judge against the version that produced most of the window's predictions.
    version = str(rows["model_version"].mode().iloc[0])
    summary = metrics.summary(rows["home_win"], rows["home_win_prob"])
    summary["games"] = float(games)
    baseline = expected(version)
    status: Status = "ok"
    message = f"Log loss {summary['logloss']:.3f} over {games} games"
    if baseline is not None:
        message += f", expected about {baseline:.3f}."
        if summary["logloss"] > baseline + TOLERANCE[model]:
            status = "warning"
            message = f"Worse than expected: {message}"
    else:
        message += "; no held-out log loss to compare with."
    if games >= ECE_MIN_GAMES and summary["ece"] > MAX_ECE:
        status = "warning"
        message += f" Calibration error {summary['ece']:.1%} is above {MAX_ECE:.0%}."
    return MonitorResult(model, window, version, games, summary, baseline, status, message)


def pregame_expected(engine: Engine) -> Callable[[str], float | None]:
    """A pre-game version's pooled walk-forward log loss, from serving.model_versions."""

    def lookup(version: str) -> float | None:
        with engine.connect() as conn:
            value = conn.scalar(
                text(
                    "SELECT (backtest->'pooled_model'->>'logloss')::float "
                    "FROM serving.model_versions WHERE version = :v"
                ),
                {"v": version},
            )
        return None if value is None else float(value)

    return lookup


def run_monitor(
    engine: Engine,
    today: date,
    ingame_expected: Callable[[str], float | None],
    min_games: dict[ModelKind, int] | None = None,
) -> list[MonitorResult]:
    """Check both models over the current season and the last 30 days; store and log."""
    season = current_season(today)
    windows = {"season": date(season, 8, 1), "last_30_days": today - timedelta(days=30)}
    sources: dict[ModelKind, tuple[str, Callable[[str], float | None]]] = {
        "pregame": (PREGAME_SQL, pregame_expected(engine)),
        "ingame": (INGAME_SQL, ingame_expected),
    }
    results = []
    for kind, (sql, expected) in sources.items():
        for window, since in windows.items():
            rows = pd.read_sql(text(sql), engine, params={"season": season, "since": since})
            minimum = (min_games or {}).get(kind)
            result = judge(kind, window, rows, expected, minimum)
            results.append(result)
            level = logging.WARNING if result.status == "warning" else logging.INFO
            log.log(level, "monitor %s/%s: %s %s", kind, window, result.status, result.message)
    with engine.begin() as conn:
        conn.execute(insert(MonitoringRun), [r.as_row() for r in results])
    return results
