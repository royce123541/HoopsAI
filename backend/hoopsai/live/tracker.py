"""Turn a growing list of play-by-play events into win-probability points."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from hoopsai.live.state import GameState, game_states, tip_off_state
from hoopsai.ml.ingame import InGameModel
from hoopsai.sources.base import PbpEventRow

# Same filter as training (hoopsai.ml.ingame): these events never change the state.
SKIPPED_EVENTS = ("Substitution", "Timeout", "Instant Replay")


@dataclass(frozen=True)
class WinProbPoint:
    action_id: int
    period: int
    clock_seconds: float
    elapsed_seconds: float
    score_home: int
    score_away: int
    possession: int
    home_win_prob: float
    description: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LiveTracker:
    """Emits a point for each new state since the last update (plus a tip-off point first).
    Feeds re-send the whole game each poll; only events past the last seen action count."""

    def __init__(
        self, model: InGameModel, pregame_prob: float, last_action_id: int | None = None
    ) -> None:
        self.model = model
        self.pregame_prob = pregame_prob
        self.last_action_id = last_action_id  # None: nothing emitted yet
        self.last_prob: float | None = None
        self.finished = False

    def _points(self, states: list[GameState], final: bool) -> list[WinProbPoint]:
        frame = pd.DataFrame([s.as_dict() for s in states])
        probs = list(self.model.predict_proba(frame, self.pregame_prob))
        if final:
            probs[-1] = self.model.predict_state(states[-1], self.pregame_prob, final=True)
        return [
            WinProbPoint(
                action_id=s.action_id,
                period=s.period,
                clock_seconds=s.clock_seconds,
                elapsed_seconds=s.elapsed_seconds,
                score_home=s.score_home,
                score_away=s.score_away,
                possession=s.possession,
                home_win_prob=float(p),
                description=s.description,
            )
            for s, p in zip(states, probs, strict=True)
        ]

    def update(self, events: Sequence[PbpEventRow], final: bool = False) -> list[WinProbPoint]:
        if self.finished:
            return []
        kept = [e for e in events if e["action_type"] not in SKIPPED_EVENTS]
        states = game_states(kept)
        if self.last_action_id is None:
            states = [tip_off_state(), *states]
            new = states
        else:
            new = [s for s in states if s.action_id > self.last_action_id]
        if final and not new and states and self.last_prob not in (0.0, 1.0):
            new = [states[-1]]  # re-emit the last state as the certain final point
        if not new:
            self.finished = final
            return []
        points = self._points(new, final)
        self.last_action_id = max(self.last_action_id or 0, points[-1].action_id)
        self.last_prob = points[-1].home_win_prob
        self.finished = final
        return points
