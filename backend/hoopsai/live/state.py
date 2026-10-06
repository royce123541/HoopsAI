"""Game states from play-by-play, shared by in-game model training and live scoring so the
two can never disagree on what a state is (docs/ARCHITECTURE.md §3.3 B).

Events are PlayByPlayV3 rows (`hoopsai.sources.base.PbpEventRow`) in feed order, i.e.
ascending `action_id`. The feed has no possession field, so possession is inferred:

  made shot by X        -> the other team      turnover by X     -> the other team
  missed shot           -> unknown (loose)     rebound by X      -> X
  last free throw made  -> the other team      last FT missed    -> unknown (loose)
  earlier FT of a trip  -> shooter keeps it    technical FT      -> unchanged
  flagrant / clear-path FT trip -> shooter keeps it (ball is inbounded by the fouled team)
  jump ball, period start       -> unknown (the jump-ball row does not say who won it)
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

PERIOD_SECONDS = 720
OT_SECONDS = 300
REGULATION_PERIODS = 4
GAME_SECONDS = REGULATION_PERIODS * PERIOD_SECONDS

HOME, AWAY, UNKNOWN = 1, -1, 0
_FT_OF = re.compile(r"(\d+) of (\d+)")


def seconds_remaining(period: int, clock_seconds: float) -> float:
    """Seconds left in regulation, or in the current overtime period."""
    if period <= REGULATION_PERIODS:
        return (REGULATION_PERIODS - period) * PERIOD_SECONDS + clock_seconds
    return clock_seconds


def elapsed_seconds(period: int, clock_seconds: float) -> float:
    """Game time since tip-off, counting overtime (x-axis of the live chart)."""
    if period <= REGULATION_PERIODS:
        return period * PERIOD_SECONDS - clock_seconds
    return GAME_SECONDS + (period - REGULATION_PERIODS) * OT_SECONDS - clock_seconds


@dataclass(frozen=True)
class GameState:
    action_id: int
    period: int
    clock_seconds: float
    score_home: int
    score_away: int
    possession: int  # HOME, AWAY or UNKNOWN, after this event
    description: str | None

    @property
    def seconds_remaining(self) -> float:
        return seconds_remaining(self.period, self.clock_seconds)

    @property
    def elapsed_seconds(self) -> float:
        return elapsed_seconds(self.period, self.clock_seconds)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "seconds_remaining": self.seconds_remaining,
            "elapsed_seconds": self.elapsed_seconds,
        }


def _side(event: Mapping[Any, Any]) -> int:
    return {"h": HOME, "v": AWAY}.get(event.get("location") or "", UNKNOWN)


def next_possession(current: int, event: Mapping[Any, Any]) -> int:
    kind, side = event.get("action_type"), _side(event)
    description = event.get("description") or ""
    if kind == "period":
        return UNKNOWN
    if kind == "Jump Ball":
        return UNKNOWN
    if side == UNKNOWN:
        return current
    if kind in ("Made Shot", "Turnover"):
        return -side
    if kind == "Missed Shot":
        return UNKNOWN
    if kind == "Rebound":
        return side
    if kind == "Free Throw":
        sub_type = event.get("sub_type") or ""
        if "Technical" in sub_type:
            return current
        m = _FT_OF.search(sub_type)
        if not m or int(m.group(1)) < int(m.group(2)):
            return side
        if "Flagrant" in sub_type or "Clear Path" in sub_type:
            return side
        return UNKNOWN if description.startswith("MISS") else -side
    return current


def game_states(events: Iterable[Mapping[Any, Any]]) -> list[GameState]:
    """One state after each event, in feed order."""
    states = []
    possession = UNKNOWN
    for e in sorted(events, key=lambda e: e["action_id"]):
        possession = next_possession(possession, e)
        states.append(
            GameState(
                action_id=int(e["action_id"]),
                period=int(e["period"]),
                clock_seconds=float(e["clock_seconds"]),
                score_home=int(e["score_home"]),
                score_away=int(e["score_away"]),
                possession=possession,
                description=e.get("description"),
            )
        )
    return states


def tip_off_state() -> GameState:
    return GameState(0, 1, float(PERIOD_SECONDS), 0, 0, UNKNOWN, "Tip-off")
