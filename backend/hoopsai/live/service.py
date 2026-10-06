"""The live pipeline: poll -> state -> in-game model -> serving.live_wp_snapshots + Redis
(docs/ARCHITECTURE.md §3.4). Replay drives the same path from a recorded game."""

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import Engine, delete, func, insert, select, text

from hoopsai.dates import EASTERN
from hoopsai.db.models import LiveWinProb
from hoopsai.features.elo import elo_home_win_probability
from hoopsai.live.channels import channel
from hoopsai.live.source import LiveGameStatus, LiveSource
from hoopsai.live.state import elapsed_seconds
from hoopsai.live.tracker import LiveTracker, WinProbPoint
from hoopsai.ml.ingame import InGameModel
from hoopsai.sources.base import GameStatus, PbpEventRow

log = logging.getLogger(__name__)

Publish = Callable[[str, str], object]  # (channel, message), e.g. redis.Redis.publish


def pregame_probability(engine: Engine, game_id: str) -> float:
    """The game's latest pre-game prediction, else its Elo estimate, else a coin flip."""
    with engine.connect() as conn:
        prob = conn.scalar(
            text("""
                SELECT home_win_prob FROM serving.predictions WHERE game_id = :g
                ORDER BY created_at DESC, id DESC LIMIT 1
            """),
            {"g": game_id},
        )
        if prob is not None:
            return float(prob)
        row = conn.execute(
            text("""
                SELECT (f.features->>'home_elo_pre')::float AS home,
                       (f.features->>'away_elo_pre')::float AS away,
                       coalesce((f.features->>'is_neutral')::float, 0) AS neutral
                FROM features.game_features f WHERE f.game_id = :g
            """),
            {"g": game_id},
        ).first()
    if row is not None and row.home is not None and row.away is not None:
        return elo_home_win_probability(row.home, row.away, bool(row.neutral))
    return 0.5


@dataclass
class Recorder:
    """Persists points and publishes them for WebSocket subscribers."""

    engine: Engine
    publish: Publish
    model_version: str
    source: str  # "live" | "replay"

    def record(self, game_id: str, points: Sequence[WinProbPoint]) -> None:
        if not points:
            return
        rows = [
            p.as_dict()
            | {"game_id": game_id, "source": self.source, "model_version": self.model_version}
            for p in points
        ]
        with self.engine.begin() as conn:
            conn.execute(insert(LiveWinProb), rows)
        message = {
            "type": "points",
            "game_id": game_id,
            "source": self.source,
            "points": [p.as_dict() for p in points],
        }
        self.publish(channel(game_id), json.dumps(message))

    def reset(self, game_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                delete(LiveWinProb).where(
                    LiveWinProb.game_id == game_id, LiveWinProb.source == self.source
                )
            )
        self.publish(channel(game_id), json.dumps({"type": "reset", "game_id": game_id}))

    def last_action_id(self, game_id: str) -> int | None:
        with self.engine.connect() as conn:
            result: int | None = conn.scalar(
                select(func.max(LiveWinProb.action_id)).where(
                    LiveWinProb.game_id == game_id, LiveWinProb.source == self.source
                )
            )
            return result


# ---------------------------------------------------------------- live polling


@dataclass
class LivePoller:
    """Polls the scoreboard every `scoreboard_every_s` and each live game's play-by-play
    every `pbp_every_s`. Only games in core.games (modelled game types) are tracked."""

    engine: Engine
    source: LiveSource
    model: InGameModel
    recorder: Recorder
    scoreboard_every_s: float = 30.0
    pbp_every_s: float = 10.0
    clock: Callable[[], float] = time.monotonic
    trackers: dict[str, LiveTracker] = field(default_factory=dict)
    _last_scoreboard: float | None = None
    _last_pbp: dict[str, float] = field(default_factory=dict)

    def _known_games(self, game_ids: list[str]) -> set[str]:
        if not game_ids:
            return set()
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("SELECT game_id FROM core.games WHERE game_id = ANY(:ids)"), {"ids": game_ids}
            )
            return {r[0] for r in rows}

    def _update_game(self, g: LiveGameStatus) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    UPDATE core.games SET status = :s, home_score = :h, away_score = :a,
                           updated_at = now()
                    WHERE game_id = :g AND status <> 'final'
                """),
                {"s": str(g.status), "h": g.home_score, "a": g.away_score, "g": g.game_id},
            )

    def poll_scoreboard(self, now_utc: datetime) -> None:
        # Late games run past midnight Eastern, so also look at yesterday's board early on.
        today = now_utc.astimezone(EASTERN)
        days: list[date] = [today.date()]
        if today.hour < 4:
            days.append(today.date() - timedelta(days=1))
        statuses = [g for d in days for g in self.source.scoreboard(d)]
        known = self._known_games([g.game_id for g in statuses])
        for g in statuses:
            if g.game_id not in known or g.status == GameStatus.SCHEDULED:
                continue
            if g.status == GameStatus.LIVE and g.game_id not in self.trackers:
                log.info("tracking live game %s", g.game_id)
                self.trackers[g.game_id] = LiveTracker(
                    self.model,
                    pregame_probability(self.engine, g.game_id),
                    last_action_id=self.recorder.last_action_id(g.game_id),
                )
            if g.status == GameStatus.FINAL and g.game_id in self.trackers:
                self.poll_game(g.game_id, final=True)
                del self.trackers[g.game_id]
                log.info("game %s final", g.game_id)
            self._update_game(g)

    def poll_game(self, game_id: str, final: bool = False) -> None:
        events = self.source.play_by_play(game_id)
        self.recorder.record(game_id, self.trackers[game_id].update(events, final=final))

    def tick(self, now_utc: datetime | None = None) -> None:
        now = self.clock()
        if self._last_scoreboard is None or now - self._last_scoreboard >= self.scoreboard_every_s:
            self._last_scoreboard = now
            self.poll_scoreboard(now_utc or datetime.now(UTC))
        for game_id in list(self.trackers):
            if now - self._last_pbp.get(game_id, -1e9) >= self.pbp_every_s:
                self._last_pbp[game_id] = now
                self.poll_game(game_id)

    def run_forever(self, sleep: Callable[[float], None] = time.sleep) -> None:
        log.info("live poller started")
        while True:
            try:
                self.tick()
            except Exception:  # keep polling: one failed request must not end the service
                log.exception("live poll failed")
            sleep(1.0)


# ---------------------------------------------------------------- replay


def replay(
    events: Sequence[PbpEventRow],
    game_id: str,
    model: InGameModel,
    pregame_prob: float,
    recorder: Recorder,
    *,
    speed: float = 30.0,
    step_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Re-run a finished game through the live path: events are revealed as simulated game
    time passes at `speed` x real time. Returns the number of points recorded."""
    recorder.reset(game_id)
    tracker = LiveTracker(model, pregame_prob)
    ordered = sorted(events, key=lambda e: e["action_id"])
    elapsed = [elapsed_seconds(e["period"], e["clock_seconds"]) for e in ordered]
    game_time, shown, total = 0.0, 0, 0
    while shown < len(ordered):
        while shown < len(ordered) and elapsed[shown] <= game_time:
            shown += 1
        points = tracker.update(ordered[:shown])
        recorder.record(game_id, points)
        total += len(points)
        game_time += speed * step_s
        if shown < len(ordered):
            sleep(step_s)
    final_points = tracker.update(ordered, final=True)
    recorder.record(game_id, final_points)
    return total + len(final_points)
