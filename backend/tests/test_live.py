"""Game states, the in-game model, the live tracker/poller, replay, and the live API."""

import json
from collections.abc import Iterator
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest
import redis
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from hoopsai.api.main import create_app
from hoopsai.ingest.pipeline import Ingestor
from hoopsai.live import service
from hoopsai.live.source import LiveGameStatus
from hoopsai.live.state import (
    AWAY,
    HOME,
    UNKNOWN,
    elapsed_seconds,
    game_states,
    next_possession,
    seconds_remaining,
    tip_off_state,
)
from hoopsai.live.tracker import LiveTracker
from hoopsai.ml import ingame
from hoopsai.sources.base import GameStatus, PbpEventRow
from hoopsai.sources.nba_stats.parse import parse_play_by_play
from tests.conftest import load_fixture
from tests.fixture_source import FixtureSource
from tests.synthetic import make_pbp_game

GAME = "0022600002"  # NYK-PHI from the schedule_2026 fixture


# ---------------------------------------------------------------- state


def test_clock_conversions() -> None:
    assert seconds_remaining(1, 720) == 2880
    assert seconds_remaining(4, 30.5) == 30.5
    assert seconds_remaining(5, 120) == 120  # overtime: time left in the OT period
    assert elapsed_seconds(1, 720) == 0
    assert elapsed_seconds(4, 0) == 2880
    assert elapsed_seconds(5, 0) == 3180


@pytest.mark.parametrize(
    ("event", "before", "after"),
    [
        ({"action_type": "Made Shot", "location": "h"}, HOME, AWAY),
        ({"action_type": "Missed Shot", "location": "v"}, AWAY, UNKNOWN),
        ({"action_type": "Rebound", "location": "v"}, UNKNOWN, AWAY),
        ({"action_type": "Turnover", "location": "h"}, HOME, AWAY),
        (
            {"action_type": "Free Throw", "location": "h", "sub_type": "Free Throw 1 of 2"},
            AWAY,
            HOME,
        ),
        (
            {"action_type": "Free Throw", "location": "h", "sub_type": "Free Throw 2 of 2"},
            HOME,
            AWAY,
        ),
        (
            {
                "action_type": "Free Throw",
                "location": "h",
                "sub_type": "Free Throw 2 of 2",
                "description": "MISS Brunson Free Throw 2 of 2",
            },
            HOME,
            UNKNOWN,
        ),
        (
            {"action_type": "Free Throw", "location": "v", "sub_type": "Free Throw Technical"},
            HOME,
            HOME,
        ),
        (
            {
                "action_type": "Free Throw",
                "location": "v",
                "sub_type": "Free Throw Flagrant 2 of 2",
            },
            HOME,
            AWAY,
        ),
        ({"action_type": "Jump Ball", "location": "h"}, HOME, UNKNOWN),
        ({"action_type": "period", "location": None}, HOME, UNKNOWN),
        ({"action_type": "Foul", "location": "v"}, HOME, HOME),
    ],
)
def test_possession_rules(event: dict[str, Any], before: int, after: int) -> None:
    assert next_possession(before, event) == after


def test_states_from_recorded_game() -> None:
    events = parse_play_by_play(load_fixture("pbp_0022400001"))
    states = game_states(events)
    assert [s.action_id for s in states] == sorted(e["action_id"] for e in events)
    first_make = next(s for s in states if s.score_home + s.score_away > 0)
    assert first_make.possession != UNKNOWN
    assert (states[-1].score_home, states[-1].score_away) == (116, 117)


# ---------------------------------------------------------------- in-game model


def synthetic_frame(n_games: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_games):
        edge = float(rng.normal())
        events, home_won = make_pbp_game(rng, f"g{seed}-{i}", edge)
        rows = [tip_off_state(), *game_states(events)]
        frame = pd.DataFrame([s.as_dict() for s in rows])
        frame["game_id"], frame["season"] = f"g{seed}-{i}", 2020 + i % 3
        frame["pregame_prob"] = 1 / (1 + np.exp(-0.8 * edge))
        frame["home_win"] = float(home_won)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def trained() -> tuple[ingame.InGameModel, ingame.InGameReport]:
    params = {**ingame.DEFAULT_PARAMS, "n_estimators": 60, "min_child_samples": 50}
    return ingame.train_and_evaluate(synthetic_frame(240, seed=3), params)


def test_ingame_model_beats_pregame_only(
    trained: tuple[ingame.InGameModel, ingame.InGameReport],
) -> None:
    _, report = trained
    assert report.test_season == 2022
    assert report.model["logloss"] < report.pregame_only["logloss"]
    assert set(report.by_quarter) == {"Q1", "Q2", "Q3", "Q4"}
    # Late in games the score dominates: far more accurate than at tip-off.
    assert report.by_quarter["Q4"]["logloss"] < report.by_quarter["Q1"]["logloss"]


def state_frame(diff: int, remaining: float, possession: int = UNKNOWN) -> pd.DataFrame:
    period = 4 if remaining <= 720 else 4 - int(remaining // 720)
    clock = remaining - (4 - period) * 720
    return pd.DataFrame(
        [
            {
                "period": period,
                "clock_seconds": clock,
                "seconds_remaining": remaining,
                "score_home": 100 + diff,
                "score_away": 100,
                "possession": possession,
            }
        ]
    )


def test_ingame_model_is_monotone_and_certain_at_the_buzzer(
    trained: tuple[ingame.InGameModel, ingame.InGameReport],
) -> None:
    model, _ = trained
    probs = [float(model.predict_proba(state_frame(d, 300), 0.5)[0]) for d in range(-15, 16, 3)]
    assert probs == sorted(probs)
    assert float(model.predict_proba(state_frame(2, 0), 0.2)[0]) == 1.0
    assert float(model.predict_proba(state_frame(-1, 0), 0.9)[0]) == 0.0
    with_ball = float(model.predict_proba(state_frame(0, 60, HOME), 0.5)[0])
    without = float(model.predict_proba(state_frame(0, 60, AWAY), 0.5)[0])
    assert with_ball >= without


def test_tracker_emits_tip_off_then_only_new_states(
    trained: tuple[ingame.InGameModel, ingame.InGameReport],
) -> None:
    model, _ = trained
    events, home_won = make_pbp_game(np.random.default_rng(1), GAME, 0.0)
    tracker = LiveTracker(model, pregame_prob=0.6)

    first = tracker.update(events[:10])
    assert first[0].action_id == 0 and first[0].description == "Tip-off"
    assert len(first) == 11
    assert tracker.update(events[:10]) == []  # nothing new
    later = tracker.update(events[:15])
    assert [p.action_id for p in later] == [e["action_id"] for e in events[10:15]]

    final = tracker.update(events, final=True)
    assert final[-1].home_win_prob == (1.0 if home_won else 0.0)
    assert tracker.update(events, final=True) == []


def test_tracker_does_not_repeat_an_already_certain_final_point(
    trained: tuple[ingame.InGameModel, ingame.InGameReport],
) -> None:
    model, _ = trained
    events, home_won = make_pbp_game(np.random.default_rng(1), GAME, 0.0)
    tracker = LiveTracker(model, pregame_prob=0.5)
    points = tracker.update(events)  # the buzzer state with a lead is already certain
    assert points[-1].home_win_prob == (1.0 if home_won else 0.0)
    assert tracker.update(events, final=True) == []
    assert tracker.finished


# ---------------------------------------------------------------- poller, replay, API


class FakeLiveSource:
    def __init__(self, events: list[PbpEventRow]) -> None:
        self.events = events
        self.revealed = 0
        self.status = GameStatus.LIVE

    def scoreboard(self, day: date) -> list[LiveGameStatus]:
        last = self.events[self.revealed - 1] if self.revealed else None
        return [
            LiveGameStatus(
                GAME,
                self.status,
                int(last["period"]) if last else 1,
                int(last["score_home"]) if last else 0,
                int(last["score_away"]) if last else 0,
            ),
            LiveGameStatus("0012600999", GameStatus.LIVE, 1, 0, 0),  # preseason: not tracked
        ]

    def play_by_play(self, game_id: str) -> list[PbpEventRow]:
        return self.events[: self.revealed]


class Published(list[tuple[str, dict[str, Any]]]):
    def __call__(self, channel: str, message: str) -> None:
        self.append((channel, json.loads(message)))


@pytest.fixture
def games_db(db: Engine) -> Engine:
    Ingestor(db, FixtureSource()).backfill(
        2025, 2026, pbp=False, force=False, today=date(2026, 10, 7)
    )
    return db


def snapshot_count(engine: Engine, source: str) -> int:
    with engine.connect() as conn:
        return int(
            conn.scalar(
                text("SELECT count(*) FROM serving.live_wp_snapshots WHERE source = :s"),
                {"s": source},
            )
        )


def test_poller_tracks_live_game_to_final(
    games_db: Engine, trained: tuple[ingame.InGameModel, ingame.InGameReport]
) -> None:
    model, _ = trained
    events, home_won = make_pbp_game(np.random.default_rng(2), GAME, 0.5)
    source = FakeLiveSource(events)
    published = Published()
    recorder = service.Recorder(games_db, published, "1", source="live")
    now = [0.0]
    poller = service.LivePoller(games_db, source, model, recorder, clock=lambda: now[0])
    tip = datetime(2026, 10, 21, 0, 30, tzinfo=UTC)

    source.revealed = 20
    poller.tick(tip)
    assert list(poller.trackers) == [GAME]  # the preseason game is ignored
    assert snapshot_count(games_db, "live") == 21  # tip-off + 20 events

    source.revealed = 30
    now[0] = 5.0
    poller.tick(tip)  # too soon for another play-by-play poll
    now[0] = 11.0
    poller.tick(tip)
    assert snapshot_count(games_db, "live") == 31

    source.revealed = len(events)
    source.status = GameStatus.FINAL
    now[0] = 45.0
    poller.tick(tip)
    assert poller.trackers == {}
    with games_db.connect() as conn:
        status, home, away = conn.execute(
            text("SELECT status, home_score, away_score FROM core.games WHERE game_id = :g"),
            {"g": GAME},
        ).one()
        last = conn.scalar(
            text(
                "SELECT home_win_prob FROM serving.live_wp_snapshots "
                "ORDER BY action_id DESC, id DESC LIMIT 1"
            )
        )
    assert status == "final" and (home > away) == home_won
    assert last == (1.0 if home_won else 0.0)
    assert all(ch == f"game:{GAME}" and msg["type"] == "points" for ch, msg in published)


def test_replay_resets_and_records_whole_game(
    games_db: Engine, trained: tuple[ingame.InGameModel, ingame.InGameReport]
) -> None:
    model, _ = trained
    events, _ = make_pbp_game(np.random.default_rng(4), GAME, 0.0)
    published = Published()
    recorder = service.Recorder(games_db, published, "1", source="replay")

    n = service.replay(events, GAME, model, 0.5, recorder, speed=600, sleep=lambda s: None)
    again = service.replay(events, GAME, model, 0.5, recorder, speed=600, sleep=lambda s: None)

    assert n == again == snapshot_count(games_db, "replay")  # reset before the second run
    assert published[0][1]["type"] == "reset"
    assert sum(len(m.get("points", [])) for _, m in published) == 2 * n


@pytest.fixture
def redis_client() -> Iterator[redis.Redis]:
    client = redis.Redis.from_url("redis://127.0.0.1:6379/0", decode_responses=True)
    try:
        client.ping()
    except redis.ConnectionError:
        pytest.skip("Redis unavailable; run `docker compose up -d redis`")
    yield client
    client.close()


def test_winprob_rest_and_websocket_stream(
    games_db: Engine,
    trained: tuple[ingame.InGameModel, ingame.InGameReport],
    redis_client: redis.Redis,
) -> None:
    model, _ = trained
    events, _ = make_pbp_game(np.random.default_rng(5), GAME, 0.0)
    recorder = service.Recorder(games_db, redis_client.publish, "1", source="live")
    tracker = LiveTracker(model, 0.5)
    recorder.record(GAME, tracker.update(events[:5]))

    with TestClient(create_app()) as client:
        series = client.get(f"/api/games/{GAME}/winprob").json()
        assert series["source"] == "live" and len(series["points"]) == 6
        assert client.get("/api/games/0022600001/winprob").json()["points"] == []

        with client.websocket_connect(f"/api/ws/games/{GAME}") as ws:
            snapshot = ws.receive_json()
            assert snapshot["type"] == "snapshot" and len(snapshot["points"]) == 6
            recorder.record(GAME, tracker.update(events[:8]))
            update = ws.receive_json()
            assert update["type"] == "points"
            assert [p["action_id"] for p in update["points"]] == [6, 7, 8]


def test_build_training_frame_from_database(db: Engine) -> None:
    # 2024 game logs + 2025 schedule (11 final games), each with the recorded play-by-play.
    Ingestor(db, FixtureSource()).backfill(
        2024, 2025, pbp=True, force=False, today=date(2026, 10, 7)
    )

    frame = ingame.build_training_frame(db, [2025])

    assert frame["season"].unique().tolist() == [2025]
    assert frame["game_id"].nunique() == 5
    per_game = frame.groupby("game_id").size()
    assert (per_game == per_game.iloc[0]).all()  # same recorded events for each game
    assert (frame.groupby("game_id")["action_id"].min() == 0).all()  # tip-off state first
    assert frame["pregame_prob"].between(0, 1, inclusive="neither").all()
    assert set(frame["home_win"].unique()) <= {0.0, 1.0}
    assert "description" not in frame  # per-play text is not kept for training
