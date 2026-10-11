"""ESPN parsers against a trimmed real response (2026 Finals Game 5, Knicks at Spurs)."""

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from hoopsai.live.state import AWAY, HOME, UNKNOWN, game_states, next_possession
from hoopsai.sources.base import GameStatus, SeasonType
from hoopsai.sources.espn.parse import (
    ESPN_TO_NBA_TEAM,
    parse_clock,
    parse_plays,
    parse_scoreboard,
    parse_team_box,
)

FIXTURES = Path(__file__).parent / "fixtures" / "espn"
NYK, SAS = 1610612752, 1610612759


def load(name: str) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((FIXTURES / f"{name}.json").read_text())
    return payload


def test_every_espn_team_maps_to_a_distinct_nba_team() -> None:
    assert len(ESPN_TO_NBA_TEAM) == 30
    assert len(set(ESPN_TO_NBA_TEAM.values())) == 30


def test_scoreboard() -> None:
    [game] = parse_scoreboard(load("scoreboard_20260613"))
    assert game.espn_id == "401859967"
    assert (game.season, game.season_type) == (2025, SeasonType.PLAYOFFS)
    assert (game.home_team_id, game.away_team_id) == (SAS, NYK)
    assert (game.home_score, game.away_score, game.status) == (90, 94, GameStatus.FINAL)
    # Tip 00:30 UTC on June 14 is the evening of June 13 in New York.
    assert game.tip_time_utc == datetime(2026, 6, 14, 0, 30, tzinfo=UTC)
    assert game.game_date == date(2026, 6, 13)


def test_scoreboard_skips_games_that_do_not_count() -> None:
    payload = load("scoreboard_20260613")
    preseason = json.loads(json.dumps(payload["events"][0]))
    preseason["season"]["type"] = 1
    tbd = json.loads(json.dumps(payload["events"][0]))
    tbd["competitions"][0]["competitors"][0]["team"]["id"] = "-1"
    postponed = json.loads(json.dumps(payload["events"][0]))
    postponed["status"]["type"]["name"] = "STATUS_POSTPONED"  # state stays "post", 0-0
    cup_final = json.loads(json.dumps(payload["events"][0]))
    cup_final["competitions"][0]["type"] = {"id": "39", "abbreviation": "CC"}
    payload["events"] += [preseason, tbd, postponed, cup_final]
    assert [g.espn_id for g in parse_scoreboard(payload)] == ["401859967"]


def test_scheduled_games_have_no_score() -> None:
    payload = load("scoreboard_20260613")
    payload["events"][0]["status"]["type"]["state"] = "pre"
    [game] = parse_scoreboard(payload)
    assert (game.status, game.home_score, game.away_score) == (GameStatus.SCHEDULED, None, None)


def test_team_box_matches_nba_definitions() -> None:
    rows = {r["team_id"]: r for r in parse_team_box(load("summary_401859967"), "G5")}
    nyk = rows[NYK]
    assert (nyk["is_home"], nyk["pts"], nyk["minutes"]) == (False, 94, 240)
    assert (nyk["fgm"], nyk["fga"], nyk["fg3m"], nyk["fg3a"]) == (31, 87, 12, 37)
    assert (nyk["ftm"], nyk["fta"], nyk["oreb"], nyk["dreb"]) == (20, 28, 13, 35)
    assert nyk["tov"] == 14  # totalTurnovers (incl. team turnovers), as the NBA counts TOV
    assert (nyk["plus_minus"], rows[SAS]["plus_minus"]) == (4, -4)


@pytest.mark.parametrize(
    ("display", "seconds"), [("12:00", 720.0), ("11:35", 695.0), ("45.3", 45.3), ("0.0", 0.0)]
)
def test_parse_clock(display: str, seconds: float) -> None:
    assert parse_clock(display) == pytest.approx(seconds)


def test_plays_translate_to_v3_vocabulary() -> None:
    events = parse_plays(load("summary_401859967"), "G5")
    by_kind = {e["action_type"] for e in events}
    assert {"Jump Ball", "Missed Shot", "Rebound"} <= by_kind
    assert [e["action_id"] for e in events] == list(range(1, len(events) + 1))
    jump = next(e for e in events if e["action_type"] == "Jump Ball")
    assert (jump["period"], jump["clock_seconds"]) == (1, 720.0)
    last = events[-1]
    assert (last["action_type"], last["score_home"], last["score_away"]) == ("period", 90, 94)
    # Rebounds and shots carry the team side used for possession.
    rebound = next(e for e in events if e["action_type"] == "Rebound")
    assert rebound["location"] in ("h", "v") and rebound["team_id"] in (NYK, SAS)


@pytest.mark.parametrize(
    ("espn_type", "scoring", "expected"),
    [
        ("Free Throw - 1 of 2", True, HOME),  # earlier FT of a trip: shooter keeps it
        ("Free Throw - 2 of 2", True, AWAY),  # last FT made: ball changes hands
        ("Free Throw - 2 of 2", False, UNKNOWN),  # last FT missed: loose ball
        ("Free Throw - Flagrant 1 of 1", True, HOME),  # flagrant: fouled team keeps it
        ("Free Throw - Technical", True, AWAY),  # technical: possession unchanged
    ],
)
def test_free_throws_follow_the_possession_rules(
    espn_type: str, scoring: bool, expected: int
) -> None:
    summary = load("summary_401859967")
    home_id = next(
        c["team"]["id"]
        for c in summary["header"]["competitions"][0]["competitors"]
        if c["homeAway"] == "home"
    )
    play = dict(summary["plays"][1], type={"id": "0", "text": espn_type}, scoringPlay=scoring)
    play["team"] = {"id": home_id}
    [event] = parse_plays({**summary, "plays": [play]}, "G5")
    assert next_possession(AWAY, event) == expected


def test_states_build_from_espn_plays() -> None:
    states = game_states(parse_plays(load("summary_401859967"), "G5"))
    assert (states[-1].score_home, states[-1].score_away) == (90, 94)
    assert any(s.possession != UNKNOWN for s in states)
