"""Parsers against trimmed real stats.nba.com responses (tests/fixtures/nba_stats)."""

import copy
from datetime import UTC, date, datetime

import pytest

from hoopsai.sources.base import GameStatus, SeasonType
from hoopsai.sources.nba_stats.parse import (
    parse_clock,
    parse_play_by_play,
    parse_schedule,
    parse_team_game_logs,
    season_type_of,
)
from tests.conftest import load_fixture

LAL, MIN, BOS, NYK = 1610612747, 1610612750, 1610612738, 1610612752


# ---------------------------------------------------------------- game logs


def test_game_logs_pair_home_and_away_rows() -> None:
    parsed = parse_team_game_logs(load_fixture("gamelog_2024_regular"))

    assert len(parsed.games) == 4
    assert len(parsed.team_stats) == 8
    game = next(g for g in parsed.games if g["game_id"] == "0022400062")
    assert game == {
        "game_id": "0022400062",
        "season": 2024,
        "season_type": SeasonType.REGULAR,
        "game_date": date(2024, 10, 22),
        "home_team_id": LAL,
        "away_team_id": MIN,
        "status": GameStatus.FINAL,
        "home_score": 110,
        "away_score": 103,
    }
    # Rows arrive in arbitrary order; home is decided by "vs.", not position.
    boston = next(g for g in parsed.games if g["game_id"] == "0022400061")
    assert (boston["home_team_id"], boston["away_team_id"]) == (BOS, NYK)


def test_game_logs_box_score_columns() -> None:
    parsed = parse_team_game_logs(load_fixture("gamelog_2024_regular"))
    lal = next(s for s in parsed.team_stats if s["game_id"] == "0022400062" and s["is_home"])
    assert lal["team_id"] == LAL
    assert lal["pts"] == 110
    assert lal["minutes"] == 240
    assert lal["plus_minus"] == 7
    assert (lal["fgm"], lal["fga"], lal["fg3m"], lal["fg3a"]) == (42, 95, 5, 30)
    assert (lal["ftm"], lal["fta"], lal["oreb"], lal["dreb"], lal["tov"]) == (21, 25, 15, 31, 7)
    overtime = next(s for s in parsed.team_stats if s["game_id"] == "0022400071")
    assert overtime["minutes"] == 265


def test_game_logs_play_in_season_type() -> None:
    parsed = parse_team_game_logs(load_fixture("gamelog_2024_playin"))
    assert {g["season_type"] for g in parsed.games} == {SeasonType.PLAYIN}
    assert {g["season"] for g in parsed.games} == {2024}


def test_game_logs_skip_unpaired_and_incomplete_games() -> None:
    payload = load_fixture("gamelog_2024_regular")
    rows = payload["resultSets"][0]["rowSet"]
    headers = payload["resultSets"][0]["headers"]
    # Drop MIN's row (unpaired) and null a stat in BOS's game (incomplete box score).
    rows[:] = [r for r in rows if not (r[4] == "0022400062" and r[1] == MIN)]
    next(r for r in rows if r[4] == "0022400061")[headers.index("OREB")] = None

    parsed = parse_team_game_logs(payload)

    assert {g["game_id"] for g in parsed.games} == {"0022400068", "0022400071"}
    assert len(parsed.team_stats) == 4


def test_game_logs_skip_cancelled_games() -> None:
    payload = load_fixture("gamelog_2024_regular")
    headers = payload["resultSets"][0]["headers"]
    for row in payload["resultSets"][0]["rowSet"]:
        if row[4] == "0022400062":  # as stats.nba.com lists the cancelled BOS-IND 2013 game
            for col in headers[headers.index("MIN") :]:
                row[headers.index(col)] = 0
            row[headers.index("WL")] = None

    parsed = parse_team_game_logs(payload)

    assert "0022400062" not in {g["game_id"] for g in parsed.games}
    assert len(parsed.games) == 3


def test_game_logs_resolve_neutral_site_games_via_schedule() -> None:
    payload = load_fixture("gamelog_2024_regular")
    # Neutral-site games list both teams with "@" (e.g. "LAL @ MIN" and "MIN @ LAL").
    for row in payload["resultSets"][0]["rowSet"]:
        if row[4] == "0022400062":
            row[6] = row[6].replace(" vs. ", " @ ")

    skipped = parse_team_game_logs(payload)
    resolved = parse_team_game_logs(payload, home_team_ids={"0022400062": LAL})

    assert "0022400062" not in {g["game_id"] for g in skipped.games}
    game = next(g for g in resolved.games if g["game_id"] == "0022400062")
    assert (game["home_team_id"], game["away_team_id"], game["home_score"]) == (LAL, MIN, 110)


# ---------------------------------------------------------------- schedule


def test_schedule_filters_to_modelled_game_types() -> None:
    rows = parse_schedule(load_fixture("schedule_2025"))
    ids = [r["game_id"] for r in rows]
    assert "0012500008" not in ids  # preseason
    assert ids == ["0022500001", "0022500002", "0022500003", "0022500004", "0042500121"]
    assert {r["season"] for r in rows} == {2025}
    assert rows[-1]["season_type"] == SeasonType.PLAYOFFS


def test_schedule_final_game_fields() -> None:
    game = parse_schedule(load_fixture("schedule_2025"))[1]
    assert game["game_id"] == "0022500002"
    # Late tip: the Eastern game date differs from the UTC date.
    assert game["game_date"] == date(2025, 10, 21)
    assert game["tip_time_utc"] == datetime(2025, 10, 22, 2, 0, tzinfo=UTC)
    assert game["status"] == GameStatus.FINAL
    assert (game["home_team_id"], game["home_score"], game["away_score"]) == (LAL, 109, 119)
    assert game["is_neutral"] is False


def test_schedule_future_games_have_no_score() -> None:
    rows = parse_schedule(load_fixture("schedule_2026"))
    assert [r["status"] for r in rows] == [GameStatus.SCHEDULED, GameStatus.SCHEDULED]
    assert all(r["home_score"] is None and r["away_score"] is None for r in rows)
    assert rows[0]["season"] == 2026


def test_schedule_skips_unresolved_matchups_and_tolerates_bad_tip_time() -> None:
    payload = copy.deepcopy(load_fixture("schedule_2025"))
    games = payload["leagueSchedule"]["gameDates"][1]["games"]
    games[0]["awayTeam"]["teamId"] = 0  # "if necessary" game, opponent TBD
    games[1]["gameDateTimeUTC"] = "TBD"

    rows = {r["game_id"]: r for r in parse_schedule(payload)}

    assert "0022500001" not in rows
    assert rows["0022500002"]["tip_time_utc"] is None


# ---------------------------------------------------------------- play-by-play


@pytest.mark.parametrize(
    ("clock", "seconds"),
    [("PT12M00.00S", 720.0), ("PT10M50.00S", 650.0), ("PT00M04.30S", 4.3), ("", 0.0)],
)
def test_parse_clock(clock: str, seconds: float) -> None:
    assert parse_clock(clock) == pytest.approx(seconds)


def test_play_by_play_forward_fills_scores() -> None:
    events = parse_play_by_play(load_fixture("pbp_0022400001"))

    assert len(events) == 40
    # Feed order is chronological and is preserved; action_id increases with it.
    assert [e["action_id"] for e in events] == sorted(e["action_id"] for e in events)
    by_id = {e["action_id"]: e for e in events}
    assert (by_id[1]["score_home"], by_id[1]["score_away"]) == (0, 0)
    assert (by_id[17]["score_home"], by_id[17]["score_away"]) == (0, 3)
    # Action 474 is a non-scoring team rebound: carries the previous score forward.
    assert (by_id[474]["score_home"], by_id[474]["score_away"]) == (116, 117)
    assert (by_id[475]["score_home"], by_id[475]["score_away"]) == (116, 117)


def test_play_by_play_normalises_empty_fields() -> None:
    by_id = {e["action_id"]: e for e in parse_play_by_play(load_fixture("pbp_0022400001"))}
    period_start = by_id[1]
    assert period_start["team_id"] is None
    assert period_start["location"] is None
    assert period_start["person_id"] is None
    assert period_start["shot_value"] is None
    assert period_start["clock_seconds"] == 720.0
    made_three = by_id[17]
    assert made_three["game_id"] == "0022400001"
    assert (made_three["location"], made_three["shot_value"]) == ("v", 3)
    assert (made_three["shot_result"], made_three["is_field_goal"]) == ("Made", True)


def test_season_type_of() -> None:
    assert season_type_of("0022400001") is SeasonType.REGULAR
    assert season_type_of("0042400101") is SeasonType.PLAYOFFS
    assert season_type_of("0052400101") is SeasonType.PLAYIN
    assert season_type_of("0012400001") is None
    assert season_type_of("0062400001") is None
