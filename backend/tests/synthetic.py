"""Synthetic league data in the shape of core.games / core.team_game_stats."""

from datetime import date, timedelta
from typing import cast

import numpy as np
import pandas as pd

from hoopsai.sources.base import PbpEventRow

ATL, BOS, CLE, NOP = 1610612737, 1610612738, 1610612739, 1610612740
TEAMS = [ATL, BOS, CLE, NOP]
# Round-robin pairings cycled over game days; every team plays on every game day.
PAIRINGS = [((ATL, BOS), (CLE, NOP)), ((BOS, CLE), (NOP, ATL)), ((ATL, CLE), (BOS, NOP))]


def box_score(rng: np.random.Generator) -> dict[str, int]:
    fga = int(rng.integers(75, 95))
    fgm = int(rng.integers(32, 48))
    fg3a = int(rng.integers(25, 40))
    fg3m = int(rng.integers(8, min(18, fgm)))
    fta = int(rng.integers(15, 30))
    ftm = int(rng.integers(10, fta))
    oreb, dreb = int(rng.integers(6, 15)), int(rng.integers(28, 40))
    return {
        "minutes": 240,
        "pts": 2 * fgm + fg3m + ftm,
        "fgm": fgm,
        "fga": fga,
        "fg3m": fg3m,
        "fg3a": fg3a,
        "ftm": ftm,
        "fta": fta,
        "oreb": oreb,
        "dreb": dreb,
        "reb": oreb + dreb,
        "ast": int(rng.integers(18, 30)),
        "stl": int(rng.integers(4, 12)),
        "blk": int(rng.integers(2, 8)),
        "tov": int(rng.integers(8, 18)),
        "pf": int(rng.integers(15, 25)),
    }


def make_league(
    seasons: tuple[int, ...] = (2021, 2022),
    days_per_season: int = 24,
    seed: int = 7,
    scheduled_days: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Games every other day from Oct 20; the last `scheduled_days` of the final season are
    unplayed (status "scheduled", no box score)."""
    rng = np.random.default_rng(seed)
    games, stats = [], []
    for season in seasons:
        start = date(season, 10, 20)
        for day in range(days_per_season):
            game_date = start + timedelta(days=2 * day)
            unplayed = season == seasons[-1] and day >= days_per_season - scheduled_days
            for n, (home, away) in enumerate(PAIRINGS[day % len(PAIRINGS)]):
                game_id = f"002{season % 100:02d}{day:03d}{n:02d}"
                row = {
                    "game_id": game_id,
                    "season": season,
                    "season_type": "regular",
                    "game_date": game_date,
                    "home_team_id": home,
                    "away_team_id": away,
                    "status": "scheduled" if unplayed else "final",
                    "home_score": None,
                    "away_score": None,
                    "is_neutral": False,
                }
                if not unplayed:
                    hb, ab = box_score(rng), box_score(rng)
                    if hb["pts"] == ab["pts"]:  # no ties in basketball
                        hb["ftm"] += 1
                        hb["fta"] = max(hb["fta"], hb["ftm"])
                        hb["pts"] += 1
                    row["home_score"], row["away_score"] = hb["pts"], ab["pts"]
                    margin = hb["pts"] - ab["pts"]
                    stats.append(
                        {
                            "game_id": game_id,
                            "team_id": home,
                            "is_home": True,
                            **hb,
                            "plus_minus": margin,
                        }
                    )
                    stats.append(
                        {
                            "game_id": game_id,
                            "team_id": away,
                            "is_home": False,
                            **ab,
                            "plus_minus": -margin,
                        }
                    )
                games.append(row)
    return pd.DataFrame(games), pd.DataFrame(stats)


def _event(game_id: str, action_id: int, period: int, clock: float, **kw: object) -> PbpEventRow:
    base: dict[str, object] = {
        "game_id": game_id,
        "action_id": action_id,
        "action_number": action_id,
        "period": period,
        "clock_seconds": max(clock, 0.0),
        "team_id": None,
        "location": None,
        "person_id": None,
        "action_type": None,
        "sub_type": None,
        "description": None,
        "is_field_goal": False,
        "shot_value": None,
        "shot_result": None,
        "score_home": 0,
        "score_away": 0,
    }
    return cast(PbpEventRow, base | kw)


def make_pbp_game(
    rng: np.random.Generator, game_id: str, home_edge: float, possessions_per_period: int = 25
) -> tuple[list[PbpEventRow], bool]:
    """A play-by-play stream in PlayByPlayV3 row shape: alternating possessions that end in a
    made shot (possession switches) or a miss and a defensive rebound. `home_edge` shifts
    shooting percentages. Overtime periods are added until the game is decided."""
    events: list[PbpEventRow] = []
    home = away = 0
    offense = 1 if rng.random() < 0.5 else -1
    period = 0
    while period < 4 or home == away:
        period += 1
        length = 720.0 if period <= 4 else 300.0
        n = possessions_per_period if period <= 4 else possessions_per_period // 2
        events.append(
            _event(
                game_id,
                len(events) + 1,
                period,
                length,
                action_type="period",
                sub_type="start",
                score_home=home,
                score_away=away,
            )
        )
        for i in range(n):
            clock = length * (1 - (i + 1) / n)
            side = "h" if offense == 1 else "v"
            p_make = 0.5 + 0.06 * home_edge * offense
            if rng.random() < p_make:
                pts = 3 if rng.random() < 0.3 else 2
                home, away = (home + pts, away) if offense == 1 else (home, away + pts)
                events.append(
                    _event(
                        game_id,
                        len(events) + 1,
                        period,
                        clock,
                        location=side,
                        action_type="Made Shot",
                        is_field_goal=True,
                        shot_value=pts,
                        score_home=home,
                        score_away=away,
                    )
                )
            else:
                events.append(
                    _event(
                        game_id,
                        len(events) + 1,
                        period,
                        clock,
                        location=side,
                        action_type="Missed Shot",
                        is_field_goal=True,
                        description="MISS",
                        score_home=home,
                        score_away=away,
                    )
                )
                events.append(
                    _event(
                        game_id,
                        len(events) + 1,
                        period,
                        clock,
                        location="v" if offense == 1 else "h",
                        action_type="Rebound",
                        score_home=home,
                        score_away=away,
                    )
                )
            offense = -offense
        if period >= 4 and home == away and period >= 7:  # cap overtimes: decide by a free throw
            home += 1
            events.append(
                _event(
                    game_id,
                    len(events) + 1,
                    period,
                    0.0,
                    location="h",
                    action_type="Free Throw",
                    sub_type="Free Throw 1 of 1",
                    score_home=home,
                    score_away=away,
                )
            )
    events.append(
        _event(
            game_id,
            len(events) + 1,
            period,
            0.0,
            action_type="period",
            sub_type="end",
            score_home=home,
            score_away=away,
        )
    )
    return events, home > away
