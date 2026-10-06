"""Team Elo ratings (FiveThirtyEight's NBA variant): margin-of-victory multiplier with
autocorrelation correction, home-court offset, and regression to the mean between seasons.

Elo is both a model feature and the baseline the learned model must beat."""

import math
from dataclasses import dataclass

import pandas as pd

INITIAL = 1500.0
K = 20.0
HOME_ADVANTAGE = 100.0  # Elo points; not applied at neutral sites
SEASON_CARRYOVER = 0.75
SEASON_MEAN = 1505.0


def win_probability(elo_diff: float) -> float:
    """P(team A wins) given elo_A - elo_B, including any home-court offset."""
    return 1.0 / (1.0 + float(10.0 ** (-elo_diff / 400.0)))


def mov_multiplier(margin: int, winner_elo_diff: float) -> float:
    """Bigger wins move ratings more, damped when the favourite wins (autocorrelation)."""
    return float((abs(margin) + 3.0) ** 0.8) / (7.5 + 0.006 * winner_elo_diff)


def regress_to_mean(elo: float) -> float:
    return SEASON_CARRYOVER * elo + (1.0 - SEASON_CARRYOVER) * SEASON_MEAN


def home_offset(is_neutral: bool) -> float:
    return 0.0 if is_neutral else HOME_ADVANTAGE


@dataclass(frozen=True)
class EloUpdate:
    home_post: float
    away_post: float


def update(home_elo: float, away_elo: float, home_margin: int, is_neutral: bool) -> EloUpdate:
    """Ratings after one game. Zero-sum: the winner gains exactly what the loser drops."""
    diff = home_elo + home_offset(is_neutral) - away_elo
    expected_home = win_probability(diff)
    actual_home = 1.0 if home_margin > 0 else 0.0
    winner_diff = diff if home_margin > 0 else -diff
    shift = K * mov_multiplier(home_margin, winner_diff) * (actual_home - expected_home)
    return EloUpdate(home_post=home_elo + shift, away_post=away_elo - shift)


def run_elo(games: pd.DataFrame) -> pd.DataFrame:
    """Pre-game (and, for final games, post-game) Elo for every game.

    `games` needs: game_id, season, game_date, home_team_id, away_team_id, status,
    home_score, away_score, is_neutral. A team's pre-game rating uses only its earlier final
    games (a team plays at most once per date), so this is point-in-time correct for
    scheduled games too. Returns one row per game with home/away pre and post ratings."""
    ratings: dict[int, float] = {}
    rated_season: dict[int, int] = {}
    rows: list[tuple[str, float, float, float, float]] = []

    ordered = games.sort_values(["game_date", "game_id"], kind="stable")
    columns = ["game_id", "season", "home_team_id", "away_team_id", "status"]
    columns += ["home_score", "away_score", "is_neutral"]
    for game_id, season, home, away, status, home_score, away_score, neutral in zip(
        *(ordered[c].tolist() for c in columns), strict=True
    ):
        pre = []
        for team in (int(home), int(away)):
            elo = ratings.get(team, INITIAL)
            if team in rated_season and rated_season[team] != season:
                elo = regress_to_mean(elo)
            pre.append(elo)
        home_pre, away_pre = pre

        if status == "final":
            post = update(home_pre, away_pre, int(home_score) - int(away_score), bool(neutral))
            home_post, away_post = post.home_post, post.away_post
            for team, elo in ((int(home), home_post), (int(away), away_post)):
                ratings[team] = elo
                rated_season[team] = int(season)
        else:
            home_post = away_post = math.nan
        rows.append((str(game_id), home_pre, away_pre, home_post, away_post))

    return pd.DataFrame(
        rows, columns=["game_id", "home_elo_pre", "away_elo_pre", "home_elo_post", "away_elo_post"]
    )


def elo_home_win_probability(home_elo: float, away_elo: float, is_neutral: bool) -> float:
    """The Elo baseline prediction."""
    return win_probability(home_elo + home_offset(is_neutral) - away_elo)
