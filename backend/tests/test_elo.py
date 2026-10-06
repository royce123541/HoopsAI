import math

import pandas as pd
import pytest

from hoopsai.features import elo
from tests.synthetic import make_league


def test_win_probability_is_symmetric() -> None:
    assert elo.win_probability(0) == 0.5
    assert elo.win_probability(100) + elo.win_probability(-100) == pytest.approx(1.0)
    assert elo.win_probability(100) == pytest.approx(0.640, abs=1e-3)


@pytest.mark.parametrize("margin", [1, 7, 25, -3, -18])
@pytest.mark.parametrize("neutral", [True, False])
def test_update_is_zero_sum(margin: int, neutral: bool) -> None:
    post = elo.update(1550.0, 1480.0, margin, neutral)
    assert post.home_post - 1550.0 == pytest.approx(-(post.away_post - 1480.0))
    assert (post.home_post > 1550.0) == (margin > 0)


def test_neutral_update_is_mirror_symmetric() -> None:
    a = elo.update(1600.0, 1500.0, 10, is_neutral=True)
    b = elo.update(1500.0, 1600.0, -10, is_neutral=True)
    assert a.home_post == pytest.approx(b.away_post)
    assert a.away_post == pytest.approx(b.home_post)


def test_home_court_reduces_reward_for_home_win() -> None:
    at_home = elo.update(1500.0, 1500.0, 5, is_neutral=False)
    neutral = elo.update(1500.0, 1500.0, 5, is_neutral=True)
    assert 0 < at_home.home_post - 1500.0 < neutral.home_post - 1500.0


def test_bigger_margin_moves_ratings_more() -> None:
    close = elo.update(1500.0, 1500.0, 2, is_neutral=True).home_post
    blowout = elo.update(1500.0, 1500.0, 30, is_neutral=True).home_post
    assert blowout > close > 1500.0


def test_regress_to_mean() -> None:
    assert elo.regress_to_mean(1700.0) == pytest.approx(0.75 * 1700 + 0.25 * 1505)
    assert elo.regress_to_mean(elo.SEASON_MEAN) == pytest.approx(elo.SEASON_MEAN)


def test_run_elo_chains_post_to_next_pre_and_regresses_between_seasons() -> None:
    games, _ = make_league(seasons=(2021, 2022), days_per_season=6, scheduled_days=1)
    ratings = run_and_join(games)

    long = pd.concat(
        [
            ratings[
                ["game_id", "season", "game_date", "home_team_id", "home_elo_pre", "home_elo_post"]
            ].set_axis(["game_id", "season", "game_date", "team", "pre", "post"], axis=1),
            ratings[
                ["game_id", "season", "game_date", "away_team_id", "away_elo_pre", "away_elo_post"]
            ].set_axis(["game_id", "season", "game_date", "team", "pre", "post"], axis=1),
        ]
    ).sort_values(["team", "game_date"])
    for _, team_games in long.groupby("team"):
        rows = team_games.to_dict("records")
        assert rows[0]["pre"] == elo.INITIAL
        for prev, cur in zip(rows, rows[1:], strict=False):
            same_season = prev["season"] == cur["season"]
            expected = prev["post"] if same_season else elo.regress_to_mean(prev["post"])
            assert cur["pre"] == pytest.approx(expected)
        assert math.isnan(rows[-1]["post"])  # the last game is scheduled: no post rating

    # League average is conserved within a season (zero-sum updates).
    first_season = long[long["season"] == 2021]
    final = first_season.sort_values("game_date").groupby("team")["post"].last()
    assert final.mean() == pytest.approx(elo.INITIAL)


def run_and_join(games: pd.DataFrame) -> pd.DataFrame:
    return games.merge(elo.run_elo(games), on="game_id")
