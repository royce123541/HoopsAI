from datetime import date

import numpy as np
import pandas as pd
import pytest

from hoopsai.features import elo
from hoopsai.features.arenas import ARENAS, haversine_km
from hoopsai.features.build import MAX_REST_DAYS, MODEL_FEATURES, build_features
from tests.synthetic import ATL, BOS, CLE, box_score, make_league


@pytest.fixture(scope="module")
def league() -> tuple[pd.DataFrame, pd.DataFrame]:
    return make_league()


@pytest.fixture(scope="module")
def features(league: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    return build_features(*league)


def team_view(features: pd.DataFrame, team: int) -> pd.DataFrame:
    """One row per game of `team`, with that team's features (unprefixed)."""
    rows = []
    for side in ("home", "away"):
        part = features[features[f"{side}_team_id"] == team]
        cols = {c: c.removeprefix(f"{side}_") for c in part.columns if c.startswith(f"{side}_")}
        rows.append(part.rename(columns=cols)[["game_id", "season", "game_date", *cols.values()]])
    return pd.concat(rows).sort_values("game_date").reset_index(drop=True)


# ---------------------------------------------------------------- leakage


def test_features_never_use_results_from_the_same_day_or_later(
    league: tuple[pd.DataFrame, pd.DataFrame], features: pd.DataFrame
) -> None:
    """The core point-in-time guarantee. Rewrite every result from a cutoff date onward (and
    add a future game): features of games up to and including the cutoff must not change."""
    games, stats = (frame.copy() for frame in league)
    cutoff = date(2022, 11, 7)
    later = set(games.loc[games["game_date"] >= cutoff, "game_id"])
    rng = np.random.default_rng(999)
    for game_id in later & set(stats["game_id"]):
        for idx in stats.index[stats["game_id"] == game_id]:
            for col, value in box_score(rng).items():
                stats.at[idx, col] = value
    # Flip the winner of every later game.
    mask = games["game_id"].isin(later) & (games["status"] == "final")
    games.loc[mask, ["home_score", "away_score"]] = games.loc[
        mask, ["away_score", "home_score"]
    ].to_numpy()
    extra = games.iloc[[-1]].assign(game_id="0022299999", game_date=date(2023, 3, 1))
    games = pd.concat([games, extra], ignore_index=True)

    rebuilt = build_features(games, stats)

    upto = features["game_date"] <= pd.Timestamp(cutoff)
    assert upto.sum() > 0 and (~upto).sum() > 0
    before = features[upto].set_index("game_id")[MODEL_FEATURES]
    after = rebuilt.set_index("game_id").loc[before.index, MODEL_FEATURES]
    pd.testing.assert_frame_equal(before, after)
    # Sanity: the perturbation is visible after the cutoff, so the test can fail.
    later_ids = features.loc[features["game_date"] > pd.Timestamp(cutoff), "game_id"]
    changed = rebuilt.set_index("game_id").loc[later_ids, "home_std_net"]
    assert not np.allclose(changed, features.set_index("game_id").loc[later_ids, "home_std_net"])


# ---------------------------------------------------------------- values


def test_season_to_date_ratings_match_hand_computation(
    league: tuple[pd.DataFrame, pd.DataFrame], features: pd.DataFrame
) -> None:
    games, stats = league
    atl = team_view(features, ATL)
    third = atl[atl["season"] == 2021].iloc[2]
    first_two = atl[atl["season"] == 2021].iloc[:2]["game_id"]

    own = stats[(stats["team_id"] == ATL) & stats["game_id"].isin(first_two)]
    opp = stats[(stats["team_id"] != ATL) & stats["game_id"].isin(first_two)]
    poss = (
        (own["fga"] + 0.44 * own["fta"] - own["oreb"] + own["tov"]).sum()
        + (opp["fga"] + 0.44 * opp["fta"] - opp["oreb"] + opp["tov"]).sum()
    ) / 2
    assert third["std_ortg"] == pytest.approx(100 * own["pts"].sum() / poss)
    assert third["std_drtg"] == pytest.approx(100 * opp["pts"].sum() / poss)
    assert third["std_efg"] == pytest.approx(
        (own["fgm"].sum() + 0.5 * own["fg3m"].sum()) / own["fga"].sum()
    )
    assert third["std_orb_pct"] == pytest.approx(
        own["oreb"].sum() / (own["oreb"].sum() + opp["dreb"].sum())
    )
    assert third["games_played"] == 2

    results = []
    for game_id in first_two:
        g = games[games["game_id"] == game_id].iloc[0]
        home = g["home_team_id"] == ATL
        results.append((g["home_score"] > g["away_score"]) == home)
    assert third["win_pct"] == pytest.approx(np.mean(results))
    expected_streak = (1 if results[1] else -1) * (2 if results[0] == results[1] else 1)
    assert third["streak"] == expected_streak


def test_new_season_resets_form_but_keeps_elo_and_previous_net(features: pd.DataFrame) -> None:
    atl = team_view(features, ATL)
    last_2021 = atl[atl["season"] == 2021].iloc[-1]
    opener = atl[atl["season"] == 2022].iloc[0]

    assert opener["games_played"] == 0
    assert (
        np.isnan(opener["std_net"]) and np.isnan(opener["l10_net"]) and np.isnan(opener["win_pct"])
    )
    assert opener["elo_pre"] == pytest.approx(elo.regress_to_mean(last_2021["elo_post"]))
    # prev_net = full 2021 season net rating, i.e. the state after ATL's final 2021 game.
    assert atl[atl["season"] == 2022]["prev_net"].nunique() == 1
    assert not np.isnan(opener["prev_net"])


def test_schedule_features(features: pd.DataFrame) -> None:
    bos = team_view(features, BOS)
    season = bos[bos["season"] == 2021]
    assert season.iloc[0]["rest_days"] == MAX_REST_DAYS  # season opener
    assert (season.iloc[1:]["rest_days"] == 2).all()  # synthetic league plays every other day
    assert (season["b2b"] == 0).all()
    # Games 2, 4 and 6 days earlier all fall inside the previous 7 days.
    assert season["games_last7"].tolist()[:4] == [0, 1, 2, 3]
    assert (season.iloc[3:]["games_last7"] == 3).all()
    # Travel from the previous game's site (the home team's arena) to this one.
    row = season.iloc[1]
    prev_site_team = features.loc[
        features["game_id"] == season.iloc[0]["game_id"], "home_team_id"
    ].item()
    this_site_team = features.loc[features["game_id"] == row["game_id"], "home_team_id"].item()
    assert row["travel_km"] == pytest.approx(
        haversine_km(ARENAS[prev_site_team], ARENAS[this_site_team])
    )


def test_back_to_back_detected() -> None:
    games, stats = make_league(seasons=(2021,), days_per_season=4, scheduled_days=0)
    # Move day 1's games to the day right after day 0.
    day1 = games["game_date"] == date(2021, 10, 22)
    games.loc[day1, "game_date"] = date(2021, 10, 21)
    feats = build_features(games, stats)
    second = team_view(feats, CLE).iloc[1]
    assert (second["rest_days"], second["b2b"]) == (1, 1)


def test_scheduled_games_get_features_but_no_label(features: pd.DataFrame) -> None:
    upcoming = features[features["status"] == "scheduled"]
    assert len(upcoming) == 4
    assert upcoming["home_win"].isna().all()
    assert upcoming[["home_elo_pre", "home_std_net", "diff_elo_pre"]].notna().all().all()
    final = features[features["status"] == "final"]
    assert set(final["home_win"].unique()) <= {0.0, 1.0}


def test_bubble_games_are_neutral() -> None:
    games, stats = make_league(seasons=(2019,), days_per_season=3, scheduled_days=0)
    games.loc[games.index[-2:], "game_date"] = date(2020, 8, 1)
    feats = build_features(games, stats)
    assert feats["is_neutral"].tolist()[-2:] == [1.0, 1.0]
    assert feats["is_neutral"].tolist()[:-2] == [0.0] * (len(feats) - 2)
