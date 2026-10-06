"""Point-in-time pre-game features, one row per game (docs/ARCHITECTURE.md §3.2).

Rule: every feature of game G uses only games that finished on an earlier date than G.
Teams play at most once per date, so "earlier date" never drops a relevant game, and no
same-day result can leak in. Built as:

  1. per team, a post-game *state* after each final game (running season totals, last-10 and
     last-5 windows, record, streak);
  2. for every game (final or scheduled), the team's latest state strictly before the game
     date (`merge_asof`, allow_exact_matches=False), reset at season boundaries;
  3. schedule features (rest, back-to-backs, travel) from game dates, which are known in
     advance;
  4. Elo from `hoopsai.features.elo`, home/away/diff columns per game.
"""

from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from hoopsai.features.arenas import ORLANDO, haversine_km, home_site
from hoopsai.features.elo import run_elo

FEATURE_VERSION = "v1"

# NBA restart bubble at Walt Disney World: every game from 2020-07-30 was at a neutral site.
BUBBLE_START = pd.Timestamp(date(2020, 7, 30))
BUBBLE_SEASON = 2019

MAX_REST_DAYS = 7  # cap; also used for a team's first game of a season

# Per-team features (emitted as home_<f> and away_<f>).
TEAM_FEATURES = [
    "elo_pre",
    "games_played",
    "win_pct",
    "streak",
    "std_ortg",
    "std_drtg",
    "std_net",
    "std_pace",
    "std_efg",
    "std_tov_pct",
    "std_orb_pct",
    "std_ftr",
    "std_opp_efg",
    "std_opp_tov_pct",
    "std_drb_pct",
    "std_opp_ftr",
    "l10_ortg",
    "l10_drtg",
    "l10_net",
    "l10_efg",
    "l10_opp_efg",
    "l5_net",
    "prev_net",
    "rest_days",
    "b2b",
    "games_last7",
    "travel_km",
]
# Home-minus-away differences (emitted as diff_<f>).
DIFF_FEATURES = [
    "elo_pre",
    "win_pct",
    "std_net",
    "l10_net",
    "l5_net",
    "prev_net",
    "rest_days",
    "std_efg",
    "std_opp_efg",
    "travel_km",
]
GAME_FEATURES = ["is_neutral", "is_playoffs"]

MODEL_FEATURES = (
    [f"home_{f}" for f in TEAM_FEATURES]
    + [f"away_{f}" for f in TEAM_FEATURES]
    + [f"diff_{f}" for f in DIFF_FEATURES]
    + GAME_FEATURES
)

# Box-score sums carried in the post-game state.
_COMPONENTS = [
    "pts",
    "opp_pts",
    "poss",
    "minutes",
    "fgm",
    "fg3m",
    "fga",
    "fta",
    "ftm",
    "tov",
    "oreb",
    "dreb",
    "opp_fgm",
    "opp_fg3m",
    "opp_fga",
    "opp_fta",
    "opp_ftm",
    "opp_tov",
    "opp_oreb",
    "opp_dreb",
]
_WINDOWS = {"std": None, "l10": 10, "l5": 5}


def load_core(engine: Engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_sql(
        "SELECT game_id, season, season_type, game_date, home_team_id, away_team_id, status, "
        "home_score, away_score, is_neutral FROM core.games",
        engine,
    )
    stats = pd.read_sql("SELECT * FROM core.team_game_stats", engine)
    return games, stats


def build_features(games: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    games = _prepare_games(games)
    team_games = _team_games(games)
    state = _post_game_state(team_games, stats)
    team_feats = _attach_state(team_games, state)
    team_feats = _schedule_features(team_feats)
    return _assemble(games, team_feats)


# ---------------------------------------------------------------- steps


def _prepare_games(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    g["game_date"] = pd.to_datetime(g["game_date"])
    bubble = (g["season"] == BUBBLE_SEASON) & (g["game_date"] >= BUBBLE_START)
    g["is_neutral"] = g["is_neutral"].fillna(False).astype(bool) | bubble
    g["is_playoffs"] = g["season_type"] == "playoffs"
    sites = [
        ORLANDO if in_bubble else home_site(int(team), int(season))
        for team, season, in_bubble in zip(g["home_team_id"], g["season"], bubble, strict=True)
    ]
    g["site_lat"] = [s[0] for s in sites]
    g["site_lon"] = [s[1] for s in sites]
    return g.sort_values(["game_date", "game_id"], kind="stable").reset_index(drop=True)


def _team_games(games: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game: one from each team's perspective."""
    base = ["game_id", "season", "game_date", "status", "site_lat", "site_lon"]
    home = games[[*base, "home_team_id", "away_team_id", "home_score", "away_score"]].set_axis(
        [*base, "team_id", "opp_id", "score", "opp_score"], axis=1
    )
    away = games[[*base, "away_team_id", "home_team_id", "away_score", "home_score"]].set_axis(
        [*base, "team_id", "opp_id", "score", "opp_score"], axis=1
    )
    home["is_home"], away["is_home"] = True, False
    tg = pd.concat([home, away], ignore_index=True)
    final = tg["status"] == "final"
    tg["win"] = np.where(final, (tg["score"] > tg["opp_score"]).astype(float), np.nan)
    return tg.sort_values(["team_id", "game_date"], kind="stable").reset_index(drop=True)


def _box_components(stats: pd.DataFrame) -> pd.DataFrame:
    """Per team-game box sums, joined with the opponent's box score."""
    cols = ["pts", "minutes", "fgm", "fg3m", "fga", "fta", "ftm", "tov", "oreb", "dreb"]
    s = stats[["game_id", "team_id", *cols]]
    opp = s.rename(columns={c: f"opp_{c}" for c in cols} | {"team_id": "opp_team_id"})
    box = s.merge(opp, on="game_id")
    box = box[box["team_id"] != box["opp_team_id"]].drop(columns=["opp_team_id", "opp_minutes"])
    # Possessions: average of both teams' estimates (Oliver's basic formula).
    team_poss = box["fga"] + 0.44 * box["fta"] - box["oreb"] + box["tov"]
    opp_poss = box["opp_fga"] + 0.44 * box["opp_fta"] - box["opp_oreb"] + box["opp_tov"]
    box["poss"] = (team_poss + opp_poss) / 2
    return box


def _ratios(sums: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Rates from summed box components (ratio of sums, i.e. possession-weighted)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            "ortg": 100 * sums["pts"] / sums["poss"],
            "drtg": 100 * sums["opp_pts"] / sums["poss"],
            "pace": 240 * sums["poss"] / sums["minutes"],
            "efg": (sums["fgm"] + 0.5 * sums["fg3m"]) / sums["fga"],
            "tov_pct": sums["tov"] / (sums["fga"] + 0.44 * sums["fta"] + sums["tov"]),
            "orb_pct": sums["oreb"] / (sums["oreb"] + sums["opp_dreb"]),
            "ftr": sums["ftm"] / sums["fga"],
            "opp_efg": (sums["opp_fgm"] + 0.5 * sums["opp_fg3m"]) / sums["opp_fga"],
            "opp_tov_pct": sums["opp_tov"]
            / (sums["opp_fga"] + 0.44 * sums["opp_fta"] + sums["opp_tov"]),
            "drb_pct": sums["dreb"] / (sums["dreb"] + sums["opp_oreb"]),
            "opp_ftr": sums["opp_ftm"] / sums["opp_fga"],
        }
    frame = pd.DataFrame(out)
    frame["net"] = frame["ortg"] - frame["drtg"]
    return frame.add_prefix(f"{prefix}_")


def _post_game_state(team_games: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Each team's season-to-date and rolling form *after* each of its final games."""
    final = team_games[team_games["status"] == "final"]
    played = final.merge(_box_components(stats), on=["game_id", "team_id"], how="inner")
    played = played.sort_values(["team_id", "game_date"], kind="stable").reset_index(drop=True)
    by_team_season = played.groupby(["team_id", "season"], sort=False)

    parts = [played[["team_id", "season", "game_date"]]]
    for prefix, window in _WINDOWS.items():
        if window is None:
            sums = by_team_season[_COMPONENTS].cumsum()
        else:
            sums = by_team_season[_COMPONENTS].rolling(window, min_periods=1).sum()
            sums = sums.reset_index(level=[0, 1], drop=True).sort_index()
        parts.append(_ratios(sums, prefix))
    state = pd.concat(parts, axis=1)
    state["games_played"] = by_team_season.cumcount() + 1
    state["win_pct"] = by_team_season["win"].cumsum() / state["games_played"]
    state["streak"] = _streaks(played)
    return state.rename(columns={"game_date": "state_date", "season": "state_season"})


def _streaks(played: pd.DataFrame) -> pd.Series:
    """Signed current streak after each game (+3 = three straight wins), per team-season."""
    out = np.zeros(len(played))
    prev_key, streak = None, 0
    for i, (team, season, win) in enumerate(
        zip(played["team_id"], played["season"], played["win"], strict=True)
    ):
        if (team, season) != prev_key:
            prev_key, streak = (team, season), 0
        step = 1 if win == 1 else -1
        streak = streak + step if streak * step > 0 else step
        out[i] = streak
    return pd.Series(out, index=played.index)


def _attach_state(team_games: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """Latest post-game state strictly before each game's date; season stats reset."""
    left = team_games.sort_values("game_date", kind="stable")
    right = state.sort_values("state_date", kind="stable")
    merged = pd.merge_asof(
        left,
        right,
        left_on="game_date",
        right_on="state_date",
        by="team_id",
        allow_exact_matches=False,
        direction="backward",
    )
    stateful = [c for c in state.columns if c not in ("team_id", "state_date", "state_season")]
    new_season = merged["state_season"] != merged["season"]  # includes "no state yet"
    merged.loc[new_season, stateful] = np.nan
    merged["games_played"] = merged["games_played"].fillna(0)

    # Previous season's final net rating: the strongest early-season signal besides Elo.
    season_end = state.sort_values("state_date").groupby(["team_id", "state_season"]).last()
    prev = season_end["std_net"].rename("prev_net").reset_index()
    prev["season"] = prev["state_season"] + 1
    merged = merged.merge(
        prev[["team_id", "season", "prev_net"]], on=["team_id", "season"], how="left"
    )
    return merged


def _schedule_features(tf: pd.DataFrame) -> pd.DataFrame:
    """Rest, back-to-backs, recent load and travel: functions of game dates and sites only."""
    tf = tf.sort_values(["team_id", "game_date", "game_id"], kind="stable").reset_index(drop=True)
    by_team = tf.groupby("team_id", sort=False)
    prev_date = by_team["game_date"].shift(1)
    prev_season = by_team["season"].shift(1)
    same_season = prev_season == tf["season"]

    rest = (tf["game_date"] - prev_date).dt.days
    tf["rest_days"] = rest.where(same_season, MAX_REST_DAYS).clip(upper=MAX_REST_DAYS)
    tf["b2b"] = (tf["rest_days"] == 1).astype(float)

    games_last7 = np.zeros(len(tf))
    for _, idx in by_team.indices.items():
        dates = tf["game_date"].to_numpy()[idx]
        earlier = np.searchsorted(dates, dates - np.timedelta64(7, "D"), side="left")
        games_last7[idx] = np.arange(len(idx)) - earlier
    tf["games_last7"] = games_last7

    prev_lat, prev_lon = by_team["site_lat"].shift(1), by_team["site_lon"].shift(1)
    travel = [
        haversine_km((a, b), (c, d)) if same else 0.0
        for a, b, c, d, same in zip(
            prev_lat, prev_lon, tf["site_lat"], tf["site_lon"], same_season, strict=True
        )
    ]
    tf["travel_km"] = travel
    return tf


def _assemble(games: pd.DataFrame, tf: pd.DataFrame) -> pd.DataFrame:
    elo = run_elo(games)
    per_team = [f for f in TEAM_FEATURES if f != "elo_pre"]
    home = tf[tf["is_home"]].set_index("game_id")[per_team].add_prefix("home_")
    away = tf[~tf["is_home"]].set_index("game_id")[per_team].add_prefix("away_")

    out = games[
        [
            "game_id",
            "season",
            "season_type",
            "game_date",
            "status",
            "home_team_id",
            "away_team_id",
            "home_score",
            "away_score",
            *GAME_FEATURES,
        ]
    ].merge(elo, on="game_id")
    out = out.join(home, on="game_id").join(away, on="game_id")
    for f in DIFF_FEATURES:
        out[f"diff_{f}"] = out[f"home_{f}"] - out[f"away_{f}"]
    for f in GAME_FEATURES:
        out[f] = out[f].astype(float)
    final = out["status"] == "final"
    out["home_win"] = np.where(final, (out["home_score"] > out["away_score"]).astype(float), np.nan)
    return out.sort_values(["game_date", "game_id"], kind="stable").reset_index(drop=True)
