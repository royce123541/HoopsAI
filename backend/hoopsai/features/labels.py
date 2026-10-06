"""Human-readable names for model features, used by the API's "why" factors."""

from typing import Literal

BASE_LABELS = {
    "elo_pre": "Elo rating",
    "games_played": "games played",
    "win_pct": "win %",
    "streak": "current streak",
    "std_ortg": "offensive rating",
    "std_drtg": "defensive rating",
    "std_net": "net rating",
    "std_pace": "pace",
    "std_efg": "effective FG%",
    "std_tov_pct": "turnover rate",
    "std_orb_pct": "offensive rebound rate",
    "std_ftr": "free-throw rate",
    "std_opp_efg": "opponent effective FG%",
    "std_opp_tov_pct": "forced turnover rate",
    "std_drb_pct": "defensive rebound rate",
    "std_opp_ftr": "opponent free-throw rate",
    "l10_ortg": "offensive rating, last 10",
    "l10_drtg": "defensive rating, last 10",
    "l10_net": "net rating, last 10",
    "l10_efg": "effective FG%, last 10",
    "l10_opp_efg": "opponent effective FG%, last 10",
    "l5_net": "net rating, last 5",
    "prev_net": "last season's net rating",
    "rest_days": "days of rest",
    "b2b": "back-to-back",
    "games_last7": "games in the last 7 days",
    "travel_km": "travel distance (km)",
    "is_neutral": "neutral site",
    "is_playoffs": "playoff game",
}

Side = Literal["home", "away"]


def feature_label(feature: str, home: str, away: str) -> str:
    """`home_std_net` -> "BOS net rating"; `diff_prev_net` -> "Last season's net rating
    difference"."""
    for prefix, name in (("home_", home), ("away_", away)):
        if feature.startswith(prefix):
            return f"{name} {BASE_LABELS.get(feature.removeprefix(prefix), feature)}"
    if feature.startswith("diff_"):
        label = f"{BASE_LABELS.get(feature.removeprefix('diff_'), feature)} difference"
    else:
        label = BASE_LABELS.get(feature, feature)
    return label[:1].upper() + label[1:]
