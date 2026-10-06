from datetime import date
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import RowMapping, text
from sqlalchemy.ext.asyncio import AsyncConnection

from hoopsai.api.deps import Conn, load_teams
from hoopsai.api.schemas import (
    ComparisonRow,
    Factor,
    GameDetail,
    GameSummary,
    PredictionSummary,
    Slate,
    TeamRef,
)
from hoopsai.dates import eastern_today
from hoopsai.features.labels import feature_label

router = APIRouter(tags=["games"])

SHOWN_FACTORS = 5

# Latest prediction per game via LATERAL (index: serving.predictions(game_id, created_at)).
GAMES_SQL = """
    SELECT g.game_id, g.game_date, g.tip_time_utc, g.status, g.season, g.season_type,
           coalesce(g.is_neutral, false) AS is_neutral, g.home_team_id, g.away_team_id,
           g.home_score, g.away_score,
           p.home_win_prob, p.model_version, p.created_at, p.made_before_tip, p.factors
    FROM core.games g
    LEFT JOIN LATERAL (
        SELECT * FROM serving.predictions p
        WHERE p.game_id = g.game_id ORDER BY p.created_at DESC, p.id DESC LIMIT 1
    ) p ON true
"""

# (feature suffix, label, higher is better) for the side-by-side comparison table.
COMPARISON = [
    ("elo_pre", "Elo rating", True),
    ("win_pct", "Win %", True),
    ("std_net", "Net rating", True),
    ("std_ortg", "Offensive rating", True),
    ("std_drtg", "Defensive rating", False),
    ("l10_net", "Net rating, last 10", True),
    ("std_pace", "Pace", True),
    ("prev_net", "Last season's net rating", True),
    ("rest_days", "Days of rest", True),
    ("travel_km", "Travel since last game (km)", False),
]


def _summary(row: RowMapping, teams: dict[int, TeamRef]) -> dict[str, Any]:
    prediction = None
    if row["home_win_prob"] is not None:
        prediction = PredictionSummary(
            home_win_prob=row["home_win_prob"],
            model_version=row["model_version"],
            created_at=row["created_at"],
            made_before_tip=row["made_before_tip"],
        )
    return {
        "game_id": row["game_id"],
        "game_date": row["game_date"],
        "tip_time_utc": row["tip_time_utc"],
        "status": row["status"],
        "season": row["season"],
        "season_type": row["season_type"],
        "is_neutral": row["is_neutral"],
        "home": teams[row["home_team_id"]],
        "away": teams[row["away_team_id"]],
        "home_score": row["home_score"],
        "away_score": row["away_score"],
        "prediction": prediction,
    }


async def _adjacent_date(conn: AsyncConnection, day: date, direction: str) -> date | None:
    op, agg = ("<", "max") if direction == "prev" else (">", "min")
    result: date | None = await conn.scalar(
        text(f"SELECT {agg}(game_date) FROM core.games WHERE game_date {op} :d"), {"d": day}
    )
    return result


@router.get("/games", response_model=Slate)
async def list_games(conn: Conn, on: Annotated[date | None, Query(alias="date")] = None) -> Slate:
    """Games on a date (US Eastern; default today) with their latest pre-game prediction."""
    day = on or eastern_today()
    teams = await load_teams(conn)
    rows = await conn.execute(
        text(GAMES_SQL + " WHERE g.game_date = :d ORDER BY g.tip_time_utc NULLS LAST, g.game_id"),
        {"d": day},
    )
    return Slate(
        date=day,
        prev_date=await _adjacent_date(conn, day, "prev"),
        next_date=await _adjacent_date(conn, day, "next"),
        games=[GameSummary(**_summary(r, teams)) for r in rows.mappings()],
    )


def factors_for(raw: list[dict[str, Any]] | None, home: str, away: str) -> list[Factor]:
    """Top factors with a known value (a missing early-season stat is not a reason a fan can
    act on), labelled for display. Positive contribution favours the home team."""
    out = []
    for f in raw or []:
        if f.get("value") is None:
            continue
        out.append(
            Factor(
                feature=f["feature"],
                label=feature_label(f["feature"], home, away),
                value=f["value"],
                contribution=f["contribution"],
                favors="home" if f["contribution"] > 0 else "away",
            )
        )
    return out[:SHOWN_FACTORS]


def comparison_for(features: dict[str, Any] | None) -> list[ComparisonRow]:
    features = features or {}
    return [
        ComparisonRow(
            key=key,
            label=label,
            home=features.get(f"home_{key}"),
            away=features.get(f"away_{key}"),
            higher_is_better=better,
        )
        for key, label, better in COMPARISON
    ]


@router.get("/games/{game_id}", response_model=GameDetail)
async def get_game(game_id: str, conn: Conn) -> GameDetail:
    teams = await load_teams(conn)
    row = (
        (await conn.execute(text(GAMES_SQL + " WHERE g.game_id = :id"), {"id": game_id}))
        .mappings()
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"game {game_id} not found")
    features = await conn.scalar(
        text("SELECT features FROM features.game_features WHERE game_id = :id"), {"id": game_id}
    )
    summary = _summary(row, teams)
    home, away = summary["home"].abbreviation, summary["away"].abbreviation
    return GameDetail(
        **summary,
        factors=factors_for(row["factors"], home, away),
        comparison=comparison_for(features),
    )
