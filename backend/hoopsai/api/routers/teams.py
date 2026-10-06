from fastapi import APIRouter, HTTPException
from sqlalchemy import RowMapping, text

from hoopsai.api.deps import Conn, load_teams
from hoopsai.api.schemas import EloPoint, TeamDetail, TeamGame, TeamListItem, TeamRef

router = APIRouter(tags=["teams"])

ELO_HISTORY_SEASONS = 3
RECENT_GAMES = 10
UPCOMING_GAMES = 5

# A team's current rating: the pre-game Elo of its next game (which already includes the
# between-seasons regression), else the post-game Elo after its latest game.
CURRENT_ELO_SQL = """
    WITH next_game AS (
        SELECT DISTINCT ON (team_id) team_id, elo_pre AS elo
        FROM features.elo_ratings WHERE elo_post IS NULL
        ORDER BY team_id, game_date
    ), last_game AS (
        SELECT DISTINCT ON (team_id) team_id, elo_post AS elo
        FROM features.elo_ratings WHERE elo_post IS NOT NULL
        ORDER BY team_id, game_date DESC
    )
    SELECT team_id, coalesce(n.elo, l.elo) AS elo
    FROM last_game l FULL JOIN next_game n USING (team_id)
"""

TEAM_GAMES_SQL = """
    SELECT g.game_id, g.game_date, g.status, g.home_team_id, g.away_team_id,
           g.home_score, g.away_score, p.home_win_prob
    FROM core.games g
    LEFT JOIN LATERAL (
        SELECT home_win_prob FROM serving.predictions p
        WHERE p.game_id = g.game_id ORDER BY p.created_at DESC, p.id DESC LIMIT 1
    ) p ON true
    WHERE (g.home_team_id = :t OR g.away_team_id = :t) AND {where}
    ORDER BY g.game_date {order}, g.game_id {order}
    LIMIT :n
"""


@router.get("/teams", response_model=list[TeamListItem])
async def list_teams(conn: Conn) -> list[TeamListItem]:
    teams = await load_teams(conn)
    elo = {r.team_id: r.elo for r in await conn.execute(text(CURRENT_ELO_SQL))}
    items = [TeamListItem(**t.model_dump(), elo=elo.get(t.team_id)) for t in teams.values()]
    return sorted(items, key=lambda t: -(t.elo or 0))


def _team_game(row: RowMapping, team_id: int, teams: dict[int, TeamRef]) -> TeamGame:
    is_home = row["home_team_id"] == team_id
    opponent = row["away_team_id"] if is_home else row["home_team_id"]
    prob = row["home_win_prob"]
    return TeamGame(
        game_id=row["game_id"],
        game_date=row["game_date"],
        status=row["status"],
        is_home=is_home,
        opponent=teams[opponent],
        team_score=row["home_score"] if is_home else row["away_score"],
        opponent_score=row["away_score"] if is_home else row["home_score"],
        win_prob=None if prob is None else (prob if is_home else 1 - prob),
    )


@router.get("/teams/{team_id}", response_model=TeamDetail)
async def get_team(team_id: int, conn: Conn) -> TeamDetail:
    teams = await load_teams(conn)
    if team_id not in teams:
        raise HTTPException(status_code=404, detail=f"team {team_id} not found")

    history = await conn.execute(
        text("""
            SELECT game_date, season, elo_post AS elo FROM features.elo_ratings
            WHERE team_id = :t AND elo_post IS NOT NULL
              AND season > (SELECT max(season) FROM features.elo_ratings
                            WHERE team_id = :t AND elo_post IS NOT NULL) - :n
            ORDER BY game_date
        """),
        {"t": team_id, "n": ELO_HISTORY_SEASONS},
    )
    elo_history = [EloPoint.model_validate(r._mapping) for r in history]

    async def games(where: str, order: str, n: int) -> list[TeamGame]:
        sql = TEAM_GAMES_SQL.format(where=where, order=order)
        rows = await conn.execute(text(sql), {"t": team_id, "n": n})
        return [_team_game(r, team_id, teams) for r in rows.mappings()]

    recent = await games("g.status = 'final'", "DESC", RECENT_GAMES)
    upcoming = await games("g.status <> 'final'", "ASC", UPCOMING_GAMES)
    current = {r.team_id: r.elo for r in await conn.execute(text(CURRENT_ELO_SQL))}
    return TeamDetail(
        team=teams[team_id],
        elo=current.get(team_id),
        elo_history=elo_history,
        recent=recent,
        upcoming=upcoming,
    )
