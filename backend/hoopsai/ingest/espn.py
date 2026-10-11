"""Daily ingestion from ESPN, for hosts stats.nba.com blocks (GitHub Actions, cloud VMs).

History stays as loaded from stats.nba.com. An ESPN game is linked to the existing row for the
same date and teams, keeping its NBA game id; a game nobody has loaded yet gets the id
`espn-<event id>`. Task keys: `espn:schedule:<season>:<YYYYMM>`, `espn:day:<YYYYMMDD>`,
`espn:game:<game_id>`."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from sqlalchemy import Connection, Engine, and_, or_, select, text

from hoopsai.db.models import Game
from hoopsai.ingest import store
from hoopsai.ingest.pipeline import RunSummary, run_task
from hoopsai.sources.base import GameStatus, RawResponse, current_season
from hoopsai.sources.espn.client import EspnClient
from hoopsai.sources.espn.parse import EspnGame, parse_plays, parse_scoreboard, parse_team_box
from hoopsai.sources.nba_stats.client import NbaStatsSource

log = logging.getLogger(__name__)

SEASON_MONTHS = (10, 11, 12, 1, 2, 3, 4, 5, 6)  # opening night (October) to the Finals (June)
RECENT_DAYS = 3  # re-read the last few days: late finals, stat corrections


def new_game_id(espn_id: str) -> str:
    return f"espn-{espn_id}"


def resolve_game_ids(conn: Connection, games: Sequence[EspnGame]) -> dict[str, str]:
    """ESPN event id -> game id: the row already linked to the event, else the unlinked row
    with the same date and teams (an NBA-loaded game), else a new `espn-` id."""
    if not games:
        return {}
    # (Not dict(result): a Result has .keys(), so dict() would treat it as a mapping.)
    linked = {
        espn_id: game_id
        for espn_id, game_id in conn.execute(
            select(Game.espn_id, Game.game_id).where(Game.espn_id.in_([g.espn_id for g in games]))
        )
    }
    unlinked = [g for g in games if g.espn_id not in linked]
    by_matchup: dict[tuple[date, int, int], str] = {}
    if unlinked:
        rows = conn.execute(
            select(Game.game_date, Game.home_team_id, Game.away_team_id, Game.game_id).where(
                Game.espn_id.is_(None),
                or_(
                    *(
                        and_(
                            Game.game_date == g.game_date,
                            Game.home_team_id == g.home_team_id,
                            Game.away_team_id == g.away_team_id,
                        )
                        for g in unlinked
                    )
                ),
            )
        )
        by_matchup = {(d, h, a): gid for d, h, a, gid in rows}
    return {
        g.espn_id: linked.get(g.espn_id)
        or by_matchup.get((g.game_date, g.home_team_id, g.away_team_id))
        or new_game_id(g.espn_id)
        for g in games
    }


def game_rows(games: Sequence[EspnGame], ids: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "game_id": ids[g.espn_id],
            "espn_id": g.espn_id,
            "season": g.season,
            "season_type": str(g.season_type),
            "game_date": g.game_date,
            "tip_time_utc": g.tip_time_utc,
            "home_team_id": g.home_team_id,
            "away_team_id": g.away_team_id,
            "status": str(g.status),
            "home_score": g.home_score,
            "away_score": g.away_score,
            "is_neutral": g.is_neutral,
        }
        for g in games
    ]


def load_scoreboard(conn: Connection, resp: RawResponse) -> int:
    games = parse_scoreboard(resp.payload)
    return store.upsert_games(conn, game_rows(games, resolve_game_ids(conn, games)))


@dataclass
class EspnIngestor:
    engine: Engine
    client: EspnClient = field(default_factory=EspnClient)
    summary: RunSummary = field(default_factory=RunSummary)
    # Off on a small database: play-by-play and raw game payloads (see Settings.store_pbp).
    store_pbp: bool = True

    def seed_teams(self) -> None:
        """Team rows come bundled with nba_api (no request), keyed by NBA team id."""
        with self.engine.begin() as conn:
            store.upsert_teams(conn, NbaStatsSource().list_teams())

    def schedule(self, season: int) -> None:
        """The whole season, one request per month."""
        for month in SEASON_MONTHS:
            year = season if month >= 8 else season + 1
            run_task(
                self.engine,
                self.summary,
                f"espn:schedule:{season}:{year}{month:02d}",
                lambda y=year, m=month: self.client.fetch_scoreboard_month(y, m),  # type: ignore[misc]
                load_scoreboard,
                skip_if_done=False,
            )

    def day(self, day: date) -> None:
        run_task(
            self.engine,
            self.summary,
            f"espn:day:{day:%Y%m%d}",
            lambda: self.client.fetch_scoreboard_day(day),
            load_scoreboard,
            skip_if_done=False,
        )

    def game(self, game_id: str, espn_id: str, *, skip_if_done: bool = True) -> None:
        """Team box scores (and play-by-play, if stored) of a finished game."""

        def load(conn: Connection, resp: RawResponse) -> int:
            rows = store.upsert_team_game_stats(conn, parse_team_box(resp.payload, game_id))
            if self.store_pbp:
                rows += store.upsert_pbp_events(conn, parse_plays(resp.payload, game_id))
            return rows

        run_task(
            self.engine,
            self.summary,
            f"espn:game:{game_id}",
            lambda: self.client.fetch_summary(espn_id),
            load,
            skip_if_done=skip_if_done,
            keep_raw=self.store_pbp,
        )

    def days_to_refresh(self, today: date) -> list[date]:
        """The last few days, plus any earlier day of this season with a game not yet final
        (postponed, or missed while the job wasn't running)."""
        recent = {today - timedelta(days=n) for n in range(RECENT_DAYS + 1)}
        with self.engine.connect() as conn:
            stale = conn.scalars(
                select(Game.game_date)
                .where(
                    Game.season == current_season(today),
                    Game.status != GameStatus.FINAL,
                    Game.game_date < today,
                )
                .distinct()
            )
            return sorted(recent | set(stale))

    def games_missing_box(self, season: int) -> list[tuple[str, str]]:
        """(game_id, espn_id) of final ESPN-linked games without a box score yet."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT g.game_id, g.espn_id FROM core.games g
                    WHERE g.season = :season AND g.status = 'final' AND g.espn_id IS NOT NULL
                      AND NOT EXISTS (
                        SELECT 1 FROM core.team_game_stats s WHERE s.game_id = g.game_id)
                    ORDER BY g.game_date, g.game_id
                """),
                {"season": season},
            )
            return [(gid, eid) for gid, eid in rows]

    def daily(self, today: date) -> RunSummary:
        """Recent results and tip times, then box scores and play-by-play of new finals."""
        self.seed_teams()
        for day in self.days_to_refresh(today):
            self.day(day)
        missing = self.games_missing_box(current_season(today))
        log.info("espn: %d finished games to load", len(missing))
        for game_id, espn_id in missing:
            self.game(game_id, espn_id, skip_if_done=False)
        return self.summary
