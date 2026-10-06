"""Ingestion tasks: fetch -> store raw -> parse -> upsert core, one checkpointed task each.

Task keys: `schedule:<season>`, `gamelog:<season>:<type>`, `pbp:<game_id>`. A failed task is
recorded and skipped over, so one bad game never aborts a multi-hour backfill."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import Connection, Engine, select, text

from hoopsai.db.models import Game
from hoopsai.ingest import store
from hoopsai.sources.base import DataSource, RawResponse, SeasonType, current_season, season_label

log = logging.getLogger(__name__)

SEASON_TYPES = (SeasonType.REGULAR, SeasonType.PLAYIN, SeasonType.PLAYOFFS)


@dataclass
class RunSummary:
    succeeded: int = 0
    skipped: int = 0
    failed: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        failed = f" ({', '.join(self.failed[:5])}{'...' if len(self.failed) > 5 else ''})"
        return (
            f"{self.succeeded} succeeded, {self.skipped} skipped, "
            f"{len(self.failed)} failed{failed if self.failed else ''}"
        )


@dataclass
class Ingestor:
    engine: Engine
    source: DataSource
    summary: RunSummary = field(default_factory=RunSummary)

    # ------------------------------------------------------------ task runner

    def _run(
        self,
        task_key: str,
        fetch: Callable[[], RawResponse],
        load: Callable[[Connection, RawResponse], int],
        *,
        skip_if_done: bool,
    ) -> None:
        if skip_if_done:
            with self.engine.connect() as conn:
                if store.is_done(conn, task_key):
                    self.summary.skipped += 1
                    return
        try:
            resp = fetch()
            with self.engine.begin() as conn:
                store.save_raw(conn, resp)
                rows = load(conn, resp)
                store.mark(conn, task_key, store.SUCCESS, rows=rows)
        except Exception as exc:
            log.exception("task %s failed", task_key)
            with self.engine.begin() as conn:
                store.mark(conn, task_key, store.FAILED, error=f"{type(exc).__name__}: {exc}")
            self.summary.failed.append(task_key)
            return
        self.summary.succeeded += 1
        log.info("task %s: %d rows", task_key, rows)

    # ------------------------------------------------------------ tasks

    def seed_teams(self) -> None:
        with self.engine.begin() as conn:
            store.upsert_teams(conn, self.source.list_teams())

    def schedule(self, season: int, *, skip_if_done: bool = False) -> None:
        def load(conn: Connection, resp: RawResponse) -> int:
            return store.upsert_games(conn, self.source.parse_schedule(resp.payload))

        self._run(
            f"schedule:{season}",
            lambda: self.source.fetch_schedule(season),
            load,
            skip_if_done=skip_if_done,
        )

    def game_logs(
        self, season: int, season_type: SeasonType, *, skip_if_done: bool = False
    ) -> None:
        def load(conn: Connection, resp: RawResponse) -> int:
            # The schedule (loaded first) names the home team of neutral-site games.
            home_team_ids = {
                game_id: home_id
                for game_id, home_id in conn.execute(
                    select(Game.game_id, Game.home_team_id).where(Game.season == season)
                )
            }
            parsed = self.source.parse_team_game_logs(resp.payload, home_team_ids)
            store.upsert_games(conn, parsed.games)
            return store.upsert_team_game_stats(conn, parsed.team_stats)

        self._run(
            f"gamelog:{season}:{season_type}",
            lambda: self.source.fetch_team_game_logs(season, season_type),
            load,
            skip_if_done=skip_if_done,
        )

    def play_by_play(self, game_id: str, *, skip_if_done: bool = True) -> None:
        def load(conn: Connection, resp: RawResponse) -> int:
            return store.upsert_pbp_events(conn, self.source.parse_play_by_play(resp.payload))

        self._run(
            f"pbp:{game_id}",
            lambda: self.source.fetch_play_by_play(game_id),
            load,
            skip_if_done=skip_if_done,
        )

    def final_games(self, first: int, last: int, *, missing_pbp_only: bool) -> list[str]:
        missing = """
            AND NOT EXISTS (
              SELECT 1 FROM raw.ingestion_runs r
              WHERE r.task_key = 'pbp:' || g.game_id AND r.status = 'success')
        """
        with self.engine.connect() as conn:
            return list(
                conn.scalars(
                    text(f"""
                        SELECT g.game_id FROM core.games g
                        WHERE g.status = 'final' AND g.season BETWEEN :first AND :last
                        {missing if missing_pbp_only else ""}
                        ORDER BY g.game_date, g.game_id
                    """),
                    {"first": first, "last": last},
                )
            )

    # ------------------------------------------------------------ jobs

    def backfill(self, first: int, last: int, *, pbp: bool, force: bool, today: date) -> RunSummary:
        """Seasons `first`..`last` inclusive. Completed seasons are checkpointed and skipped on
        re-runs; the in-progress season is always refetched."""
        self.seed_teams()
        ongoing = current_season(today)
        for season in range(first, last + 1):
            skip = not force and season < ongoing
            log.info("season %s", season_label(season))
            self.schedule(season, skip_if_done=skip)
            for season_type in SEASON_TYPES:
                self.game_logs(season, season_type, skip_if_done=skip)
            if pbp:
                self._play_by_play_for(season, season, force=force)
        return self.summary

    def daily(self, today: date) -> RunSummary:
        """Nightly incremental update for the current season."""
        season = current_season(today)
        self.seed_teams()
        self.schedule(season)
        for season_type in SEASON_TYPES:
            self.game_logs(season, season_type)
        self._play_by_play_for(season, season, force=False)
        return self.summary

    def _play_by_play_for(self, first: int, last: int, *, force: bool) -> None:
        game_ids = self.final_games(first, last, missing_pbp_only=not force)
        log.info("play-by-play: %d games to fetch", len(game_ids))
        for i, game_id in enumerate(game_ids, 1):
            self.play_by_play(game_id, skip_if_done=False)
            if i % 100 == 0:
                log.info("play-by-play progress: %d/%d", i, len(game_ids))
