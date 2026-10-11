"""Keep live win-probability history small enough for a free 0.5 GB database.

A finished game has ~500 snapshots (~300 MB a season). Thinning keeps an evenly spaced
subset (always the first and last point) that still draws the same chart, and live charts of
games older than the retention window are dropped."""

import logging

from sqlalchemy import Engine, text

log = logging.getLogger(__name__)

THIN_SQL = """
    WITH ranked AS (
        SELECT s.id,
               row_number() OVER w - 1 AS i,
               count(*) OVER (PARTITION BY s.game_id, s.source) AS n
        FROM serving.live_wp_snapshots s
        JOIN core.games g USING (game_id)
        WHERE g.status = 'final'
        WINDOW w AS (PARTITION BY s.game_id, s.source ORDER BY s.action_id, s.id)
    )
    DELETE FROM serving.live_wp_snapshots s
    USING ranked r
    WHERE s.id = r.id AND r.n > :keep
      AND r.i <> r.n - 1
      -- keep point i when it is the first at or past one of `keep` evenly spaced marks
      AND floor(r.i * (:keep - 1)::float / (r.n - 1))
          = floor((r.i - 1) * (:keep - 1)::float / (r.n - 1))
"""

DROP_OLD_SQL = """
    DELETE FROM serving.live_wp_snapshots s
    USING core.games g
    WHERE s.game_id = g.game_id AND g.season < :first_kept
"""


def prune_live_history(engine: Engine, *, keep_points: int, current: int, seasons: int) -> int:
    """Thin finished games to about `keep_points` points; keep only the last `seasons`
    seasons (up to `current`). Returns the number of snapshots deleted."""
    with engine.begin() as conn:
        thinned = conn.execute(text(THIN_SQL), {"keep": keep_points}).rowcount
        dropped = conn.execute(text(DROP_OLD_SQL), {"first_kept": current - seasons + 1}).rowcount
    log.info("live history: %d points thinned, %d from old seasons dropped", thinned, dropped)
    return thinned + dropped
