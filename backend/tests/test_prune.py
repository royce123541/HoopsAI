"""Thinning and retention of live win-probability history."""

from datetime import date

from sqlalchemy import Engine, text

from hoopsai.ingest import store
from hoopsai.live.prune import prune_live_history
from hoopsai.sources.nba_stats.client import NbaStatsSource

BOS, NYK = 1610612738, 1610612752


def seed(db: Engine, game_id: str, season: int, status: str, points: int) -> None:
    with db.begin() as conn:
        store.upsert_teams(conn, NbaStatsSource().list_teams())
        store.upsert_games(
            conn,
            [
                {
                    "game_id": game_id,
                    "season": season,
                    "season_type": "regular",
                    "game_date": date(season + 1, 1, 10),
                    "home_team_id": BOS,
                    "away_team_id": NYK,
                    "status": status,
                    "home_score": None,
                    "away_score": None,
                }
            ],
        )
        conn.execute(
            text("""
                INSERT INTO serving.live_wp_snapshots (game_id, source, model_version, action_id,
                    period, clock_seconds, elapsed_seconds, score_home, score_away, possession,
                    home_win_prob)
                SELECT :g, 'live', '1', i, 1 + i / 150, 720, i, i, i, 0, 0.5
                FROM generate_series(0, :n - 1) i
            """),
            {"g": game_id, "n": points},
        )


def action_ids(db: Engine, game_id: str) -> list[int]:
    with db.connect() as conn:
        return list(
            conn.scalars(
                text(
                    "SELECT action_id FROM serving.live_wp_snapshots WHERE game_id = :g "
                    "ORDER BY action_id"
                ),
                {"g": game_id},
            )
        )


def test_finished_games_are_thinned_evenly(db: Engine) -> None:
    seed(db, "final", 2026, "final", 500)
    seed(db, "live", 2026, "live", 500)
    seed(db, "short", 2026, "final", 100)

    prune_live_history(db, keep_points=150, current=2026, seasons=2)

    kept = action_ids(db, "final")
    assert len(kept) == 150
    assert (kept[0], kept[-1]) == (0, 499)  # tip-off and the final point survive
    gaps = {b - a for a, b in zip(kept, kept[1:], strict=False)}
    assert gaps <= {3, 4}  # evenly spaced
    assert len(action_ids(db, "live")) == 500  # a game in progress is left alone
    assert len(action_ids(db, "short")) == 100

    prune_live_history(db, keep_points=150, current=2026, seasons=2)
    assert action_ids(db, "final") == kept  # idempotent


def test_old_seasons_are_dropped(db: Engine) -> None:
    seed(db, "old", 2024, "final", 10)
    seed(db, "last", 2025, "final", 10)
    seed(db, "now", 2026, "final", 10)

    prune_live_history(db, keep_points=150, current=2026, seasons=2)

    assert [len(action_ids(db, g)) for g in ("old", "last", "now")] == [0, 10, 10]
