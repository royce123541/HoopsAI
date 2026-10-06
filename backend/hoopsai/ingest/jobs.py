"""Entry points shared by the CLI and the worker scheduler."""

import logging
from datetime import UTC, datetime

from hoopsai.db.session import get_sync_engine
from hoopsai.ingest.pipeline import Ingestor, RunSummary
from hoopsai.sources.base import current_season
from hoopsai.sources.nba_stats.client import NbaStatsSource

log = logging.getLogger(__name__)


def make_ingestor() -> Ingestor:
    return Ingestor(engine=get_sync_engine(), source=NbaStatsSource())


def ingest_daily() -> RunSummary:
    """Previous day's finals, box scores and play-by-play for the current season."""
    summary = make_ingestor().daily(datetime.now(UTC).date())
    log.info("ingest_daily: %s", summary)
    return summary


def daily_pipeline() -> RunSummary:
    """Nightly: ingest yesterday's games, then rebuild features (Elo, form, schedule)."""
    from hoopsai.features.store import rebuild_features

    summary = ingest_daily()
    rebuild_features(get_sync_engine())
    return summary


def retrain() -> None:
    """Weekly: backtest, refit and register; promotion is gated in hoopsai.ml.registry."""
    from hoopsai.config import get_settings
    from hoopsai.ml.train import format_report, train_from_db

    result = train_from_db(
        get_sync_engine(),
        today=datetime.now(UTC).date(),
        tracking_uri=get_settings().mlflow_tracking_uri,
    )
    log.info("retrain backtest:\n%s", format_report(result.report))


def ingest_schedule() -> RunSummary:
    """Refresh the current season's schedule (tip times, postponements, playoff matchups)."""
    ingestor = make_ingestor()
    ingestor.seed_teams()
    ingestor.schedule(current_season(datetime.now(UTC).date()))
    log.info("ingest_schedule: %s", ingestor.summary)
    return ingestor.summary
