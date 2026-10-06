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
    """Nightly: ingest yesterday's games, rebuild features (Elo, form, schedule), re-score."""
    from hoopsai.features.store import rebuild_features

    summary = ingest_daily()
    rebuild_features(get_sync_engine())
    predict_pregame()
    return summary


def predict_pregame() -> None:
    """Score the coming week's games; unchanged predictions are not rewritten."""
    from hoopsai.config import get_settings
    from hoopsai.predict import pregame

    pregame.predict_pregame(get_sync_engine(), get_settings().mlflow_tracking_uri)


def retrain() -> None:
    """Weekly: backtest, refit and register both models; promotion is gated in
    hoopsai.ml.registry. In-game training reuses the seasons that have play-by-play."""
    from hoopsai.config import get_settings
    from hoopsai.ml import ingame, registry
    from hoopsai.ml.train import format_report, train_from_db

    engine, tracking_uri = get_sync_engine(), get_settings().mlflow_tracking_uri
    result = train_from_db(engine, today=datetime.now(UTC).date(), tracking_uri=tracking_uri)
    log.info("retrain backtest:\n%s", format_report(result.report))

    seasons = [s for s in ingame.seasons_with_pbp(engine) if s >= ingame.FIRST_TRAINING_SEASON]
    if len(seasons) < 2:
        log.warning("in-game retrain skipped: play-by-play for %s only", seasons)
        return
    model, report = ingame.train_and_evaluate(ingame.build_training_frame(engine, seasons))
    log.info("in-game retrain:\n%s", ingame.format_report(report))
    registry.log_and_register_ingame(model, report, ingame.DEFAULT_PARAMS, tracking_uri)


def ingest_schedule() -> RunSummary:
    """Refresh the current season's schedule (tip times, postponements, playoff matchups)."""
    ingestor = make_ingestor()
    ingestor.seed_teams()
    ingestor.schedule(current_season(datetime.now(UTC).date()))
    log.info("ingest_schedule: %s", ingestor.summary)
    return ingestor.summary
