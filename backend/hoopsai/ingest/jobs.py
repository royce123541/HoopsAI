"""Entry points shared by the CLI and the worker scheduler. Scheduled jobs are `@recorded`,
so each run's start, end and outcome lands in serving.job_runs (see GET /api/status)."""

import functools
import logging
from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy import insert, update

from hoopsai.db.models import JobRun
from hoopsai.db.session import get_sync_engine
from hoopsai.ingest.pipeline import Ingestor, RunSummary
from hoopsai.sources.base import current_season
from hoopsai.sources.nba_stats.client import NbaStatsSource

log = logging.getLogger(__name__)

DETAIL_CHARS = 2000


def recorded[**P, R](job: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Record a job run (running -> success | failed) around the wrapped function."""

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            engine = get_sync_engine()
            with engine.begin() as conn:
                run_id = conn.scalar(
                    insert(JobRun).values(job=job, status="running").returning(JobRun.id)
                )
            status, detail = "failed", None
            try:
                result = fn(*args, **kwargs)
                status, detail = "success", None if result is None else str(result)
                return result
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                with engine.begin() as conn:
                    conn.execute(
                        update(JobRun)
                        .where(JobRun.id == run_id)
                        .values(
                            status=status,
                            detail=detail[:DETAIL_CHARS] if detail else None,
                            finished_at=datetime.now(UTC),
                        )
                    )

        return wrapper

    return decorate


def make_ingestor() -> Ingestor:
    return Ingestor(engine=get_sync_engine(), source=NbaStatsSource())


def ingest_daily() -> RunSummary:
    """Previous day's finals, box scores and play-by-play for the current season."""
    summary = make_ingestor().daily(datetime.now(UTC).date())
    log.info("ingest_daily: %s", summary)
    return summary


@recorded("daily_pipeline")
def daily_pipeline() -> RunSummary:
    """Nightly: ingest yesterday's games, rebuild features (Elo, form, schedule), re-score,
    then check the models against the newly finished games."""
    from hoopsai.features.store import rebuild_features

    summary = ingest_daily()
    rebuild_features(get_sync_engine())
    _predict_pregame()
    _monitor()
    return summary


def _predict_pregame() -> None:
    from hoopsai.config import get_settings
    from hoopsai.predict import pregame

    pregame.predict_pregame(get_sync_engine(), get_settings().mlflow_tracking_uri)


@recorded("predict_pregame")
def predict_pregame() -> None:
    """Score the coming week's games; unchanged predictions are not rewritten."""
    _predict_pregame()


def _monitor() -> str:
    from hoopsai.config import get_settings
    from hoopsai.ml import registry
    from hoopsai.monitor import run_monitor

    uri = get_settings().mlflow_tracking_uri
    results = run_monitor(
        get_sync_engine(),
        datetime.now(UTC).date(),
        ingame_expected=lambda v: registry.heldout_logloss(uri, registry.INGAME_MODEL_NAME, v),
    )
    return "; ".join(f"{r.model}/{r.window}: {r.status}" for r in results)


@recorded("monitor")
def monitor() -> str:
    """Compare both models' real-world accuracy with their held-out evaluation."""
    return _monitor()


@recorded("retrain")
def retrain() -> None:
    """Weekly: backtest, refit and register both models; promotion is gated in
    hoopsai.ml.registry. In-game training reuses the seasons that have play-by-play and
    needs ~750 MB of memory; set HOOPSAI_INGAME_RETRAIN=false to skip it."""
    from hoopsai.config import get_settings
    from hoopsai.ml import ingame, registry
    from hoopsai.ml.train import format_report, train_from_db

    settings = get_settings()
    engine, tracking_uri = get_sync_engine(), settings.mlflow_tracking_uri
    result = train_from_db(engine, today=datetime.now(UTC).date(), tracking_uri=tracking_uri)
    log.info("retrain backtest:\n%s", format_report(result.report))

    if not settings.ingame_retrain:
        log.info("in-game retrain disabled (HOOPSAI_INGAME_RETRAIN=false)")
        return
    seasons = [s for s in ingame.seasons_with_pbp(engine) if s >= ingame.FIRST_TRAINING_SEASON]
    if len(seasons) < 2:
        log.warning("in-game retrain skipped: play-by-play for %s only", seasons)
        return
    model, report = ingame.train_and_evaluate(ingame.build_training_frame(engine, seasons))
    log.info("in-game retrain:\n%s", ingame.format_report(report))
    registry.log_and_register_ingame(model, report, ingame.DEFAULT_PARAMS, tracking_uri)


@recorded("ingest_schedule")
def ingest_schedule() -> RunSummary:
    """Refresh the current season's schedule (tip times, postponements, playoff matchups)."""
    ingestor = make_ingestor()
    ingestor.seed_teams()
    ingestor.schedule(current_season(datetime.now(UTC).date()))
    log.info("ingest_schedule: %s", ingestor.summary)
    return ingestor.summary
