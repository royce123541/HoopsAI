"""Scheduled pipeline jobs (docs/ARCHITECTURE.md §3.5). Monitoring joins in M5."""

import logging
from datetime import UTC

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from hoopsai.ingest import jobs

log = logging.getLogger(__name__)


def build_scheduler() -> BlockingScheduler:
    scheduler = BlockingScheduler(
        timezone=UTC,
        # One instance per job; if the worker was down at fire time, run once on recovery.
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 3600},
    )
    # 10:00 UTC is after the last West Coast games end (~07:00 UTC).
    scheduler.add_job(
        jobs.daily_pipeline, CronTrigger(hour=10, minute=0, timezone=UTC), id="daily_pipeline"
    )
    # Re-score every 2 hours: picks up schedule changes, tip times and newly promoted models.
    scheduler.add_job(
        jobs.predict_pregame,
        CronTrigger(hour="*/2", minute=5, timezone=UTC),
        id="predict_pregame",
    )
    scheduler.add_job(
        jobs.retrain,
        CronTrigger(day_of_week="mon", hour=11, minute=0, timezone=UTC),
        id="retrain",
    )
    scheduler.add_job(
        jobs.ingest_schedule, CronTrigger(hour=12, minute=0, timezone=UTC), id="ingest_schedule"
    )
    return scheduler


def run() -> None:
    scheduler = build_scheduler()
    for job in scheduler.get_jobs():
        log.info("scheduled %s: %s", job.id, job.trigger)
    scheduler.start()
