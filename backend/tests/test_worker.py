from datetime import UTC, datetime

from hoopsai.ingest import jobs
from hoopsai.worker.main import build_scheduler


def test_scheduler_registers_ingestion_jobs_in_utc() -> None:
    scheduler = build_scheduler()
    by_id = {job.id: job for job in scheduler.get_jobs()}

    assert by_id["daily_pipeline"].func is jobs.daily_pipeline
    assert by_id["ingest_schedule"].func is jobs.ingest_schedule
    assert by_id["retrain"].func is jobs.retrain
    assert by_id["predict_pregame"].func is jobs.predict_pregame

    after = datetime(2026, 10, 6, 11, 0, tzinfo=UTC)
    assert by_id["daily_pipeline"].trigger.get_next_fire_time(None, after) == datetime(
        2026, 10, 7, 10, 0, tzinfo=UTC
    )
    # 2026-10-06 is a Tuesday: next Monday 11:00 UTC.
    assert by_id["retrain"].trigger.get_next_fire_time(None, after) == datetime(
        2026, 10, 12, 11, 0, tzinfo=UTC
    )
    assert by_id["ingest_schedule"].trigger.get_next_fire_time(None, after) == datetime(
        2026, 10, 6, 12, 0, tzinfo=UTC
    )
    assert by_id["predict_pregame"].trigger.get_next_fire_time(None, after) == datetime(
        2026, 10, 6, 12, 5, tzinfo=UTC
    )
