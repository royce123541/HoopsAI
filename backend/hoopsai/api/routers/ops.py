"""Operational visibility: model monitoring results, job runs and models in use."""

from fastapi import APIRouter
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from hoopsai.api.deps import Conn
from hoopsai.api.schemas import JobStatus, MonitoringResult, SystemStatus

router = APIRouter(tags=["ops"])


async def latest_monitoring(conn: AsyncConnection) -> list[MonitoringResult]:
    rows = await conn.execute(
        text("""
            SELECT DISTINCT ON (model, "window") * FROM serving.monitoring_runs
            ORDER BY model DESC, "window" DESC, run_at DESC, id DESC
        """)
    )
    return [MonitoringResult.model_validate(dict(r)) for r in rows.mappings()]


@router.get("/model/monitoring", response_model=list[MonitoringResult])
async def get_monitoring(conn: Conn) -> list[MonitoringResult]:
    """Latest accuracy check per model and window (pre-game first; season before 30 days)."""
    return await latest_monitoring(conn)


@router.get("/status", response_model=SystemStatus)
async def get_status(conn: Conn) -> SystemStatus:
    jobs = await conn.execute(
        text("""
            SELECT DISTINCT ON (job) job, status, started_at, finished_at, detail
            FROM serving.job_runs ORDER BY job, started_at DESC, id DESC
        """)
    )
    pregame = await conn.scalar(
        text("SELECT version FROM serving.model_versions ORDER BY first_used_at DESC LIMIT 1")
    )
    ingame = await conn.scalar(
        text("SELECT model_version FROM serving.live_wp_snapshots ORDER BY id DESC LIMIT 1")
    )
    return SystemStatus(
        models={"pregame": pregame, "ingame": ingame},
        jobs=[JobStatus.model_validate(dict(r)) for r in jobs.mappings()],
        monitoring=await latest_monitoring(conn),
    )
