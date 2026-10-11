"""`hoopsai` command-line entrypoint. Pipeline commands are added per milestone (see plan §7)."""

import logging
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated

import typer
import uvicorn

from hoopsai.sources.base import current_season, parse_season

app = typer.Typer(help="HoopsAI data pipeline and service commands.", no_args_is_help=True)
ingest_app = typer.Typer(help="Incremental ingestion jobs (also run by the worker).")
app.add_typer(ingest_app, name="ingest")

# psycopg's async mode cannot run on Windows' default ProactorEventLoop.
EVENT_LOOP = "asyncio:SelectorEventLoop" if sys.platform == "win32" else "auto"


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    # MLflow prints emoji; Windows consoles default to cp1252 and would crash on them.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    from hoopsai.config import get_settings
    from hoopsai.logs import configure_logging

    configure_logging(logging.DEBUG if verbose else logging.INFO, get_settings().log_format)


def _publisher(redis_url: str) -> Callable[[str, str], object]:
    """Redis publish for live WebSocket updates; a no-op without Redis (clients poll)."""
    if not redis_url:
        return lambda channel, message: None
    import redis

    return redis.Redis.from_url(redis_url).publish


def _season(value: str) -> int:
    try:
        return parse_season(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Run the FastAPI server."""
    # log_config=None: uvicorn's own loggers (access log included) go through the root
    # handler set up in main(), so they follow HOOPSAI_LOG_FORMAT too.
    uvicorn.run(
        "hoopsai.api.main:app",
        host=host,
        port=port,
        reload=reload,
        loop=EVENT_LOOP,
        log_config=None,
    )


@app.command()
def worker() -> None:
    """Run the job scheduler (ingestion now; features, predictions, retraining later)."""
    from hoopsai.worker.main import run

    run()


@app.command()
def backfill(
    from_season: Annotated[
        str, typer.Option("--from", help="First season, e.g. 2004-05 or 2004.")
    ] = "2004-05",
    to_season: Annotated[
        str | None, typer.Option("--to", help="Last season (default: current).")
    ] = None,
    pbp: Annotated[
        bool, typer.Option(help="Also fetch play-by-play (~1 request per game, slow).")
    ] = False,
    force: Annotated[bool, typer.Option(help="Refetch tasks that already succeeded.")] = False,
) -> None:
    """Load historical schedules, team box scores and (optionally) play-by-play. Resumable."""
    from hoopsai.ingest.jobs import make_ingestor

    today = datetime.now(UTC).date()
    first = _season(from_season)
    last = _season(to_season) if to_season else current_season(today)
    if first > last:
        raise typer.BadParameter(f"--from {from_season} is after --to {last}")
    summary = make_ingestor().backfill(first, last, pbp=pbp, force=force, today=today)
    typer.echo(f"backfill done: {summary}")
    if summary.failed:
        raise typer.Exit(1)


@ingest_app.command("daily")
def ingest_daily() -> None:
    """Current-season schedule, box scores and missing play-by-play."""
    from hoopsai.ingest import jobs

    if jobs.ingest_daily().failed:
        raise typer.Exit(1)


@ingest_app.command("schedule")
def ingest_schedule() -> None:
    """Refresh the current season's schedule."""
    from hoopsai.ingest import jobs

    if jobs.ingest_schedule().failed:
        raise typer.Exit(1)


@ingest_app.command("pbp")
def ingest_pbp(game_ids: list[str], force: bool = False) -> None:
    """Fetch play-by-play for specific games."""
    from hoopsai.ingest.jobs import make_ingestor

    ingestor = make_ingestor()
    for game_id in game_ids:
        ingestor.play_by_play(game_id, skip_if_done=not force)
    typer.echo(str(ingestor.summary))
    if ingestor.summary.failed:
        raise typer.Exit(1)


@app.command()
def features() -> None:
    """Rebuild point-in-time features and Elo for every game (features.* tables)."""
    from hoopsai.db.session import get_sync_engine
    from hoopsai.features.store import rebuild_features

    rebuild_features(get_sync_engine())


@app.command()
def train(
    register: Annotated[
        bool, typer.Option(help="Log to MLflow and register (and maybe promote) the model.")
    ] = True,
    test_seasons: Annotated[
        int, typer.Option(help="Most recent completed seasons to backtest on.")
    ] = 4,
) -> None:
    """Walk-forward backtest vs Elo, then fit the pre-game model on all completed seasons."""
    from hoopsai.config import get_settings
    from hoopsai.db.session import get_sync_engine
    from hoopsai.ml.train import format_report, train_from_db

    result = train_from_db(
        get_sync_engine(),
        today=datetime.now(UTC).date(),
        tracking_uri=get_settings().mlflow_tracking_uri if register else None,
        n_test_seasons=test_seasons,
    )
    typer.echo(format_report(result.report))
    if result.registration:
        r = result.registration
        typer.echo(f"\nregistered hoopsai-pregame v{r.version} (run {r.run_id})")
        typer.echo(f"promoted to production: {r.promoted} ({r.reason})")


@app.command("train-ingame")
def train_ingame(
    from_season: Annotated[
        str, typer.Option("--from", help="First season of play-by-play to train on.")
    ] = "2015-16",  # hoopsai.ml.ingame.FIRST_TRAINING_SEASON
    register: Annotated[bool, typer.Option(help="Log to MLflow and register the model.")] = True,
) -> None:
    """Train the in-game model on play-by-play states; the newest season is held out."""
    from hoopsai.config import get_settings
    from hoopsai.db.session import get_sync_engine
    from hoopsai.ml import ingame, registry

    engine = get_sync_engine()
    seasons = [s for s in ingame.seasons_with_pbp(engine) if s >= _season(from_season)]
    if len(seasons) < 2:
        raise typer.BadParameter(f"need play-by-play for 2+ seasons, have {seasons}")
    frame = ingame.build_training_frame(engine, seasons)
    model, report = ingame.train_and_evaluate(frame)
    typer.echo(ingame.format_report(report))
    if register:
        r = registry.log_and_register_ingame(
            model, report, ingame.DEFAULT_PARAMS, get_settings().mlflow_tracking_uri
        )
        typer.echo(f"\nregistered hoopsai-ingame v{r.version}; promoted: {r.promoted} ({r.reason})")


@app.command()
def live(
    watch: Annotated[
        str | None,
        typer.Option(help="Print win probability for one game without storing anything."),
    ] = None,
    until_idle: Annotated[
        bool, typer.Option(help="Stop once every game today is final (scheduled hosts).")
    ] = False,
    max_hours: Annotated[
        float | None, typer.Option(help="Stop after this many hours (scheduled hosts).")
    ] = None,
) -> None:
    """Run the live poller: in-game win probability for every live game."""
    from hoopsai.config import get_settings
    from hoopsai.db.session import get_sync_engine
    from hoopsai.live import service
    from hoopsai.live.source import make_live_source
    from hoopsai.live.tracker import LiveTracker
    from hoopsai.ml.registry import (
        INGAME_MODEL_NAME,
        load_production_ingame,
        production_version,
    )

    settings = get_settings()
    engine = get_sync_engine()
    while True:
        try:
            model, version = load_production_ingame(settings.mlflow_tracking_uri)
            break
        except LookupError:
            if watch:
                raise
            logging.getLogger(__name__).warning(
                "no production in-game model yet (run `hoopsai train-ingame`); retrying in 60s"
            )
            time.sleep(60)
    source = make_live_source(engine)
    if watch:
        tracker = LiveTracker(model, service.pregame_probability(engine, watch))
        points = tracker.update(source.play_by_play(watch))
        for p in points[-10:]:
            typer.echo(
                f"Q{p.period} {p.clock_seconds:5.1f}s  {p.score_away}-{p.score_home}  "
                f"home {p.home_win_prob:.1%}  {p.description or ''}"
            )
        typer.echo(f"{len(points)} states")
        return
    recorder = service.Recorder(engine, _publisher(settings.redis_url), version, source="live")
    uri = settings.mlflow_tracking_uri
    models = service.ModelSource(
        current_version=lambda: production_version(uri, INGAME_MODEL_NAME),
        load=lambda: load_production_ingame(uri),
    )
    service.LivePoller(engine, source, model, recorder, models=models).run(
        max_seconds=None if max_hours is None else max_hours * 3600,
        stop_when_idle=until_idle,
    )


@app.command()
def replay(
    game_id: str,
    speed: Annotated[float, typer.Option(help="Game seconds per real second.")] = 30.0,
) -> None:
    """Replay a finished game through the live pipeline (snapshots, Redis, WebSocket)."""
    from sqlalchemy import select

    from hoopsai.config import get_settings
    from hoopsai.db.models import ApiResponse
    from hoopsai.db.session import get_sync_engine
    from hoopsai.ingest.store import decompress_payload
    from hoopsai.live import service
    from hoopsai.ml.registry import load_production_ingame
    from hoopsai.sources.base import request_key
    from hoopsai.sources.nba_stats.client import NbaStatsSource
    from hoopsai.sources.nba_stats.parse import parse_play_by_play

    settings = get_settings()
    engine = get_sync_engine()
    key = request_key("nba_stats", "playbyplayv3", {"game_id": game_id})
    with engine.connect() as conn:
        blob = conn.scalar(select(ApiResponse.payload_gz).where(ApiResponse.request_key == key))
    payload = (
        decompress_payload(blob) if blob else NbaStatsSource().fetch_play_by_play(game_id).payload
    )
    events = parse_play_by_play(payload)
    model, version = load_production_ingame(settings.mlflow_tracking_uri)
    recorder = service.Recorder(engine, _publisher(settings.redis_url), version, source="replay")
    minutes = max(e["period"] for e in events) * 12 * 60 / speed / 60
    typer.echo(f"replaying {game_id} ({len(events)} events) at {speed:g}x, about {minutes:.1f} min")
    n = service.replay(
        events, game_id, model, service.pregame_probability(engine, game_id), recorder, speed=speed
    )
    typer.echo(f"done: {n} points")


@app.command()
def predict(
    on: Annotated[
        str | None,
        typer.Option(
            "--date",
            help="Score every game on this date (YYYY-MM-DD), even finished ones. "
            "Default: upcoming games from today (US Eastern).",
        ),
    ] = None,
    days: Annotated[int, typer.Option(help="Days ahead to score (without --date).")] = 7,
) -> None:
    """Score games with the production model and store predictions (serving.predictions)."""
    from datetime import date as date_type

    from hoopsai.config import get_settings
    from hoopsai.db.session import get_sync_engine
    from hoopsai.predict.pregame import predict_pregame

    try:
        start = date_type.fromisoformat(on) if on else None
    except ValueError as exc:
        raise typer.BadParameter(f"--date {on!r} is not YYYY-MM-DD") from exc
    summary = predict_pregame(
        get_sync_engine(),
        get_settings().mlflow_tracking_uri,
        start=start,
        days=1 if start else days,
        only_scheduled=start is None,
    )
    typer.echo(str(summary))


@app.command()
def monitor() -> None:
    """Check both models' accuracy on finished games against their held-out evaluation."""
    from hoopsai.ingest import jobs

    typer.echo(jobs.monitor())


JOBS = ("daily_pipeline", "predict_pregame", "retrain", "ingest_schedule", "monitor")


@app.command()
def job(name: Annotated[str, typer.Argument(help=" | ".join(JOBS))]) -> None:
    """Run one scheduled job once, recorded like the worker's (for cron or GitHub Actions)."""
    from hoopsai.ingest import jobs

    if name not in JOBS:
        raise typer.BadParameter(f"unknown job {name!r}; one of {', '.join(JOBS)}")
    result = getattr(jobs, name)()
    typer.echo(f"{name}: {result}")
    if getattr(result, "failed", None):
        raise typer.Exit(1)


@app.command()
def publish(
    target_url: Annotated[str, typer.Argument(help="e.g. Neon's postgresql://... URL")],
    models: Annotated[
        bool, typer.Option(help="Copy MLflow's production models into the database first")
    ] = True,
) -> None:
    """Copy the website's data and the production models to another Postgres (e.g. Neon).
    Play-by-play and raw API responses are left out to fit a free 0.5 GB database."""
    from hoopsai.config import get_settings
    from hoopsai.db.session import get_sync_engine
    from hoopsai.ml import registry
    from hoopsai.publish import migrate
    from hoopsai.publish import publish as publish_to

    migrate(get_settings().database_url)  # the local registry table must exist
    if models:
        uri = get_settings().mlflow_tracking_uri
        for name in (registry.MODEL_NAME, registry.INGAME_MODEL_NAME):
            typer.echo(f"{name}: v{registry.copy_production_to_db(uri, name)} -> database")
    url = target_url.replace("postgresql://", "postgresql+psycopg://", 1)
    for table, rows in publish_to(get_sync_engine(), url).items():
        typer.echo(f"{table}: {rows} rows")


@app.command()
def version() -> None:
    """Print the package version."""
    from hoopsai import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
