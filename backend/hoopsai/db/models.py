"""ORM models for the raw (bronze) and core (silver) data layers. See docs/ARCHITECTURE.md §3.1."""

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    SmallInteger,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from hoopsai.db.base import Base


class TimestampMixin:
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# ---------------------------------------------------------------- raw


class ApiResponse(Base):
    """Latest raw response per request, gzip-compressed so a full play-by-play backfill
    (~28k games) stays well under 1 GB. Parsers can be re-run from here without refetching."""

    __tablename__ = "api_responses"
    __table_args__ = {"schema": "raw"}

    request_key: Mapped[str] = mapped_column(Text, primary_key=True)
    source: Mapped[str] = mapped_column(Text)
    endpoint: Mapped[str] = mapped_column(Text, index=True)
    params: Mapped[dict[str, Any]] = mapped_column(JSONB)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    payload_gz: Mapped[bytes] = mapped_column(LargeBinary)
    payload_bytes: Mapped[int] = mapped_column(Integer)


class IngestionRun(TimestampMixin, Base):
    """Checkpoint per ingestion task (e.g. `pbp:0022400001`), so backfills can resume."""

    __tablename__ = "ingestion_runs"
    __table_args__ = {"schema": "raw"}

    task_key: Mapped[str] = mapped_column(Text, primary_key=True)
    status: Mapped[str] = mapped_column(Text)  # "success" | "failed"
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    rows: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)


# ---------------------------------------------------------------- core


class Team(Base):
    __tablename__ = "teams"
    __table_args__ = {"schema": "core"}

    team_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    abbreviation: Mapped[str] = mapped_column(Text)
    city: Mapped[str] = mapped_column(Text)
    nickname: Mapped[str] = mapped_column(Text)
    full_name: Mapped[str] = mapped_column(Text)


class Game(TimestampMixin, Base):
    __tablename__ = "games"
    __table_args__ = {"schema": "core"}

    game_id: Mapped[str] = mapped_column(Text, primary_key=True)
    season: Mapped[int] = mapped_column(SmallInteger, index=True)  # start year: 2024 = 2024-25
    season_type: Mapped[str] = mapped_column(Text)  # regular | playoffs | playin
    game_date: Mapped[date] = mapped_column(Date, index=True)  # US Eastern calendar date
    tip_time_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    home_team_id: Mapped[int] = mapped_column(ForeignKey("core.teams.team_id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("core.teams.team_id"))
    status: Mapped[str] = mapped_column(Text)  # scheduled | live | final
    home_score: Mapped[int | None] = mapped_column(SmallInteger)
    away_score: Mapped[int | None] = mapped_column(SmallInteger)
    is_neutral: Mapped[bool | None] = mapped_column(Boolean)


class TeamGameStats(Base):
    """One team's box score in one game. Derived metrics (possessions, ratings, four
    factors) are computed in the features layer, not stored here."""

    __tablename__ = "team_game_stats"
    __table_args__ = {"schema": "core"}

    game_id: Mapped[str] = mapped_column(ForeignKey("core.games.game_id"), primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("core.teams.team_id"), primary_key=True)
    is_home: Mapped[bool] = mapped_column(Boolean)
    minutes: Mapped[int] = mapped_column(SmallInteger)
    pts: Mapped[int] = mapped_column(SmallInteger)
    fgm: Mapped[int] = mapped_column(SmallInteger)
    fga: Mapped[int] = mapped_column(SmallInteger)
    fg3m: Mapped[int] = mapped_column(SmallInteger)
    fg3a: Mapped[int] = mapped_column(SmallInteger)
    ftm: Mapped[int] = mapped_column(SmallInteger)
    fta: Mapped[int] = mapped_column(SmallInteger)
    oreb: Mapped[int] = mapped_column(SmallInteger)
    dreb: Mapped[int] = mapped_column(SmallInteger)
    reb: Mapped[int] = mapped_column(SmallInteger)
    ast: Mapped[int] = mapped_column(SmallInteger)
    stl: Mapped[int] = mapped_column(SmallInteger)
    blk: Mapped[int] = mapped_column(SmallInteger)
    tov: Mapped[int] = mapped_column(SmallInteger)
    pf: Mapped[int] = mapped_column(SmallInteger)
    plus_minus: Mapped[int] = mapped_column(SmallInteger)


class PbpEvent(Base):
    """One play-by-play event. Order a game's events by `action_id`: it is the feed's
    chronological sequence (1..N), whereas `action_number` is not (late-logged events such
    as substitutions and technical free throws get higher numbers)."""

    __tablename__ = "pbp_events"
    __table_args__ = {"schema": "core"}

    game_id: Mapped[str] = mapped_column(ForeignKey("core.games.game_id"), primary_key=True)
    action_id: Mapped[int] = mapped_column(Integer, primary_key=True)  # chronological, 1..N
    action_number: Mapped[int] = mapped_column(Integer)  # NBA event number; not unique
    period: Mapped[int] = mapped_column(SmallInteger)
    clock_seconds: Mapped[float] = mapped_column(Float)  # remaining in the period
    team_id: Mapped[int | None] = mapped_column(Integer)
    location: Mapped[str | None] = mapped_column(Text)  # "h" | "v"
    person_id: Mapped[int | None] = mapped_column(BigInteger)
    action_type: Mapped[str | None] = mapped_column(Text)
    sub_type: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    is_field_goal: Mapped[bool] = mapped_column(Boolean)
    shot_value: Mapped[int | None] = mapped_column(SmallInteger)
    shot_result: Mapped[str | None] = mapped_column(Text)
    score_home: Mapped[int] = mapped_column(SmallInteger)  # forward-filled running score
    score_away: Mapped[int] = mapped_column(SmallInteger)


# ---------------------------------------------------------------- features


class GameFeatures(Base):
    """Pre-game features for one game (hoopsai.features.build), rebuilt after each ingest."""

    __tablename__ = "game_features"
    __table_args__ = {"schema": "features"}

    game_id: Mapped[str] = mapped_column(ForeignKey("core.games.game_id"), primary_key=True)
    feature_version: Mapped[str] = mapped_column(Text)
    features: Mapped[dict[str, Any]] = mapped_column(JSONB)
    built_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EloRating(Base):
    """A team's Elo before (and, once final, after) each game."""

    __tablename__ = "elo_ratings"
    __table_args__ = (
        Index("ix_elo_ratings_team_date", "team_id", "game_date"),
        {"schema": "features"},
    )

    game_id: Mapped[str] = mapped_column(ForeignKey("core.games.game_id"), primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("core.teams.team_id"), primary_key=True)
    season: Mapped[int] = mapped_column(SmallInteger)
    game_date: Mapped[date] = mapped_column(Date)
    elo_pre: Mapped[float] = mapped_column(Float)
    elo_post: Mapped[float | None] = mapped_column(Float)


# ---------------------------------------------------------------- serving


class ModelVersionSnapshot(Base):
    """Metadata of a registered model version used for predictions, copied from MLflow so the
    API never depends on the MLflow server at request time."""

    __tablename__ = "model_versions"
    __table_args__ = {"schema": "serving"}

    version: Mapped[str] = mapped_column(Text, primary_key=True)
    run_id: Mapped[str] = mapped_column(Text)
    feature_version: Mapped[str] = mapped_column(Text)
    calibration: Mapped[str] = mapped_column(Text)
    train_seasons: Mapped[str] = mapped_column(Text)  # e.g. "2005-2025"
    backtest: Mapped[dict[str, Any]] = mapped_column(JSONB)
    importance: Mapped[dict[str, Any]] = mapped_column(JSONB)
    first_used_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Prediction(Base):
    """A pre-game win probability. Re-scoring appends a row only when the probability or the
    model changes; the latest row per game is the current prediction."""

    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_predictions_game_created", "game_id", "created_at"),
        {"schema": "serving"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(ForeignKey("core.games.game_id"))
    model_version: Mapped[str] = mapped_column(ForeignKey("serving.model_versions.version"))
    feature_version: Mapped[str] = mapped_column(Text)
    home_win_prob: Mapped[float] = mapped_column(Float)
    factors: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)  # TreeSHAP, top by |impact|
    made_before_tip: Mapped[bool] = mapped_column(Boolean)  # False for after-the-fact scoring
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
