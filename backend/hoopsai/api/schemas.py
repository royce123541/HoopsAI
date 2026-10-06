"""Response models. These define the OpenAPI schema the web app's TypeScript types are
generated from (web: `npm run gen:api`)."""

import datetime as dt
from typing import Literal

from pydantic import BaseModel


class TeamRef(BaseModel):
    team_id: int
    abbreviation: str
    city: str
    nickname: str
    full_name: str


class PredictionSummary(BaseModel):
    home_win_prob: float
    model_version: str
    created_at: dt.datetime
    made_before_tip: bool


class GameSummary(BaseModel):
    game_id: str
    game_date: dt.date
    tip_time_utc: dt.datetime | None
    status: Literal["scheduled", "live", "final"]
    season: int
    season_type: Literal["regular", "playoffs", "playin"]
    is_neutral: bool
    home: TeamRef
    away: TeamRef
    home_score: int | None
    away_score: int | None
    prediction: PredictionSummary | None
    live: "LiveSummary | None" = None  # latest in-game probability while a game is live


class Slate(BaseModel):
    date: dt.date
    prev_date: dt.date | None  # nearest earlier date with games
    next_date: dt.date | None
    games: list[GameSummary]


class Factor(BaseModel):
    feature: str
    label: str
    value: float
    contribution: float  # log-odds impact on P(home win)
    favors: Literal["home", "away"]


class ComparisonRow(BaseModel):
    key: str
    label: str
    home: float | None
    away: float | None
    higher_is_better: bool


class GameDetail(GameSummary):
    factors: list[Factor]
    comparison: list[ComparisonRow]


class EloPoint(BaseModel):
    game_date: dt.date
    season: int
    elo: float


class TeamGame(BaseModel):
    game_id: str
    game_date: dt.date
    status: Literal["scheduled", "live", "final"]
    is_home: bool
    opponent: TeamRef
    team_score: int | None
    opponent_score: int | None
    win_prob: float | None  # the team's pre-game win probability, if predicted


class TeamListItem(TeamRef):
    elo: float | None


class TeamDetail(BaseModel):
    team: TeamRef
    elo: float | None
    elo_history: list[EloPoint]
    recent: list[TeamGame]
    upcoming: list[TeamGame]


class Metrics(BaseModel):
    logloss: float
    brier: float
    accuracy: float
    ece: float
    n: float


class SeasonMetrics(BaseModel):
    season: int
    model: Metrics
    elo: Metrics


class ReliabilityBin(BaseModel):
    bin_low: float
    bin_high: float
    mean_pred: float
    observed: float
    count: int


class FeatureImportance(BaseModel):
    feature: str
    label: str
    share: float  # share of total split gain


class ModelInfo(BaseModel):
    version: str
    run_id: str
    feature_version: str
    calibration: str
    train_seasons: str
    first_used_at: dt.datetime
    seasons: list[SeasonMetrics]
    pooled_model: Metrics
    pooled_elo: Metrics
    reliability: list[ReliabilityBin]
    acceptance: dict[str, bool]
    top_features: list[FeatureImportance]


class WinProbPoint(BaseModel):
    action_id: int
    period: int
    clock_seconds: float
    elapsed_seconds: float
    score_home: int
    score_away: int
    possession: int  # 1 home, -1 away, 0 unknown
    home_win_prob: float
    description: str | None


class WinProbSeries(BaseModel):
    game_id: str
    source: Literal["live", "replay"] | None  # live data wins over a replay
    model_version: str | None
    points: list[WinProbPoint]


class LiveSummary(BaseModel):
    home_win_prob: float
    period: int
    clock_seconds: float
    score_home: int
    score_away: int
