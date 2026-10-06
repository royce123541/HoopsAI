# HoopsAI: Architecture & Data Pipeline Plan

## Context
HoopsAI is a new web app that uses machine learning to predict NBA win probabilities. The project directory (`C:\Users\Cinnamoroll\Downloads\HoopsAI`) is empty, so this plan starts from nothing. It is not a git repo yet.
Decisions confirmed with the user:
- **NBA only** for v1
- **Pre-game and live in-game** win probability
- **Docker Compose** for local development, built so it can move to the cloud later

The plan has two parts:
1. A modern stack that suits a data-driven, real-time app
2. A pipeline that ingests historical stats, builds features, trains models, and serves predictions both before and during games

---

## 1. Technology Stack

| Layer | Choice | Why |
|---|---|---|
| Frontend | **Next.js (App Router) + TypeScript**, Tailwind CSS, shadcn/ui, TanStack Query, Recharts | SSR for the slate pages, strong typing, and fast UI building. Recharts draws the win-probability curves. |
| Realtime to the browser | **Native WebSockets** (FastAPI to the browser), with Redis pub/sub behind them | Simple. Several API replicas can share one live feed. |
| Backend API | **FastAPI** (Python 3.12), Pydantic v2, SQLAlchemy 2.0 (async), Alembic | Same language as the ML code, so models and feature code are shared rather than rewritten. Async suits the WebSocket fan-out. |
| Database | **PostgreSQL 16** | Relational game and stat data, JSONB for raw payloads, and enough capacity for live snapshots (about 600k rows a season). |
| Cache and message bus | **Redis 7** | Pub/sub for live probability updates, plus an API response cache. |
| Jobs and scheduling | **APScheduler** in a `worker` service | Runs the nightly ingest, daily predictions and weekly retrain. Lighter than Celery or Airflow at this scale. |
| ML | **pandas, scikit-learn, LightGBM**, and **MLflow** (client: `mlflow-skinny`) for tracking and the model registry | Gradient-boosted trees are strong on tabular data. LightGBM's built-in TreeSHAP (`pred_contrib`) explains the main factors behind each prediction without the separate `shap` package. MLflow versions every model. |
| Data source | **`nba_api`**: stats.nba.com for history, cdn.nba.com live endpoints for in-game data | Free and thorough. It sits behind a `DataSource` interface so a paid provider can replace it later. |
| Tooling | `uv` (Python), `npm` (Node), Ruff and mypy, ESLint and Prettier, pytest, Vitest, Playwright | |
| Runtime | Docker Compose: `db`, `redis`, `api`, `worker`, `live`, `web`, `mlflow` | Runs on Docker Desktop with WSL2 on Windows and moves unchanged to Fly.io, Render or a VPS. |

**Single Python package, several entrypoints.** One backend image, run with different commands for the `api`, `worker` and `live` services. The ingestion, feature and model code then exists only once, and training and serving cannot drift apart.

---

## 2. Repository Layout
```
HoopsAI/
├─ docker-compose.yml          # db, redis, api, worker, live, web, mlflow
├─ .env.example
├─ backend/
│  ├─ pyproject.toml           # uv-managed
│  ├─ alembic/                 # migrations
│  ├─ hoopsai/
│  │  ├─ config.py             # pydantic-settings
│  │  ├─ db/                   # SQLAlchemy models, session
│  │  ├─ sources/              # DataSource interface + NbaApiSource (+ ReplaySource for tests)
│  │  ├─ ingest/               # backfill + incremental loaders (raw → core)
│  │  ├─ features/             # point-in-time feature builders, Elo
│  │  ├─ ml/                   # train_pregame.py, train_ingame.py, evaluate.py, registry.py
│  │  ├─ predict/              # batch pregame scoring, live in-game scorer
│  │  ├─ api/                  # FastAPI app, routers, websocket
│  │  ├─ worker/               # APScheduler job definitions
│  │  ├─ live/                 # live poller loop
│  │  └─ cli.py                # `hoopsai api|backfill|train|predict|replay ...` (Typer)
│  └─ tests/                   # unit + fixtures of recorded nba_api JSON
├─ web/                        # Next.js app
│  ├─ app/(routes)/            # /, /games/[id], /teams/[id], /model
│  ├─ components/              # GameCard, WinProbBar, WinProbChart, FactorList
│  └─ lib/                     # api client, ws hook (useLiveWinProb)
└─ notebooks/                  # exploration only; production code lives in hoopsai/
```

---

## 3. Data Pipeline

```
 stats.nba.com ──(backfill/nightly)──► raw.*  ──parse──► core.* ──build──► features.* ──► train (MLflow)
                                                                    │                        │
                                                                    └──► pregame scorer ◄────┘ (registered model)
                                                                              │
                                                                     predictions table ──► FastAPI REST ──► Next.js
 cdn.nba.com live ──(poll 5–10s)──► live poller ──► in-game model ──► live_wp_snapshots
                                                     └──► Redis pub/sub ──► FastAPI WS ──► Next.js live chart
```

### 3.1 Ingestion (Bronze → Silver)
- **Raw layer (`raw` schema):** the latest response for each request is stored unchanged, gzip-compressed (`bytea`), with its endpoint, parameters and `fetched_at`. Compression keeps the full play-by-play backfill well under 1 GB. Parsers can then be re-run without fetching again.
- **Core layer (`core` schema):** the parsed, typed tables:
  - `teams`, and `games` (id, season, season type, date, tip time, home/away, status, final score, neutral-site flag). The season is a column, not a table.
  - `team_game_stats` (box score and advanced stats: possessions, ORtg/DRtg, the four factors)
  - `pbp_events` (period, clock, score, event type, possession)
- **Backfill** with `hoopsai backfill --from 2004-05`. 2004-05 is roughly where play-by-play data becomes reliable. Requests are throttled to about 1 every 0.6s, retried with exponential backoff, and the job can resume from `raw.ingestion_runs` checkpoints.
- **Neutral-site games:** for international games and NBA Cup knockouts, the game log lists both teams as away ("@"). The designated home team is taken from the schedule, which is loaded first.
- **Game types modelled:** regular season (game-id prefix 002), play-in (005) and playoffs (004). Preseason, All-Star and the NBA Cup final (006) are excluded.
- **Incremental:** a nightly job at about 10:00 UTC, after West Coast games end, pulls the previous day's finals and box scores. A second job pulls the day's schedule.
- **Caveat:** stats.nba.com often blocks cloud IP ranges. The historical backfill should run locally, or through the `DataSource` swap if that is blocked too. The cdn.nba.com live endpoints are more permissive.

### 3.2 Feature Engineering (Gold, `features` schema)
**Point-in-time correctness is the main rule.** Every feature for game G uses only games that finished on an earlier date than G. A team plays at most once per date, so no relevant game is lost and no same-day result can leak in.

How the builder (`hoopsai/features/build.py`) works:
- It builds each team's post-game state after every final game.
- It attaches the latest state strictly before each game with `merge_asof(allow_exact_matches=False)`.
- Season-level stats reset each season.

`tests/test_features_build.py` rewrites every result from a cutoff date onward and checks that features up to and including that date are unchanged. Introducing same-day leakage makes it fail.

Games in the 2020 restart bubble (from 2020-07-30) count as neutral-site.

Features per team, used as home minus away differences plus raw values:
- **Elo rating:** a custom implementation with margin-of-victory adjustment, home-court offset, and season regression toward the mean. It also serves as the baseline model.
- **Rolling net rating, ORtg, DRtg and pace** over the last 5 and last 10 games and season to date
- **The four factors** (eFG%, TOV%, ORB%, FT rate), offense and defense
- **Schedule:** rest days, back-to-back flag, games in the last 7 days, travel distance (from a static table of arena coordinates)
- **Context:** home court, win percentage, current streak, games played (lets the model discount early-season noise)
- **v2:** injury and availability data from the official NBA injury report, and player-level lineup ratings

### 3.3 Models
**A. Pre-game model**
- **Estimator:** LightGBM binary classifier predicting P(home win).
- **Calibration:** each backtest compares no calibration with isotonic calibration fitted on the preceding season, and keeps the one with lower pooled log loss. With v1 features, no calibration wins: pooled log loss 0.617 against 0.625, and the raw model's calibration error is about 0.013.
- **Validation:** walk-forward by season. Train on seasons up to N-1 and validate on season N, for several values of N.
- **Metrics:**
  - Log loss (primary) and Brier score, both of which must beat the Elo baseline
  - Accuracy, which is typically 66–69% for the NBA
  - Expected calibration error and a reliability curve
- **Explainability:** TreeSHAP contributions in log-odds, from LightGBM `pred_contrib`, give the top factors for each game (`PregameModel.explain`). They are stored with each prediction, and the UI shows the top 5.

**B. In-game model**
- **Training data:** every play-by-play state, labeled with the game's final outcome
- **Features:**
  - Score difference and seconds remaining
  - `diff / sqrt(seconds_remaining + 1)`, the key interaction
  - Which team has possession, and the period
  - The pre-game model's logit, which lets team strength fade as the game goes on
- **Estimator:** LightGBM with a monotone constraint on score difference, then calibration
- **Split:** by game, never by row, so states from one game don't leak across train and test
- **Hard limits:** the probability is pinned to exactly 0 or 1 once the game is final.

**Registry:** both models are logged to MLflow with their metrics, feature list, data window and backtest report. A new version becomes the `production` alias only if both hold:
- it passes the acceptance checks;
- it beats the current production version's out-of-sample log loss on the same most recent backtest season.

That log loss is stored at full precision as a version tag. Each prediction row records `model_version`.

### 3.4 Serving
- **Pre-game (batch):** the worker scores the day's games every morning and rescores every 2 hours until tipoff. Results are written to `predictions` (game_id, model_version, home_wp, shap_top JSONB, created_at). The API only reads from this table, so there is no inference on the request path.
- **Live (streaming):**
  - The `live` service checks the live scoreboard about every 30s. For each game in progress, it polls the box score and play-by-play every 5–10s.
  - When the state changes, it runs the in-game model, inserts a row into `live_wp_snapshots`, and publishes JSON to the Redis channel `game:{id}`.
  - The FastAPI endpoint `WS /ws/games/{id}` subscribes and pushes updates. On connect it sends a snapshot of the history so far, then the deltas.
- **Replay mode:** `hoopsai replay <game_id> --speed 20x` feeds a recorded play-by-play through the same live path via `ReplaySource`. This allows development and demos when no games are being played.

### 3.5 Scheduled Jobs (`worker`)

| Job | When | Does |
|---|---|---|
| `ingest_daily` | 10:00 UTC | Pulls the previous day's finals and box scores, then updates Elo and features |
| `ingest_schedule` | 12:00 UTC | Pulls the day's schedule |
| `predict_pregame` | 13:00 UTC, then every 2h until tip | Scores the day's games |
| `retrain` | Weekly (Mon 11:00 UTC) | Trains, evaluates, and promotes the model if it improved |
| `monitor` | Daily | Tracks rolling Brier and log loss of live predictions and logs a warning on drift |

---

## 4. API Surface (FastAPI, `/api`)
- `GET /games?date=YYYY-MM-DD`: the day's slate with pre-game win probability and status
- `GET /games/{id}`: game detail, pre-game win probability, top SHAP factors, and team comparison
- `GET /games/{id}/winprob`: the full live win-probability timeline
- `WS /ws/games/{id}`: live win-probability stream
- `GET /teams`, `GET /teams/{id}`: team profile, Elo history and recent form
- `GET /model`: current model versions, backtest metrics and calibration curve data
- `GET /health`

The OpenAPI schema generates TypeScript types for `web/lib/api` with `openapi-typescript`, which keeps the frontend and backend contract in sync.

## 5. Frontend Pages
- **`/` Today's slate:** one card per game with a win-probability bar. Live games show score, clock and a moving win probability.
- **`/games/[id]`:**
  - Pre-game probability and its "why" factors (SHAP)
  - Team comparison table
  - Live win-probability chart over game time, updated in real time through `useLiveWinProb`
- **`/teams/[id]`:** Elo over time and recent form
- **`/model`:** model transparency, with the calibration plot, season backtest metrics and model version

---

## 6. Database Tables (core and serving)
- **Raw and audit:** `raw.api_responses`, `raw.ingestion_runs`
- **Core data:** `teams`, `games`, `team_game_stats`, `pbp_events`
- **Features:** `features.elo_ratings` (pre and post Elo for each team and game) and `features.game_features` (one JSONB row per game with the 66 model features, versioned as `v1`). Both are fully rebuilt in one transaction.
- **Serving:** `model_versions`, `predictions`, `live_wp_snapshots`
- **Indexes:** `(game_date)`, `(game_id, created_at)`, `(game_id, ts)`

---

## 7. Build Milestones
1. **M0, scaffold:**
   - `git init`, the monorepo layout and `docker-compose.yml`
   - Postgres, Redis and MLflow running
   - A FastAPI `/health` endpoint, a Next.js page that calls it, and the first Alembic migration
2. **M1, ingestion:** the `DataSource` interface, `NbaApiSource`, the raw and core loaders, a backfill CLI with checkpoints, and recorded JSON fixtures
3. **M2, features and pre-game model:**
   - Elo, the point-in-time feature builder and the leakage test
   - LightGBM training with walk-forward evaluation, calibration, SHAP and MLflow registration
4. **M3, pre-game serving and UI:** the batch scorer, the REST endpoints, and the slate, game detail and model pages
5. **M4, live:**
   - In-game model training, the live poller, Redis pub/sub and the WebSocket endpoint
   - The live chart and replay mode
6. **M5, operations:** worker schedules, the drift monitor, structured logging, a CI workflow (lint, typecheck and tests) and a README

---

## 8. Verification
- **Unit tests (pytest):**
  - Elo math, including symmetry and zero-sum
  - Four-factor calculations against known box scores
  - **Leakage test:** a feature row for game G must not change when games after G are added
  - In-game model limits, with the probability at 0 or 1 once the game is final
- **Integration tests:** ingestion parsers run against recorded nba_api JSON fixtures with no network. API tests use `httpx.AsyncClient` against a test Postgres.
- **Model acceptance:** `hoopsai train --evaluate` prints a backtest report. It passes when log loss and Brier beat Elo on the 2 most recent full seasons and calibration error is at most 0.03.
- **End-to-end:**
  1. `docker compose up`, then `hoopsai backfill --from 2022-23` for a small test run
  2. `hoopsai train`, then `hoopsai predict --date <past date>`
  3. Open `http://localhost:3000`
  4. Run `hoopsai replay <game_id> --speed 30x` and watch the live chart update over WebSocket
- **Frontend:** Vitest for components and one Playwright smoke test (slate loads, game page opens, a replayed WebSocket update moves the chart)
