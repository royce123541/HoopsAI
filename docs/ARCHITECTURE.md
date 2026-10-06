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
| Frontend | **Next.js (App Router) + TypeScript**, Tailwind CSS, Recharts, and API types generated with `openapi-typescript` | Server-rendered pages read the API directly, so M3 needed neither shadcn/ui nor TanStack Query. TanStack Query will be reconsidered for live updates in M4. Recharts draws the Elo and calibration charts; simple bars are plain HTML. |
| Realtime to the browser | **Native WebSockets** (FastAPI to the browser), with Redis pub/sub behind them | Simple. Several API replicas can share one live feed. |
| Backend API | **FastAPI** (Python 3.12), Pydantic v2, SQLAlchemy 2.0 (async), Alembic | Same language as the ML code, so models and feature code are shared rather than rewritten. Async suits the WebSocket fan-out. |
| Database | **PostgreSQL 16** | Relational game and stat data, JSONB for raw payloads, and enough capacity for live snapshots (about 600k rows a season). |
| Cache and message bus | **Redis 7** | Pub/sub for live probability updates, plus an API response cache. |
| Jobs and scheduling | **APScheduler** in a `worker` service | Runs the nightly ingest, daily predictions and weekly retrain. Lighter than Celery or Airflow at this scale. |
| ML | **pandas, scikit-learn, LightGBM**, and **MLflow** (client: `mlflow-skinny`) for tracking and the model registry | Gradient-boosted trees are strong on tabular data. LightGBM's built-in TreeSHAP (`pred_contrib`) explains the main factors behind each prediction without the separate `shap` package. MLflow versions every model. |
| Data source | **`nba_api`**: stats.nba.com for history and for live games (`ScoreboardV3`, `PlayByPlayV3`) | Free and thorough. It sits behind `DataSource` and `LiveSource` interfaces so a paid provider can replace it later. The originally planned cdn.nba.com live feed returns 403 (Akamai "Access Denied") from this deployment's network, for browsers too. |
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
 stats.nba.com live ─(poll 10s)───► live poller ──► in-game model ──► live_wp_snapshots
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
- **Caveat:** stats.nba.com often blocks cloud IP ranges. The historical backfill should run locally, or through the `DataSource` swap if that is blocked too. Live polling uses the same host, so the `live` service must run where stats.nba.com is reachable too.

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
- **Training data:** every play-by-play state, labelled with the game's final outcome.
  - States come from `hoopsai/live/state.py`, the same code the live service uses.
  - Substitutions, timeouts and replay reviews are dropped, since they never change the state.
- **Possession:** PlayByPlayV3 has no possession field, so it is inferred from events.
  - Rules: made shots and turnovers switch possession, rebounds give it to the rebounder, and the last made free throw of a trip switches it. Technical free throws don't change it, and flagrant or clear-path trips keep it with the fouled team.
  - Jump balls make it unknown, because the jump-ball row doesn't say who won the tip.
  - On 35,609 real shots, the inferred possession was the shooting team 97.8% of the time, unknown 2.1% and wrong 0.1%.
- **Features:**
  - score difference and seconds remaining (in regulation, or in the current overtime)
  - `diff / sqrt(seconds_remaining + 1)`
  - possession and period
  - the pre-game logit, plus that logit times the share of the game left, so team strength fades as the game goes on
- **Pre-game input:** for training, each season's pre-game probabilities come from a pre-game model fitted only on earlier seasons. An in-sample probability would be overconfident, and the in-game model would learn to trust it too much. Live, the input is the game's stored pre-game prediction.
- **Estimator:** LightGBM, monotone increasing in score difference, scaled difference, possession and pre-game strength.
- **Split:** the newest season is held out for evaluation, and the production model is then refitted on all seasons. No game is ever split across training and test.
- **Acceptance:** better log loss than the pre-game probability alone, and calibration error of at most 0.03.
- **Hard limits:** the probability is exactly 0 or 1 when a game is final, or when the clock reads zero in the 4th quarter or overtime with one team ahead.

**Registry:** both models are logged to MLflow with their metrics, feature list, data window and backtest report. A new version becomes the `production` alias only if both hold:
- it passes the acceptance checks;
- it beats the current production version's out-of-sample log loss on the same most recent backtest season.

That log loss is stored at full precision as a version tag. Each prediction row records `model_version`.

### 3.4 Serving
- **Pre-game (batch):** `hoopsai predict` (`hoopsai/predict/pregame.py`) scores the coming week's scheduled games.
  - **When:** after the nightly ingest and every 2 hours.
  - **Inputs:** stored features; they are rebuilt only if missing or from a different feature version.
  - **Writes:** a row in `serving.predictions` (game_id, model_version, home_win_prob, top-10 TreeSHAP `factors`, `made_before_tip`, created_at), but only when a game's probability or model changes.
  - **Model snapshot:** the backtest report and feature importance of each model version used are copied from MLflow into `serving.model_versions`, so the API never calls MLflow.
  - **Reads:** the API only reads these tables, so there is no inference on the request path.
- **Live (streaming):**
  - **Polling:** the `live` service (`hoopsai live`) reads `ScoreboardV3` every 30s and only tracks games that are in `core.games`, so preseason games are ignored. For each live game it polls `PlayByPlayV3` every 10s.
  - **Scoring:** new events become states, the in-game model scores them, and the points go into `serving.live_wp_snapshots`.
  - **Publishing:** points are published as `{"type": "points", ...}` to the Redis channel `game:{id}`. The game's status and score are also updated in `core.games`.
  - **Restarts:** after a restart the poller resumes after the last stored action, so nothing is recorded twice.
  - **WebSocket:** `WS /api/ws/games/{id}` subscribes to the channel before reading the snapshot it sends first, so no update falls in between. Clients de-duplicate by `action_id`.
- **Replay mode:** `hoopsai replay <game_id> --speed 30` re-runs a finished game through the same recorder, Redis and WebSocket path, with game time running at 30× real time. It uses the play-by-play stored by the backfill, or fetches it. Replay points are stored with `source = 'replay'` and cleared at the start of each replay. The API prefers live points over replayed ones.

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
- `WS /api/ws/games/{id}`: live win-probability stream (snapshot, then `points` and `reset` messages)
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
- **Serving:** `serving.model_versions`, `serving.predictions`; `live_wp_snapshots` comes in M4
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
