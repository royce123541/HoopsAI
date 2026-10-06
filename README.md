# HoopsAI

Machine-learned win probabilities for NBA games, before tip-off and live during play.

- **Architecture and data pipeline:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Stack:** Next.js (TypeScript, Tailwind) · FastAPI (Python 3.12) · PostgreSQL 16 · Redis 7 · LightGBM and MLflow

## Status

| Milestone | Scope | State |
|---|---|---|
| M0 | Scaffold: compose stack, FastAPI `/api/health`, Next.js status page, Alembic | ✅ done |
| M1 | Ingestion: `DataSource`, nba_api loaders, backfill CLI, `worker` service | ✅ done |
| M2 | Elo, point-in-time features, pre-game LightGBM model, MLflow | ✅ done |
| M3 | Pre-game serving: REST endpoints, slate, game and model pages | ✅ done |
| M4 | Live: in-game model, `live` poller, Redis to WebSocket, live chart, replay | ⏳ next |
| M5 | Ops: schedules, drift monitor, CI | |

## Run everything (Docker)

Requires Docker Desktop with the WSL2 backend on Windows.

```sh
docker compose up --build
```

| Service | URL |
|---|---|
| Web | http://localhost:3000 |
| API (OpenAPI docs at `/docs`) | http://localhost:8000/api/health |
| MLflow | http://localhost:5000 |
| Postgres / Redis | `localhost:5433` / `localhost:6379` |

The API container runs `alembic upgrade head` on start. Copy `.env.example` to `.env` to override the defaults.

Postgres is published on host port **5433**, not 5432, so it can run alongside a native Postgres install. Set `POSTGRES_HOST_PORT` to change it.

## Load data

Ingestion pulls from stats.nba.com through `nba_api`. It is throttled to about one request every 0.6 s and retried with backoff. stats.nba.com often blocks cloud IP ranges, so run the historical backfill from a home connection.

```sh
cd backend

# Every season's schedule and team box scores since 2004-05: ~90 requests, a few minutes
uv run hoopsai backfill --from 2004-05

# Add play-by-play: 1 request per game (~28k games, several hours). Resumable, so stop and
# re-run at any time.
uv run hoopsai backfill --from 2004-05 --pbp

uv run hoopsai ingest daily              # current season: schedule, box scores, new play-by-play
uv run hoopsai ingest schedule           # current season's schedule only
uv run hoopsai ingest pbp 0022400061     # specific games
```

Each task (`schedule:<season>`, `gamelog:<season>:<type>`, `pbp:<game_id>`) records its result in `raw.ingestion_runs`:
- Completed past seasons and fetched games are skipped on re-runs, unless you pass `--force`.
- The current season is always refetched.
- A failure is logged and the run continues; failed tasks are retried on the next run.

Raw responses are stored gzip-compressed in `raw.api_responses`, and the parsed tables are in `core.*`.

## Features and model

```sh
cd backend
uv run hoopsai features             # rebuild Elo and point-in-time features (features.* tables)
uv run hoopsai train                # backtest vs Elo, fit on all completed seasons, register in MLflow
uv run hoopsai train --no-register  # backtest and fit only
```

`hoopsai train` does three things:
- **Backtest:** walk-forward on the 4 most recent completed seasons. Each test season is predicted by a model that never saw it.
- **Report:** prints log loss, Brier score, accuracy and calibration error for both the model and the Elo baseline.
- **Register:** adds a new version of `hoopsai-pregame` in MLflow and points the `production` alias at it only if it beats Elo and the current production model. The comparison uses log loss on the same held-out season.

Backtest from 2026-10-06 (model v2):

| Season | Log loss (Elo) | Brier (Elo) | Accuracy (Elo) |
|---|---|---|---|
| 2022-23 | 0.6435 (0.6557) | 0.2259 (0.2307) | 64.2% (63.1%) |
| 2023-24 | 0.6131 (0.6251) | 0.2126 (0.2182) | 64.7% (64.2%) |
| 2024-25 | 0.6116 (0.6259) | 0.2121 (0.2181) | 65.8% (64.8%) |
| 2025-26 | 0.6011 (0.6121) | 0.2066 (0.2110) | 68.4% (67.4%) |
| Pooled | 0.6174 (0.6297) | 0.2143 (0.2195) | 65.8% (64.9%) |

Pooled calibration error is 0.013.

In Docker, the `worker` service runs three jobs:
- **10:00 UTC daily:** ingest the previous day's games, then rebuild features.
- **12:00 UTC daily:** refresh the current season's schedule.
- **Monday 11:00 UTC:** retrain.

## Develop natively

Start only the infrastructure in Docker, then run the apps on the host with hot reload:

```sh
docker compose up -d db redis mlflow

# backend
cd backend
uv sync
uv run alembic upgrade head
uv run hoopsai api --reload        # http://localhost:8000

# web, in a second terminal
cd web
npm install
npm run dev                        # http://localhost:3000, API_INTERNAL_URL defaults to localhost:8000
```

On Windows, `hoopsai api` switches uvicorn to a selector event loop, because psycopg's async mode can't run on the default Proactor loop.

## Predictions and the web app

```sh
cd backend
uv run hoopsai predict               # score the next 7 days' scheduled games (US Eastern dates)
uv run hoopsai predict --days 21     # look further ahead
uv run hoopsai predict --date 2025-06-22   # score a past date (flagged as made after tip-off)
```

Predictions are stored in `serving.predictions`. A new row is written only when a game's probability or the production model changes.

| Page | Shows |
|---|---|
| `/` (`?date=YYYY-MM-DD`) | The day's games, each with its win-probability split and links to the previous and next game days |
| `/games/{id}` | The prediction, the factors behind it (TreeSHAP), and a team comparison |
| `/teams`, `/teams/{id}` | Teams ranked by Elo, plus each team's Elo history, upcoming games and recent results |
| `/model` | Backtest results against Elo, calibration, and feature importance |

The API serves the same data as JSON: `/api/games`, `/api/games/{id}`, `/api/teams`, `/api/teams/{id}` and `/api/model`. The web app's TypeScript types are generated from its OpenAPI schema; with the API running, refresh them with `cd web && npm run gen:api`.

## Checks

```sh
# backend
cd backend
uv run ruff check . && uv run ruff format --check . && uv run mypy hoopsai tests && uv run pytest

# web
cd web
npm run lint && npm run typecheck && npx prettier --check . && npm run build
```
