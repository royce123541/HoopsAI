# Deploying HoopsAI for free: Vercel + Neon + GitHub Actions

This setup costs **$0**, needs no card, and does not need your PC to stay on.

| Part | Runs on | Free limit to keep an eye on |
|---|---|---|
| Website (Next.js) and API (FastAPI) | **Vercel** Hobby, one project ([`vercel.json`](../vercel.json)) | function invocations and bandwidth |
| Database the site reads | **Neon** free Postgres | 0.5 GB storage, monthly compute hours |
| Daily update, predictions, weekly retrain, live tracking | **GitHub Actions** schedules ([`.github/workflows`](../.github/workflows)) | free for public repositories |
| Models | Postgres table `serving.registered_models` (tracking URI `db`) | ~2 MB per version |

Free tiers change, so re-check these limits once a year.

## How it differs from the Docker setup

- **Data source:** stats.nba.com blocks cloud servers, so the cloud jobs use ESPN's public API (`HOOPSAI_DATA_SOURCE=espn`). ESPN games link to the NBA game ids already in the database.
- **No MLflow server:** models, metrics and reports live in Postgres (`HOOPSAI_MLFLOW_TRACKING_URI=db`).
- **No Redis or WebSockets:** live charts re-fetch every 10 seconds (`API_PUBLIC_WS_URL=poll`).
- **Smaller database:**
  - play-by-play stays on your PC (`HOOPSAI_STORE_PBP=false`);
  - finished games' live charts are thinned to 150 points;
  - only two seasons of live charts are kept.
- **In-game model retraining** needs the full play-by-play, so it runs only on your PC (`hoopsai train-ingame`, then `hoopsai publish`).

## One-time setup

### 1. Neon

1. Sign up at [neon.tech](https://neon.tech) and create a project. Set Vercel's function region (`regions` in `vercel.json`, now `sin1`, Singapore) to the region nearest the database.
2. Copy two connection strings from **Connect**:
   - the **pooled** one (host contains `-pooler`), for Vercel;
   - the **direct** one, for GitHub Actions and `hoopsai publish`.
3. Copy your local data and production models to Neon. Run this from `backend/` in PowerShell, with Docker running:

   ```powershell
   $env:HOOPSAI_DATABASE_URL = "postgresql+psycopg://hoopsai:hoopsai@127.0.0.1:5433/hoopsai"
   $env:HOOPSAI_MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
   uv run hoopsai publish 'DIRECT-NEON-URL'
   ```

### 2. GitHub Actions

1. Go to the repository's **Settings → Secrets and variables → Actions → New repository secret**.
2. Add a secret named `NEON_DATABASE_URL`, set to the **direct** Neon URL.
3. Under **Actions**, run **Daily update** once by hand to check it.

The schedules then run by themselves (times in UTC):

| Workflow | When | What |
|---|---|---|
| Daily update | 10:00 | schedule, results, features, predictions, monitoring, live-chart thinning |
| Refresh predictions | every 3 hours | re-scores the coming week |
| Weekly retrain | Mondays 11:00 | pre-game model; promoted only if it beats production |
| Live tracking | 16:00, 21:30, 03:00 | polls until that day's games are final |

The daily workflow also re-enables the scheduled workflows. GitHub would otherwise disable them after 60 days without repository activity.

### 3. Vercel

1. Sign up at [vercel.com](https://vercel.com) with GitHub, then **Add New → Project** and import this repository.
2. Leave **Root Directory** as the repository root. Vercel reads `vercel.json`, which defines the `web` and `api` services.
3. Under **Environment Variables**, add:

   | Name | Value |
   |---|---|
   | `HOOPSAI_DATABASE_URL` | the **pooled** Neon URL |
   | `HOOPSAI_REDIS_URL` | `off` |
   | `HOOPSAI_MLFLOW_TRACKING_URI` | `db` |
   | `API_PUBLIC_WS_URL` | `poll` |

4. Click **Deploy**.
5. To check it, open `https://<project>.vercel.app/api/health`. It should report `"database": "ok"`.

## Updating

Pushing to `main` redeploys Vercel automatically. Schema changes are applied by the next GitHub Actions job (`alembic upgrade head` runs before every job).
