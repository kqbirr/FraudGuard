# FraudGuard — Real-Time Fraud Detection Service

Phase 1 establishes the repository structure and a minimal, runnable FastAPI service.
Fraud prediction, database integration, model training, dataset loading and the
frontend application are **intentionally not implemented yet**. The `frontend/`
directory is reserved for a future React + TypeScript + Vite application.

## Requirements

- Windows 10 or 11
- Python 3.13
- Git
- Visual Studio Code (recommended)
- Docker Desktop 4.x (optional: only for the container workflow in
  [Containers](#containers-docker); nothing else needs it)

The dependencies in `requirements.txt` are all selected for Python 3.13 support on
Windows (wheels are published for CPython 3.13).

## Repository structure

```text
FraudGuard-Phase1/
├── app/                 # FastAPI application package
│   ├── api/             # Route modules (empty in Phase 1)
│   ├── database/        # Database layer (empty in Phase 1)
│   ├── models/          # ORM / model definitions (empty in Phase 1)
│   ├── services/        # Business logic (empty in Phase 1)
│   ├── config.py        # Environment-driven settings
│   ├── main.py          # FastAPI application entry point
│   └── schemas.py       # Pydantic response schemas
├── data/
│   ├── raw/             # Original, immutable datasets (not committed)
│   └── processed/       # Cleaned / feature-engineered data (not committed)
├── notebooks/           # Exploratory notebooks (empty in Phase 1)
├── tests/               # Test package (no tests in Phase 1)
├── scripts/             # Utility and automation scripts
├── frontend/            # React + TypeScript + Vite dashboard (own Dockerfile)
├── artifacts/           # Trained model files and generated outputs (not committed)
├── .github/workflows/   # GitHub Actions CI (tests, build, image smoke tests)
├── Dockerfile           # Backend image (FastAPI + model artifact)
├── .dockerignore        # Backend build-context exclusions
├── docker-compose.yml   # Local stack: backend + dashboard
├── docker-compose.sqlite.yml  # Optional SQLite override for safe testing
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

## Set up the virtual environment (Windows PowerShell)

The virtual environment is created once, at the project root, in `.venv`:

```powershell
py -3.13 -m venv .venv
```

Activate it. The prompt is prefixed with `(.venv)` when activation succeeded:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for the current user, allow locally created scripts
for this session only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the interpreter and installed packages:

```powershell
python --version
python -m pip list
```

## Configure the environment

The application runs with safe development defaults and does not require a `.env`
file in Phase 1. Create one when you want to override values:

```powershell
Copy-Item .env.example .env
```

`app/config.py` reads `APP_NAME`, `APP_ENV`, `LOG_LEVEL`, `DATABASE_URL`,
`CORS_ORIGINS` and `MODEL_PATH` from the environment (via `python-dotenv`). The
database URL in the example is a placeholder.

`CORS_ORIGINS` is a comma-separated allow-list of browser origins, and it defaults
to the Vite dev server (`http://localhost:5173,http://127.0.0.1:5173`). Origins
are always matched explicitly — never a wildcard — and no credentials are allowed
in cross-origin requests. Add the deployed dashboard's origin here when a real
frontend host exists; nothing else in the API needs to change.

## Run the application

Start the development server from the project root (the folder containing `app/`):

```powershell
python -m uvicorn app.main:app --reload
```

- API base URL: `http://127.0.0.1:8000`
- Interactive docs (Swagger UI): `http://127.0.0.1:8000/docs`
- Alternative docs (ReDoc): `http://127.0.0.1:8000/redoc`

Open the docs directly from PowerShell:

```powershell
Start-Process http://127.0.0.1:8000/docs
```

Stop the server with `Ctrl+C`.

## Verify the endpoints

In a second PowerShell window, with the server still running:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
```

Expected output:

```text
status
------
ok
```

`status` is `ok` when the model is resident. The endpoint also reports model
information, and returns HTTP 503 with `"status":"degraded"` if the model could
not be loaded. You can confirm the exact JSON with:

```powershell
(Invoke-WebRequest -Uri http://127.0.0.1:8000/health).Content
```

Check the root endpoint:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/
```

## Persistence: Neon PostgreSQL

Fraud decisions are stored in a PostgreSQL `flagged_transactions` table. Legitimate
predictions are deliberately not stored.

1. Create a project at [neon.tech](https://neon.tech) (free tier).
2. Copy the **pooled** connection string (its host contains `-pooler`), not the
   direct one - see the note below.
3. Put it in `.env`. Both of these work:

```text
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@ep-xxx-pooler.REGION.aws.neon.tech/fraudguard?sslmode=require
# ...or exactly the string Neon prints. A driver-less postgresql:// URL is
# pinned to psycopg 3 when the engine is built, so it needs no editing.
DATABASE_URL=postgresql://USER:PASSWORD@ep-xxx-pooler.REGION.aws.neon.tech/fraudguard?sslmode=require
```

4. Create the table (non-destructive and safe to re-run):

```powershell
python scripts/init_db.py
```

The script issues `CREATE TABLE IF NOT EXISTS` for missing tables and prints the
before/after table list; it never drops, alters or truncates anything. The
application itself never creates tables at import or during startup - schema
creation is always this explicit step.

Use the **pooled** string for the application and the **direct** string for
migrations/admin work.

### Stored fields

| Column                  | Type          | Notes                                                            |
| ----------------------- | ------------- | ---------------------------------------------------------------- |
| `id`                    | UUID          | Primary key, generated by the application |
| `transaction_id`        | varchar(64)   | Nullable. Reserved for an upstream reference; `/predict` has no such field in its contract today, so API alerts store `null` |
| `fraud_probability`     | float         | Model output, not rounded |
| `decision`              | varchar(16)   | `fraud` for every row written by `/predict` |
| `threshold`             | float         | Threshold in force at decision time, so old alerts stay explainable after a retrain |
| `model_name`            | varchar(64)   | e.g. `xgboost` |
| `prediction_latency_ms` | float         | Model scoring time only - database time is excluded |
| `transaction_time`      | float, null   | See the caveat below |
| `transaction_amount`    | float, null   | Echoed from the request for display |
| `created_at`            | timestamptz   | UTC, set by the application (microsecond precision) |

Indexes: `created_at`, `fraud_probability`, `decision`, and a composite
`(decision, created_at)` for the dashboard query.

**The 30 model features are not stored.** `V1`-`V28` are PCA components of real
cardholder behaviour - pseudonymised, not anonymous - so keeping them would turn
every alert row into cardholder data and widen both breach impact and regulatory
scope. The accepted trade-off is that a decision cannot be replayed exactly from
this table; the transaction store, not the alert table, remains the system of
record. See `app/models/flagged.py` for the full reasoning.

> **`Time` is not a real timestamp.** It is the Kaggle dataset's elapsed seconds
> since the first transaction in the file, so `transaction_time` is a relative
> offset rather than a usable wall-clock time. Live traffic would need a real
> ingestion timestamp column before this field means anything operationally.

### Failure policy

| Situation                          | Behaviour                                                      |
| ---------------------------------- | -------------------------------------------------------------- |
| `DATABASE_URL` empty               | Prediction returns **200**; no alert is written; a warning is logged |
| Configured, and the fraud write fails | **503**, body says no alert was recorded; error logged redacted |

An unrecorded fraud is unrecoverable, so the endpoint refuses to answer 200 for a
fraud decision it could not store. The cost is that a database outage also fails
fraud scoring for that request; the client is expected to retry. Prediction
latency is measured before the insert, so `latency_ms` never includes database IO.

### Listing alerts

Newest first (`created_at DESC`, then `id DESC` so bursts written in the same
microsecond keep a stable order). `limit` defaults to 20, must be 1-100 (`422`
otherwise); `offset` must be >= 0. `total` counts all stored alerts, not the page.

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/flagged?limit=20&offset=0"
```

```json
{
  "total": 1,
  "limit": 20,
  "offset": 0,
  "items": [
    {
      "id": "0f8fad5b-d9cb-469f-a165-70867728950e",
      "transaction_id": null,
      "fraud_probability": 0.9987,
      "decision": "fraud",
      "threshold": 0.5383065342903137,
      "model_name": "xgboost",
      "prediction_latency_ms": 2.41,
      "transaction_time": 406.0,
      "transaction_amount": 0.0,
      "created_at": "2026-11-05T09:14:22.418365Z"
    }
  ]
}
```

## Frontend dashboard

`frontend/` is a React + TypeScript + Vite app: a compact operations console for
scoring a transaction sample, watching session counters, and triaging stored
alerts. It talks to this API over HTTP through one typed client
(`frontend/src/api/client.ts`); nothing is mocked and no model features, labels or
credentials are bundled.

Start the API first, then the dashboard in a second terminal:

```powershell
cd frontend
npm install          # once
npm run dev          # http://127.0.0.1:5173
```

`npm run build` type-checks and produces `frontend/dist/`. The API base URL comes
from `VITE_API_BASE_URL` (copy `frontend/.env.example` to `frontend/.env.local` to
override the `http://127.0.0.1:8000` default). Only public configuration belongs in
that variable: Vite inlines it into the bundle, so the database URL, the model
artifact and any secret stay on the server.

Panels and where their numbers come from:

| Panel               | Source                          | Notes                                                            |
| ------------------- | ------------------------------- | ---------------------------------------------------------------- |
| Header              | `GET /health`, polled 15 s      | Connection state, `database_configured`, last poll time          |
| Session metrics     | `GET /metrics-lite`, polled 10 s | Process-local counters; they reset on restart and are not fleet totals |
| Model tile          | `GET /health`                   | Name and the artifact's stored decision threshold, available on first load |
| Transaction replay  | `POST /predict`                 | Manual only; pause/stop supported; never runs on page load        |
| Decision mix        | `GET /metrics-lite`             | Split bar of this process's fraud/legitimate counts plus latency   |
| Flagged transactions | `GET /flagged`                 | Paginated 10/20/50/100, manual and post-replay refresh            |

### Demo replay data

The replay rows live in `frontend/src/data/demoTransactions.json`, generated from
the sanitised sample by:

```powershell
python scripts/export_demo_transactions.py
```

That script writes the 30 features only and drops the `Class` column, so the label
is never bundled or sent. The API would reject it anyway (`extra="forbid"`), but
the point is that the dashboard shows the model's decisions rather than comparing
them to the dataset's answers. The 12-row sample contains fraud rows the model
misses — the first one scores well below the threshold — so a replay can add fewer
alerts than the sample has fraud rows. That is real model behaviour, not a bug.

While exploring, a throwaway database keeps demo writes out of Neon:

```powershell
$env:DATABASE_URL = "sqlite:///./artifacts/dashboard_check.db"
python scripts/init_db.py
python -m uvicorn app.main:app --reload
```

A SQLite file is enough because the schema is portable; restore the Neon URL by
removing the variable `(Remove-Item Env:DATABASE_URL)`.

## Containers (Docker)

The repository ships two images and a Compose stack for reproducible local runs:

| Image | Base | Contents | Serves |
| --- | --- | --- | --- |
| `fraudguard-backend` | `python:3.13-slim-bookworm` | `app/`, `scripts/init_db.py`, `artifacts/models/fraud_model.joblib` | Uvicorn on port 8000, as non-root user `appuser` (uid 10001) |
| `fraudguard-frontend` | `node:24-alpine` build → `nginx:1.29-alpine` | the static `dist/` bundle only | Nginx on port 80 (published as 8080) |

Neither image contains `.env`, `data/`, `notebooks/`, `tests/`, `.venv/`,
`node_modules/` or Git metadata; `.dockerignore` enforces that shape and the CI
workflow asserts the exclusion after building.

The backend image is large because `requirements.txt` also carries the EDA,
notebook, training and load-testing stacks. That file is used unchanged so image
dependencies match the versions everything else was verified against; splitting a
runtime-only requirements file is a separate change.

### Prerequisites

- Docker Desktop 4.x with Compose v2 (`docker compose version`).
- A trained artifact at `artifacts/models/fraud_model.joblib`. It is generated by
  `python scripts/train_models.py` and is git-ignored, so a fresh clone must train
  (or copy) it before the backend image can build.
- Optional: a `DATABASE_URL` in `.env` for Neon. Without it the API still scores
  transactions and stores no alerts — the documented Phase 5 behaviour.

### Build and start

```powershell
docker compose build
docker compose up            # foreground, logs from both services
docker compose up -d         # detached
```

Compose starts the dashboard only after the backend's health check passes, and the
health check returns 200 only when the model is resident.

To build the images individually:

```powershell
docker build -t fraudguard-backend .
docker build -t fraudguard-frontend --build-arg VITE_API_BASE_URL=http://127.0.0.1:8000 ./frontend
docker run --rm -p 8000:8000 -e DATABASE_URL= fraudguard-backend
```

### Local URLs

| What | URL |
| --- | --- |
| Dashboard | http://localhost:8080 (also http://127.0.0.1:8080) |
| API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Health | http://localhost:8000/health |

`VITE_API_BASE_URL` is inlined into the JavaScript bundle by Vite, so it must be a
URL the **browser** can reach. The Compose stack builds it as
`http://127.0.0.1:8000` (the published backend port), never `http://backend:8000`,
which only resolves inside the Docker network.

### Environment configuration

Compose interpolates the root `.env` (git-ignored) at run time; no value is copied
into an image layer.

| Variable | Default in Compose | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | empty | Neon pooled URL; empty disables persistence |
| `CORS_ORIGINS` | `http://localhost:8080,http://127.0.0.1:8080` | the Nginx-served dashboard origin |
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | browser-visible API URL (build arg, public) |
| `APP_ENV`, `LOG_LEVEL` | `development`, `INFO` | logging and environment label |
| `DB_CONNECT_TIMEOUT`, `DB_ECHO` | `10`, `false` | database behaviour |

Changing `VITE_API_BASE_URL` requires rebuilding the frontend image, because the
value lives in the built assets: `docker compose build frontend`.

### SQLite instead of Neon

For a throwaway database that never touches Neon:

```powershell
docker compose -f docker-compose.yml -f docker-compose.sqlite.yml up --build
```

The override points the API at `sqlite:////data/fraudguard.db` in the named volume
`sqlite-data`, runs the idempotent `scripts/init_db.py` before starting Uvicorn,
and leaves the default stack untouched. The file persists across
`docker compose down`; delete it (and the data) with:

```powershell
docker compose -f docker-compose.yml -f docker-compose.sqlite.yml down -v
```

### Stop and inspect

```powershell
docker compose down           # stop and remove containers/networks
docker compose down -v        # also delete the SQLite volume (if used)
docker compose ps             # container and health status
docker compose logs -f backend
```

## Continuous integration (GitHub Actions)

`.github/workflows/ci.yml` runs on every push and pull request. It needs no
secrets and deploys nothing.

| Job | What it does |
| --- | --- |
| `backend-tests` | Python 3.13, `pip install -r requirements.txt`, `python -m pytest -q`, with `DATABASE_URL` forced empty |
| `frontend-build` | Node 24, `npm ci`, `npm run build` (`tsc --noEmit && vite build`), then scans `dist/` for `DATABASE_URL`, `postgresql://` or `neon.tech` strings |
| `containers` | `docker compose config`, builds and smoke-tests the frontend image (app shell plus SPA fallback), and builds/smoke-tests the backend image **when the model artifact is present** |

The backend image is skipped on a fresh checkout because `artifacts/*` is
git-ignored: the model is not in the repository by design. Supplying it to CI
(private artifact download, registry image, or versioning the bundle) is a
Phase 9 packaging decision. Output includes a workflow notice when that happens.

## Run the tests

```powershell
python -m pytest -q                       # 92 tests, no database required
cd frontend; npm run build                # type-check + production bundle
```

Load testing (see `tests/load/locustfile.py`) runs against a local API on port
8010, never against Neon:

```powershell
python -m locust -f tests/load/locustfile.py --headless -u 10 -r 5 -t 30s `
    --host http://127.0.0.1:8010 --csv artifacts/performance/locust_10u --only-summary
```

## Local Docker vs the eventual Vercel deployment

This stack is for local reproducibility, not a preview of Vercel's runtime:

| | Local Compose (Phase 8) | Vercel (Phase 9) |
| --- | --- | --- |
| Dashboard | Nginx container serving the built SPA | static build on Vercel's CDN |
| API | long-running Uvicorn container, one worker | serverless function (or container), cold starts, per-instance memory |
| Model artifact | baked into the image at build time | must fit the deployment bundle and cold-start budget |
| `/metrics-lite` | per-container counters | per-instance counters; a cold start resets them |
| Database | Neon via `DATABASE_URL`, or SQLite volume | Neon pooled URL from project environment variables |
| CORS | `http://localhost:8080` | the deployed dashboard origin |
| Transport | plain HTTP on localhost | HTTPS only |
| Secrets | root `.env`, never baked in | Vercel project environment variables |

Nothing in Phase 8 deploys to Vercel; the images exist to make the application
runnable and testable in one command.

## Initialize Git

Only needed once, if the folder is not already a repository:

```powershell
git init
git add .
git status
git commit -m "Initialize FraudGuard Phase 1"
```

`.gitignore` excludes local secrets (`.env`), virtual environments, raw and
processed datasets, generated model artifacts, logs and caches, plus the
frontend's `node_modules/`, `dist/` and local `.env` files. Commit both
`.env.example` templates as configuration references; never commit a real `.env`.

## API routes

| Method | Path            | Description                                  | Response                          |
| ------ | --------------- | -------------------------------------------- | --------------------------------- |
| GET    | `/`             | Service welcome message                      | `{"message": "..."}`              |
| GET    | `/health`       | API + model health (503 when degraded)       | `{"status": "ok", ...}`           |
| POST   | `/predict`      | Score one transaction                        | `{"fraud_probability", "decision", ...}` |
| GET    | `/metrics-lite` | In-memory per-process counters               | `{"total_requests", ...}`          |
| GET    | `/flagged`      | Stored fraud alerts, newest first, paginated | `{"total", "limit", "offset", "items"}` |

### Scoring a transaction

`POST /predict` accepts exactly the 30 model features (`Time`, `V1`-`V28`,
`Amount`). The `Class` label is rejected, NaN and infinity are rejected, and
unknown fields are rejected. Build a payload from the demo sample in memory, so
no temp file (and no PowerShell encoding surprise) is involved:

```powershell
$payload = (python -c "import json,pandas as pd; d=pd.read_csv('artifacts/models/demo_transactions.csv'); r=d[d.Class==1].iloc[0]; print(json.dumps({k: float(r[k]) for k in d.columns if k!='Class'}))" ) -join ''
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -Body $payload -ContentType 'application/json'
```

The response looks like:

```json
{
  "fraud_probability": 0.9999,
  "decision": "fraud",
  "threshold": 0.538307,
  "latency_ms": 1.8
}
```

The decision is `fraud` when `fraud_probability >= threshold`. The threshold comes
from the trained artifact and is not adjustable per request.

`/metrics-lite` counters are held in process memory only: they reset on restart
and are not aggregated across instances, so they are a smoke-test aid rather than
a monitoring solution.

## Out of scope for Phase 1

- Loading the Kaggle fraud dataset
- Exploratory data analysis
- Model training and serialization
- A `/predict` endpoint
- Database connection and persistence
- Frontend implementation
