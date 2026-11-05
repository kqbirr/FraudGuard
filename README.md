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
├── frontend/            # Reserved for the future React + TypeScript + Vite app
├── artifacts/           # Trained model files and generated outputs (not committed)
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

`app/config.py` reads `APP_NAME`, `APP_ENV`, `LOG_LEVEL`, `DATABASE_URL` and
`MODEL_PATH` from the environment (via `python-dotenv`). The database URL in the
example is a placeholder; no database connection is opened in this phase.

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

The raw JSON body is `{"status":"ok"}`. You can confirm the exact JSON with:

```powershell
(Invoke-WebRequest -Uri http://127.0.0.1:8000/health).Content
```

Check the root endpoint:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/
```

## Initialize Git

Only needed once, if the folder is not already a repository:

```powershell
git init
git add .
git status
git commit -m "Initialize FraudGuard Phase 1"
```

`.gitignore` excludes local secrets (`.env`), virtual environments, raw and
processed datasets, generated model artifacts, logs and caches. Commit
`.env.example` as the configuration template; never commit `.env`.

## Initial API routes

| Method | Path      | Description                          | Response              |
| ------ | --------- | ------------------------------------ | --------------------- |
| GET    | `/`       | Service welcome message              | `{"message": "..."}`  |
| GET    | `/health` | Process health check                 | `{"status": "ok"}`    |

## Out of scope for Phase 1

- Loading the Kaggle fraud dataset
- Exploratory data analysis
- Model training and serialization
- A `/predict` endpoint
- Database connection and persistence
- Frontend implementation
