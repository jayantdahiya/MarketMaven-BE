# MarketMaven-BE

Backend for the Market Maven forecasting application: config-driven LSTM baseline (Phase 0), PyTorch, date-based train/val/test splits, and FastAPI. Forecasts are served from pre-trained checkpoints; no training at request time.

## Architecture

- Frontend sends requests to the backend (e.g. via Cloudflare Tunnel / NGINX).
- FastAPI serves the API; entrypoint is `api.app:app`.
- Daily forecasts use a trained LSTM (`lstm_baseline`) or legacy Prophet; models are loaded from checkpoints under `artifacts/checkpoints/`.
- Auth and ticker metadata use Supabase; ticker list can be cached in Redis.

## Installation and setup

1. Clone this repository.
2. Install dependencies:
   - **With uv:** `uv sync` (or `uv sync --extra dev` for Ruff and pytest-cov).
   - **With pip:** `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and set `SUPABASE_URL`, `SUPABASE_KEY`, and optionally `REDIS_*` if using the tickers cache.

## Running the API

```bash
uv run uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Or with pip: `uvicorn api.app:app --host 0.0.0.0 --port 8000`

Interactive API docs: **http://localhost:8000/docs**

## Main endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/forecast/daily` | Daily return forecast. Body: `asset_id`, `horizon_days` (1, 5, or 10), `model` (e.g. `lstm_baseline`), optional `as_of_date`. |
| GET | `/prophet` | Legacy Prophet forecast. Query: `ticker`, optional `horizon_days`, `end_date`. |
| POST | `/signup`, `/login` | Auth (Supabase). |
| GET | `/tickers` | Ticker list (Supabase + optional Redis cache). |
| GET | `/metrics/latest` | Latest evaluation metrics summary (e.g. MAE, RMSE, Sharpe). |

Request/response schemas: see OpenAPI at `/docs`.

## Scripts (Phase 0)

- **Fetch raw data:**  
  `uv run python scripts/fetch_market_data.py --config config/phase_0.yaml`  
  → writes `artifacts/data/phase0/raw_daily.parquet`

- **Train model:**  
  `uv run python scripts/train.py --config config/phase_0.yaml`  
  → multi-seed training, checkpoints under `artifacts/checkpoints/phase0/`

- **Evaluate checkpoint:**  
  `uv run python scripts/evaluate.py --config config/phase_0.yaml --checkpoint <path>`  
  → writes `metrics_summary.json`, `regime_metrics.csv`, `backtest_report.csv` under `artifacts/reports/phase0/`

## Linting and formatting (Ruff)

- **Check:** `uv run ruff check .`
- **Format:** `uv run ruff format .`
- **Auto-fix:** `uv run ruff check . --fix`

Ruff is configured in `pyproject.toml` under `[tool.ruff]` and is included in the dev extra.

## Git hooks (optional)

To run Ruff on staged Python files before each commit:

```bash
git config core.hooksPath .githooks
```

The `.githooks/pre-commit` script runs `ruff check --fix` and `ruff format` on staged `.py` files and blocks the commit if violations remain.

## Tests

```bash
uv run pytest tests/unit/ tests/integration/ -v
```

## Docker

- **Build:** `docker build -t market-maven-be .`
- **Run:** `docker compose up` (mounts `./config` and `./artifacts`; see `docker-compose.yml`).

The app runs as `uvicorn api.app:app --host 0.0.0.0 --port 8000` inside the container.
