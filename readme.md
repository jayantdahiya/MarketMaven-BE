# MarketMaven-BE

Backend for the Market Maven forecasting application: config-driven LSTM baseline (Phase 0), PyTorch, date-based train/val/test splits, and FastAPI.

## Architecture

- Frontend sends requests to the backend (e.g. via Cloudflare Tunnel / NGINX).
- FastAPI serves the API; entrypoint is `api.app:app`.
- Daily forecasts use a trained LSTM or legacy Prophet; no training at request time.

## Installation and setup

1. Clone this repository.
2. Install dependencies:
   - **With uv:** `uv sync`
   - **With pip:** `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and set `SUPABASE_URL`, `SUPABASE_KEY`, and optionally `REDIS_*` if using tickers cache.

## Running the API

```bash
uv run uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Or with pip:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for the OpenAPI UI.

## Main endpoints

- **POST /forecast/daily** — Request body: `asset_id`, `horizon_days` (1, 5, or 10), `model` (`lstm_baseline` or `prophet`), optional `as_of_date`. Returns predicted returns and signals.
- **GET /prophet?ticker=...** — Legacy Prophet forecast for the given ticker.
- **GET /**, **/signup**, **/login**, **/logout**, **GET /tickers**, **GET /metrics/latest** — As implemented in the app.

## Scripts (Phase 0)

- **Fetch raw data:** `uv run python scripts/fetch_market_data.py --config config/phase_0.yaml` → writes `artifacts/data/phase0/raw_daily.parquet`
- **Train:** `uv run python scripts/train.py --config config/phase_0.yaml` (optional `--seed N` for a single seed)
- **Evaluate:** `uv run python scripts/evaluate.py --config config/phase_0.yaml --checkpoint <path>` → writes reports under `artifacts/reports/phase0/`

## Docker

```bash
docker compose up --build
```

Uses `api.app:app` on port 8000; mounts `./config` and `./artifacts`.
