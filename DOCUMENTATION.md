# MarketMaven-BE — Full Codebase Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Tech Stack](#3-tech-stack)
4. [Environment Setup](#4-environment-setup)
5. [Configuration System](#5-configuration-system)
6. [Data Pipeline](#6-data-pipeline)
7. [ML Models](#7-ml-models)
8. [Training Pipeline](#8-training-pipeline)
9. [API — Application Startup](#9-api--application-startup)
10. [API Routes Reference](#10-api-routes-reference)
    - [GET /](#get-)
    - [POST /signup](#post-signup)
    - [POST /login](#post-login)
    - [GET /logout](#get-logout)
    - [GET /tickers](#get-tickers)
    - [POST /forecast/daily](#post-forecastdaily)
    - [POST /forecast/multimodal](#post-forecastmultimodal)
    - [GET /forecast/prophet](#get-forecastprophet)
    - [GET /alphas](#get-alphas)
    - [GET /metrics/latest](#get-metricslatest)
    - [POST /forecast/lob](#post-forecastlob)
11. [Scripts](#11-scripts)
12. [Testing](#12-testing)
13. [Linting and Formatting](#13-linting-and-formatting)
14. [Docker](#14-docker)
15. [Project Phases Roadmap](#15-project-phases-roadmap)

---

## 1. Project Overview

MarketMaven-BE is a FastAPI-based stock market forecasting backend that serves trained deep learning model predictions over a REST API. The system is designed around an offline training / online inference separation: models are trained as a batch process and serialised to checkpoint files; at runtime the API simply loads these checkpoints and runs inference on request.

Key characteristics:

- **No online training** — the API never trains; it only loads pre-trained `best_epoch_*.pt` files.
- **Date-based splits** — train/val/test boundaries are fixed calendar dates (never random percentages) to prevent look-ahead bias.
- **Multi-seed robustness** — every training run is repeated across 5 seeds to produce stable results.
- **Phased roadmap** — Phase 0 (LSTM baseline) is currently active; later phases add CNN-Transformer, Mamba/SSM, multimodal fusion, and LOB pipelines, all switchable via YAML config.

---

## 2. Repository Structure

```
MarketMaven-BE/
├── api/                        # All production Python source code
│   ├── app.py                  # FastAPI app factory (CORS, routers, services)
│   ├── main.py                 # Uvicorn entrypoint
│   ├── routes/
│   │   ├── forecast.py         # POST /forecast/daily, /multimodal; GET /forecast/prophet
│   │   ├── auth.py             # POST /signup, /login; GET /logout
│   │   ├── tickers.py          # GET /tickers
│   │   ├── metrics.py          # GET /metrics/latest
│   │   ├── alphas.py           # GET /alphas  (Phase 3+)
│   │   └── lob.py              # POST /forecast/lob  (Phase 4, conditional)
│   ├── services/
│   │   ├── forecast_service.py # Model loading, inference, response formatting
│   │   ├── auth_service.py     # Supabase auth wrapper
│   │   ├── ticker_service.py   # Supabase + Redis ticker list
│   │   ├── alpha_service.py    # LLM-based alpha factor generation (Phase 3+)
│   │   └── lob_service.py      # LOB inference service (Phase 4)
│   ├── models/
│   │   ├── factory.py          # MODEL_REGISTRY + create_model()
│   │   ├── lstm_baseline.py    # Phase 0 LSTM model
│   │   ├── cnn_transformer.py  # Phase 1 CNN-Transformer
│   │   ├── mamba_ssm.py        # Phase 2 Mamba/SSM model
│   │   ├── multimodal.py       # Phase 3 multimodal fusion model
│   │   ├── lob_models.py       # Phase 4 LOB models (TLOBForecaster, LobCNNBaseline)
│   │   └── prophet_forecaster.py  # Legacy Prophet wrapper
│   ├── data/
│   │   ├── sources.py          # yfinance OHLCV fetcher
│   │   ├── features.py         # SMA, RSI, MACD, OBV calculations
│   │   ├── targets.py          # Log-return target generation
│   │   ├── normalization.py    # Per-asset z-score scaler (clip=5.0)
│   │   ├── splitters.py        # Date-based train/val/test splitters
│   │   ├── datasets.py         # PyTorch Dataset classes
│   │   └── pipeline.py         # DataPipeline orchestrator + load_config()
│   ├── training/
│   │   ├── train_loop.py       # Trainer: early stopping, grad clipping, LR scheduling
│   │   ├── losses.py           # MSE + Sharpe surrogate composite loss
│   │   └── checkpointing.py    # Checkpoint save/load helpers
│   └── schemas/
│       ├── forecast.py         # DailyForecastRequest/Response, MultimodalForecast*
│       ├── auth.py             # SignUpRequest, LoginRequest, AuthResponse
│       ├── metrics.py          # MetricsSummaryResponse
│       ├── lob.py              # LOBForecastRequest/Response
│       └── alpha.py            # AlphaResponse
├── scripts/
│   ├── fetch_market_data.py    # Pull OHLCV data from yfinance
│   ├── train.py                # Multi-seed training entrypoint
│   ├── evaluate.py             # Checkpoint evaluation + report writer
│   ├── benchmark.py            # Cross-phase model comparison
│   ├── fetch_sentiment.py      # Finnhub news → sentiment parquet
│   ├── generate_alphas.py      # LLM alpha factor batch generation
│   └── build_lob_dataset.py    # LOB event data → numpy arrays
├── config/
│   ├── base.yaml               # Shared defaults
│   └── phase_0.yaml            # Phase 0 (LSTM) full config (inherits base)
├── tests/
│   ├── conftest.py             # Shared pytest fixtures
│   ├── unit/                   # Unit tests (15 files)
│   └── integration/            # Integration tests (9 files)
├── supabase/
│   └── migrations/
│       └── 202603150001_create_public_tickers.sql
├── artifacts/                  # Generated at runtime (gitignored)
│   ├── data/
│   ├── checkpoints/
│   └── reports/
├── .env.example
├── pyproject.toml
├── docker-compose.yml
└── Dockerfile
```

---

## 3. Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI + Uvicorn |
| ML framework | PyTorch |
| Data fetching | yfinance |
| Legacy forecasting | Prophet |
| Database / Auth | Supabase (PostgreSQL + GoTrue) |
| Caching | Redis |
| LLM alpha factors | OpenAI API |
| Experiment tracking | MLflow (optional) |
| Package manager | `uv` (not pip) |
| Linting / formatting | Ruff |
| Testing | pytest |
| Containerisation | Docker + Docker Compose |

---

## 4. Environment Setup

### Prerequisites

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) package manager
- A Supabase project (for auth and ticker list)
- Redis instance (optional; falls back gracefully)

### Step-by-step

```bash
# 1. Clone the repository
git clone <repo-url>
cd MarketMaven-BE

# 2. Copy and populate environment variables
cp .env.example .env
# Edit .env — set SUPABASE_URL and SUPABASE_KEY at minimum

# 3. Install all dependencies
uv sync

# 4. (Optional) Apply the Supabase database migration
# Run supabase/migrations/202603150001_create_public_tickers.sql
# in your Supabase SQL editor to create the tickers table

# 5. Fetch historical market data
python scripts/fetch_market_data.py

# 6. Train the model
python scripts/train.py --config config/phase_0.yaml

# 7. Start the API
python api/main.py
# OR with hot-reload:
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
# OR via Docker:
docker compose up
```

### `.env` variables

| Variable | Required | Description |
|---|---|---|
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_KEY` | Yes | Supabase anon/service key |
| `REDIS_HOST` | No | Redis hostname (default: `localhost`) |
| `REDIS_PORT` | No | Redis port (default: `6379`) |
| `REDIS_PASSWORD` | No | Redis password |
| `ALLOWED_ORIGINS` | No | Comma-separated CORS origins |

---

## 5. Configuration System

All configuration is YAML-based. `config/base.yaml` defines shared defaults; phase-specific files such as `config/phase_0.yaml` override individual keys and reference `_base_: config/base.yaml`.

### Key sections of `config/phase_0.yaml`

```yaml
data:
  assets: [AAPL, MSFT, GOOGL, AMZN, SPY]   # 5 tracked symbols
  seq_len: 60                                 # lookback window (60 trading days)
  feature_cols: [close, volume, sma_15, sma_45, rsi_14, macd, macd_signal, obv]
  split:
    train_end: "2021-12-31"
    val_end:   "2023-12-31"
    test_end:  "2025-12-31"

model:
  type: lstm_baseline
  input_dim: 8
  hidden_dim: 128
  num_layers: 2
  dropout: 0.2

training:
  epochs: 40
  batch_size: 64
  lr: 0.001
  early_stopping_patience: 8
  loss:
    type: mse_plus_sharpe
    lambda_sharpe: 0.1
    warmup_epochs: 3      # first 3 epochs are MSE-only
  seeds: [42, 123, 456, 789, 1024]
```

The API reads this config at startup via `api/data/pipeline.load_config()` and stores it on `app.state.config`.

---

## 6. Data Pipeline

The data pipeline lives entirely in `api/data/` and is orchestrated by `DataPipeline` (`pipeline.py`).

### Stages

```
yfinance (sources.py)
  → raw OHLCV DataFrame
  → features.py   (add_technical_features)
       SMA-15, SMA-45, RSI-14, MACD, MACD-signal, OBV
  → targets.py    (add_log_return_targets)
       log_return_1d = log(close_t / close_t-1)
  → splitters.py  (date-based split — NO random shuffle)
       train ≤ 2021-12-31 / val ≤ 2023-12-31 / test ≤ 2025-12-31
  → normalization.py (FeatureScaler)
       per-asset z-score, fitted on train only, clipped at ±5σ
       scaler state is serialised into every checkpoint
  → datasets.py   (StockDataset — PyTorch Dataset)
       sliding windows of shape [T, F] → target scalar
```

### Inference window

At request time `ForecastService` calls `pipeline.build_inference_window(asset_id, as_of_date, scaler)`, which fetches the most recent 60-day window for the asset, applies the saved scaler from the checkpoint, and returns a `[60, 8]` numpy array ready for the model.

---

## 7. ML Models

All models are registered in `api/models/factory.py` via `MODEL_REGISTRY` and instantiated with `create_model(name, config)`.

| Registry key | Class | Phase | Input shape | Output |
|---|---|---|---|---|
| `lstm_baseline` | `LSTMBaseline` | 0 | `[B, T, 8]` | `[B, 1]` log-return |
| `cnn_transformer` | `CNNTransformer` | 1 | `[B, T, F]` | `[B, 1]` log-return |
| `mamba_ssm` | `MambaSSM` | 2 | `[B, T, F]` | `[B, 1]` log-return |
| `cnn_transformer_multimodal` | `MultimodalCNNTransformer` | 3 | `[B, T, F]` + mask | `[B, 1]` log-return |
| `tlob_forecaster` | `TLOBForecaster` | 4 | LOB book tensor | `[B, 3]` class probs |
| `lob_cnn` | `LobCNNBaseline` | 4 | LOB book tensor | `[B, 3]` class probs |

### Phase 0 — LSTM Baseline

A two-layer LSTM followed by a linear projection head. Processes the full 60-step sequence and reads the hidden state at the final timestep to predict the next-day log return.

### Phase 1 — CNN-Transformer

Applies a 1D convolutional front-end to extract local patterns, then feeds them into a Transformer encoder. Attention weights across timesteps are verified to sum to 1. Supports variable horizon.

### Phase 2 — Mamba/SSM

State-space model optionally augmented by a correlation graph. When `graph.enabled: true` in config, a relational context tensor (`[A, 3]`) describing inter-asset correlations is passed alongside the sequence.

### Phase 3 — Multimodal Fusion

Extends the CNN-Transformer with gated modality fusion. Accepts four modality streams (price+technicals, macro context, sentiment, LLM alpha factors) controlled by a `[1, 4]` binary mask supplied per request.

### Phase 4 — LOB Models

Two architectures for mid-price direction forecasting from limit-order-book snapshots, outputting a 3-class probability distribution (down / flat / up).

---

## 8. Training Pipeline

Training is handled offline via `scripts/train.py`. It is never triggered by API requests.

```bash
# Fetch data first (if not already done)
python scripts/fetch_market_data.py

# Train all 5 seeds for Phase 0
python scripts/train.py --config config/phase_0.yaml

# Train a single seed
python scripts/train.py --config config/phase_0.yaml --seed 42
```

### What happens during training

1. `DataPipeline` loads raw data, builds features, splits by date, fits the scaler on the train split only.
2. `Trainer` (in `api/training/train_loop.py`) runs the training loop with:
   - **Composite loss**: MSE + 0.1 × Sharpe surrogate (first 3 epochs are MSE-only warm-up).
   - **Gradient clipping**: `norm=1.0`.
   - **LR scheduling**: `ReduceLROnPlateau` (factor 0.5, patience 3, min_lr 1e-6).
   - **Early stopping**: patience 8 epochs on validation loss.
3. `checkpointing.save_checkpoint()` saves `best_epoch_<N>.pt` under `artifacts/checkpoints/phase0/seed_<SEED>_<timestamp>/`, embedding the scaler state so inference never needs the raw training data.

### Evaluate a checkpoint

```bash
python scripts/evaluate.py \
  --config config/phase_0.yaml \
  --checkpoint artifacts/checkpoints/phase0/seed_42_<run>/best_epoch_<N>.pt
```

---

## 9. API — Application Startup

The app is assembled by the `create_app()` factory in `api/app.py`. On startup it:

1. Loads `.env` via `python-dotenv`.
2. Configures CORS from `ALLOWED_ORIGINS` env var (defaults to localhost variants).
3. Reads `config/phase_0.yaml` and stores it on `app.state.config`.
4. Instantiates `ForecastService` (loads no model yet — models are lazy-loaded on first request).
5. Connects to Supabase; if credentials exist, instantiates `AuthService` and `TickerService`.
6. Connects to Redis; falls back silently if unavailable.
7. Conditionally creates `AlphaService` (only when `multimodal.alpha.enabled: true` in config).
8. Conditionally imports and registers the LOB router (only when `lob.api_enabled: true` in config).
9. Registers all routers and a root `GET /` health check.

---

## 10. API Routes Reference

Base URL: `http://localhost:8000`

---

### GET /

**Health check / welcome.**

```bash
curl http://localhost:8000/
```

**Response**

```json
{ "message": "Welcome to the Market Maven API" }
```

---

### POST /signup

**Register a new user via Supabase Auth.**

**Request body**

```json
{
  "email": "user@example.com",
  "password": "securepassword"
}
```

| Field | Type | Constraints |
|---|---|---|
| `email` | string | valid email address |
| `password` | string | minimum 8 characters |

**Example**

```bash
curl -X POST http://localhost:8000/signup \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "mypassword123"}'
```

**Response (201)**

```json
{
  "user_id": "uuid-string",
  "access_token": "jwt-token",
  "message": "Signup successful"
}
```

**Under the hood**

`api/routes/auth.py` → `AuthService.signup()` → calls `supabase.auth.sign_up()`. The Supabase GoTrue service creates the user, triggers a confirmation email (depending on your project settings), and returns a session object. The route extracts `user.id` and `session.access_token` from the response and wraps them in `AuthResponse`.

If Supabase credentials are not configured, the service is `None` and the route returns `503`.

---

### POST /login

**Authenticate an existing user.**

**Request body**

```json
{
  "email": "user@example.com",
  "password": "mypassword123"
}
```

**Example**

```bash
curl -X POST http://localhost:8000/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "mypassword123"}'
```

**Response (200)**

```json
{
  "user_id": "uuid-string",
  "access_token": "jwt-token",
  "message": "Login successful"
}
```

**Error (401)**

```json
{ "detail": "Invalid credentials" }
```

**Under the hood**

`api/routes/auth.py` → `AuthService.login()` → calls `supabase.auth.sign_in_with_password()`. The Supabase session contains a short-lived JWT (`access_token`) which the client should include in subsequent requests as a Bearer token. If the credentials are wrong, Supabase raises an exception which the service catches and the route maps to a 401.

---

### GET /logout

**Invalidate the current Supabase session.**

**Example**

```bash
curl http://localhost:8000/logout
```

**Response (200)**

```json
{ "message": "Signed out" }
```

**Under the hood**

`api/routes/auth.py` → `AuthService.logout()` → calls `supabase.auth.sign_out()`, which revokes the refresh token server-side.

---

### GET /tickers

**Return the list of tracked ticker symbols.**

**Example**

```bash
curl http://localhost:8000/tickers
```

**Response (200)**

```json
[
  { "id": 1, "symbol": "AAPL", "name": "Apple Inc.", "is_active": true },
  { "id": 2, "symbol": "MSFT", "name": "Microsoft Corp.", "is_active": true },
  { "id": 3, "symbol": "GOOGL", "name": "Alphabet Inc.", "is_active": true },
  { "id": 4, "symbol": "AMZN", "name": "Amazon.com Inc.", "is_active": true },
  { "id": 5, "symbol": "SPY", "name": "SPDR S&P 500 ETF", "is_active": true }
]
```

**Under the hood**

`api/routes/tickers.py` → `TickerService.get_tickers()`:

1. Checks Redis for a cached JSON blob under the key `"tickers"` (TTL: 1 hour).
2. On cache hit, deserialises and returns immediately.
3. On cache miss (or Redis unavailable), queries `supabase.table('tickers').select('*')`.
4. Stores the result in Redis with a 1-hour TTL for subsequent requests.
5. Returns the list of ticker records.

The `public.tickers` table is seeded by the migration in `supabase/migrations/`.

---

### POST /forecast/daily

**Run a single-step price return forecast for a given asset.**

**Request body**

```json
{
  "asset_id": "AAPL",
  "horizon_days": 1,
  "model": "lstm_baseline",
  "as_of_date": "2025-01-15"
}
```

| Field | Type | Allowed values | Default |
|---|---|---|---|
| `asset_id` | string | 1–15 uppercase alphanumeric chars | — (required) |
| `horizon_days` | integer | `1`, `5`, or `10` | `1` |
| `model` | string | `lstm_baseline`, `cnn_transformer`, `mamba_ssm`, `prophet` | `lstm_baseline` |
| `as_of_date` | date (ISO) | any past date | today |

**Example**

```bash
curl -X POST http://localhost:8000/forecast/daily \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "AAPL",
    "horizon_days": 1,
    "model": "lstm_baseline"
  }'
```

**Response (200)**

```json
{
  "asset_id": "AAPL",
  "model": "lstm_baseline",
  "horizon_days": 1,
  "generated_at": "2025-01-15T10:30:00",
  "predictions": [
    {
      "timestamp": "2025-01-16T00:00:00",
      "predicted_return": 0.0042,
      "predicted_price": null,
      "signal": "long",
      "confidence": 0.67
    }
  ]
}
```

**Error codes**

| HTTP | `code` | Cause |
|---|---|---|
| 400 | `invalid_model` | `model` not in allowed list |
| 422 | — | Pydantic validation failure (bad `asset_id`, invalid `horizon_days`) |
| 503 | `model_unavailable` | Checkpoint file not found or model load error |
| 502 | `data_unavailable` | Insufficient market data for the inference window |

**Under the hood**

```
POST /forecast/daily
  ↓
api/routes/forecast.py   — validates model name, delegates to service
  ↓
ForecastService.predict_daily()
  ↓
  1. _load_model(model_name)
       - checks _model_cache (avoids reloading on every request)
       - calls create_model() from MODEL_REGISTRY to build architecture
       - scans artifacts/checkpoints/phase0/ for the latest run dir
       - finds best_epoch_*.pt inside that run dir
       - loads checkpoint via checkpointing.load_checkpoint()
       - restores FeatureScaler from checkpoint's scaler_state
       - sets model.eval()
  ↓
  2. pipeline.build_inference_window(asset_id, as_of_date, scaler)
       - fetches recent OHLCV via yfinance
       - computes SMA, RSI, MACD, OBV features
       - applies saved scaler (z-score, clip ±5σ)
       - returns numpy array [60, 8]
  ↓
  3. torch.no_grad() forward pass → scalar log-return prediction
  ↓
  4. _build_signal():  signal="long" if pred > threshold(0.0) else "flat"
                       confidence = sigmoid(|pred| / 0.01)
  ↓
  5. Wraps in DailyForecastResponse and returns
```

---

### POST /forecast/multimodal

**Run a multimodal forecast using Phase 3 model with selectable modality streams.**

> This route is only active when `multimodal.enabled: true` in the config.

**Request body**

```json
{
  "asset_id": "MSFT",
  "horizon_days": 1,
  "model": "cnn_transformer",
  "include_context": true,
  "include_sentiment": true,
  "include_alpha": true,
  "as_of_date": "2025-01-15"
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `asset_id` | string | — (required) | Ticker symbol |
| `horizon_days` | int | `1` | Must be 1, 5, or 10 |
| `model` | `cnn_transformer` or `mamba_ssm` | `cnn_transformer` | Base architecture |
| `include_context` | bool | `true` | Include macro context modality |
| `include_sentiment` | bool | `true` | Include news sentiment modality |
| `include_alpha` | bool | `true` | Include LLM alpha factors modality |

**Example**

```bash
curl -X POST http://localhost:8000/forecast/multimodal \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "MSFT",
    "horizon_days": 1,
    "include_context": true,
    "include_sentiment": false,
    "include_alpha": true
  }'
```

**Response (200)**

```json
{
  "asset_id": "MSFT",
  "model": "cnn_transformer_multimodal",
  "horizon_days": 1,
  "generated_at": "2025-01-15T10:30:00",
  "predictions": [
    {
      "timestamp": "2025-01-16T00:00:00",
      "predicted_return": 0.0031,
      "predicted_price": null,
      "signal": "long",
      "confidence": 0.62
    }
  ],
  "used_modalities": ["price_tech", "context", "alpha"],
  "alpha_version": null
}
```

**Error codes**

| HTTP | `code` | Cause |
|---|---|---|
| 400 | `no_modalities` | All `include_*` flags are `false` |
| 404 | `multimodal_disabled` | `multimodal.enabled` is `false` in config |
| 503 | `model_unavailable` | Checkpoint not found |

**Under the hood**

```
POST /forecast/multimodal
  ↓
api/routes/forecast.py   — checks multimodal enabled, at least one modality active
  ↓
ForecastService.predict_multimodal()
  ↓
  1. Loads cnn_transformer_multimodal checkpoint (same lazy-cache mechanism)
  ↓
  2. Builds 60-day inference window (same as /forecast/daily)
  ↓
  3. Constructs a modality_mask tensor [1, 4]:
       [1.0,  # price_tech — always 1
        1.0 if include_context else 0.0,
        1.0 if include_sentiment else 0.0,
        1.0 if include_alpha else 0.0]
  ↓
  4. torch.no_grad() forward pass with modality_mask
       — the MultimodalCNNTransformer gates each modality stream by this mask
       — zeroed-out modalities contribute nothing to the fused representation
  ↓
  5. Builds signal and response; lists used_modalities from the request flags
```

---

### GET /forecast/prophet

**Legacy time-series forecast using Facebook Prophet.**

**Query parameters**

| Parameter | Type | Description |
|---|---|---|
| `ticker` | string | Ticker symbol, e.g. `AAPL` |

**Example**

```bash
curl "http://localhost:8000/forecast/prophet?ticker=AAPL"
```

**Response (200)**

```json
{
  "timestamps": ["2025-01-16", "2025-01-17", "..."],
  "trend": [228.5, 229.1, "..."]
}
```

**Under the hood**

`api/routes/forecast.py (GET /forecast/prophet)` → `ForecastService.predict_daily()` with `model='prophet'` → `forecast_prophet()` in `api/models/prophet_forecaster.py`. Prophet is a curve-fitting model that decomposes the price series into trend, seasonality, and holiday components, then extrapolates forward. Unlike the deep learning models, Prophet does not require a pre-trained checkpoint — it fits on the fly from historical yfinance data.

This endpoint can be disabled by setting `allow_legacy_prophet: false` in config, which causes it to return `410 Gone`.

---

### GET /alphas

**Retrieve LLM-generated alpha factors for an asset (Phase 3+).**

> Only available when `multimodal.alpha.enabled: true` in config.

**Query parameters**

| Parameter | Type | Description |
|---|---|---|
| `asset_id` | string | Ticker symbol |
| `date` | date (ISO) | Date to generate alphas for (defaults to today) |

**Example**

```bash
curl "http://localhost:8000/alphas?asset_id=AAPL&date=2025-01-15"
```

**Response (200)**

```json
{
  "asset_id": "AAPL",
  "date": "2025-01-15",
  "alpha_values": {
    "alpha_1": 0.42,
    "alpha_2": -0.15,
    "alpha_3": 0.08,
    "alpha_4": 0.31,
    "alpha_5": -0.22,
    "alpha_6": 0.17,
    "alpha_7": -0.05,
    "alpha_8": 0.29
  },
  "alpha_version": "v1",
  "cached": true,
  "generated_at": "2025-01-15T09:00:00",
  "rationale": "Momentum strong; sentiment neutral; macro headwinds moderate."
}
```

**Under the hood**

`api/routes/alphas.py` → `AlphaService`:

1. Checks a JSONL cache file (`artifacts/data/phase3/alpha_cache.jsonl`) keyed by `(asset_id, date)`.
2. On cache hit, returns the stored alpha values directly (`cached: true`).
3. On cache miss, calls the OpenAI API (model configurable, default `gpt-4.1-mini`) with a structured prompt asking for 8 alpha factors in `[-1, 1]`.
4. On LLM error, falls back to `deterministic_fallback_alphas()` which derives rule-based signals from recent price data.
5. Appends new results to the JSONL cache for future requests.

---

### GET /metrics/latest

**Return the most recent evaluation metrics from the reports directory.**

**Query parameters**

| Parameter | Type | Description |
|---|---|---|
| `model` | string (optional) | Model name label to include in the response |

**Example**

```bash
curl "http://localhost:8000/metrics/latest?model=lstm_baseline"
```

**Response (200)**

```json
{
  "mae": 0.0093,
  "rmse": 0.0121,
  "directional_accuracy": 0.543,
  "sharpe": 1.24,
  "sortino": 1.87,
  "max_drawdown": -0.082,
  "model": "lstm_baseline",
  "run_id": "artifacts/reports/phase0/seed_42_20250110"
}
```

If no report files exist, all metric values default to `0.0`.

**Under the hood**

`api/routes/metrics.py` reads `artifacts/reports/phase0/` (path comes from `config.paths.reports_dir`). It recursively scans for `metrics_summary.json` files, picks the most recently modified one, parses it, and returns it as a `MetricsSummaryResponse`. The Sharpe and Sortino values prefer after-cost versions (`sharpe_after_cost`, `sortino_after_cost`) when available.

---

### POST /forecast/lob

**Predict mid-price movement direction from a limit order book snapshot (Phase 4).**

> Only available when `lob.api_enabled: true` in `config/phase_4.yaml`.

**Request body**

```json
{
  "asset_id": "AAPL",
  "as_of_timestamp": "2025-01-15T14:30:00",
  "horizon_events": 20,
  "model": "tlob_forecaster"
}
```

| Field | Type | Allowed values | Default |
|---|---|---|---|
| `asset_id` | string | 1–15 uppercase alphanumeric | — (required) |
| `horizon_events` | int | 1–100 | `20` |
| `model` | string | `tlob_forecaster`, `lob_cnn` | `tlob_forecaster` |

**Example**

```bash
curl -X POST http://localhost:8000/forecast/lob \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "AAPL",
    "horizon_events": 20,
    "model": "tlob_forecaster"
  }'
```

**Response (200)**

```json
{
  "asset_id": "AAPL",
  "as_of_timestamp": "2025-01-15T14:30:00",
  "predicted_class": 2,
  "class_probabilities": [0.12, 0.25, 0.63],
  "expected_mid_move_ticks": 0.38,
  "confidence": 0.63
}
```

The `predicted_class` values are: `0` = price down, `1` = price flat, `2` = price up.

**Error codes**

| HTTP | Cause |
|---|---|
| 503 | `lob.api_enabled` is false or LOB service failed to initialise |
| 422 | Insufficient LOB history for the requested window |

**Under the hood**

```
POST /forecast/lob
  ↓
api/routes/lob.py   — checks lob_service is available
  ↓
LOBService.predict()
  ↓
  1. Loads LOB snapshot data for the asset up to as_of_timestamp
  ↓
  2. Normalises the order book depth features via LOBPipeline
  ↓
  3. Runs forward pass through TLOBForecaster or LobCNNBaseline
       - Output: [B, 3] softmax probabilities over (down, flat, up)
  ↓
  4. Computes expected_mid_move_ticks as weighted sum of class tick values
  ↓
  5. Returns LOBForecastResponse
```

---

## 11. Scripts

All scripts are CLI entrypoints, not imported by the API.

### `scripts/fetch_market_data.py`

Downloads daily OHLCV data for all configured assets from yfinance and saves it to `artifacts/data/phase0/raw_daily.parquet`.

```bash
python scripts/fetch_market_data.py
```

### `scripts/train.py`

Multi-seed training for any phase. Supports optional MLflow tracking and three LR scheduler types (`reduce_lr_on_plateau`, `cosine_annealing`, `one_cycle_lr`). Automatically routes to LOB training when the config task is `lob`.

```bash
# All seeds
python scripts/train.py --config config/phase_0.yaml

# Single seed
python scripts/train.py --config config/phase_0.yaml --seed 42
```

### `scripts/evaluate.py`

Evaluates a checkpoint against the test split and writes a `metrics_summary.json` to the reports directory.

```bash
python scripts/evaluate.py \
  --config config/phase_0.yaml \
  --checkpoint artifacts/checkpoints/phase0/seed_42_<run>/best_epoch_20.pt
```

### `scripts/benchmark.py`

Compares Phase 0 vs Phase 1 (or Phase 2) models on MAE ratio, RMSE ratio, and Sharpe delta. Exits with code `0` (pass), `1` (fail), or `2` (Phase 0 missing).

```bash
python scripts/benchmark.py
```

### `scripts/fetch_sentiment.py`

Fetches news from the Finnhub API, aggregates sentiment per asset per day, and saves to `artifacts/data/phase3/sentiment_daily.parquet`.

### `scripts/generate_alphas.py`

Batch-generates 8 LLM alpha factors for all assets and dates in a range, writing a JSONL cache that the `AlphaService` reads at inference time.

### `scripts/build_lob_dataset.py`

Converts raw LOB event data into ready-to-train numpy arrays: `x_book.npy` (order book features), `x_aux.npy` (auxiliary features), `y_class.npy` (direction labels), and `meta.parquet`.

---

## 12. Testing

Tests live in `tests/` and mirror the production structure.

```bash
# Full suite
pytest

# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Single test file
pytest tests/unit/test_phase0_features.py

# Single test by name
pytest tests/unit/test_phase0_features.py::test_calculate_sma_expected_values
```

### Key fixtures (`tests/conftest.py`)

| Fixture | Description |
|---|---|
| `phase0_cfg` | Full Phase 0 config with paths redirected to `tmp_path` |
| `sample_daily_df` | 500-row synthetic OHLCV DataFrame (2 assets) |
| `tmp_artifact_dir` | Isolated temp directory for artifacts |
| `sample_lstm_model` | Small `LSTMBaseline` (input_dim=8) |

All unit tests use synthetic data — no live API calls or real checkpoints required.

---

## 13. Linting and Formatting

The project uses [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting.

```bash
ruff check .          # Lint
ruff format .         # Format
ruff check . --fix    # Lint and auto-fix
```

**Config** (in `pyproject.toml`):

- Line length: 88 characters
- Quote style: single quotes
- Enabled rules: `E` (pycodestyle), `F` (pyflakes), `B` (bugbear), `I` (isort), `UP` (pyupgrade), `SIM` (simplify)
- `E501` (line too long) is ignored
- `assert` statements allowed in test files

---

## 14. Docker

```bash
# Build and start the API on port 8000
docker compose up

# Rebuild after dependency changes
docker compose up --build
```

The `Dockerfile` installs dependencies via `uv` and starts Uvicorn. The `docker-compose.yml` mounts the project directory and passes through environment variables from `.env`.

---

## 15. Project Phases Roadmap

| Phase | Model | Status | Config |
|---|---|---|---|
| 0 | LSTM Baseline | Active | `config/phase_0.yaml` |
| 1 | CNN-Transformer | Implemented | `config/phase_1.yaml` |
| 2 | Mamba/SSM + Graph | Implemented | `config/phase_2.yaml` |
| 3 | Multimodal Fusion + LLM Alphas | Implemented | `config/phase_3.yaml` |
| 4 | Limit Order Book (LOB) | Implemented | `config/phase_4.yaml` |

Switching phases requires only a config change — no code changes. Each phase's API endpoints are conditionally registered at startup based on the active config flags (`multimodal.enabled`, `lob.api_enabled`).

Rollback is config-only: pointing the active config back to an earlier phase restores the previous behaviour without touching source code or checkpoints.
