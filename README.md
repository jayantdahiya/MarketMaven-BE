# MarketMaven-BE

Production-grade FastAPI backend for stock market forecasting with PyTorch deep learning models. Serves trained model predictions via REST API — no training at request time.

Currently implements three model architectures across three completed phases:

| Phase | Model | Architecture | Status |
|-------|-------|-------------|--------|
| 0 | `lstm_baseline` | 2-layer LSTM + LayerNorm + Dropout + Linear head | Complete |
| 1 | `cnn_transformer` | 2x Conv1D + 3x TransformerEncoder + attention pooling + MLP | Complete |
| 2 | `mamba_ssm` | 4x MambaBlock (S4D selective scan) + asset-correlation graph fusion + MLP | Complete |
| 3 | Multimodal | Sentiment, VIX, LLM-generated alphas, gated modality fusion | Planned |
| 4 | LOB / Microstructure | Limit Order Book pipeline + TLOBForecaster for intraday | Planned |

Each phase produces a self-contained, deployable increment. Rollback between phases is config-only.

## Architecture

```
api/
  app.py                  # FastAPI app factory
  main.py                 # uvicorn entrypoint
  routes/                 # forecast, auth, tickers, metrics
  services/               # forecast_service, auth_service, ticker_service
  models/                 # lstm_baseline, cnn_transformer, mamba_ssm, prophet
  data/                   # sources, features, targets, normalization, splitters,
                          #   datasets, pipeline, asset_graph
  training/               # train_loop, losses, checkpointing, metrics, backtest
  schemas/                # Pydantic request/response models

config/                   # YAML configs: base.yaml, phase_0/1/2.yaml
scripts/                  # fetch_market_data, train, evaluate, benchmark
artifacts/                # data/, checkpoints/, reports/ (generated)
tests/                    # unit/ and integration/ (68 tests)
```

Key design decisions:
- **Date-based train/val/test splits** (train <= 2021, val <= 2023, test <= 2025) — never percentage-based
- **Scalers fit on train only**, then applied to val/test
- **Composite loss**: MSE + Sharpe surrogate (with configurable rolling window)
- **Multi-seed robustness**: 5 seeds `[42, 123, 456, 789, 1024]` by default
- **5 assets**: AAPL, MSFT, GOOGL, AMZN, SPY with 8 features (close, volume, sma_15, sma_45, rsi_14, macd, macd_signal, obv)

## Setup

1. Clone the repository
2. Install dependencies (uses [uv](https://docs.astral.sh/uv/)):
   ```bash
   uv sync
   ```
3. Copy `.env.example` to `.env` and set:
   - **Required**: `SUPABASE_URL`, `SUPABASE_KEY`
   - **Optional**: `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`

### Supabase ticker migration

If `/tickers` returns a Supabase `PGRST205` error, apply the migration in
`supabase/migrations/202603150001_create_public_tickers.sql`.

- Supabase SQL Editor: paste the migration and run it.
- Supabase CLI (if configured in this repo): `supabase db push`

## Running the API

```bash
# Development with hot reload
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# Or via script
python api/main.py

# Or with Docker
docker compose up
```

Interactive docs at **http://localhost:8000/docs**

## Running Frontend + Backend Together

Use the root startup script to run both services in one command:

```bash
./start.sh
```

This starts:
- Backend API at `http://localhost:8000`
- Frontend app at `http://localhost:3000`

Press `Ctrl+C` to stop both processes.

If backend startup fails with `uvicorn: command not found`, install dependencies first:

```bash
uv sync
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/forecast/daily` | Daily return forecast. Body: `asset_id`, `model` (`lstm_baseline`, `cnn_transformer`, `mamba_ssm`, or `prophet`), optional `horizon_days`, `as_of_date` |
| GET | `/prophet` | Legacy Prophet forecast. Query: `ticker` |
| POST | `/signup`, `/login` | Auth (Supabase) |
| GET | `/tickers` | Ticker list (Supabase + optional Redis cache) |
| GET | `/metrics/latest` | Latest evaluation metrics (MAE, RMSE, Sharpe) |

## Training and Evaluation

All training is offline. The API loads from checkpoint files only.

```bash
# 1. Fetch OHLCV data
python scripts/fetch_market_data.py --config config/phase_0.yaml

# 2. Train (runs all 5 seeds by default)
python scripts/train.py --config config/phase_0.yaml
python scripts/train.py --config config/phase_1.yaml
python scripts/train.py --config config/phase_2.yaml

# Single seed
python scripts/train.py --config config/phase_2.yaml --seed 42

# 3. Evaluate a checkpoint
python scripts/evaluate.py --config config/phase_0.yaml \
  --checkpoint artifacts/checkpoints/phase0/seed_42_*/best_epoch_*.pt

# 4. Cross-model benchmark
python scripts/benchmark.py
```

## Models

### Phase 0 — LSTM Baseline
Config: `config/phase_0.yaml` | seq_len=60 | ReduceLROnPlateau scheduler

Standard 2-layer LSTM with LayerNorm, dropout, and linear head. Serves as the performance baseline.

### Phase 1 — CNN-Transformer
Config: `config/phase_1.yaml` | seq_len=60 | CosineAnnealingLR scheduler

Two Conv1D layers (kernel=3, GELU+BN) extract local patterns, followed by 3 TransformerEncoder layers with learnable attention pooling.

Acceptance: MAE <= 0.97 x Phase0 MAE, Sharpe >= Phase0 Sharpe + 0.10

### Phase 2 — Mamba/SSM with Graph Fusion
Config: `config/phase_2.yaml` | seq_len=90 | OneCycleLR scheduler

Four MambaBlock layers using selective state-space modeling (S4D). Optionally fuses asset-correlation graph context (60-day rolling Pearson correlation, threshold >= 0.5) via a learned projection.

- **CUDA**: Uses `mamba-ssm` package when available
- **CPU/MPS**: Falls back to pure-PyTorch S4D implementation automatically
- **Graph features**: degree centrality, 20-day rolling return mean, 20-day rolling volatility

Acceptance: MAE <= 0.99 x Phase1 MAE, Sharpe >= Phase1 Sharpe + 0.05

## Tests

```bash
# Full suite (68 tests)
pytest

# By category
pytest tests/unit/
pytest tests/integration/

# Specific test
pytest tests/unit/test_phase2_model.py -k "test_forward_shape_with_graph"
```

## Linting and Formatting

Uses [Ruff](https://docs.astral.sh/ruff/) (configured in `pyproject.toml`):

```bash
ruff check .          # Lint
ruff format .         # Format
ruff check . --fix    # Lint with auto-fix
```

Optional pre-commit hook:
```bash
git config core.hooksPath .githooks
```

## Docker

```bash
docker build -t market-maven-be .
docker compose up    # Mounts ./config and ./artifacts
```

## Configuration

All hyperparameters, paths, and feature definitions live in YAML configs under `config/`. Phase configs extend `config/base.yaml`:

```yaml
data:
  assets: [AAPL, MSFT, GOOGL, AMZN, SPY]
  seq_len: 90
  feature_cols: [close, volume, sma_15, sma_45, rsi_14, macd, macd_signal, obv]
  split:
    train_end: "2021-12-31"
    val_end: "2023-12-31"
    test_end: "2025-12-31"

model:
  type: mamba_ssm
  d_model: 128
  num_layers: 4

training:
  epochs: 55
  batch_size: 64
  seeds: [42, 123, 456, 789, 1024]
  loss: { type: mse_plus_sharpe, lambda_sharpe: 0.12, sharpe_window: 20 }
  scheduler: { type: one_cycle_lr, max_lr: 0.0006 }
```
