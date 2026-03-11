# AGENTS.md — Agentic Coding Guidelines for MarketMaven-BE

## Overview

MarketMaven-BE is a FastAPI stock forecasting backend with PyTorch deep learning models. It serves trained model predictions via REST API. Currently in **Phase 0** (LSTM baseline); future phases add CNN-Transformer, Mamba/SSM, multimodal fusion, and LOB pipelines. Each phase is configurable via YAML and rollback is config-only.

---

## Build, Lint & Test Commands

### Package Management (uses uv, not pip)
```bash
uv sync              # Install all dependencies
uv sync --locked     # Install with exact lockfile versions
uv add <package>     # Add a dependency
```

### Running the API
```bash
python api/main.py                                      # Default: localhost:8000
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload  # With hot reload
docker compose up                                      # Runs API on port 8000
```

### Training & Evaluation
```bash
python scripts/fetch_market_data.py                      # Pull OHLCV data → artifacts/data/phase0/raw_daily.parquet
python scripts/train.py --config config/phase_0.yaml     # Train (runs 5 seeds by default)
python scripts/train.py --config config/phase_0.yaml --seed 42  # Single seed
python scripts/evaluate.py --config config/phase_0.yaml --checkpoint artifacts/checkpoints/phase0/seed_42_*/best_epoch_*.pt
```

### Testing
```bash
pytest                      # Full suite (verbose by default)
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
pytest tests/unit/test_phase0_features.py -k "test_name"  # Single test by name
pytest tests/unit/test_phase0_features.py::test_calculate_sma_expected_values  # Specific test
```

### Linting & Formatting (Ruff)
```bash
ruff check .         # Lint only
ruff format .        # Format only
ruff check . --fix   # Lint and auto-fix
```

---

## Architecture

All production code lives under `api/`. No root-level modules.

- **`api/app.py`** — FastAPI app factory. `api/main.py` is the uvicorn entrypoint.
- **`api/routes/`** — Endpoint modules: `forecast.py` (POST `/forecast/daily`), `auth.py` (signup/login/logout), `tickers.py`, `metrics.py`
- **`api/services/`** — Business logic: `forecast_service.py` loads checkpoints and runs inference, `auth_service.py` wraps Supabase auth, `ticker_service.py` uses Redis caching (TTL 1hr)
- **`api/models/`** — PyTorch model definitions. `factory.py` provides a `MODEL_REGISTRY` mapping string names → classes. Currently only `lstm_baseline`.
- **`api/data/`** — Full data pipeline: `sources.py` (yfinance fetch) → `features.py` (SMA, RSI, MACD, OBV) → `targets.py` (log returns) → `normalization.py` (per-asset z-score, clip=5.0) → `splitters.py` (date-based) → `datasets.py` (PyTorch Dataset) → `pipeline.py` (orchestrator)
- **`api/training/`** — `train_loop.py` (Trainer with early stopping, gradient clipping, ReduceLROnPlateau), `losses.py` (MSE + Sharpe surrogate composite loss with warmup), `checkpointing.py`
- **`scripts/`** — CLI entrypoints for fetch, train, evaluate (not imported by API)
- **`config/`** — YAML configs. `base.yaml` for shared defaults, `phase_0.yaml` for LSTM hyperparameters.
- **`artifacts/`** — Generated outputs: `data/`, `checkpoints/`, `reports/`

---

## Key Design Decisions

- **No training during API requests** — training is offline only; API loads from checkpoint files (`best_epoch_*.pt`)
- **Date-based train/val/test splits** (train ≤ 2021-12-31, val ≤ 2023-12-31, test ≤ 2025-12-31) — never percentage-based, to prevent look-ahead bias
- **Scalers fit on train only**, then applied to val/test. Scaler state is serialized into checkpoints.
- **Composite loss**: MSE + 0.1·Sharpe surrogate with 3-epoch MSE-only warmup
- **Multi-seed robustness**: default seeds [42, 123, 456, 789, 1024]
- **5 assets**: AAPL, MSFT, GOOGL, AMZN, SPY with 8 features (close, volume, sma_15, sma_45, rsi_14, macd, macd_signal, obv)

---

## Code Style Guidelines

### General
- **Target Python**: 3.10+
- **Line length**: 88 characters
- **Indent**: 4 spaces
- **Quotes**: Single quotes (`'`) — configured in ruff
- **Preview features**: Enabled for ruff

### Imports
- Use absolute imports from `api` package (e.g., `from api.routes import forecast`)
- Order: standard library → third-party → local, with blank lines between groups
- Sort imports with ruff (default behavior)

```python
# Correct
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from api.data.pipeline import DataPipeline
from api.schemas.forecast import DailyForecastRequest
```

### Type Hints
- Use Python 3.10+ union syntax: `str | None` instead of `Optional[str]`
- Use `dict` for generic dicts, or `dict[str, int]` for parameterized
- Return types required for public functions

```python
# Correct
def predict_daily(self, req: DailyForecastRequest) -> DailyForecastResponse:
    ...

# Incorrect
def predict_daily(self, req):  # Missing type hints
```

### Naming Conventions
- **Functions/variables**: `snake_case` (e.g., `calculate_sma`, `horizon_days`)
- **Classes**: `PascalCase` (e.g., `LSTMBaseline`, `ForecastService`)
- **Private methods**: Leading underscore (e.g., `_load_model`, `_get_pipeline`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_HORIZON = 1`)

### File Organization
- Production code under `api/` only
- Routes in `api/routes/`, services in `api/services/`, models in `api/models/`
- Schemas in `api/schemas/`, data pipeline in `api/data/`, training in `api/training/`
- Tests mirror structure: `tests/unit/`, `tests/integration/`

### Error Handling
- Use specific exception types (e.g., `FileNotFoundError`, `ValueError`, `RuntimeError`)
- Chain exceptions with `from e` to preserve traceback
- In routes, raise `HTTPException` with structured detail:

```python
raise HTTPException(
    status_code=400,
    detail={'code': 'invalid_model', 'message': f'Unknown model: {req.model}'},
)
```

### FastAPI Routes
- Use `async def` for route handlers
- Access services via `request.app.state` (dependency injection pattern)
- Response models via `response_model=...`

```python
@router.post('/daily', response_model=DailyForecastResponse)
async def forecast_daily(
    req: DailyForecastRequest, request: Request
) -> DailyForecastResponse:
    service = request.app.state.forecast_service
    ...
```

### Pydantic Schemas
- Use v2 style (from `pydantic import BaseModel`)
- Define request/response schemas in `api/schemas/`
- Use `Field` for validation

### PyTorch Models
- Inherit from `nn.Module`
- Implement `forward(self, x: torch.Tensor) -> torch.Tensor`
- Validate input shapes in forward
- Use `torch.no_grad()` during inference
- Set `model.eval()` before inference

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError('Input must be 3D [B, T, F]')
    ...
    with torch.no_grad():
        y_hat = model(x)
```

### Testing Conventions
- Use pytest with fixtures defined in `tests/conftest.py`
- Use `tmp_path` fixture for temporary directories
- Use `tmp_artifact_dir` fixture for artifact directories
- Use descriptive test names: `test_<function>_<expected_behavior>`
- Prefer synthetic data fixtures over real API calls in unit tests

```python
def test_calculate_sma_expected_values():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = features.calculate_sma(close, 2)
    expected_last = (4.0 + 5.0) / 2
    assert np.allclose(result.dropna().iloc[-1], expected_last, atol=1e-6)
```

### Ruff Configuration (pyproject.toml)
- Rules: E (pycodestyle errors), F (pyflakes), B (flake8-bugbear), I (isort), UP (pyupgrade), SIM (simplify)
- Ignore: E501 (line too long)
- `__init__.py` files allow F401/E402 (unused imports, module level import not at top)
- Test files allow S101 (assert statements)

### Documentation
- Use docstrings for public modules and classes
- Use Google-style docstrings:

```python
def calculate_sma(close: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average.

    Args:
        close: Close price series.
        window: Window size for SMA calculation.

    Returns:
        SMA series with same index as input.
    """
```

---

## Environment

Copy `.env.example` → `.env`. Required: `SUPABASE_URL`, `SUPABASE_KEY`. Optional: `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`.