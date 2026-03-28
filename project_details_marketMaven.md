# Project: MarketMaven-BE

**Duration:** March 2022 – March 2026 (4 years, active development)
**GitHub/Demo:** N/A (private repo)
**Stack:** Python 3.10+, FastAPI, PyTorch 2.2+, LSTM, CNN+Transformer (with attention pooling), Mamba/SSM (CUDA + PyTorch fallback), TLOB Dual-Attention Transformer, Prophet, OpenAI GPT-4.1-mini, Finnhub Sentiment API, NetworkX (correlation graph), Supabase (auth + database), Redis (caching), MLflow, yfinance, pandas, pyarrow, scikit-learn, joblib, einops, tenacity, Next.js 16, React 19, Tailwind CSS 4, Recharts, SWR, Docker (+ CUDA image for mamba-ssm), uv, Ruff, pytest

---

## What It Does

MarketMaven-BE is a production-grade stock forecasting system that serves deep learning model predictions via a REST API, with zero training at inference time. It solves the problem of building a rigorous, multi-architecture ML forecasting backend by enforcing date-based (leak-free) train/val/test splits, per-asset normalization, and multi-seed robustness — designed for senior engineers and quantitative researchers who need explainable, reproducible results. The system covers 5 equity assets (AAPL, MSFT, GOOGL, AMZN, SPY) and supports a 4-phase model upgrade roadmap (LSTM → CNN-Transformer → Mamba/SSM + graph fusion → multimodal + LOB) via config-only rollback.

---

## Architecture & Key Engineering Work

- **Multi-model checkpoint routing** (`api/services/forecast_service.py`, `api/models/factory.py`): A `MODEL_REGISTRY` maps string names (`lstm_baseline`, `cnn_transformer`, `cnn_transformer_multimodal`, `mamba_ssm`, `tlob_forecaster`) to PyTorch classes. `ForecastService` discovers best checkpoints via glob (`best_epoch_*.pt`), deserializes per-asset `FeatureScaler` state stored inside checkpoint dicts, and routes inference accordingly — all without re-training at request time.

- **Composite loss with financial reward signal** (`api/training/losses.py`): `CompositeForecastLoss` combines MSE with a `SharpeSurrogateLoss` (tanh-based position sizing, annualization factor 252) and a `RollingSharpeSurrogateLoss`, with a configurable MSE-only warmup period (e.g., 3 epochs in `phase_2.yaml`). Acceptance criteria between phases enforce MAE ratio ≤ 0.97× and Sharpe delta ≥ +0.10 from Phase 0 → 1, tracked in `scripts/benchmark.py`.

- **Multimodal gated fusion architecture** (`api/models/cnn_transformer.py`): `CnnTransForecaster` in multimodal mode instantiates 4 separate `ModalityEncoder` branches (price/technical, market context, sentiment, LLM alpha factors) and combines their outputs via a learned `softmax gate` (`gate_params` Linear layer), fusing 27 features across modalities. Supports `get_attention_weights()` for interpretability; gate regularization via `lambda_gate_l1 = 0.001` in `phase_3.yaml`.

- **Mamba/SSM with CUDA + PyTorch fallback** (`api/models/mamba_ssm.py`): `MambaForecaster` attempts to import `mamba_ssm.Mamba` (CUDA) and falls back to a custom `MambaBlockFallback` implementing the selective-scan recurrence in pure PyTorch. Adds an optional graph projection pathway using a 60-day rolling correlation graph (threshold |ρ| ≥ 0.5 from `phase_2.yaml`) built with `api/data/asset_graph.py`, passing `G=3` graph-context features as auxiliary input to the final MLP.

- **LOB dual-attention Transformer** (`api/models/lob_models.py`): `TLOBForecaster` processes Level-2 order book tensors shaped `[B, T, 4, L]` (10 levels) through spatial `MultiheadAttention` blocks across the price-level dimension, then temporal `TransformerEncoder` layers, concatenated with auxiliary features before a 3-class classification head (up/flat/down). Training uses `WeightedCEFocalLoss` with class weights `[1.2, 0.8, 1.2]` and `lambda_focal = 0.25` to handle class imbalance; best checkpoint selected on balanced accuracy (`fit_lob` in `api/training/train_loop.py`).

- **LLM alpha generation pipeline** (`api/services/alpha_service.py`, `scripts/generate_alphas.py`): `AlphaService` calls OpenAI GPT-4.1-mini to generate 8 structured alpha factors (`alpha_1`–`alpha_8`) in `[-3, 3]` scale per asset per date, with JSONL caching, `tenacity`-powered retries (5 attempts), deterministic fallback for offline use, and a dedicated `/alphas` endpoint. Alpha features are merged into the multimodal pipeline in `api/data/alpha_features.py`.

---

## Performance & Scale Signals

- **5 random seeds** `[42, 123, 456, 789, 1024]` for multi-seed training robustness (per README and `scripts/train.py`)
- **5 assets**, **8 base features** (LSTM/Phase 0), **27 multimodal features** (Phase 3) — sourced from `config/phase_3.yaml` group definitions
- **Sequence lengths**: 60 days (Phase 0/1), 90 days (Phase 2 Mamba), 100 events (Phase 4 LOB) — from respective YAML configs
- **Redis TTL 3600s** for ticker cache — `api/app.py` `TickerService` constructor
- **Transaction costs modeled at 5 bps + 2 bps slippage** in `Backtester` — from `config/phase_0.yaml`
- **Annualization factor 252** throughout Sharpe calculations — `config/base.yaml` and `api/training/losses.py`
- **68 tests** (per README) covering unit + integration paths with `tmp_path` fixtures
- **Early stopping patience 10 epochs**, `grad_clip 1.0` (Phase 0) / `0.8` (Phase 1) — from phase YAML configs
- **Historical returns endpoint** returns up to 504 data points (2 years of daily data) — `api/routes/history.py` default `limit`
- **LOB rolling normalization window: 5000 events**, clip 6.0 — `config/phase_4.yaml`
- No explicit req/sec or latency benchmarks found in codebase. [inferred] Single-model inference is CPU/GPU bound by one forward pass over a 60–100 step window with batch size 1 at request time.

---

## Resume Bullets

- **Architected a 4-phase production stock forecasting backend** (FastAPI + PyTorch) with a config-driven model registry routing inference across LSTM, CNN-Transformer, Mamba/SSM, and a dual-attention LOB Transformer — with phase acceptance criteria enforcing ≥ +0.10 Sharpe improvement per upgrade and zero-downtime rollback via YAML config swap.

- **Engineered a multimodal gated fusion pipeline** combining 27 features across price/technicals, market indices/VIX, Finnhub sentiment, and GPT-4.1-mini–generated alpha factors, with a learned softmax gate layer, composite MSE + Sharpe surrogate loss (annualized at 252), and a CUDA/pure-PyTorch fallback Mamba SSM trained across 5 random seeds for statistical robustness.

- **Built a full-stack ML inference system** with FastAPI, Supabase auth, Redis-cached ticker data (TTL 3600s), MLflow experiment tracking, Docker + CUDA GPU images for mamba-ssm, and a Next.js 16 / React 19 frontend featuring retail vs. quant UX modes and live model comparison across 4 forecasting architectures.

- **Implemented a leak-proof financial ML training framework** enforcing strict date-based train/val/test splits (train ≤ 2021, val ≤ 2023, test ≤ 2025), per-asset z-score normalization with scaler state serialized into checkpoints, and a Level-2 order book pipeline processing 10-level book tensors with weighted focal loss to correct for class imbalance in 3-class direction prediction.

---

## What's Outdated in the Current Resume Entry

Based on codebase analysis, the following are missing, understated, or potentially inaccurate in any prior resume bullets describing this project:

1. **Missing: Mamba/SSM architecture** — this is a highly relevant 2024-era architecture that signals awareness of state-space models; must be included explicitly.
2. **Missing: LOB (Level-2 order book) pipeline** — `TLOBForecaster`, `LOBPipeline`, `WeightedCEFocalLoss`, and the execution-style backtest helpers are significant and not trivial to implement.
3. **Missing: LLM alpha generation** — calling GPT-4.1-mini to generate structured alpha factors and feeding them as modality inputs into the fusion model is a genuinely novel architecture decision worth highlighting.
4. **Missing: Multimodal gated fusion** — the 4-branch `ModalityEncoder` + softmax gate architecture in `cnn_transformer.py` is one of the most architecturally impressive parts of the codebase.
5. **Missing: Sharpe surrogate loss** — composite MSE + financial reward signal is a non-trivial ML training decision that most resume reviewers in quant/ML roles will recognize as sophisticated.
6. **Missing: Finnhub sentiment integration** — shows real-world data engineering across multiple heterogeneous APIs.
7. **Understated: Multi-seed robustness protocol** — 5-seed training with benchmarking acceptance criteria (MAE ratio, Sharpe delta thresholds) signals production-grade ML discipline.
8. **Understated: Frontend** — the Next.js 16 / React 19 frontend with SWR polling, Recharts model comparison, and dual retail/quant UX modes may be omitted if only describing this as a "backend" project.
9. **Potentially inaccurate: Port** — `api/main.py` uses port 9000 while Docker and README state 8000; avoid quoting a specific port unless verified.
10. **Potentially inaccurate: `get_feature_window`** — the LOB route calls `lob_service.get_feature_window()` but this method was not found in `LOBService`; avoid claiming a fully end-to-end LOB inference API unless this gap is resolved.
