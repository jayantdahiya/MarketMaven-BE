"""
FastAPI app factory: CORS, routers, state (forecast_service, auth_service, ticker_service).
"""

import os
from pathlib import Path

import redis
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client

from api.data.pipeline import load_config
from api.routes import alphas, auth, forecast, metrics, tickers
from api.services.alpha_service import AlphaService
from api.services.auth_service import AuthService
from api.services.forecast_service import ForecastService
from api.services.ticker_service import TickerService


def create_app() -> FastAPI:
    load_dotenv()
    app = FastAPI(title='Market Maven API')
    origins = os.environ.get(
        'ALLOWED_ORIGINS',
        'http://localhost,http://localhost:3000,http://localhost:8000',
    ).split(',')
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )
    config_path = Path('config/phase_0.yaml')
    if config_path.exists():
        try:
            config = load_config(str(config_path))
        except Exception:
            config = {}
    else:
        config = {}
    # Inject per-model checkpoint overrides so LSTM (phase0) and
    # CNN-Transformer (phase1) are served simultaneously without editing YAMLs.
    config.setdefault('model_checkpoints', {})
    config['model_checkpoints'].setdefault(
        'lstm_baseline', 'artifacts/checkpoints/phase0'
    )
    config['model_checkpoints'].setdefault(
        'cnn_transformer', 'artifacts/checkpoints/phase1'
    )
    app.state.config = config
    app.state.allow_legacy_prophet = config.get('api', {}).get(
        'allow_legacy_prophet', True
    )
    app.state.forecast_service = ForecastService(config=config)
    supabase_url = os.environ.get('SUPABASE_URL') or ''
    supabase_key = os.environ.get('SUPABASE_KEY') or ''
    if supabase_url and supabase_key:
        supabase_client = create_client(supabase_url, supabase_key)
        app.state.auth_service = AuthService(supabase_url, supabase_key)
    else:
        supabase_client = None
        app.state.auth_service = None
    try:
        redis_host = (os.environ.get('REDIS_HOST') or 'localhost').strip()
        if redis_host.startswith('https://'):
            redis_host = redis_host.removeprefix('https://').split('/')[0]
            redis_ssl = True
        elif redis_host.startswith('http://'):
            redis_host = redis_host.removeprefix('http://').split('/')[0]
            redis_ssl = False
        else:
            redis_ssl = os.environ.get('REDIS_SSL', '').lower() == 'true'
        redis_client = redis.Redis(
            host=redis_host,
            port=int(os.environ.get('REDIS_PORT', '6379')),
            password=os.environ.get('REDIS_PASSWORD') or None,
            ssl=redis_ssl,
        )
        redis_client.ping()
    except Exception:
        redis_client = None
    app.state.ticker_service = (
        TickerService(supabase_client, redis_client, 3600) if supabase_client else None
    )
    # Alpha service (Phase 3+): initialise when multimodal alpha is enabled
    multimodal_cfg = config.get('multimodal', {})
    alpha_cfg = multimodal_cfg.get('alpha', {})
    if multimodal_cfg.get('enabled', False) and alpha_cfg.get('enabled', False):
        app.state.alpha_service = AlphaService(
            model_name=alpha_cfg.get('llm_model', 'gpt-4.1-mini'),
            cache_path=alpha_cfg.get(
                'cache_path', 'artifacts/data/phase3/alpha_cache.jsonl'
            ),
            temperature=alpha_cfg.get('temperature', 0.1),
        )
    else:
        app.state.alpha_service = None

    # LOB service (Phase 4): conditionally register when lob.api_enabled is true
    lob_cfg = config.get('lob', {})
    if lob_cfg.get('api_enabled', False):
        try:
            from api.routes import lob as lob_route  # noqa: PLC0415
            from api.services.lob_service import LOBService  # noqa: PLC0415

            lob_config_path = Path('config/phase_4.yaml')
            lob_full_cfg = (
                load_config(str(lob_config_path))
                if lob_config_path.exists()
                else config
            )
            app.state.lob_service = LOBService(cfg=lob_full_cfg)
            app.include_router(lob_route.router)
        except Exception:
            app.state.lob_service = None
    else:
        app.state.lob_service = None

    app.include_router(forecast.router)
    app.include_router(auth.router)
    app.include_router(tickers.router)
    app.include_router(metrics.router)
    app.include_router(alphas.router)

    @app.get('/')
    def index():
        return {'message': 'Welcome to the Market Maven API'}

    # Legacy: GET /prophet (backward compat; forecast router has /forecast/prophet)
    from fastapi import HTTPException, Request

    from api.schemas.forecast import DailyForecastRequest

    @app.get('/prophet')
    async def legacy_prophet_root(ticker: str, request: Request):
        if not getattr(request.app.state, 'allow_legacy_prophet', True):
            raise HTTPException(
                status_code=410, detail='Legacy Prophet endpoint disabled'
            )
        service = request.app.state.forecast_service
        req = DailyForecastRequest(
            asset_id=ticker.upper(), horizon_days=10, model='prophet'
        )
        try:
            resp = service.predict_daily(req)
            return {
                'timestamps': [p.timestamp.isoformat() for p in resp.predictions],
                'trend': [p.predicted_price or 0.0 for p in resp.predictions],
            }
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e)) from e

    return app


app = create_app()
