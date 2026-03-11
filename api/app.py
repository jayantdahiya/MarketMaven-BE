"""
FastAPI app factory: CORS, routers, state (forecast_service, auth_service, ticker_service).
"""

import os
from pathlib import Path

import redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client

from api.data.pipeline import load_config
from api.routes import auth, forecast, metrics, tickers
from api.services.auth_service import AuthService
from api.services.forecast_service import ForecastService
from api.services.ticker_service import TickerService


def create_app() -> FastAPI:
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
        redis_client = redis.Redis(
            host=os.environ.get('REDIS_HOST', 'localhost'),
            port=int(os.environ.get('REDIS_PORT', '6379')),
            password=os.environ.get('REDIS_PASSWORD') or None,
            ssl=os.environ.get('REDIS_SSL', '').lower() == 'true',
        )
    except Exception:
        redis_client = None
    app.state.ticker_service = (
        TickerService(supabase_client, redis_client, 3600) if supabase_client else None
    )
    app.include_router(forecast.router)
    app.include_router(auth.router)
    app.include_router(tickers.router)
    app.include_router(metrics.router)

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
