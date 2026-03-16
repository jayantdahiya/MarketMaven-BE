"""
Forecast routes: POST /forecast/daily, POST /forecast/multimodal, GET /prophet (legacy).
"""

from fastapi import APIRouter, HTTPException, Request

from api.schemas.forecast import (
    DailyForecastRequest,
    DailyForecastResponse,
    MultimodalForecastRequest,
    MultimodalForecastResponse,
)

router = APIRouter(prefix='/forecast', tags=['forecast'])


@router.post('/daily', response_model=DailyForecastResponse)
async def forecast_daily(
    req: DailyForecastRequest, request: Request
) -> DailyForecastResponse:
    service = request.app.state.forecast_service
    if req.model not in (
        'lstm_baseline',
        'cnn_transformer',
        'cnn_transformer_multimodal',
        'mamba_ssm',
        'prophet',
    ):
        raise HTTPException(
            status_code=400,
            detail={'code': 'invalid_model', 'message': f'Unknown model: {req.model}'},
        )
    try:
        return service.predict_daily(req)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503, detail={'code': 'model_unavailable', 'message': str(e)}
        ) from e
    except RuntimeError as e:
        if 'insufficient' in str(e).lower() or 'data' in str(e).lower():
            raise HTTPException(
                status_code=502, detail={'code': 'data_unavailable', 'message': str(e)}
            ) from e
        raise HTTPException(
            status_code=503, detail={'code': 'model_unavailable', 'message': str(e)}
        ) from e


@router.post('/multimodal', response_model=MultimodalForecastResponse)
async def forecast_multimodal(
    req: MultimodalForecastRequest, request: Request
) -> MultimodalForecastResponse:
    """Multimodal forecast endpoint (Phase 3+).

    Returns predictions with modality usage metadata.
    """
    multimodal_enabled = request.app.state.config.get('multimodal', {}).get(
        'enabled', False
    )
    if not multimodal_enabled:
        raise HTTPException(
            status_code=404,
            detail={
                'code': 'multimodal_disabled',
                'message': 'Multimodal forecasting is not enabled',
            },
        )
    if not any([req.include_context, req.include_sentiment, req.include_alpha]):
        raise HTTPException(
            status_code=400,
            detail={
                'code': 'no_modalities',
                'message': 'At least one modality must be enabled',
            },
        )
    service = request.app.state.forecast_service
    try:
        return service.predict_multimodal(req)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503, detail={'code': 'model_unavailable', 'message': str(e)}
        ) from e
    except RuntimeError as e:
        if 'insufficient' in str(e).lower() or 'data' in str(e).lower():
            raise HTTPException(
                status_code=502, detail={'code': 'data_unavailable', 'message': str(e)}
            ) from e
        raise HTTPException(
            status_code=503, detail={'code': 'model_unavailable', 'message': str(e)}
        ) from e


@router.get('/prophet')
async def legacy_prophet(ticker: str, request: Request) -> dict:
    """Legacy endpoint: GET /prophet?ticker=AAPL. Returns Prophet forecast dict."""
    if not getattr(request.app.state, 'allow_legacy_prophet', True):
        raise HTTPException(status_code=410, detail='Legacy Prophet endpoint disabled')
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
