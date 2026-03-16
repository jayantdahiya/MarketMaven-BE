"""
LOB forecast route: POST /forecast/lob.

Returns 503 if lob.api_enabled is false in config.
Returns 422 if insufficient LOB history (x_book/x_aux not provided or wrong shape).
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

from api.schemas.lob import LOBForecastRequest, LOBForecastResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/forecast', tags=['lob'])


@router.post('/lob', response_model=LOBForecastResponse)
async def forecast_lob(
    req: LOBForecastRequest,
    request: Request,
) -> LOBForecastResponse:
    """Run LOB mid-price direction forecast.

    Requires:
        - lob.api_enabled: true in config (else 503).
        - lob_service present on app.state (else 503).
        - x_book / x_aux pre-assembled by service layer (else 422).

    Returns:
        LOBForecastResponse with predicted_class, class_probabilities,
        expected_mid_move_ticks, confidence.
    """
    config = getattr(request.app.state, 'config', {})
    lob_enabled = config.get('lob', {}).get('api_enabled', False)
    if not lob_enabled:
        raise HTTPException(
            status_code=503,
            detail={
                'code': 'lob_module_disabled',
                'message': 'LOB forecasting module is disabled. Set lob.api_enabled: true in config.',
            },
        )

    lob_service = getattr(request.app.state, 'lob_service', None)
    if lob_service is None:
        raise HTTPException(
            status_code=503,
            detail={
                'code': 'lob_module_disabled',
                'message': 'LOB service is not initialised.',
            },
        )

    # Build input tensors from service (raises ValueError on insufficient history)
    import torch  # noqa: PLC0415

    try:
        x_book, x_aux = lob_service.get_feature_window(req)
    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=422,
            detail={
                'code': 'insufficient_lob_history',
                'message': str(exc),
            },
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                'code': 'lob_module_disabled',
                'message': f'LOB data source unavailable: {exc}',
            },
        ) from exc

    try:
        response = lob_service.predict(req, x_book, x_aux)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                'code': 'lob_module_disabled',
                'message': str(exc),
            },
        ) from exc

    return response
