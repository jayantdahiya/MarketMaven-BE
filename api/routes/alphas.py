"""
Alpha factor routes: GET /alphas.
"""

from datetime import date, datetime

from fastapi import APIRouter, HTTPException, Request

from api.schemas.alpha import AlphaResponse

router = APIRouter(tags=['alphas'])


@router.get('/alphas', response_model=AlphaResponse)
async def get_alphas(
    asset_id: str,
    date: date,
    request: Request,
) -> AlphaResponse:
    """Return cached alpha factors for a given asset and date.

    Does NOT trigger LLM generation — returns 404 if no cached entry exists.
    """
    alpha_service = getattr(request.app.state, 'alpha_service', None)
    if alpha_service is None:
        raise HTTPException(
            status_code=503,
            detail={
                'code': 'alpha_service_unavailable',
                'message': 'Alpha service not configured',
            },
        )

    cached = alpha_service.get_cached_alphas(asset_id, date.isoformat())
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail={
                'code': 'alpha_not_found',
                'message': f'No cached alphas for {asset_id} on {date}',
            },
        )

    alpha_values = {
        f'alpha_{i}': float(cached.get(f'alpha_{i}', 0.0)) for i in range(1, 9)
    }
    alpha_version = request.app.state.config.get('api', {}).get(
        'alpha_version', 'unknown'
    )
    generated_at_str = cached.get('generated_at', datetime.utcnow().isoformat())
    return AlphaResponse(
        asset_id=asset_id,
        date=date,
        alpha_values=alpha_values,
        alpha_version=alpha_version,
        cached=True,
        generated_at=datetime.fromisoformat(generated_at_str),
        rationale=cached.get('rationale'),
    )
