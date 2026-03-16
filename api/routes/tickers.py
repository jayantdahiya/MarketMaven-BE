"""
Tickers route: GET /tickers.
"""

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(tags=['tickers'])


@router.get('/tickers')
async def get_tickers(request: Request) -> list:
    service = getattr(request.app.state, 'ticker_service', None)
    if service is None:
        raise HTTPException(
            status_code=503, detail='Ticker service unavailable (missing Supabase)'
        )
    try:
        return service.get_tickers()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={'code': 'ticker_source_unavailable', 'message': str(e)},
        ) from e
