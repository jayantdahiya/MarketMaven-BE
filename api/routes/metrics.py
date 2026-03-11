"""
Metrics route: GET /metrics/latest.
"""

from pathlib import Path

from fastapi import APIRouter, Request

from api.schemas.metrics import MetricsSummaryResponse

router = APIRouter(tags=['metrics'])


@router.get('/metrics/latest', response_model=MetricsSummaryResponse)
async def metrics_latest(
    request: Request, model: str | None = None
) -> MetricsSummaryResponse:
    """Return latest evaluation metrics from reports dir (if any)."""
    cfg = getattr(request.app.state, 'config', {})
    reports_dir = Path(
        cfg.get('paths', {}).get('reports_dir', 'artifacts/reports/phase0')
    )
    if not reports_dir.exists():
        return MetricsSummaryResponse(
            mae=0.0,
            rmse=0.0,
            directional_accuracy=0.0,
            sharpe=0.0,
            sortino=0.0,
            max_drawdown=0.0,
            model=model,
            run_id=None,
        )
    # Find latest metrics_summary.json
    best = None
    for p in reports_dir.rglob('metrics_summary.json'):
        if best is None or p.stat().st_mtime > best.stat().st_mtime:
            best = p
    if best is None:
        return MetricsSummaryResponse(
            mae=0.0,
            rmse=0.0,
            directional_accuracy=0.0,
            sharpe=0.0,
            sortino=0.0,
            max_drawdown=0.0,
            model=model,
            run_id=None,
        )
    import json

    with open(best) as f:
        data = json.load(f)
    return MetricsSummaryResponse(
        mae=data.get('mae', 0.0),
        rmse=data.get('rmse', 0.0),
        directional_accuracy=data.get('directional_accuracy', 0.0),
        sharpe=data.get('sharpe_after_cost', data.get('sharpe', 0.0)),
        sortino=data.get('sortino_after_cost', data.get('sortino', 0.0)),
        max_drawdown=data.get('max_drawdown', 0.0),
        model=model,
        run_id=str(best.parent),
    )
