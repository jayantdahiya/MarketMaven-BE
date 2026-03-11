"""
Metrics response schema for GET /metrics/latest.
"""
from pydantic import BaseModel


class MetricsSummaryResponse(BaseModel):
    mae: float
    rmse: float
    directional_accuracy: float
    sharpe: float
    sortino: float
    max_drawdown: float
    model: str | None = None
    run_id: str | None = None
