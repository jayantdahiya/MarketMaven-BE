"""
History response schemas: historical returns and predictions.
"""

from pydantic import BaseModel


class ReturnPoint(BaseModel):
    """A single daily log-return data point."""

    date: str  # ISO format YYYY-MM-DD
    log_return: float


class HistoricalReturnsResponse(BaseModel):
    """Response for GET /history/returns."""

    asset_id: str
    returns: list[ReturnPoint]
    rolling_volatility: list[ReturnPoint] | None = (
        None  # 30-day rolling std, quant mode
    )


class PredictionPoint(BaseModel):
    """A single historical prediction data point."""

    date: str  # ISO format YYYY-MM-DD
    predicted_return: float
    actual_return: float
    signal: str  # 'long' | 'flat'


class HistoricalPredictionsResponse(BaseModel):
    """Response for GET /history/predictions."""

    asset_id: str
    model: str
    predictions: list[PredictionPoint]
