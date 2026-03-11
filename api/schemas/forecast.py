"""
Forecast request/response schemas.
"""
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DailyForecastRequest(BaseModel):
    asset_id: str = Field(..., pattern=r"^[A-Z0-9._-]{1,15}$")
    horizon_days: int = Field(1, ge=1, le=10)
    model: str = "lstm_baseline"
    as_of_date: date | None = None

    @field_validator("horizon_days")
    @classmethod
    def horizon_allowed(cls, v: int) -> int:
        if v not in (1, 5, 10):
            raise ValueError("horizon_days must be 1, 5, or 10")
        return v


class DailyForecastPoint(BaseModel):
    timestamp: datetime
    predicted_return: float
    predicted_price: float | None = None
    signal: Literal["long", "flat"]
    confidence: float


class DailyForecastResponse(BaseModel):
    asset_id: str
    model: str
    horizon_days: int
    generated_at: datetime
    predictions: list[DailyForecastPoint]
