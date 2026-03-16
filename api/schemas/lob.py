"""
LOB forecast request/response schemas (Phase 4).
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class LOBForecastRequest(BaseModel):
    """Request schema for POST /forecast/lob."""

    asset_id: str = Field(..., pattern=r'^[A-Z0-9._-]{1,15}$')
    as_of_timestamp: datetime | None = None
    horizon_events: int = Field(20, ge=1, le=100)
    model: Literal['tlob_forecaster', 'lob_cnn'] = 'tlob_forecaster'


class LOBForecastResponse(BaseModel):
    """Response schema for POST /forecast/lob."""

    asset_id: str
    as_of_timestamp: datetime
    predicted_class: int = Field(..., ge=0, le=2, description='0=down, 1=flat, 2=up')
    class_probabilities: list[float] = Field(
        ..., description='[p_down, p_flat, p_up]; must sum to ~1.0'
    )
    expected_mid_move_ticks: float = Field(
        ..., description='Signed expected mid-price move in ticks'
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description='Max class probability')

    @field_validator('class_probabilities')
    @classmethod
    def probabilities_sum_to_one(cls, v: list[float]) -> list[float]:
        total = sum(v)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f'class_probabilities must sum to 1.0; got {total:.8f}')
        return v
