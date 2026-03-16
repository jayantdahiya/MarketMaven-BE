"""
Alpha factor API response schema.
"""

from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator


class AlphaResponse(BaseModel):
    """Response schema for the ``GET /alphas`` endpoint.

    Attributes:
        asset_id: Ticker symbol.
        date: Date the alphas were generated for.
        alpha_values: Dict mapping ``alpha_1`` through ``alpha_8`` to float values.
        alpha_version: Version identifier from config.
        cached: Whether the result was served from cache.
        generated_at: Timestamp when the alphas were generated.
        rationale: Optional text rationale from the LLM.
    """

    asset_id: str
    date: date
    alpha_values: dict[str, float] = Field(
        ..., description='Keys alpha_1 through alpha_8, each a float'
    )
    alpha_version: str
    cached: bool
    generated_at: datetime
    rationale: str | None = None

    @field_validator('alpha_values')
    @classmethod
    def must_have_eight_factors(cls, v: dict[str, float]) -> dict[str, float]:
        expected = {f'alpha_{i}' for i in range(1, 9)}
        if set(v.keys()) != expected:
            raise ValueError(
                f'alpha_values must contain exactly keys {sorted(expected)}, '
                f'got {sorted(v.keys())}'
            )
        return v
