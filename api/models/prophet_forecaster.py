"""
Legacy Prophet forecaster: yfinance + Prophet fit, return structured dict.
"""

from datetime import date, timedelta
from typing import Any

import pandas as pd
import yfinance as yf
from prophet import Prophet


class DataUnavailableError(Exception):
    """Raised when yfinance returns no data."""


def forecast_prophet(
    asset_id: str,
    horizon_days: int = 10,
    end_date: date | None = None,
) -> dict[str, Any]:
    """
    Download recent data via yfinance, fit Prophet, return prediction dict.
    Returns dict with keys e.g. timestamps, trend (list of values), not raw JSON string.
    """
    end = end_date or date.today()
    ticker = yf.Ticker(asset_id)
    data = ticker.history(period='max', interval='1d')
    if data is None or data.empty or len(data) < 30:
        raise DataUnavailableError(f'Insufficient data for {asset_id}')
    df = data.reset_index()
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    df = df[['ds', 'y']].dropna()
    if len(df) < 30:
        raise DataUnavailableError(f'Insufficient valid rows for {asset_id}')
    model = Prophet()
    model.fit(df)
    future_end = end + timedelta(days=horizon_days)
    dates = pd.date_range(start=end, end=future_end, freq='D')
    df_future = pd.DataFrame({'ds': dates})
    forecast = model.predict(df_future)
    prediction = forecast.tail(horizon_days)
    return {
        'timestamps': prediction['ds'].dt.strftime('%Y-%m-%d').tolist(),
        'trend': prediction['yhat'].tolist(),
    }
