"""Unit tests for Phase 3 sentiment ingestion and aggregation."""

import numpy as np
import pandas as pd
import pytest
import responses

from api.data.sentiment import (
    FinnhubSentimentClient,
    aggregate_daily_sentiment,
    fill_neutral_defaults,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_news_items() -> list[dict]:
    """Synthetic Finnhub news articles spanning 2 days."""
    from datetime import datetime

    day1 = int(datetime(2024, 1, 15, 10, 0, 0).timestamp())
    day2 = int(datetime(2024, 1, 16, 14, 30, 0).timestamp())
    return [
        {'datetime': day1, 'headline': 'Earnings beat', 'sentiment': 0.8},
        {'datetime': day1, 'headline': 'Revenue growth', 'sentiment': 0.3},
        {'datetime': day1, 'headline': 'Downgrade', 'sentiment': -0.5},
        {'datetime': day2, 'headline': 'New product', 'sentiment': 0.6},
        {'datetime': day2, 'headline': 'Market rally', 'sentiment': 0.2},
    ]


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_sentiment_aggregation_fields_present(sample_news_items: list[dict]) -> None:
    """Output DataFrame contains all 4 required sentiment columns."""
    df = aggregate_daily_sentiment(sample_news_items)
    required_cols = {
        'sentiment_mean',
        'sentiment_std',
        'sentiment_pos_ratio',
        'sentiment_count',
    }
    assert required_cols.issubset(set(df.columns)), (
        f'Missing columns: {required_cols - set(df.columns)}'
    )
    # Should have 2 days of data
    assert len(df) == 2, f'Expected 2 rows (2 days), got {len(df)}'
    # Verify sentiment_count sums correctly
    assert df['sentiment_count'].sum() == 5


def test_sentiment_no_news_defaults_to_neutral() -> None:
    """Neutral defaults when article list is empty."""
    df = aggregate_daily_sentiment([])
    assert df.empty, 'Expected empty DataFrame when no news articles provided'

    # fill_neutral_defaults should fill NaN sentinel values with neutral defaults
    base = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-15']),
        'asset_id': ['AAPL'],
        'sentiment_mean': [np.nan],
        'sentiment_std': [np.nan],
        'sentiment_pos_ratio': [np.nan],
        'sentiment_count': [np.nan],
    })
    filled = fill_neutral_defaults(base)
    assert filled['sentiment_mean'].iloc[0] == 0.0
    assert filled['sentiment_std'].iloc[0] == 0.0
    assert filled['sentiment_pos_ratio'].iloc[0] == 0.5
    assert filled['sentiment_count'].iloc[0] == 0


@responses.activate
def test_sentiment_retry_on_429() -> None:
    """FinnhubSentimentClient retries on HTTP 429 and succeeds on subsequent call."""
    base_url = 'https://finnhub.io/api/v1'

    # First call returns 429, second returns 200 with empty list
    responses.add(
        responses.GET,
        f'{base_url}/company-news',
        status=429,
        body='Rate limit exceeded',
    )
    responses.add(
        responses.GET,
        f'{base_url}/company-news',
        json=[],
        status=200,
    )

    client = FinnhubSentimentClient(api_key='test_key', base_url=base_url)
    result = client.fetch_company_news('AAPL', '2024-01-01', '2024-01-31')

    # Should have retried and returned empty list
    assert result == []
    # Should have made 2 requests (1 failed + 1 succeeded)
    assert len(responses.calls) == 2
