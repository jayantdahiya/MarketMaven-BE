"""
Sentiment data ingestion via Finnhub company news API.

Provides FinnhubSentimentClient for fetching news and aggregate_daily_sentiment
for computing per-asset daily sentiment features: sentiment_mean, sentiment_std,
sentiment_pos_ratio, sentiment_count.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when Finnhub returns HTTP 429."""


class FinnhubSentimentClient:
    """Fetch company news from Finnhub with exponential backoff on rate limits.

    Args:
        api_key: Finnhub API key.
        base_url: Finnhub API base URL.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = 'https://finnhub.io/api/v1',
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def fetch_company_news(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """Retrieve news articles for a ticker from Finnhub.

        Args:
            ticker: Stock ticker (e.g. 'AAPL').
            start_date: ISO date string (YYYY-MM-DD).
            end_date: ISO date string (YYYY-MM-DD).

        Returns:
            List of news article dicts with keys: datetime, headline,
            source, summary, url, etc.

        Raises:
            RateLimitError: On HTTP 429 (triggers tenacity retry).
            RuntimeError: On non-retryable HTTP errors.
        """
        url = f'{self.base_url}/company-news'
        params = {
            'symbol': ticker,
            'from': start_date,
            'to': end_date,
            'token': self.api_key,
        }
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            raise RateLimitError(f'Finnhub rate limit for {ticker}')
        if resp.status_code != 200:
            raise RuntimeError(
                f'Finnhub API error {resp.status_code} for {ticker}: {resp.text}'
            )
        return resp.json()


def _score_article(article: dict) -> float:
    """Derive a simple sentiment score from a Finnhub news article.

    Uses a basic heuristic: the Finnhub API returns a 'sentiment' field
    when available, otherwise we assign a neutral 0.0 score.

    Returns:
        Float in [-1, 1].
    """
    # Finnhub company-news endpoint doesn't always include sentiment;
    # use it if present, otherwise default to neutral.
    score = article.get('sentiment', 0.0)
    if score is None:
        return 0.0
    return float(np.clip(score, -1.0, 1.0))


def aggregate_daily_sentiment(news_items: list[dict]) -> pd.DataFrame:
    """Aggregate news articles into daily sentiment features.

    Args:
        news_items: List of Finnhub news dicts. Each must have a
            'datetime' field (unix timestamp).

    Returns:
        DataFrame with columns: date, sentiment_mean, sentiment_std,
        sentiment_pos_ratio, sentiment_count.
    """
    if not news_items:
        return pd.DataFrame(
            columns=[
                'date',
                'sentiment_mean',
                'sentiment_std',
                'sentiment_pos_ratio',
                'sentiment_count',
            ]
        )

    records = []
    for article in news_items:
        ts = article.get('datetime')
        if ts is None:
            continue
        dt = datetime.utcfromtimestamp(ts).date()
        score = _score_article(article)
        records.append({'date': dt, 'score': score})

    if not records:
        return pd.DataFrame(
            columns=[
                'date',
                'sentiment_mean',
                'sentiment_std',
                'sentiment_pos_ratio',
                'sentiment_count',
            ]
        )

    df = pd.DataFrame(records)
    grouped = df.groupby('date')['score']
    result = pd.DataFrame({
        'date': grouped.mean().index,
        'sentiment_mean': grouped.mean().values.astype(np.float32),
        'sentiment_std': grouped.std().fillna(0.0).values.astype(np.float32),
        'sentiment_pos_ratio': grouped.apply(
            lambda s: (s > 0).sum() / max(len(s), 1)
        ).values.astype(np.float32),
        'sentiment_count': grouped.count().values.astype(np.int32),
    })
    return result


def build_sentiment_table(
    tickers: list[str],
    start_date: str,
    end_date: str,
    client: FinnhubSentimentClient,
) -> pd.DataFrame:
    """Build a complete sentiment table for multiple tickers.

    Args:
        tickers: List of stock tickers.
        start_date: ISO date string.
        end_date: ISO date string.
        client: FinnhubSentimentClient instance.

    Returns:
        DataFrame with columns: timestamp, asset_id, sentiment_mean,
        sentiment_std, sentiment_pos_ratio, sentiment_count.
    """
    frames = []
    for ticker in tickers:
        logger.info(
            'Fetching sentiment for %s (%s to %s)', ticker, start_date, end_date
        )
        try:
            news = client.fetch_company_news(ticker, start_date, end_date)
        except Exception:
            logger.warning(
                'Failed to fetch sentiment for %s, using neutral defaults',
                ticker,
                exc_info=True,
            )
            news = []
        daily = aggregate_daily_sentiment(news)
        if daily.empty:
            # Create a single row with neutral defaults
            daily = pd.DataFrame({
                'date': [pd.Timestamp(start_date).date()],
                'sentiment_mean': [np.float32(0.0)],
                'sentiment_std': [np.float32(0.0)],
                'sentiment_pos_ratio': [np.float32(0.5)],
                'sentiment_count': [np.int32(0)],
            })
        daily['asset_id'] = ticker
        daily = daily.rename(columns={'date': 'timestamp'})
        daily['timestamp'] = pd.to_datetime(daily['timestamp'])
        frames.append(daily)
    if not frames:
        raise RuntimeError('No sentiment data retrieved for any ticker')
    return pd.concat(frames, ignore_index=True)


def fill_neutral_defaults(
    df: pd.DataFrame,
    neutral_defaults: dict | None = None,
) -> pd.DataFrame:
    """Fill missing sentiment columns with neutral defaults.

    Args:
        df: DataFrame that may have NaN sentiment columns.
        neutral_defaults: Dict of column→default value. Uses spec defaults
            if not provided.

    Returns:
        DataFrame with NaN sentiment values replaced.
    """
    defaults = neutral_defaults or {
        'sentiment_mean': 0.0,
        'sentiment_std': 0.0,
        'sentiment_pos_ratio': 0.5,
        'sentiment_count': 0,
    }
    for col, default_val in defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default_val)
    return df
