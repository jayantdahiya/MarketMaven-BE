"""
TickerService: Supabase ticker list with Redis cache.
"""
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class TickerService:
    def __init__(self, supabase_client: Any, redis_client: Any | None, ttl_seconds: int = 3600):
        self._supabase = supabase_client
        self._redis = redis_client
        self._ttl = ttl_seconds

    def get_tickers(self) -> list[dict]:
        if self._redis is not None:
            try:
                cached = self._redis.get("tickers")
                if cached is not None:
                    if isinstance(cached, bytes):
                        cached = cached.decode("utf-8")
                    return json.loads(cached)
            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                logger.warning("Redis cache decode error, refetching tickers: %s", e)
        try:
            result = self._supabase.table("tickers").select("*").execute()
            data = result.data if hasattr(result, "data") else []
            if self._redis is not None:
                try:
                    self._redis.setex("tickers", self._ttl, json.dumps(data))
                except Exception as e:
                    logger.warning("Redis set failed: %s", e)
            return data
        except Exception as e:
            logger.error("Supabase ticker fetch failed: %s", e)
            raise
