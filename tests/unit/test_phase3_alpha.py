"""Unit tests for Phase 3 AlphaService: caching, fallback, and output contract."""

import json

import pytest

from api.services.alpha_service import (
    AlphaService,
    deterministic_fallback_alphas,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def alpha_service(tmp_path) -> AlphaService:
    """AlphaService with cache pointing at tmp_path."""
    cache_path = str(tmp_path / 'alpha_cache.jsonl')
    return AlphaService(
        model_name='gpt-4.1-mini',
        cache_path=cache_path,
        temperature=0.1,
    )


@pytest.fixture
def cached_alpha_service(tmp_path) -> AlphaService:
    """AlphaService with a pre-populated cache entry for AAPL/2024-01-15."""
    cache_path = tmp_path / 'alpha_cache.jsonl'
    entry = {
        'cache_key': 'AAPL_2024-01-15',
        'asset_id': 'AAPL',
        'date': '2024-01-15',
        'alpha_1': 0.5,
        'alpha_2': -0.3,
        'alpha_3': 0.7,
        'alpha_4': 0.1,
        'alpha_5': -0.2,
        'alpha_6': 0.9,
        'alpha_7': -0.1,
        'alpha_8': 0.4,
        'rationale': None,
        'generated_at': '2024-01-15T06:00:00',
        'cached': False,
    }
    cache_path.write_text(json.dumps(entry) + '\n')
    return AlphaService(
        model_name='gpt-4.1-mini',
        cache_path=str(cache_path),
        temperature=0.1,
    )


@pytest.fixture
def feature_snapshot() -> dict:
    """Minimal feature snapshot for prompt construction."""
    return {
        'close': 185.5,
        'volume': 5_000_000,
        'rsi_14': 55.2,
        'macd': 0.45,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_alpha_service_returns_eight_factors(feature_snapshot: dict) -> None:
    """deterministic_fallback_alphas returns exactly 8 float alpha keys."""
    result = deterministic_fallback_alphas(feature_snapshot)
    expected_keys = {f'alpha_{i}' for i in range(1, 9)}
    assert set(result.keys()) == expected_keys, (
        f'Expected keys {sorted(expected_keys)}, got {sorted(result.keys())}'
    )
    for key, val in result.items():
        assert isinstance(val, float), f'{key} is not a float: {type(val)}'
        assert -1.0 <= val <= 1.0, f'{key}={val} is outside [-1, 1]'


def test_alpha_service_cache_hit_skips_llm_call(
    cached_alpha_service: AlphaService,
    feature_snapshot: dict,
) -> None:
    """When cache entry exists, generate_alphas returns it without calling LLM."""
    # generate_alphas should find the cache entry and return it
    result = cached_alpha_service.generate_alphas(
        asset_id='AAPL',
        as_of_date='2024-01-15',
        feature_snapshot=feature_snapshot,
    )
    assert result['cached'] is True
    assert result['asset_id'] == 'AAPL'
    # Verify all 8 alpha keys are present
    for i in range(1, 9):
        key = f'alpha_{i}'
        assert key in result, f'{key} missing from cached result'
        assert isinstance(result[key], (int, float)), f'{key} is not numeric'


def test_alpha_service_fallback_on_llm_error(
    alpha_service: AlphaService,
    feature_snapshot: dict,
) -> None:
    """When LLM is unavailable, generate_alphas uses deterministic fallback."""
    # AlphaService._get_client() will fail since there's no OPENAI_API_KEY
    # This should trigger the deterministic fallback path
    result = alpha_service.generate_alphas(
        asset_id='MSFT',
        as_of_date='2024-02-01',
        feature_snapshot=feature_snapshot,
    )
    # Should have all 8 alpha keys (from deterministic fallback)
    for i in range(1, 9):
        key = f'alpha_{i}'
        assert key in result, f'{key} missing from fallback result'
        val = result[key]
        assert isinstance(val, (int, float)), f'{key} is not numeric: {type(val)}'
    # Should be marked as not cached (freshly generated)
    assert result['cached'] is False
    assert result['asset_id'] == 'MSFT'
    assert result['date'] == '2024-02-01'
