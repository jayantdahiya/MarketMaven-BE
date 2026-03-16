"""Integration tests for Phase 3 multimodal API endpoints."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.schemas.forecast import (
    DailyForecastPoint,
    MultimodalForecastResponse,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def multimodal_client(tmp_path):
    """TestClient with multimodal enabled and mocked services."""
    app = create_app()

    # Enable multimodal in config
    app.state.config['multimodal'] = {
        'enabled': True,
        'alpha': {'enabled': True},
    }
    app.state.config.setdefault('api', {})['alpha_version'] = 'phase3-v1'

    # Mock forecast_service.predict_multimodal
    mock_service = MagicMock()
    mock_service.predict_multimodal.return_value = MultimodalForecastResponse(
        asset_id='AAPL',
        model='cnn_transformer_multimodal',
        horizon_days=1,
        generated_at=datetime(2026, 3, 12, 12, 0, 0),
        predictions=[
            DailyForecastPoint(
                timestamp=datetime(2026, 3, 13, 0, 0, 0),
                predicted_return=0.0062,
                predicted_price=213.02,
                signal='long',
                confidence=0.69,
            )
        ],
        used_modalities=['price_tech', 'context', 'sentiment', 'alpha'],
        alpha_version='phase3-v1',
    )
    app.state.forecast_service = mock_service

    # Mock alpha_service
    mock_alpha = MagicMock()
    app.state.alpha_service = mock_alpha

    return TestClient(app)


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_forecast_multimodal_success(multimodal_client) -> None:
    """POST /forecast/multimodal returns 200 with used_modalities."""
    resp = multimodal_client.post(
        '/forecast/multimodal',
        json={
            'asset_id': 'AAPL',
            'horizon_days': 1,
            'model': 'cnn_transformer',
            'include_context': True,
            'include_sentiment': True,
            'include_alpha': True,
        },
    )
    assert resp.status_code == 200, f'Expected 200, got {resp.status_code}: {resp.text}'
    data = resp.json()
    assert 'used_modalities' in data
    assert 'price_tech' in data['used_modalities']
    assert data['alpha_version'] == 'phase3-v1'
    assert len(data['predictions']) == 1
    assert data['predictions'][0]['signal'] in ('long', 'flat')


def test_forecast_multimodal_rejects_all_modalities_false(multimodal_client) -> None:
    """POST /forecast/multimodal returns 400 when all include flags are false."""
    resp = multimodal_client.post(
        '/forecast/multimodal',
        json={
            'asset_id': 'AAPL',
            'horizon_days': 1,
            'model': 'cnn_transformer',
            'include_context': False,
            'include_sentiment': False,
            'include_alpha': False,
        },
    )
    assert resp.status_code == 400, f'Expected 400, got {resp.status_code}: {resp.text}'
    detail = resp.json()['detail']
    assert detail['code'] == 'no_modalities'


def test_alphas_endpoint_returns_cached_payload(multimodal_client) -> None:
    """GET /alphas returns 200 with cached alpha payload."""
    # Configure mock alpha_service to return a cached result
    mock_alpha = multimodal_client.app.state.alpha_service
    mock_alpha.get_cached_alphas.return_value = {
        'cache_key': 'AAPL_2024-01-15',
        'asset_id': 'AAPL',
        'date': '2024-01-15',
        'alpha_1': 0.42,
        'alpha_2': -0.18,
        'alpha_3': 0.75,
        'alpha_4': -0.03,
        'alpha_5': 0.51,
        'alpha_6': -0.29,
        'alpha_7': 0.12,
        'alpha_8': 0.08,
        'generated_at': '2024-01-15T06:00:00',
        'rationale': 'Strong momentum',
    }

    resp = multimodal_client.get(
        '/alphas',
        params={'asset_id': 'AAPL', 'date': '2024-01-15'},
    )
    assert resp.status_code == 200, f'Expected 200, got {resp.status_code}: {resp.text}'
    data = resp.json()
    assert data['cached'] is True
    assert data['asset_id'] == 'AAPL'
    assert data['alpha_version'] == 'phase3-v1'
    # Verify all 8 alpha keys present
    alpha_vals = data['alpha_values']
    assert len(alpha_vals) == 8
    for i in range(1, 9):
        assert f'alpha_{i}' in alpha_vals
