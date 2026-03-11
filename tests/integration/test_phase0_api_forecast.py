"""Integration: forecast API contract."""

import pytest
from fastapi.testclient import TestClient

from api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_index_route_200(client):
    r = client.get('/')
    assert r.status_code == 200
    assert 'message' in r.json()


def test_forecast_daily_invalid_asset_422(client):
    r = client.post(
        '/forecast/daily',
        json={'asset_id': 'invalid ticker!', 'model': 'lstm_baseline'},
    )
    assert r.status_code == 422


def test_forecast_daily_invalid_model_400(client):
    r = client.post(
        '/forecast/daily', json={'asset_id': 'AAPL', 'model': 'unknown_model'}
    )
    assert r.status_code == 400


def test_forecast_daily_no_checkpoint_503(client):
    """When no checkpoint exists, LSTM forecast returns 503."""
    r = client.post(
        '/forecast/daily', json={'asset_id': 'AAPL', 'model': 'lstm_baseline'}
    )
    assert r.status_code == 503


def test_legacy_prophet_route_enabled(client):
    """GET /prophet returns 200 and expected keys (or 502 if yfinance fails)."""
    r = client.get('/prophet', params={'ticker': 'AAPL'})
    assert r.status_code in (200, 502)
    if r.status_code == 200:
        data = r.json()
        assert 'timestamps' in data or 'trend' in data
