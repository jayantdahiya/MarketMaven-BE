"""
Integration tests for POST /forecast/lob (Phase 4 LOB API).

Tests: successful forecast, 503 when module disabled, 422 on insufficient history.
The LOBService is fully mocked so no real model or data is needed.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from api.routes.lob import router as lob_router
from api.schemas.lob import LOBForecastResponse


# ---------------------------------------------------------------------------
# Helper: build a minimal FastAPI app with the LOB router + mocked state
# ---------------------------------------------------------------------------


def _build_app(
    lob_enabled: bool = True,
    lob_service=None,
) -> FastAPI:
    app = FastAPI()
    app.include_router(lob_router)
    app.state.config = {'lob': {'api_enabled': lob_enabled, 'tick_size': 0.01}}
    app.state.lob_service = lob_service
    return app


def _make_mock_service(
    get_window_raises: Exception | None = None,
) -> MagicMock:
    """Return a mock LOBService whose get_feature_window/predict behave as specified."""
    svc = MagicMock()

    T, C, L, E = 50, 4, 10, 4
    x_book = torch.randn(1, T, C, L)
    x_aux = torch.randn(1, T, E)

    if get_window_raises is not None:
        svc.get_feature_window.side_effect = get_window_raises
    else:
        svc.get_feature_window.return_value = (x_book, x_aux)

    svc.predict.return_value = LOBForecastResponse(
        asset_id='AAPL',
        as_of_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        predicted_class=2,
        class_probabilities=[0.1, 0.2, 0.7],
        expected_mid_move_ticks=0.014,
        confidence=0.7,
    )
    return svc


# ---------------------------------------------------------------------------
# Test 1 — Successful forecast returns 200 with valid LOBForecastResponse
# ---------------------------------------------------------------------------


def test_forecast_lob_success():
    """POST /forecast/lob returns 200 and a valid LOBForecastResponse."""
    mock_svc = _make_mock_service()
    app = _build_app(lob_enabled=True, lob_service=mock_svc)
    client = TestClient(app, raise_server_exceptions=False)

    payload = {'asset_id': 'AAPL', 'horizon_events': 20, 'model': 'tlob_forecaster'}
    resp = client.post('/forecast/lob', json=payload)

    assert resp.status_code == 200, f'Expected 200, got {resp.status_code}: {resp.text}'
    data = resp.json()
    assert data['asset_id'] == 'AAPL'
    assert data['predicted_class'] == 2
    assert len(data['class_probabilities']) == 3
    assert abs(sum(data['class_probabilities']) - 1.0) < 1e-5
    assert 'confidence' in data
    assert 'expected_mid_move_ticks' in data


# ---------------------------------------------------------------------------
# Test 2 — 503 returned when LOB module is disabled in config
# ---------------------------------------------------------------------------


def test_forecast_lob_503_when_disabled():
    """POST /forecast/lob returns 503 with lob_module_disabled when api_enabled=false."""
    app = _build_app(lob_enabled=False, lob_service=None)
    client = TestClient(app, raise_server_exceptions=False)

    payload = {'asset_id': 'AAPL', 'horizon_events': 20, 'model': 'tlob_forecaster'}
    resp = client.post('/forecast/lob', json=payload)

    assert resp.status_code == 503, f'Expected 503, got {resp.status_code}'
    detail = resp.json().get('detail', {})
    assert detail.get('code') == 'lob_module_disabled'


# ---------------------------------------------------------------------------
# Test 3 — 422 returned when get_feature_window raises ValueError (insufficient history)
# ---------------------------------------------------------------------------


def test_forecast_lob_422_on_insufficient_history():
    """POST /forecast/lob returns 422 with insufficient_lob_history when history is too short."""
    mock_svc = _make_mock_service(
        get_window_raises=ValueError('Not enough LOB events: need 50, got 10')
    )
    app = _build_app(lob_enabled=True, lob_service=mock_svc)
    client = TestClient(app, raise_server_exceptions=False)

    payload = {'asset_id': 'AAPL', 'horizon_events': 20, 'model': 'tlob_forecaster'}
    resp = client.post('/forecast/lob', json=payload)

    assert resp.status_code == 422, f'Expected 422, got {resp.status_code}: {resp.text}'
    detail = resp.json().get('detail', {})
    assert detail.get('code') == 'insufficient_lob_history'
    assert 'Not enough LOB events' in detail.get('message', '')
