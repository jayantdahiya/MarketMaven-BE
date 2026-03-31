"""Integration tests for runtime config resolution and app boot behavior."""

from __future__ import annotations

from pathlib import Path

from api.app import create_app, resolve_config_path


def _route_paths(app) -> set[str]:
    return {route.path for route in app.router.routes}


def test_resolve_config_path_defaults_to_phase0(monkeypatch) -> None:
    monkeypatch.delenv('MARKET_MAVEN_CONFIG', raising=False)
    assert resolve_config_path() == Path('config/phase_0.yaml')


def test_resolve_config_path_uses_env_when_present(monkeypatch) -> None:
    monkeypatch.setenv('MARKET_MAVEN_CONFIG', 'config/phase_1.yaml')
    assert resolve_config_path() == Path('config/phase_1.yaml')


def test_create_app_default_runtime_is_phase0(monkeypatch) -> None:
    monkeypatch.delenv('MARKET_MAVEN_CONFIG', raising=False)
    app = create_app()

    assert app.state.config_path.endswith('config/phase_0.yaml')
    assert app.state.config['project']['phase'] == 0
    assert app.state.config['api']['default_model'] == 'lstm_baseline'
    assert app.state.config.get('multimodal', {}).get('enabled', False) is False
    assert '/forecast/lob' not in _route_paths(app)


def test_create_app_uses_env_phase1(monkeypatch) -> None:
    monkeypatch.setenv('MARKET_MAVEN_CONFIG', 'config/phase_1.yaml')
    app = create_app()

    assert app.state.config_path.endswith('config/phase_1.yaml')
    assert app.state.config['project']['phase'] == 1
    assert app.state.config['api']['default_model'] == 'cnn_transformer'
    assert app.state.alpha_service is None
    assert '/forecast/lob' not in _route_paths(app)


def test_create_app_explicit_config_overrides_env(monkeypatch) -> None:
    monkeypatch.setenv('MARKET_MAVEN_CONFIG', 'config/phase_1.yaml')
    app = create_app('config/phase_3.yaml')

    assert app.state.config_path.endswith('config/phase_3.yaml')
    assert app.state.config['project']['phase'] == 3
    assert app.state.config['multimodal']['enabled'] is True
    assert app.state.alpha_service is not None
    assert '/forecast/lob' not in _route_paths(app)
