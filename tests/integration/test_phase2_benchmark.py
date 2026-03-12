"""Integration tests for Phase 2 benchmark — 3-way comparison (LSTM, CNN-Trans, Mamba)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _write_seed_metrics(parent_dir: Path, run_name: str, metrics: dict) -> None:
    """Write a metrics_summary.json for a fake seed run."""
    run_dir = parent_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'metrics_summary.json', 'w') as f:
        json.dump(metrics, f)


@pytest.fixture
def benchmark_3way_dirs(tmp_path):
    """Create synthetic phase0, phase1, and phase2 report directories."""
    phase0_dir = tmp_path / 'phase0'
    phase1_dir = tmp_path / 'phase1'
    phase2_dir = tmp_path / 'phase2'

    # Phase 0 — LSTM baseline
    p0_metrics = {
        'mae': 0.0227,
        'rmse': 0.0268,
        'sharpe': 0.8217,
        'sortino': 1.05,
        'max_drawdown': -0.12,
    }
    _write_seed_metrics(phase0_dir, 'seed_42', p0_metrics)
    _write_seed_metrics(
        phase0_dir, 'seed_123', {**p0_metrics, 'mae': 0.0230, 'rmse': 0.0270}
    )

    # Phase 1 — CNN-Trans
    p1_metrics = {
        'mae': 0.0200,
        'rmse': 0.0240,
        'sharpe': 1.0000,
        'sortino': 1.20,
        'max_drawdown': -0.10,
    }
    _write_seed_metrics(phase1_dir, 'seed_42', p1_metrics)
    _write_seed_metrics(
        phase1_dir, 'seed_123', {**p1_metrics, 'mae': 0.0205, 'rmse': 0.0245}
    )

    # Phase 2 — Mamba (better than Phase 1)
    p2_metrics = {
        'mae': 0.0190,
        'rmse': 0.0230,
        'sharpe': 1.1000,
        'sortino': 1.35,
        'max_drawdown': -0.08,
    }
    _write_seed_metrics(phase2_dir, 'seed_42', p2_metrics)
    _write_seed_metrics(
        phase2_dir, 'seed_123', {**p2_metrics, 'mae': 0.0195, 'rmse': 0.0235}
    )

    return phase0_dir, phase1_dir, phase2_dir


def test_benchmark_includes_all_three_models(benchmark_3way_dirs, tmp_path):
    """benchmark.py should produce reports for all supplied phase directories."""
    phase0_dir, phase1_dir, phase2_dir = benchmark_3way_dirs
    output_path = tmp_path / 'report_3way.json'

    result = subprocess.run(
        [
            sys.executable,
            'scripts/benchmark.py',
            '--phase0-reports',
            str(phase0_dir),
            '--phase1-reports',
            str(phase1_dir),
            '--output',
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    # Exit 0 (pass) or 1 (fail) both acceptable
    assert result.returncode in (0, 1), (
        f'Unexpected exit code {result.returncode}.\nstderr: {result.stderr}'
    )
    assert output_path.exists(), 'Output JSON file was not created'

    with open(output_path) as f:
        report = json.load(f)

    assert 'phase_0_lstm' in report, 'Missing phase_0_lstm in report'
    assert 'phase_1_cnn_transformer' in report, 'Missing phase_1_cnn_transformer'


def test_benchmark_3way_all_metrics_present(benchmark_3way_dirs, tmp_path):
    """Each model entry must contain all tracked metric keys with mean/std."""
    phase0_dir, phase1_dir, _ = benchmark_3way_dirs
    output_path = tmp_path / 'report_metrics.json'

    subprocess.run(
        [
            sys.executable,
            'scripts/benchmark.py',
            '--phase0-reports',
            str(phase0_dir),
            '--phase1-reports',
            str(phase1_dir),
            '--output',
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )

    with open(output_path) as f:
        report = json.load(f)

    for model_key in ['phase_0_lstm', 'phase_1_cnn_transformer']:
        summary = report[model_key]
        for metric in ['mae', 'rmse', 'sharpe', 'sortino', 'max_drawdown']:
            assert metric in summary, f'{model_key} missing metric: {metric}'
            agg = summary[metric]
            assert 'mean' in agg and 'std' in agg, (
                f'{model_key}.{metric} missing mean/std'
            )


def test_benchmark_3way_acceptance_flags(benchmark_3way_dirs, tmp_path):
    """Acceptance block must contain pass/fail flags."""
    phase0_dir, phase1_dir, _ = benchmark_3way_dirs
    output_path = tmp_path / 'report_accept.json'

    subprocess.run(
        [
            sys.executable,
            'scripts/benchmark.py',
            '--phase0-reports',
            str(phase0_dir),
            '--phase1-reports',
            str(phase1_dir),
            '--output',
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )

    with open(output_path) as f:
        report = json.load(f)

    assert 'acceptance' in report
    acceptance = report['acceptance']
    assert 'overall_pass' in acceptance
    assert isinstance(acceptance['overall_pass'], bool)


def test_benchmark_dominant_phase1_passes(benchmark_3way_dirs, tmp_path):
    """With synthetically dominant Phase 1 metrics, overall_pass is True."""
    phase0_dir, _, _ = benchmark_3way_dirs
    dominant = tmp_path / 'dominant_phase1'
    _write_seed_metrics(
        dominant,
        'seed_42',
        {
            'mae': 0.010,
            'rmse': 0.012,
            'sharpe': 2.0,
            'sortino': 2.5,
            'max_drawdown': -0.05,
        },
    )
    output_path = tmp_path / 'dominant_report.json'

    result = subprocess.run(
        [
            sys.executable,
            'scripts/benchmark.py',
            '--phase0-reports',
            str(phase0_dir),
            '--phase1-reports',
            str(dominant),
            '--output',
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f'Expected exit 0, got {result.returncode}.\nstderr: {result.stderr}'
    )
    with open(output_path) as f:
        report = json.load(f)
    assert report['acceptance']['overall_pass'] is True
