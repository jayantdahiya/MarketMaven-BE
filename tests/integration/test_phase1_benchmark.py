"""Integration tests for scripts/benchmark.py — validates output JSON structure."""

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
def benchmark_dirs(tmp_path):
    """Create synthetic phase0 and phase1 report directories."""
    phase0_dir = tmp_path / 'phase0'
    phase1_dir = tmp_path / 'phase1'

    # Phase 0 — LSTM baseline: 2 synthetic seeds
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

    # Phase 1 — CNN-Trans: 2 synthetic seeds with better metrics
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

    return phase0_dir, phase1_dir


def test_benchmark_produces_valid_json(benchmark_dirs, tmp_path):
    """benchmark.py must write a JSON report with the required top-level keys."""
    phase0_dir, phase1_dir = benchmark_dirs
    output_path = tmp_path / 'report.json'

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

    # Exit 0 (pass) or 1 (fail) — both are acceptable here, 2 means missing files
    assert result.returncode in (0, 1), (
        f'Unexpected exit code {result.returncode}.\nstderr: {result.stderr}'
    )
    assert output_path.exists(), 'Output JSON file was not created'

    with open(output_path) as f:
        report = json.load(f)

    # Top-level structure
    assert 'phase_0_lstm' in report, 'Missing key: phase_0_lstm'
    assert 'phase_1_cnn_transformer' in report, 'Missing key: phase_1_cnn_transformer'
    assert 'acceptance' in report, 'Missing key: acceptance'
    assert 'thresholds_used' in report, 'Missing key: thresholds_used'


def test_benchmark_acceptance_flags_present(benchmark_dirs, tmp_path):
    """acceptance block must contain all 5 pass/fail flags."""
    phase0_dir, phase1_dir = benchmark_dirs
    output_path = tmp_path / 'report.json'

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

    acceptance = report['acceptance']
    required_flags = [
        'mae_pass',
        'rmse_pass',
        'sharpe_pass',
        'sortino_pass',
        'max_dd_pass',
        'overall_pass',
    ]
    for flag in required_flags:
        assert flag in acceptance, f'Missing acceptance flag: {flag}'
        assert isinstance(acceptance[flag], bool), (
            f'{flag} must be bool, got {type(acceptance[flag])}'
        )


def test_benchmark_model_summary_metrics_present(benchmark_dirs, tmp_path):
    """Each model summary must contain all tracked metric keys."""
    phase0_dir, phase1_dir = benchmark_dirs
    output_path = tmp_path / 'report.json'

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
                f'{model_key}.{metric} missing mean/std keys'
            )


def test_benchmark_exits_2_on_missing_phase0(tmp_path):
    """benchmark.py must exit 2 when phase0 reports directory is empty."""
    empty_dir = tmp_path / 'empty_phase0'
    empty_dir.mkdir()
    phase1_dir = tmp_path / 'phase1'
    _write_seed_metrics(
        phase1_dir,
        'seed_42',
        {
            'mae': 0.02,
            'rmse': 0.024,
            'sharpe': 1.0,
            'sortino': 1.2,
            'max_drawdown': -0.1,
        },
    )
    output_path = tmp_path / 'report.json'

    result = subprocess.run(
        [
            sys.executable,
            'scripts/benchmark.py',
            '--phase0-reports',
            str(empty_dir),
            '--phase1-reports',
            str(phase1_dir),
            '--output',
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, (
        f'Expected exit code 2 for missing phase0, got {result.returncode}'
    )


def test_benchmark_all_criteria_pass_with_better_phase1(benchmark_dirs, tmp_path):
    """With synthetically dominant Phase 1 metrics, overall_pass must be True."""
    phase0_dir, _ = benchmark_dirs
    # Build a phase1 that clearly beats all thresholds
    dominant_phase1 = tmp_path / 'dominant_phase1'
    _write_seed_metrics(
        dominant_phase1,
        'seed_42',
        {
            'mae': 0.010,  # << 0.97 * 0.0227
            'rmse': 0.012,  # << 0.97 * 0.0268
            'sharpe': 2.0,  # >> 0.8217 + 0.10
            'sortino': 2.5,  # > 1.05
            'max_drawdown': -0.05,  # better than -0.12
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
            str(dominant_phase1),
            '--output',
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f'Expected exit code 0 (all pass), got {result.returncode}.\nstdout: {result.stdout}\nstderr: {result.stderr}'
    )
    with open(output_path) as f:
        report = json.load(f)
    assert report['acceptance']['overall_pass'] is True
