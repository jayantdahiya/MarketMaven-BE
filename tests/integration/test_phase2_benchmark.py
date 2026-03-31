"""Integration tests for the Phase 2.5 three-phase benchmark leaderboard."""

from __future__ import annotations

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

    p0_metrics = {
        'mae': 0.0110,
        'rmse': 0.0160,
        'sharpe': 0.70,
        'sortino': 1.05,
        'max_drawdown': -0.12,
    }
    p1_metrics = {
        'mae': 0.0100,
        'rmse': 0.0150,
        'sharpe': 0.95,
        'sortino': 1.20,
        'max_drawdown': -0.10,
    }
    p2_metrics = {
        'mae': 0.0095,
        'rmse': 0.0147,
        'sharpe': 1.05,
        'sortino': 1.35,
        'max_drawdown': -0.08,
    }

    for seed in ['seed_42', 'seed_123', 'seed_456']:
        _write_seed_metrics(phase0_dir, seed, p0_metrics)
        _write_seed_metrics(phase1_dir, seed, p1_metrics)
        _write_seed_metrics(phase2_dir, seed, p2_metrics)

    return phase0_dir, phase1_dir, phase2_dir


def _run_benchmark(
    phase0_dir: Path,
    phase1_dir: Path,
    phase2_dir: Path,
    output_path: Path,
):
    return subprocess.run(
        [
            sys.executable,
            'scripts/benchmark.py',
            '--phase0-reports',
            str(phase0_dir),
            '--phase1-reports',
            str(phase1_dir),
            '--phase2-reports',
            str(phase2_dir),
            '--output',
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )


def test_benchmark_includes_all_three_models(benchmark_3way_dirs, tmp_path) -> None:
    """benchmark.py should emit summaries for all three daily phases."""
    phase0_dir, phase1_dir, phase2_dir = benchmark_3way_dirs
    output_path = tmp_path / 'report_3way.json'

    result = _run_benchmark(phase0_dir, phase1_dir, phase2_dir, output_path)

    assert result.returncode == 0, result.stderr
    assert output_path.exists()

    with open(output_path) as f:
        report = json.load(f)

    assert 'phase_0_lstm' in report
    assert 'phase_1_cnn_transformer' in report
    assert 'phase_2_mamba_ssm' in report
    assert report['recommended_phase'] == 'phase_2_mamba_ssm'


def test_benchmark_emits_status_fields(benchmark_3way_dirs, tmp_path) -> None:
    """Each phase summary must include implemented/validated/acceptance/runtime flags."""
    phase0_dir, phase1_dir, phase2_dir = benchmark_3way_dirs
    output_path = tmp_path / 'report_status.json'

    _run_benchmark(phase0_dir, phase1_dir, phase2_dir, output_path)

    with open(output_path) as f:
        report = json.load(f)

    for model_key in [
        'phase_0_lstm',
        'phase_1_cnn_transformer',
        'phase_2_mamba_ssm',
    ]:
        status = report[model_key]['status']
        assert set(status) == {
            'implemented',
            'validated',
            'acceptance_passed',
            'recommended_for_runtime',
        }
        assert status['implemented'] is True
        assert status['validated'] is True


def test_benchmark_emits_pairwise_acceptance_and_seed_counts(
    benchmark_3way_dirs, tmp_path
) -> None:
    """Report must contain pairwise acceptance plus per-seed pass counts."""
    phase0_dir, phase1_dir, phase2_dir = benchmark_3way_dirs
    output_path = tmp_path / 'report_acceptance.json'

    _run_benchmark(phase0_dir, phase1_dir, phase2_dir, output_path)

    with open(output_path) as f:
        report = json.load(f)

    assert 'phase1_vs_phase0' in report['pairwise_acceptance']
    assert 'phase2_vs_phase1' in report['pairwise_acceptance']
    assert report['phase_1_cnn_transformer']['seed_acceptance']['pass_count'] == 3
    assert report['phase_2_mamba_ssm']['seed_acceptance']['pass_count'] == 3


def test_benchmark_leaderboard_is_ranked(benchmark_3way_dirs, tmp_path) -> None:
    """Leaderboard rows should be present and sorted by runtime recommendation first."""
    phase0_dir, phase1_dir, phase2_dir = benchmark_3way_dirs
    output_path = tmp_path / 'report_leaderboard.json'

    _run_benchmark(phase0_dir, phase1_dir, phase2_dir, output_path)

    with open(output_path) as f:
        report = json.load(f)

    leaderboard = report['leaderboard']
    assert leaderboard[0]['phase_key'] == 'phase_2_mamba_ssm'
    assert leaderboard[0]['recommended_for_runtime'] is True
    assert leaderboard[0]['rank'] == 1
