#!/usr/bin/env python3
"""Cross-model benchmark runner.

Loads evaluation reports from two phase directories, computes acceptance
criteria (Phase 1 vs Phase 0), and outputs a ranked leaderboard JSON.

Usage::

    python scripts/benchmark.py \\
        --phase0-reports artifacts/reports/phase0 \\
        --phase1-reports artifacts/reports/phase1 \\
        --output artifacts/reports/phase1_vs_phase0.json

Exit code:
    0 — all acceptance criteria pass
    1 — one or more acceptance criteria fail
    2 — missing report files
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── Acceptance thresholds (defaults; overridable via CLI) ──────────────────────
DEFAULT_MAE_RATIO = 0.97  # cnn_trans_MAE <= 0.97 * lstm_MAE
DEFAULT_RMSE_RATIO = 0.97  # cnn_trans_RMSE <= 0.97 * lstm_RMSE
DEFAULT_SHARPE_DELTA = 0.10  # cnn_trans_Sharpe >= lstm_Sharpe + 0.10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Benchmark Phase 0 vs Phase 1 models.')
    p.add_argument(
        '--phase0-reports',
        default='artifacts/reports/phase0',
        help='Directory containing Phase 0 seed run sub-directories',
    )
    p.add_argument(
        '--phase1-reports',
        default='artifacts/reports/phase1',
        help='Directory containing Phase 1 seed run sub-directories',
    )
    p.add_argument(
        '--output',
        default='artifacts/reports/phase1_vs_phase0.json',
        help='Output path for the benchmark JSON report',
    )
    p.add_argument(
        '--mae-ratio',
        type=float,
        default=DEFAULT_MAE_RATIO,
        help='Acceptance: cnn_trans_MAE <= mae_ratio * lstm_MAE',
    )
    p.add_argument(
        '--rmse-ratio',
        type=float,
        default=DEFAULT_RMSE_RATIO,
        help='Acceptance: cnn_trans_RMSE <= rmse_ratio * lstm_RMSE',
    )
    p.add_argument(
        '--sharpe-delta',
        type=float,
        default=DEFAULT_SHARPE_DELTA,
        help='Acceptance: cnn_trans_Sharpe >= lstm_Sharpe + sharpe_delta',
    )
    return p.parse_args()


def _load_seed_metrics(reports_dir: Path) -> list[dict]:
    """Collect metrics_summary.json from all seed run sub-directories.

    Args:
        reports_dir: Parent directory; each child is a per-seed run folder.

    Returns:
        List of metric dicts, one per seed run found.
    """
    seed_runs = sorted(reports_dir.iterdir()) if reports_dir.exists() else []
    metrics_list: list[dict] = []
    for run_dir in seed_runs:
        summary_path = run_dir / 'metrics_summary.json'
        if run_dir.is_dir() and summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)
            data['_run_id'] = run_dir.name
            metrics_list.append(data)
        elif run_dir.is_file() and run_dir.suffix == '.json':
            # Flat layout fallback (single file named metrics_summary.json)
            if run_dir.name == 'metrics_summary.json':
                with open(run_dir) as f:
                    data = json.load(f)
                data['_run_id'] = 'single'
                metrics_list.append(data)
    return metrics_list


def _aggregate(metrics_list: list[dict], key: str) -> dict[str, float]:
    """Compute mean and std for a metric across seed runs.

    Args:
        metrics_list: List of per-seed metric dicts.
        key: Metric key to aggregate.

    Returns:
        Dict with 'mean', 'std', 'n', 'min', 'max'.
    """
    import statistics

    values = [m[key] for m in metrics_list if key in m]
    if not values:
        return {
            'mean': float('nan'),
            'std': float('nan'),
            'n': 0,
            'min': float('nan'),
            'max': float('nan'),
        }
    n = len(values)
    mean = sum(values) / n
    std = statistics.stdev(values) if n > 1 else 0.0
    return {'mean': mean, 'std': std, 'n': n, 'min': min(values), 'max': max(values)}


def _build_model_summary(metrics_list: list[dict], model_tag: str) -> dict:
    """Build a summary dict for a model across all seeds.

    Args:
        metrics_list: Per-seed metric dicts.
        model_tag: Label for this model (e.g. 'phase_0_lstm').

    Returns:
        Nested dict: {metric_name: {mean, std, n, min, max}}.
    """
    tracked = [
        'mae',
        'rmse',
        'directional_accuracy',
        'sharpe',
        'sortino',
        'max_drawdown',
    ]
    return {
        'model_tag': model_tag,
        'n_seeds': len(metrics_list),
        'seed_runs': [m.get('_run_id', '?') for m in metrics_list],
        **{k: _aggregate(metrics_list, k) for k in tracked},
    }


def _check_acceptance(
    phase0_summary: dict,
    phase1_summary: dict,
    mae_ratio: float,
    rmse_ratio: float,
    sharpe_delta: float,
) -> dict:
    """Evaluate whether Phase 1 meets all acceptance criteria vs Phase 0.

    Args:
        phase0_summary: Aggregated summary for Phase 0 model.
        phase1_summary: Aggregated summary for Phase 1 model.
        mae_ratio: Phase 1 MAE must be ≤ mae_ratio × Phase 0 MAE.
        rmse_ratio: Phase 1 RMSE must be ≤ rmse_ratio × Phase 0 RMSE.
        sharpe_delta: Phase 1 Sharpe must be ≥ Phase 0 Sharpe + sharpe_delta.

    Returns:
        Dict with per-criterion pass/fail flags and thresholds used.
    """
    p0_mae = phase0_summary['mae']['mean']
    p1_mae = phase1_summary['mae']['mean']
    p0_rmse = phase0_summary['rmse']['mean']
    p1_rmse = phase1_summary['rmse']['mean']
    p0_sharpe = phase0_summary['sharpe']['mean']
    p1_sharpe = phase1_summary['sharpe']['mean']

    mae_threshold = mae_ratio * p0_mae
    rmse_threshold = rmse_ratio * p0_rmse
    sharpe_threshold = p0_sharpe + sharpe_delta

    mae_pass = p1_mae <= mae_threshold
    rmse_pass = p1_rmse <= rmse_threshold
    sharpe_pass = p1_sharpe >= sharpe_threshold
    sortino_pass = (
        phase1_summary['sortino']['mean'] >= phase0_summary['sortino']['mean']
    )
    max_dd_pass = abs(phase1_summary['max_drawdown']['mean']) <= abs(
        phase0_summary['max_drawdown']['mean']
    )
    overall_pass = mae_pass and rmse_pass and sharpe_pass

    return {
        'mae_pass': mae_pass,
        'mae_phase0': p0_mae,
        'mae_phase1': p1_mae,
        'mae_threshold': mae_threshold,
        'rmse_pass': rmse_pass,
        'rmse_phase0': p0_rmse,
        'rmse_phase1': p1_rmse,
        'rmse_threshold': rmse_threshold,
        'sharpe_pass': sharpe_pass,
        'sharpe_phase0': p0_sharpe,
        'sharpe_phase1': p1_sharpe,
        'sharpe_threshold': sharpe_threshold,
        'sortino_pass': sortino_pass,
        'max_dd_pass': max_dd_pass,
        'overall_pass': overall_pass,
    }


def main() -> None:
    args = parse_args()

    phase0_dir = Path(args.phase0_reports)
    phase1_dir = Path(args.phase1_reports)
    output_path = Path(args.output)

    # Load per-seed metrics
    phase0_metrics = _load_seed_metrics(phase0_dir)
    phase1_metrics = _load_seed_metrics(phase1_dir)

    if not phase0_metrics:
        logger.error('No Phase 0 metrics found under %s', phase0_dir)
        sys.exit(2)
    if not phase1_metrics:
        logger.error('No Phase 1 metrics found under %s', phase1_dir)
        sys.exit(2)

    logger.info('Phase 0 seeds found: %d', len(phase0_metrics))
    logger.info('Phase 1 seeds found: %d', len(phase1_metrics))

    phase0_summary = _build_model_summary(phase0_metrics, 'phase_0_lstm')
    phase1_summary = _build_model_summary(phase1_metrics, 'phase_1_cnn_transformer')

    acceptance = _check_acceptance(
        phase0_summary,
        phase1_summary,
        mae_ratio=args.mae_ratio,
        rmse_ratio=args.rmse_ratio,
        sharpe_delta=args.sharpe_delta,
    )

    report = {
        'phase_0_lstm': phase0_summary,
        'phase_1_cnn_transformer': phase1_summary,
        'acceptance': acceptance,
        'thresholds_used': {
            'mae_ratio': args.mae_ratio,
            'rmse_ratio': args.rmse_ratio,
            'sharpe_delta': args.sharpe_delta,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info('Benchmark report written to %s', output_path)

    # Print summary table
    print('\n── Benchmark Summary ─────────────────────────────────────')
    for tag, summary in [
        ('Phase 0 (LSTM)', phase0_summary),
        ('Phase 1 (CNN-Trans)', phase1_summary),
    ]:
        print(
            f'  {tag:22s}  '
            f'MAE={summary["mae"]["mean"]:.4f}  '
            f'RMSE={summary["rmse"]["mean"]:.4f}  '
            f'Sharpe={summary["sharpe"]["mean"]:.3f}  '
            f'(n={summary["n_seeds"]} seeds)'
        )
    print()
    for criterion, passed in [
        ('MAE improvement', acceptance['mae_pass']),
        ('RMSE improvement', acceptance['rmse_pass']),
        ('Sharpe improvement', acceptance['sharpe_pass']),
        ('Sortino non-regression', acceptance['sortino_pass']),
        ('Max drawdown non-regression', acceptance['max_dd_pass']),
    ]:
        status = 'PASS' if passed else 'FAIL'
        print(f'  {criterion:30s}  {status}')
    overall = acceptance['overall_pass']
    print(f'\n  Overall: {"PASS" if overall else "FAIL"}')
    print('──────────────────────────────────────────────────────────\n')

    sys.exit(0 if overall else 1)


if __name__ == '__main__':
    main()
