#!/usr/bin/env python3
"""Cross-model benchmark runner for Phases 0, 1, and 2.

Generates a canonical leaderboard JSON with:
- per-phase aggregate metrics
- per-seed acceptance counts
- pairwise acceptance checks
- runtime recommendation status
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

PHASE0_DEFAULTS = {
    'mae_max': 0.0130,
    'rmse_max': 0.0185,
    'sharpe_min': 0.55,
}
PHASE1_DEFAULTS = {
    'mae_ratio': 0.97,
    'rmse_ratio': 0.97,
    'sharpe_delta': 0.10,
}
PHASE2_DEFAULTS = {
    'mae_ratio': 0.99,
    'sharpe_delta': 0.05,
}
MIN_SEED_PASS_COUNT = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Benchmark Phases 0, 1, and 2.')
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
        '--phase2-reports',
        default=None,
        help='Optional directory containing Phase 2 seed run sub-directories',
    )
    p.add_argument(
        '--output',
        default='artifacts/reports/leaderboard.json',
        help='Output path for the benchmark JSON report',
    )
    p.add_argument(
        '--phase0-mae-max',
        type=float,
        default=PHASE0_DEFAULTS['mae_max'],
        help='Acceptance: Phase 0 MAE <= threshold',
    )
    p.add_argument(
        '--phase0-rmse-max',
        type=float,
        default=PHASE0_DEFAULTS['rmse_max'],
        help='Acceptance: Phase 0 RMSE <= threshold',
    )
    p.add_argument(
        '--phase0-sharpe-min',
        type=float,
        default=PHASE0_DEFAULTS['sharpe_min'],
        help='Acceptance: Phase 0 Sharpe >= threshold',
    )
    p.add_argument(
        '--phase1-mae-ratio',
        type=float,
        default=PHASE1_DEFAULTS['mae_ratio'],
        help='Acceptance: Phase 1 MAE <= ratio * Phase 0 mean MAE',
    )
    p.add_argument(
        '--phase1-rmse-ratio',
        type=float,
        default=PHASE1_DEFAULTS['rmse_ratio'],
        help='Acceptance: Phase 1 RMSE <= ratio * Phase 0 mean RMSE',
    )
    p.add_argument(
        '--phase1-sharpe-delta',
        type=float,
        default=PHASE1_DEFAULTS['sharpe_delta'],
        help='Acceptance: Phase 1 Sharpe >= Phase 0 mean Sharpe + delta',
    )
    p.add_argument(
        '--phase2-mae-ratio',
        type=float,
        default=PHASE2_DEFAULTS['mae_ratio'],
        help='Acceptance: Phase 2 MAE <= ratio * Phase 1 mean MAE',
    )
    p.add_argument(
        '--phase2-sharpe-delta',
        type=float,
        default=PHASE2_DEFAULTS['sharpe_delta'],
        help='Acceptance: Phase 2 Sharpe >= Phase 1 mean Sharpe + delta',
    )
    return p.parse_args()


def _load_seed_metrics(reports_dir: Path) -> list[dict]:
    """Collect metrics_summary.json from all seed run sub-directories."""
    seed_runs = sorted(reports_dir.iterdir()) if reports_dir.exists() else []
    metrics_list: list[dict] = []
    for run_dir in seed_runs:
        summary_path = run_dir / 'metrics_summary.json'
        if run_dir.is_dir() and summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)
            data['_run_id'] = run_dir.name
            metrics_list.append(data)
        elif run_dir.is_file() and run_dir.name == 'metrics_summary.json':
            with open(run_dir) as f:
                data = json.load(f)
            data['_run_id'] = 'single'
            metrics_list.append(data)
    return metrics_list


def _aggregate(metrics_list: list[dict], key: str) -> dict[str, float]:
    """Compute aggregate stats for a metric across seed runs."""
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


def _phase0_seed_acceptance(metrics_list: list[dict], args: argparse.Namespace) -> dict:
    """Return per-seed acceptance for the absolute Phase 0 thresholds."""
    passing = [
        m.get('_run_id', '?')
        for m in metrics_list
        if m.get('mae', float('inf')) <= args.phase0_mae_max
        and m.get('rmse', float('inf')) <= args.phase0_rmse_max
        and m.get('sharpe', float('-inf')) >= args.phase0_sharpe_min
    ]
    return {
        'pass_count': len(passing),
        'required_pass_count': MIN_SEED_PASS_COUNT,
        'passing_runs': passing,
        'thresholds': {
            'mae_max': args.phase0_mae_max,
            'rmse_max': args.phase0_rmse_max,
            'sharpe_min': args.phase0_sharpe_min,
        },
        'acceptance_passed': len(passing) >= MIN_SEED_PASS_COUNT,
    }


def _pairwise_seed_acceptance(
    baseline_metrics: list[dict],
    candidate_metrics: list[dict],
    *,
    mae_ratio: float,
    sharpe_delta: float,
    rmse_ratio: float | None = None,
) -> dict:
    """Return per-seed acceptance counts against aggregate baseline thresholds."""
    if not baseline_metrics or not candidate_metrics:
        return {
            'pass_count': 0,
            'required_pass_count': MIN_SEED_PASS_COUNT,
            'passing_runs': [],
            'thresholds': {},
            'acceptance_passed': False,
        }

    baseline_mae = sum(m['mae'] for m in baseline_metrics) / len(baseline_metrics)
    baseline_sharpe = sum(m['sharpe'] for m in baseline_metrics) / len(baseline_metrics)
    thresholds: dict[str, float] = {
        'mae_max': mae_ratio * baseline_mae,
        'sharpe_min': baseline_sharpe + sharpe_delta,
    }
    if rmse_ratio is not None:
        baseline_rmse = sum(m['rmse'] for m in baseline_metrics) / len(baseline_metrics)
        thresholds['rmse_max'] = rmse_ratio * baseline_rmse

    passing = []
    for metrics_row in candidate_metrics:
        if metrics_row.get('mae', float('inf')) > thresholds['mae_max']:
            continue
        if metrics_row.get('sharpe', float('-inf')) < thresholds['sharpe_min']:
            continue
        if 'rmse_max' in thresholds and metrics_row.get('rmse', float('inf')) > thresholds['rmse_max']:
            continue
        passing.append(metrics_row.get('_run_id', '?'))

    return {
        'pass_count': len(passing),
        'required_pass_count': MIN_SEED_PASS_COUNT,
        'passing_runs': passing,
        'thresholds': thresholds,
        'acceptance_passed': len(passing) >= MIN_SEED_PASS_COUNT,
    }


def _build_model_summary(metrics_list: list[dict], model_tag: str, reports_dir: Path) -> dict:
    """Build a summary dict for a model across all seeds."""
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
        'reports_dir': str(reports_dir),
        'n_seeds': len(metrics_list),
        'seed_runs': [m.get('_run_id', '?') for m in metrics_list],
        **{k: _aggregate(metrics_list, k) for k in tracked},
    }


def _check_acceptance(
    baseline_summary: dict,
    candidate_summary: dict,
    *,
    mae_ratio: float,
    sharpe_delta: float,
    rmse_ratio: float | None = None,
) -> dict:
    """Evaluate whether a candidate phase meets aggregate acceptance criteria."""
    base_mae = baseline_summary['mae']['mean']
    cand_mae = candidate_summary['mae']['mean']
    base_sharpe = baseline_summary['sharpe']['mean']
    cand_sharpe = candidate_summary['sharpe']['mean']

    mae_threshold = mae_ratio * base_mae
    sharpe_threshold = base_sharpe + sharpe_delta
    mae_pass = cand_mae <= mae_threshold
    sharpe_pass = cand_sharpe >= sharpe_threshold

    rmse_threshold = None
    rmse_pass = True
    if rmse_ratio is not None:
        base_rmse = baseline_summary['rmse']['mean']
        cand_rmse = candidate_summary['rmse']['mean']
        rmse_threshold = rmse_ratio * base_rmse
        rmse_pass = cand_rmse <= rmse_threshold
    else:
        cand_rmse = candidate_summary['rmse']['mean']

    sortino_pass = (
        candidate_summary['sortino']['mean'] >= baseline_summary['sortino']['mean']
    )
    max_dd_pass = abs(candidate_summary['max_drawdown']['mean']) <= abs(
        baseline_summary['max_drawdown']['mean']
    )
    overall_pass = mae_pass and sharpe_pass and rmse_pass

    result = {
        'mae_pass': mae_pass,
        'mae_baseline': base_mae,
        'mae_candidate': cand_mae,
        'mae_threshold': mae_threshold,
        'sharpe_pass': sharpe_pass,
        'sharpe_baseline': base_sharpe,
        'sharpe_candidate': cand_sharpe,
        'sharpe_threshold': sharpe_threshold,
        'sortino_pass': sortino_pass,
        'max_dd_pass': max_dd_pass,
        'overall_pass': overall_pass,
    }
    if rmse_threshold is not None:
        result.update({
            'rmse_pass': rmse_pass,
            'rmse_baseline': baseline_summary['rmse']['mean'],
            'rmse_candidate': cand_rmse,
            'rmse_threshold': rmse_threshold,
        })
    return result


def _with_status(
    summary: dict,
    *,
    implemented: bool,
    validated: bool,
    acceptance_passed: bool,
    recommended_for_runtime: bool,
    seed_acceptance: dict,
) -> dict:
    """Attach runtime status metadata to a phase summary."""
    return {
        **summary,
        'seed_acceptance': seed_acceptance,
        'status': {
            'implemented': implemented,
            'validated': validated,
            'acceptance_passed': acceptance_passed,
            'recommended_for_runtime': recommended_for_runtime,
        },
    }


def _leaderboard_rows(report: dict) -> list[dict]:
    """Build a sortable leaderboard view from the phase summaries."""
    rows = []
    for phase_key in [
        'phase_0_lstm',
        'phase_1_cnn_transformer',
        'phase_2_mamba_ssm',
    ]:
        summary = report.get(phase_key)
        if not summary:
            continue
        status = summary.get('status', {})
        rows.append({
            'phase_key': phase_key,
            'model_tag': summary.get('model_tag'),
            'mae': summary['mae']['mean'],
            'rmse': summary['rmse']['mean'],
            'sharpe': summary['sharpe']['mean'],
            'n_seeds': summary.get('n_seeds', 0),
            'implemented': status.get('implemented', False),
            'validated': status.get('validated', False),
            'acceptance_passed': status.get('acceptance_passed', False),
            'recommended_for_runtime': status.get('recommended_for_runtime', False),
        })
    rows.sort(
        key=lambda row: (
            not row['recommended_for_runtime'],
            not row['validated'],
            not row['acceptance_passed'],
            -row['sharpe'],
            row['mae'],
        )
    )
    for idx, row in enumerate(rows, start=1):
        row['rank'] = idx
    return rows


def main() -> None:
    args = parse_args()

    phase0_dir = Path(args.phase0_reports)
    phase1_dir = Path(args.phase1_reports)
    phase2_dir = Path(args.phase2_reports) if args.phase2_reports else None
    output_path = Path(args.output)

    phase0_metrics = _load_seed_metrics(phase0_dir)
    phase1_metrics = _load_seed_metrics(phase1_dir)
    phase2_metrics = _load_seed_metrics(phase2_dir) if phase2_dir else []

    if not phase0_metrics:
        logger.error('No Phase 0 metrics found under %s', phase0_dir)
        sys.exit(2)
    if not phase1_metrics:
        logger.error('No Phase 1 metrics found under %s', phase1_dir)
        sys.exit(2)

    phase0_summary = _build_model_summary(phase0_metrics, 'phase_0_lstm', phase0_dir)
    phase1_summary = _build_model_summary(
        phase1_metrics, 'phase_1_cnn_transformer', phase1_dir
    )

    phase0_seed_acceptance = _phase0_seed_acceptance(phase0_metrics, args)
    phase1_seed_acceptance = _pairwise_seed_acceptance(
        phase0_metrics,
        phase1_metrics,
        mae_ratio=args.phase1_mae_ratio,
        rmse_ratio=args.phase1_rmse_ratio,
        sharpe_delta=args.phase1_sharpe_delta,
    )
    phase1_vs_phase0 = _check_acceptance(
        phase0_summary,
        phase1_summary,
        mae_ratio=args.phase1_mae_ratio,
        rmse_ratio=args.phase1_rmse_ratio,
        sharpe_delta=args.phase1_sharpe_delta,
    )

    phase2_summary = None
    phase2_seed_acceptance = None
    phase2_vs_phase1 = None
    if phase2_dir and phase2_metrics:
        phase2_summary = _build_model_summary(
            phase2_metrics, 'phase_2_mamba_ssm', phase2_dir
        )
        phase2_seed_acceptance = _pairwise_seed_acceptance(
            phase1_metrics,
            phase2_metrics,
            mae_ratio=args.phase2_mae_ratio,
            sharpe_delta=args.phase2_sharpe_delta,
        )
        phase2_vs_phase1 = _check_acceptance(
            phase1_summary,
            phase2_summary,
            mae_ratio=args.phase2_mae_ratio,
            sharpe_delta=args.phase2_sharpe_delta,
        )

    recommended_phase = 'phase_0_lstm'
    if (
        phase1_summary['mae']['mean'] <= phase0_summary['mae']['mean']
        and phase1_summary['sharpe']['mean'] > phase0_summary['sharpe']['mean']
    ):
        recommended_phase = 'phase_1_cnn_transformer'
    if phase2_summary and phase2_seed_acceptance and phase2_seed_acceptance['acceptance_passed']:
        recommended_phase = 'phase_2_mamba_ssm'

    report = {
        'phase_0_lstm': _with_status(
            phase0_summary,
            implemented=True,
            validated=True,
            acceptance_passed=phase0_seed_acceptance['acceptance_passed'],
            recommended_for_runtime=recommended_phase == 'phase_0_lstm',
            seed_acceptance=phase0_seed_acceptance,
        ),
        'phase_1_cnn_transformer': _with_status(
            phase1_summary,
            implemented=True,
            validated=True,
            acceptance_passed=phase1_seed_acceptance['acceptance_passed'],
            recommended_for_runtime=recommended_phase == 'phase_1_cnn_transformer',
            seed_acceptance=phase1_seed_acceptance,
        ),
        'acceptance': phase1_vs_phase0,
        'pairwise_acceptance': {'phase1_vs_phase0': phase1_vs_phase0},
        'thresholds_used': {
            'phase0': phase0_seed_acceptance['thresholds'],
            'phase1': {
                'mae_ratio': args.phase1_mae_ratio,
                'rmse_ratio': args.phase1_rmse_ratio,
                'sharpe_delta': args.phase1_sharpe_delta,
            },
        },
    }

    if phase2_summary and phase2_seed_acceptance and phase2_vs_phase1:
        report['phase_2_mamba_ssm'] = _with_status(
            phase2_summary,
            implemented=True,
            validated=True,
            acceptance_passed=phase2_seed_acceptance['acceptance_passed'],
            recommended_for_runtime=recommended_phase == 'phase_2_mamba_ssm',
            seed_acceptance=phase2_seed_acceptance,
        )
        report['pairwise_acceptance']['phase2_vs_phase1'] = phase2_vs_phase1
        report['thresholds_used']['phase2'] = {
            'mae_ratio': args.phase2_mae_ratio,
            'sharpe_delta': args.phase2_sharpe_delta,
        }

    report['leaderboard'] = _leaderboard_rows(report)
    report['recommended_phase'] = recommended_phase

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info('Benchmark report written to %s', output_path)

    print('\n-- Benchmark Summary ------------------------------------')
    for row in report['leaderboard']:
        print(
            f"  rank={row['rank']} phase={row['phase_key']} "
            f"MAE={row['mae']:.4f} RMSE={row['rmse']:.4f} "
            f"Sharpe={row['sharpe']:.3f} recommended={row['recommended_for_runtime']}"
        )
    print(f"\n  Recommended runtime phase: {report['recommended_phase']}")
    print('--------------------------------------------------------\n')

    overall_pass = phase1_vs_phase0['overall_pass']
    if phase2_vs_phase1 is not None:
        overall_pass = overall_pass and phase2_vs_phase1['overall_pass']
    sys.exit(0 if overall_pass else 1)


if __name__ == '__main__':
    main()
