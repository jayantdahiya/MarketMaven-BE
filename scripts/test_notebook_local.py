#!/usr/bin/env python3
"""Run colab_training.ipynb locally with reduced parameters for verification.

Usage:
    python scripts/test_notebook_local.py              # 2 seeds, 2 epochs
    python scripts/test_notebook_local.py --seeds 1    # 1 seed, 2 epochs
    python scripts/test_notebook_local.py --epochs 5   # 2 seeds, 5 epochs

The script patches cells for local execution:
  - Skips clone/install cells (already have repo + deps locally)
  - Rewrites REPO_DIR to CWD
  - Overrides seeds/epochs for speed
  - Uses a temp directory for artifacts
  - Uses Agg matplotlib backend (no display)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import nbclient
import nbformat


def patch_notebook(
    nb: nbformat.NotebookNode,
    *,
    repo_dir: str,
    seeds: list[int],
    epochs: int,
    artifact_dir: str,
) -> nbformat.NotebookNode:
    """Rewrite notebook cells for local headless execution."""
    device = 'mps' if sys.platform == 'darwin' else 'cpu'
    skip_titles = {'1a.', '1b.', '1c.'}

    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        src = cell.source

        # Skip GPU check, pip install, git clone cells
        first_line = src.split('\n')[0]
        if any(t in first_line for t in skip_titles):
            cell.source = '# Skipped for local run\npass'
            continue

        # Patch cell 1d: rewrite REPO_DIR, skip pip install -e .
        if '1d.' in first_line:
            cell.source = (
                f'import os, sys\n'
                f'from pathlib import Path\n'
                f'REPO_DIR = {repo_dir!r}\n'
                f'os.chdir(REPO_DIR)\n'
                f'if REPO_DIR not in sys.path:\n'
                f'    sys.path.insert(0, REPO_DIR)\n'
                f'from api.data.pipeline import DataPipeline, load_config\n'
                f'from api.models.factory import create_model\n'
                f'from api.training.train_loop import Trainer\n'
                f'from api.training import losses, evaluate\n'
                f"print('All imports successful.')\n"
            )
            continue

        # Patch cell 2a: override device, seeds, epochs, artifact paths
        if '2a.' in first_line:
            src = re.sub(
                r"DEVICE\s*=\s*'cuda'",
                f"DEVICE = '{device}'",
                src,
            )
            src = re.sub(
                r'EPOCHS\s*=\s*\d+',
                f'EPOCHS = {epochs}',
                src,
            )
            cell.source = src
            continue

        # Patch cell 7b: rewrite EXPORT_ZIP path
        if '7b.' in first_line:
            src = src.replace(
                "EXPORT_ZIP = '/content/marketmaven_artifacts.zip'",
                f"EXPORT_ZIP = '{artifact_dir}/marketmaven_artifacts.zip'",
            )
            cell.source = src
            continue

        # Patch cell 7c: skip google.colab download
        if '7c.' in first_line:
            cell.source = (
                "print(f'ZIP ready at: {EXPORT_ZIP}')\n"
                "import os; print(f'Size: {os.path.getsize(EXPORT_ZIP)/1e6:.1f} MB')\n"
            )
            continue

    # Inject seeds override into the training cell (4a) after seeds = ... line
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        if '4a.' in cell.source.split('\n')[0]:
            cell.source = cell.source.replace(
                "seeds = train_cfg.get('seeds', [42, 123, 456, 789, 1024])",
                f'seeds = {seeds}',
            )

    # Inject matplotlib Agg backend before any plotting cell
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        if '6a.' in cell.source.split('\n')[0]:
            cell.source = "import matplotlib\nmatplotlib.use('Agg')\n" + cell.source
            break

    # Replace plt.show() with plt.close() everywhere
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.source = cell.source.replace('plt.show()', 'plt.close()')

    return nb


def main():
    parser = argparse.ArgumentParser(description='Run Colab notebook locally')
    parser.add_argument(
        '--seeds', type=int, default=2, help='Number of seeds (default: 2)'
    )
    parser.add_argument('--epochs', type=int, default=2, help='Max epochs (default: 2)')
    parser.add_argument(
        '--keep-artifacts', action='store_true', help='Keep temp artifact dir'
    )
    args = parser.parse_args()

    repo_dir = str(Path(__file__).resolve().parent.parent)
    nb_path = Path(repo_dir) / 'notebooks' / 'colab_training.ipynb'

    if not nb_path.exists():
        print(f'ERROR: {nb_path} not found')
        sys.exit(1)

    all_seeds = [42, 123, 456, 789, 1024]
    seeds = all_seeds[: args.seeds]
    artifact_dir = tempfile.mkdtemp(prefix='nb_test_')

    print(f'Notebook: {nb_path}')
    print(f'Seeds:    {seeds}')
    print(f'Epochs:   {args.epochs}')
    print(f'Artifacts: {artifact_dir}')
    print('=' * 60)

    # Load and patch
    nb = nbformat.read(str(nb_path), as_version=4)
    nb = patch_notebook(
        nb,
        repo_dir=repo_dir,
        seeds=seeds,
        epochs=args.epochs,
        artifact_dir=artifact_dir,
    )

    # Execute
    client = nbclient.NotebookClient(
        nb,
        timeout=600,
        kernel_name='python3',
    )

    try:
        client.execute()
        print('\n' + '=' * 60)
        print('ALL CELLS PASSED')
        print('=' * 60)
    except nbclient.exceptions.CellExecutionError as e:
        print('\n' + '=' * 60)
        print(f'CELL EXECUTION FAILED')
        print('=' * 60)
        print(e)
        sys.exit(1)
    finally:
        if not args.keep_artifacts:
            shutil.rmtree(artifact_dir, ignore_errors=True)
            print(f'Cleaned up {artifact_dir}')
        else:
            print(f'Artifacts kept at {artifact_dir}')


if __name__ == '__main__':
    main()
