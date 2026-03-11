"""
Checkpoint save/load: model, optimizer, scheduler, config, feature_cols, scaler_state.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    epoch: int,
    config: dict,
    feature_cols: list[str],
    scaler_path: str,
    scaler_state: dict,
    val_metrics: dict,
) -> None:
    """Save checkpoint dict to path. Creates parent dirs."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
        if scheduler is not None
        else None,
        'epoch': epoch,
        'config': config,
        'feature_cols': feature_cols,
        'scaler_path': scaler_path,
        'scaler_state': scaler_state,
        'val_metrics': val_metrics,
    }
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    device: str | torch.device = 'cpu',
) -> dict:
    """Load checkpoint, apply model.load_state_dict; optionally restore optimizer/scheduler. Return full dict."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(str(path_obj.resolve()))
    state = torch.load(path_obj, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'], strict=True)
    if optimizer is not None and state.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])
    if scheduler is not None and state.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(state['scheduler_state_dict'])
    return state
