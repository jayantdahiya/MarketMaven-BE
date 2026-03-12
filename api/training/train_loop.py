"""
Trainer: fit loop with validation, early stopping, gradient clipping, checkpointing.

Supports ReduceLROnPlateau (step with val_loss), epoch-based schedulers
like CosineAnnealingLR (step with no argument), and per-batch schedulers
like OneCycleLR (step every batch inside _train_epoch).
"""

import logging
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from api.training import checkpointing, losses, metrics

logger = logging.getLogger(__name__)

# Schedulers that require val_loss as the step() argument.
_PLATEAU_SCHEDULERS = (ReduceLROnPlateau,)

# Schedulers that must be stepped every *batch* rather than every epoch.
_BATCH_SCHEDULERS = (OneCycleLR,)

# Union type for all supported scheduler kinds.
_AnyScheduler = LRScheduler | ReduceLROnPlateau


class Trainer:
    def __init__(
        self,
        cfg: dict,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: _AnyScheduler | None,
        criterion: torch.nn.Module,
        device: str | torch.device,
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device)
        self.current_epoch = 0
        self.train_cfg = cfg.get('training', {})
        self.epochs = self.train_cfg.get('epochs', 40)
        self.grad_clip = self.train_cfg.get('grad_clip_norm', 1.0)
        self.patience = self.train_cfg.get('early_stopping_patience', 8)
        self.checkpoints_dir = Path(
            cfg.get('paths', {}).get('checkpoints_dir', 'artifacts/checkpoints/phase0')
        )

    def _step_scheduler(self, val_loss: float) -> None:
        """Step the LR scheduler correctly based on its type.

        ReduceLROnPlateau requires the monitored metric; batch-level
        schedulers (OneCycleLR) are already stepped inside ``_train_epoch``;
        all other LR schedulers are stepped with no argument.
        """
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, _BATCH_SCHEDULERS):
            # Already stepped per-batch in _train_epoch — skip here.
            return
        if isinstance(self.scheduler, _PLATEAU_SCHEDULERS):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    # ------------------------------------------------------------------
    # Batch helpers — extract optional Phase-2 graph tensors from meta
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_graph_tensors(
        meta: dict | list,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return ``(asset_index, graph_context)`` from the meta dict(s).

        The default ``DataLoader`` collation turns ``meta`` into a dict of
        lists (one element per sample in the batch).  When neither field is
        present, both returned values are ``None``.
        """
        if isinstance(meta, dict):
            if 'asset_index' in meta:
                ai_list = meta['asset_index']
                if isinstance(ai_list, (list, tuple)):
                    asset_index = torch.tensor(ai_list, dtype=torch.long, device=device)
                elif isinstance(ai_list, torch.Tensor):
                    asset_index = ai_list.to(device)
                else:
                    asset_index = torch.tensor(
                        [ai_list], dtype=torch.long, device=device
                    )
            else:
                asset_index = None

            if 'graph_context' in meta:
                gc_list = meta['graph_context']
                if isinstance(gc_list, torch.Tensor):
                    # DataLoader stacks identical per-sample [A, G] tensors
                    # into [B, A, G]; take first slice to recover [A, G].
                    gc = gc_list[0] if gc_list.dim() == 3 else gc_list
                    graph_context = gc.to(device)
                elif isinstance(gc_list, (list, tuple)):
                    # Each element is the same shared tensor; take the first
                    graph_context = gc_list[0].to(device) if gc_list else None
                else:
                    graph_context = None
            else:
                graph_context = None

            return asset_index, graph_context
        return None, None

    def _model_forward(
        self,
        x: torch.Tensor,
        meta: dict | list,
    ) -> torch.Tensor:
        """Call the model, passing graph tensors when the model accepts them."""
        asset_index, graph_context = self._extract_graph_tensors(meta, self.device)
        if asset_index is not None:
            return self.model(x, asset_index=asset_index, graph_context=graph_context)
        return self.model(x)

    def _train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0.0
        n = 0
        for batch in loader:
            x, y_true, meta = batch
            x = x.to(self.device)
            y_true = y_true.to(self.device)
            self.optimizer.zero_grad()
            y_hat = self._model_forward(x, meta)
            if isinstance(self.criterion, losses.CompositeForecastLoss):
                loss = self.criterion(y_hat, y_true, current_epoch=self.current_epoch)
            else:
                loss = self.criterion(y_hat, y_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip
            )
            self.optimizer.step()
            # Per-batch scheduler stepping (e.g. OneCycleLR)
            if self.scheduler is not None and isinstance(
                self.scheduler, _BATCH_SCHEDULERS
            ):
                self.scheduler.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        return {'loss': total_loss / n if n else 0.0}

    def _validate_epoch(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_pred: list[torch.Tensor] = []
        all_true: list[torch.Tensor] = []
        n = 0
        with torch.no_grad():
            for batch in loader:
                x, y_true, meta = batch
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                y_hat = self._model_forward(x, meta)
                if isinstance(self.criterion, losses.CompositeForecastLoss):
                    loss = self.criterion(
                        y_hat, y_true, current_epoch=self.current_epoch
                    )
                else:
                    loss = self.criterion(y_hat, y_true)
                total_loss += loss.item() * x.size(0)
                all_pred.append(y_hat.cpu())
                all_true.append(y_true.cpu())
                n += x.size(0)
        y_pred = torch.cat(all_pred, dim=0).numpy().ravel()
        y_true_np = torch.cat(all_true, dim=0).numpy().ravel()
        return {
            'loss': total_loss / n if n else 0.0,
            'mae': metrics.mae(y_true_np, y_pred),
            'rmse': metrics.rmse(y_true_np, y_pred),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        run_id: str,
        feature_cols: list[str],
        scaler_path: str,
        scaler_state: dict,
    ) -> dict:
        """Run training; save best and last checkpoints. Return best_checkpoint_path and history."""
        best_val_loss = float('inf')
        patience_counter = 0
        history: list[dict] = []
        best_path = ''
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            train_m = self._train_epoch(train_loader)
            val_m = self._validate_epoch(val_loader)
            self._step_scheduler(val_m['loss'])
            history.append({
                'epoch': epoch,
                'train_loss': train_m['loss'],
                'val_loss': val_m['loss'],
                'val_mae': val_m['mae'],
                'val_rmse': val_m['rmse'],
            })
            logger.info(
                'Epoch %d train_loss=%.6f val_loss=%.6f val_mae=%.6f val_rmse=%.6f',
                epoch,
                train_m['loss'],
                val_m['loss'],
                val_m['mae'],
                val_m['rmse'],
            )
            if val_m['loss'] < best_val_loss:
                best_val_loss = val_m['loss']
                patience_counter = 0
                best_path = str(
                    self.checkpoints_dir / run_id / f'best_epoch_{epoch:03d}.pt'
                )
                checkpointing.save_checkpoint(
                    best_path,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    self.cfg,
                    feature_cols,
                    scaler_path,
                    scaler_state,
                    {'loss': val_m['loss'], 'mae': val_m['mae'], 'rmse': val_m['rmse']},
                )
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info('Early stopping at epoch %d', epoch)
                    break
        last_path = str(self.checkpoints_dir / run_id / 'last.pt')
        checkpointing.save_checkpoint(
            last_path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.current_epoch,
            self.cfg,
            feature_cols,
            scaler_path,
            scaler_state,
            history[-1] if history else {},
        )
        return {'best_checkpoint_path': best_path or last_path, 'history': history}
