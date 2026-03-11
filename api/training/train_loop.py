"""
Trainer: fit loop with validation, early stopping, gradient clipping, checkpointing.
"""

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from api.training import checkpointing, losses, metrics

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        cfg: dict,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
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

    def _train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0.0
        n = 0
        for batch in loader:
            x, y_true, _ = batch
            x = x.to(self.device)
            y_true = y_true.to(self.device)
            self.optimizer.zero_grad()
            y_hat = self.model(x)
            if isinstance(self.criterion, losses.CompositeForecastLoss):
                loss = self.criterion(y_hat, y_true, current_epoch=self.current_epoch)
            else:
                loss = self.criterion(y_hat, y_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip
            )
            self.optimizer.step()
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
                x, y_true, _ = batch
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                y_hat = self.model(x)
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
            if self.scheduler is not None:
                self.scheduler.step(val_m['loss'])
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
