"""
Trainer: fit loop with validation, early stopping, gradient clipping, checkpointing.

Supports ReduceLROnPlateau (step with val_loss), epoch-based schedulers
like CosineAnnealingLR (step with no argument), and per-batch schedulers
like OneCycleLR (step every batch inside _train_epoch).

Also supports LOB classification task via fit_lob(), which expects 4-tuple
batches (x_book, x_aux, y_class, meta) and tracks validation balanced accuracy.
"""

import logging
from pathlib import Path

import numpy as np
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

    @staticmethod
    def _extract_modality_mask(
        meta: dict | list,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Return ``modality_mask`` tensor ``[B, M]`` from the meta dict.

        The default collation stacks per-sample ``[M]`` tensors into
        ``[B, M]``.  Returns ``None`` when the field is absent.
        """
        if isinstance(meta, dict) and 'modality_mask' in meta:
            mask = meta['modality_mask']
            if isinstance(mask, torch.Tensor):
                return mask.to(device)
            if isinstance(mask, (list, tuple)):
                return torch.stack(mask).to(device)
        return None

    def _model_forward(
        self,
        x: torch.Tensor,
        meta: dict | list,
    ) -> torch.Tensor:
        """Call the model, passing graph tensors and/or multimodal mask when available."""
        asset_index, graph_context = self._extract_graph_tensors(meta, self.device)
        modality_mask = self._extract_modality_mask(meta, self.device)

        # Phase 2: graph-context path
        if asset_index is not None:
            return self.model(x, asset_index=asset_index, graph_context=graph_context)
        # Phase 3: multimodal path
        if modality_mask is not None:
            return self.model(x, modality_mask=modality_mask)
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
            # Phase 3: gate L1 penalty on modality gate parameters
            if (
                isinstance(self.criterion, losses.CompositeForecastLoss)
                and self.criterion.lambda_gate_l1 > 0.0
                and hasattr(self.model, 'gate_params')
            ):
                gate_l1 = self.model.gate_params.abs().sum()
                loss = loss + self.criterion.lambda_gate_l1 * gate_l1
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
        """Run training; save best and last checkpoints. Return best_checkpoint_path and history.

        Best checkpoint is selected by lowest val_mae (not val_loss) so that the
        composite Sharpe surrogate loss cannot drive checkpoint selection toward a
        collapsed constant-output model.  Early stopping also tracks val_mae.
        """
        best_val_mae = float('inf')
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
            if val_m['mae'] < best_val_mae:
                best_val_mae = val_m['mae']
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

    # ------------------------------------------------------------------
    # LOB classification task support
    # ------------------------------------------------------------------

    def _step_scheduler_lob(self, val_balacc: float) -> None:
        """Step the LR scheduler using validation balanced accuracy (mode='max').

        For ReduceLROnPlateau in LOB mode we negate val_balacc so that the
        scheduler (configured mode='min' externally) still fires correctly.
        Callers are responsible for providing a scheduler with mode='max' or
        we handle the negation here transparently.
        """
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, _BATCH_SCHEDULERS):
            return
        if isinstance(self.scheduler, _PLATEAU_SCHEDULERS):
            # Pass negative balacc so that ReduceLROnPlateau(mode='min') behaves
            # equivalently to mode='max' on positive balacc.
            self.scheduler.step(-val_balacc)
        else:
            self.scheduler.step()

    def _lob_train_epoch(self, loader: DataLoader) -> dict:
        """Train one epoch on LOB 4-tuple batches (x_book, x_aux, y_class, meta)."""
        self.model.train()
        total_loss = 0.0
        n = 0
        for batch in loader:
            x_book, x_aux, y_class, _meta = batch
            x_book = x_book.to(self.device)
            x_aux = x_aux.to(self.device)
            y_class = y_class.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x_book, x_aux)
            loss = self.criterion(logits, y_class)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip
            )
            self.optimizer.step()
            if self.scheduler is not None and isinstance(
                self.scheduler, _BATCH_SCHEDULERS
            ):
                self.scheduler.step()
            total_loss += loss.item() * x_book.size(0)
            n += x_book.size(0)
        return {'loss': total_loss / n if n else 0.0}

    def _lob_validate_epoch(self, loader: DataLoader) -> dict:
        """Validate one epoch on LOB 4-tuple batches; returns loss + balanced_accuracy."""
        self.model.eval()
        total_loss = 0.0
        all_pred: list[np.ndarray] = []
        all_true: list[np.ndarray] = []
        n = 0
        with torch.no_grad():
            for batch in loader:
                x_book, x_aux, y_class, _meta = batch
                x_book = x_book.to(self.device)
                x_aux = x_aux.to(self.device)
                y_class = y_class.to(self.device)
                logits = self.model(x_book, x_aux)
                loss = self.criterion(logits, y_class)
                total_loss += loss.item() * x_book.size(0)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_pred.append(preds)
                all_true.append(y_class.cpu().numpy())
                n += x_book.size(0)
        y_pred = np.concatenate(all_pred) if all_pred else np.array([])
        y_true_np = np.concatenate(all_true) if all_true else np.array([])
        balacc = metrics.balanced_accuracy_3class(y_true_np, y_pred)
        return {
            'loss': total_loss / n if n else 0.0,
            'balanced_accuracy': float(balacc) if not np.isnan(balacc) else 0.0,
        }

    def fit_lob(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        run_id: str,
    ) -> dict:
        """Run LOB classification training loop.

        Tracks validation balanced accuracy (higher is better) for checkpointing
        and early stopping. Checkpoint filename: best_balacc_epoch_{epoch:03d}.pt.

        Args:
            train_loader: DataLoader yielding (x_book, x_aux, y_class, meta).
            val_loader: DataLoader yielding (x_book, x_aux, y_class, meta).
            run_id: Unique identifier for this run (used for checkpoint subdir).

        Returns:
            Dict with 'best_checkpoint_path' and 'history'.
        """
        best_val_balacc = -1.0
        patience_counter = 0
        history: list[dict] = []
        best_path = ''

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            train_m = self._lob_train_epoch(train_loader)
            val_m = self._lob_validate_epoch(val_loader)
            self._step_scheduler_lob(val_m['balanced_accuracy'])
            history.append({
                'epoch': epoch,
                'train_loss': train_m['loss'],
                'val_loss': val_m['loss'],
                'val_balanced_accuracy': val_m['balanced_accuracy'],
            })
            logger.info(
                'LOB Epoch %d train_loss=%.6f val_loss=%.6f val_balacc=%.4f',
                epoch,
                train_m['loss'],
                val_m['loss'],
                val_m['balanced_accuracy'],
            )
            if val_m['balanced_accuracy'] > best_val_balacc:
                best_val_balacc = val_m['balanced_accuracy']
                patience_counter = 0
                best_path = str(
                    self.checkpoints_dir / run_id / f'best_balacc_epoch_{epoch:03d}.pt'
                )
                checkpointing.save_checkpoint(
                    best_path,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    self.cfg,
                    [],  # no feature_cols for LOB (book channels are fixed)
                    '',  # no scaler path
                    {'task_type': 'lob'},
                    {
                        'loss': val_m['loss'],
                        'balanced_accuracy': val_m['balanced_accuracy'],
                    },
                )
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info('LOB early stopping at epoch %d', epoch)
                    break

        last_path = str(self.checkpoints_dir / run_id / 'last_lob.pt')
        checkpointing.save_checkpoint(
            last_path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.current_epoch,
            self.cfg,
            [],
            '',
            {'task_type': 'lob'},
            history[-1] if history else {},
        )
        return {'best_checkpoint_path': best_path or last_path, 'history': history}
