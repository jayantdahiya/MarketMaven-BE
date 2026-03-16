"""
MSE, Sharpe surrogate (batch-level and rolling-window), composite forecast loss
(MSE + λ·Sharpe with warmup), and WeightedCEFocalLoss for LOB classification.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MSELossWrapper(nn.Module):
    """Standard MSE loss for regression."""

    def forward(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(y_hat, y_true)


class SharpeSurrogateLoss(nn.Module):
    """
    Differentiable Sharpe surrogate: r_s = tanh(y_hat / temperature) * y_true,
    loss = -sqrt(annualization) * mean(r_s) / (std(r_s) + eps).
    """

    def __init__(self, temperature: float = 0.02, annualization: int = 252):
        super().__init__()
        self.temperature = temperature
        self.annualization = annualization

    def forward(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        r_s = torch.tanh(y_hat / self.temperature) * y_true
        mean_r = r_s.mean()
        std_r = r_s.std() + 1e-8
        sharpe = (mean_r / std_r) * (self.annualization**0.5)
        return -sharpe


class RollingSharpeSurrogateLoss(nn.Module):
    """Rolling-window Sharpe surrogate for Phase 2.

    Instead of computing a single Sharpe ratio over the whole batch, this
    variant computes the negative Sharpe ratio within a rolling window of
    *window* samples and averages the results.  When the batch is smaller
    than the window, it falls back to the batch-level computation.

    Args:
        window: Number of samples in each rolling window.
        temperature: Scaling temperature for ``tanh`` position sizing.
        annualization: Trading-days-per-year factor for annualisation.
    """

    def __init__(
        self,
        window: int = 20,
        temperature: float = 0.02,
        annualization: int = 252,
    ):
        super().__init__()
        self.window = window
        self.temperature = temperature
        self.annualization = annualization

    def forward(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        r_s = torch.tanh(y_hat / self.temperature) * y_true
        r_flat = r_s.reshape(-1)
        N = r_flat.shape[0]

        if self.window > N:
            mean_r = r_flat.mean()
            std_r = r_flat.std() + 1e-8
            return -(mean_r / std_r) * (self.annualization**0.5)

        # Rolling-window Sharpe: average across windows
        sharpes: list[torch.Tensor] = []
        for start in range(0, N - self.window + 1, self.window):
            chunk = r_flat[start : start + self.window]
            mean_c = chunk.mean()
            std_c = chunk.std() + 1e-8
            sharpes.append(mean_c / std_c)

        avg_sharpe = torch.stack(sharpes).mean()
        return -avg_sharpe * (self.annualization**0.5)


class CompositeForecastLoss(nn.Module):
    """MSE + λ·Sharpe surrogate + optional gate L1 penalty; MSE-only for first warmup_epochs.

    When *sharpe_window* > 0 the rolling-window variant is used; otherwise
    the original batch-level ``SharpeSurrogateLoss`` is used.

    The ``lambda_gate_l1`` coefficient is stored here for the training loop
    to use when adding the gate L1 penalty (which requires access to
    ``model.gate_params``).
    """

    def __init__(
        self,
        lambda_sharpe: float = 0.1,
        temperature: float = 0.02,
        warmup_epochs: int = 3,
        annualization: int = 252,
        sharpe_window: int = 0,
        lambda_gate_l1: float = 0.0,
    ):
        super().__init__()
        self.lambda_sharpe = lambda_sharpe
        self.warmup_epochs = warmup_epochs
        self.lambda_gate_l1 = lambda_gate_l1
        self.mse = MSELossWrapper()
        if sharpe_window > 0:
            self.sharpe: nn.Module = RollingSharpeSurrogateLoss(
                window=sharpe_window,
                temperature=temperature,
                annualization=annualization,
            )
        else:
            self.sharpe = SharpeSurrogateLoss(
                temperature=temperature, annualization=annualization
            )

    def forward(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor,
        current_epoch: int = 0,
    ) -> torch.Tensor:
        if torch.isnan(y_hat).any() or torch.isnan(y_true).any():
            logger.warning('NaN in y_hat or y_true; using MSE only for this batch')
            return self.mse(y_hat, y_true)
        l_mse = self.mse(y_hat, y_true)
        if current_epoch < self.warmup_epochs:
            return l_mse
        l_sharpe = self.sharpe(y_hat, y_true)
        return l_mse + self.lambda_sharpe * l_sharpe


class WeightedCEFocalLoss(nn.Module):
    """Weighted Cross-Entropy + Focal loss for LOB classification.

    L_total = L_ce + lambda_focal * L_focal

    L_ce   = -w_c * log(p(y=c))  with per-class weights.
    L_focal = -alpha * (1 - p_t)^gamma * log(p_t)  with uniform alpha=1.

    Args:
        class_weights: Per-class weight list/tensor (length == num_classes).
        lambda_focal: Coefficient for focal loss term.
        focal_gamma: Focusing parameter gamma (>= 0).
        num_classes: Number of output classes (default 3).
    """

    def __init__(
        self,
        class_weights: list[float] | None = None,
        lambda_focal: float = 0.25,
        focal_gamma: float = 1.5,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.lambda_focal = lambda_focal
        self.focal_gamma = focal_gamma
        self.num_classes = num_classes

        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
        else:
            w = torch.ones(num_classes, dtype=torch.float32)
        self.register_buffer('class_weights', w)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined weighted CE + focal loss.

        Args:
            logits: [B, C] raw model output (un-softmaxed).
            targets: [B] integer class labels in {0, ..., C-1}.

        Returns:
            Scalar loss tensor.
        """
        # Weighted cross-entropy
        l_ce = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.class_weights.to(logits.device),  # type: ignore[arg-type]
        )

        if self.lambda_focal == 0.0:
            return l_ce

        # Focal loss component
        log_probs = nn.functional.log_softmax(logits, dim=-1)  # [B, C]
        probs = torch.exp(log_probs)  # [B, C]
        # p_t = probability of the true class
        p_t = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)  # [B]
        log_p_t = log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - p_t) ** self.focal_gamma
        l_focal = -(focal_weight * log_p_t).mean()

        return l_ce + self.lambda_focal * l_focal
