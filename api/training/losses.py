"""
MSE, Sharpe surrogate, and composite forecast loss (MSE + λ·Sharpe with warmup).
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


class CompositeForecastLoss(nn.Module):
    """MSE + λ·Sharpe surrogate; MSE-only for first warmup_epochs."""

    def __init__(
        self,
        lambda_sharpe: float = 0.1,
        temperature: float = 0.02,
        warmup_epochs: int = 3,
        annualization: int = 252,
    ):
        super().__init__()
        self.lambda_sharpe = lambda_sharpe
        self.warmup_epochs = warmup_epochs
        self.mse = MSELossWrapper()
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
