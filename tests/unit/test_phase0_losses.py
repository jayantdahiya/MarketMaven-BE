"""Unit tests for losses."""
import torch
import pytest

from api.training.losses import MSELossWrapper, SharpeSurrogateLoss, CompositeForecastLoss


def test_mse_loss_zero_on_perfect_prediction():
    criterion = MSELossWrapper()
    y = torch.tensor([[1.0], [2.0]])
    loss = criterion(y, y)
    assert loss.item() == 0.0


def test_composite_loss_warmup_mse_only():
    mse = MSELossWrapper()
    composite = CompositeForecastLoss(lambda_sharpe=0.1, warmup_epochs=3)
    y_hat = torch.tensor([[0.5], [0.5]])
    y_true = torch.tensor([[0.5], [0.5]])
    l_mse = mse(y_hat, y_true)
    l_comp = composite(y_hat, y_true, current_epoch=0)
    assert torch.allclose(l_comp, l_mse)


def test_composite_loss_finite():
    composite = CompositeForecastLoss()
    y_hat = torch.randn(8, 1)
    y_true = torch.randn(8, 1)
    loss = composite(y_hat, y_true, current_epoch=5)
    assert torch.isfinite(loss)
