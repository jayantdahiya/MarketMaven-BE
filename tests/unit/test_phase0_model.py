"""Unit tests for LSTMBaseline."""
import pytest
import torch

from api.models.lstm_baseline import LSTMBaseline


def test_lstm_forward_shape(sample_lstm_model):
    B, T, F = 4, 60, 8
    x = torch.randn(B, T, F)
    y = sample_lstm_model(x)
    assert y.shape == (B, 1)


def test_lstm_forward_finite(sample_lstm_model):
    x = torch.randn(2, 60, 8)
    y = sample_lstm_model(x)
    assert torch.isfinite(y).all()


def test_lstm_variable_batch_size(sample_lstm_model):
    for B in [1, 16, 64]:
        x = torch.randn(B, 60, 8)
        y = sample_lstm_model(x)
        assert y.shape == (B, 1)


def test_lstm_input_dim_raises():
    with pytest.raises(ValueError, match="input_dim"):
        LSTMBaseline(input_dim=0)
