"""
Unit tests for api/models/lob_models.py.

Tests: forward pass shape, invalid input raises ValueError, gradient flow.
"""

import torch
import pytest

from api.models.lob_models import TLOBForecaster, LobCNNBaseline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, T, C, L, E = 4, 50, 4, 10, 4  # batch, time, channels, levels, aux_dim


@pytest.fixture
def tlob_model():
    """Small TLOBForecaster for fast tests."""
    return TLOBForecaster(
        levels=L,
        channels=C,
        aux_dim=E,
        d_model=32,
        spatial_heads=4,
        temporal_heads=4,
        layers=2,
        ff_dim=64,
        dropout=0.0,
        num_classes=3,
    )


@pytest.fixture
def cnn_model():
    """Small LobCNNBaseline for fast tests."""
    return LobCNNBaseline(
        levels=L,
        channels=C,
        aux_dim=E,
        d_model=32,
        dropout=0.0,
        num_classes=3,
    )


def _make_inputs(batch=B, time=T, channels=C, levels=L, aux=E):
    x_book = torch.randn(batch, time, channels, levels)
    x_aux = torch.randn(batch, time, aux)
    return x_book, x_aux


# ---------------------------------------------------------------------------
# Test 1 — forward pass produces correct output shape for both models
# ---------------------------------------------------------------------------


def test_tlob_forward_shape(tlob_model):
    """TLOBForecaster forward pass: output shape [B, 3]."""
    x_book, x_aux = _make_inputs()
    tlob_model.eval()
    with torch.no_grad():
        logits = tlob_model(x_book, x_aux)
    assert logits.shape == (B, 3), f'Expected [{B}, 3], got {logits.shape}'


def test_cnn_forward_shape(cnn_model):
    """LobCNNBaseline forward pass: output shape [B, 3]."""
    x_book, x_aux = _make_inputs()
    cnn_model.eval()
    with torch.no_grad():
        logits = cnn_model(x_book, x_aux)
    assert logits.shape == (B, 3), f'Expected [{B}, 3], got {logits.shape}'


# ---------------------------------------------------------------------------
# Test 2 — invalid input raises ValueError
# ---------------------------------------------------------------------------


def test_tlob_invalid_input_raises(tlob_model):
    """Passing 3D x_book (missing channels dim) raises ValueError."""
    x_book_bad = torch.randn(B, T, L)  # 3D instead of 4D
    x_aux = torch.randn(B, T, E)
    with pytest.raises(ValueError, match='x_book must be 4D'):
        tlob_model(x_book_bad, x_aux)


def test_cnn_invalid_input_raises(cnn_model):
    """Passing 2D x_book raises ValueError."""
    x_book_bad = torch.randn(B, T)  # 2D
    x_aux = torch.randn(B, T, E)
    with pytest.raises(ValueError, match='x_book must be 4D'):
        cnn_model(x_book_bad, x_aux)


# ---------------------------------------------------------------------------
# Test 3 — gradient flows through the model (loss.backward() completes)
# ---------------------------------------------------------------------------


def test_tlob_gradient_flow(tlob_model):
    """Loss.backward() completes without errors; all params receive gradients."""
    tlob_model.train()
    x_book, x_aux = _make_inputs()
    targets = torch.randint(0, 3, (B,))
    logits = tlob_model(x_book, x_aux)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()

    params_with_grad = [
        p for p in tlob_model.parameters() if p.grad is not None and p.requires_grad
    ]
    assert len(params_with_grad) > 0, 'No parameters received gradients'


def test_cnn_gradient_flow(cnn_model):
    """Loss.backward() completes; CNN model params receive gradients."""
    cnn_model.train()
    x_book, x_aux = _make_inputs()
    targets = torch.randint(0, 3, (B,))
    logits = cnn_model(x_book, x_aux)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()

    params_with_grad = [
        p for p in cnn_model.parameters() if p.grad is not None and p.requires_grad
    ]
    assert len(params_with_grad) > 0, 'No parameters received gradients'
