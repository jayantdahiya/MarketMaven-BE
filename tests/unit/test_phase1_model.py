"""Unit tests for CnnTransForecaster (Phase 1 model)."""

import pytest
import torch

from api.models.cnn_transformer import CnnTransForecaster
from api.models.factory import create_model

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def default_model() -> CnnTransForecaster:
    """CnnTransForecaster with default hyperparameters."""
    return CnnTransForecaster()


@pytest.fixture
def phase1_model_cfg() -> dict:
    """Minimal model config matching config/phase_1.yaml model section."""
    return {
        'n_features': 8,
        'seq_len': 60,
        'horizon': 1,
        'conv_channels': 128,
        'conv_kernel': 3,
        'd_model': 128,
        'n_heads': 4,
        'n_encoder_layers': 3,
        'ff_dim': 256,
        'dropout': 0.1,
        'mlp_hidden': 64,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_forward_output_shape(default_model: CnnTransForecaster) -> None:
    """Forward pass on [4, 60, 8] input must return [4, 1]."""
    default_model.eval()
    x = torch.randn(4, 60, 8)
    with torch.no_grad():
        y = default_model(x)
    assert y.shape == (4, 1), f'Expected (4, 1), got {tuple(y.shape)}'


def test_variable_batch_size(default_model: CnnTransForecaster) -> None:
    """Output shape [B, 1] must hold for batch sizes 1, 16, 128."""
    default_model.eval()
    for B in [1, 16, 128]:
        x = torch.randn(B, 60, 8)
        with torch.no_grad():
            y = default_model(x)
        assert y.shape == (B, 1), f'B={B}: expected ({B}, 1), got {tuple(y.shape)}'


def test_variable_horizon() -> None:
    """Model instantiated with horizon=5 must return [B, 5]."""
    model = CnnTransForecaster(horizon=5)
    model.eval()
    x = torch.randn(4, 60, 8)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (4, 5), f'Expected (4, 5), got {tuple(y.shape)}'


def test_conv_padding_preserves_length() -> None:
    """padding='same' on both Conv1d blocks must preserve temporal dim T=60."""
    model = CnnTransForecaster()
    model.eval()

    # Hook into the model to capture intermediate shape after conv blocks
    captured: list[torch.Tensor] = []

    def _hook(_module, _input, output: torch.Tensor) -> None:
        captured.append(output)

    # Register hook on the second conv block (after both blocks are applied)
    handle = model.conv_block2.register_forward_hook(_hook)

    x = torch.randn(2, 60, 8)
    with torch.no_grad():
        model(x)

    handle.remove()

    assert len(captured) == 1
    # captured[0] shape: [B, conv_channels, T]
    assert captured[0].shape[-1] == 60, (
        f'Temporal dim after conv blocks: expected 60, got {captured[0].shape[-1]}'
    )


def test_factory_creates_correct_model(phase1_model_cfg: dict) -> None:
    """create_model('cnn_transformer', cfg) must return a CnnTransForecaster."""
    model = create_model('cnn_transformer', phase1_model_cfg)
    assert isinstance(model, CnnTransForecaster)


def test_gradient_flow(default_model: CnnTransForecaster) -> None:
    """All trainable parameters must have non-None gradients after backward pass."""
    default_model.train()
    x = torch.randn(4, 60, 8)
    y_hat = default_model(x)
    y_true = torch.randn_like(y_hat)
    loss = torch.nn.functional.mse_loss(y_hat, y_true)
    loss.backward()

    params_without_grad = [
        name
        for name, p in default_model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not params_without_grad, (
        f'Parameters missing gradients: {params_without_grad}'
    )


def test_deterministic_with_seed() -> None:
    """Two forward passes with identical seed and input must be bitwise identical."""
    x = torch.randn(4, 60, 8)

    torch.manual_seed(0)
    model_a = CnnTransForecaster()
    model_a.eval()
    with torch.no_grad():
        out_a = model_a(x.clone())

    torch.manual_seed(0)
    model_b = CnnTransForecaster()
    model_b.eval()
    with torch.no_grad():
        out_b = model_b(x.clone())

    assert torch.equal(out_a, out_b), 'Outputs differ despite identical seed and input'


def test_attention_pooling_weights_sum_to_one(
    default_model: CnnTransForecaster,
) -> None:
    """Attention weights returned by get_attention_weights must sum to ~1 per sample."""
    default_model.eval()
    x = torch.randn(8, 60, 8)
    attn = default_model.get_attention_weights(x)  # [B, T]
    assert attn.shape == (8, 60), f'Expected (8, 60), got {tuple(attn.shape)}'
    sums = attn.sum(dim=-1)  # [B]
    assert torch.allclose(sums, torch.ones(8), atol=1e-5), (
        f'Attention weight sums not all 1.0: {sums}'
    )


def test_init_weights_nonzero() -> None:
    """Conv1d weight tensors must have non-zero std after _init_weights."""
    model = CnnTransForecaster()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv1d):
            std = module.weight.std().item()
            assert std > 0.0, f'Conv1d layer {name!r} has all-zero weights (std=0)'


def test_invalid_input_dim_raises(default_model: CnnTransForecaster) -> None:
    """Passing a 2D input must raise ValueError."""
    x = torch.randn(4, 60)  # missing feature dim
    with pytest.raises(ValueError, match='3D'):
        default_model(x)
