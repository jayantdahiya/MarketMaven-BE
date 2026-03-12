"""Unit tests for MambaForecaster (Phase 2 model)."""

import pytest
import torch

from api.models.factory import create_model
from api.models.mamba_ssm import MambaBlockFallback, MambaForecaster

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def default_model() -> MambaForecaster:
    """MambaForecaster with default-like hyperparams (small for speed)."""
    return MambaForecaster(
        input_dim=8,
        d_model=32,
        d_state=8,
        d_conv=4,
        expand=2,
        num_layers=2,
        num_assets=5,
        graph_dim=16,
        use_graph_context=True,
        dropout=0.1,
        forecast_horizons=1,
        backend='pure_pytorch',
    )


@pytest.fixture
def no_graph_model() -> MambaForecaster:
    """MambaForecaster with graph fusion disabled."""
    return MambaForecaster(
        input_dim=8,
        d_model=32,
        d_state=8,
        d_conv=4,
        expand=2,
        num_layers=2,
        num_assets=5,
        graph_dim=16,
        use_graph_context=False,
        dropout=0.1,
        forecast_horizons=1,
        backend='pure_pytorch',
    )


@pytest.fixture
def phase2_model_cfg() -> dict:
    """Minimal model config matching config/phase_2.yaml model section."""
    return {
        'input_dim': 8,
        'd_model': 128,
        'd_state': 16,
        'd_conv': 4,
        'expand': 2,
        'num_layers': 4,
        'num_assets': 5,
        'graph_dim': 32,
        'use_graph_context': True,
        'dropout': 0.1,
        'forecast_horizons': 1,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_forward_shape_no_graph(no_graph_model: MambaForecaster) -> None:
    """Forward pass without graph context returns [B, H]."""
    no_graph_model.eval()
    x = torch.randn(4, 90, 8)
    with torch.no_grad():
        y = no_graph_model(x)
    assert y.shape == (4, 1), f'Expected (4, 1), got {tuple(y.shape)}'
    assert not torch.isnan(y).any(), 'Output contains NaN'


def test_forward_shape_with_graph(default_model: MambaForecaster) -> None:
    """Forward pass with graph context returns [B, H]."""
    default_model.eval()
    B, T, F, A, G = 4, 90, 8, 5, 3
    x = torch.randn(B, T, F)
    asset_index = torch.randint(0, A, (B,))
    graph_context = torch.randn(A, G)
    with torch.no_grad():
        y = default_model(x, asset_index=asset_index, graph_context=graph_context)
    assert y.shape == (B, 1), f'Expected ({B}, 1), got {tuple(y.shape)}'
    assert not torch.isnan(y).any(), 'Output contains NaN'


def test_variable_batch_size(no_graph_model: MambaForecaster) -> None:
    """Output shape [B, 1] must hold for batch sizes 1, 8, 32."""
    no_graph_model.eval()
    for B in [1, 8, 32]:
        x = torch.randn(B, 90, 8)
        with torch.no_grad():
            y = no_graph_model(x)
        assert y.shape == (B, 1), f'B={B}: expected ({B}, 1), got {tuple(y.shape)}'


def test_variable_horizon() -> None:
    """Model with forecast_horizons=5 must return [B, 5]."""
    model = MambaForecaster(
        input_dim=8,
        d_model=32,
        d_state=8,
        num_layers=1,
        use_graph_context=False,
        forecast_horizons=5,
        backend='pure_pytorch',
    )
    model.eval()
    x = torch.randn(4, 90, 8)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (4, 5), f'Expected (4, 5), got {tuple(y.shape)}'


def test_factory_creates_correct_model(phase2_model_cfg: dict) -> None:
    """create_model('mamba_ssm', cfg) must return a MambaForecaster."""
    model = create_model('mamba_ssm', phase2_model_cfg)
    assert isinstance(model, MambaForecaster)


def test_gradient_flow(default_model: MambaForecaster) -> None:
    """All trainable parameters must have non-None gradients after backward."""
    default_model.train()
    B, T, F, A, G = 4, 90, 8, 5, 3
    x = torch.randn(B, T, F)
    asset_index = torch.randint(0, A, (B,))
    graph_context = torch.randn(A, G)
    y_hat = default_model(x, asset_index=asset_index, graph_context=graph_context)
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


def test_deterministic_inference(no_graph_model: MambaForecaster) -> None:
    """Two eval-mode forward passes with same input yield identical output."""
    no_graph_model.eval()
    x = torch.randn(4, 90, 8)
    with torch.no_grad():
        out1 = no_graph_model(x.clone())
        out2 = no_graph_model(x.clone())
    assert torch.allclose(out1, out2), 'Outputs differ for identical input in eval mode'


def test_long_sequence_memory() -> None:
    """Forward pass with seq_len=500 completes without error on CPU."""
    model = MambaForecaster(
        input_dim=8,
        d_model=32,
        d_state=8,
        num_layers=2,
        use_graph_context=False,
        forecast_horizons=1,
        backend='pure_pytorch',
    )
    model.eval()
    x = torch.randn(2, 500, 8)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1), f'Expected (2, 1), got {tuple(y.shape)}'
    assert not torch.isnan(y).any(), 'Output contains NaN for long sequence'


def test_graph_pad_when_context_missing(default_model: MambaForecaster) -> None:
    """When use_graph_context=True but no context is passed, zero-pad is used."""
    default_model.eval()
    x = torch.randn(4, 90, 8)
    with torch.no_grad():
        y = default_model(x)  # no asset_index or graph_context
    assert y.shape == (4, 1), f'Expected (4, 1), got {tuple(y.shape)}'
    assert not torch.isnan(y).any(), 'Output contains NaN with zero-padded graph'


def test_invalid_input_dim_raises(default_model: MambaForecaster) -> None:
    """Passing a 2D input must raise ValueError."""
    x = torch.randn(4, 90)  # missing feature dim
    with pytest.raises(ValueError, match='3D'):
        default_model(x)


def test_mamba_block_fallback_shape() -> None:
    """MambaBlockFallback must preserve [B, T, d_model] shape."""
    block = MambaBlockFallback(d_model=32, d_state=8, d_conv=4, expand=2)
    x = torch.randn(4, 60, 32)
    y = block(x)
    assert y.shape == x.shape, f'Expected {x.shape}, got {y.shape}'


def test_init_weights_nonzero(default_model: MambaForecaster) -> None:
    """Linear weight tensors must have non-zero std after _init_weights."""
    for name, module in default_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            std = module.weight.std().item()
            assert std > 0.0, f'Linear layer {name!r} has all-zero weights (std=0)'
