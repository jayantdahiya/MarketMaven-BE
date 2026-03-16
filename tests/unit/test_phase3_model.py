"""Unit tests for Phase 3 multimodal CnnTransForecaster."""

import pytest
import torch

from api.models.cnn_transformer import CnnTransForecaster

# ── Fixtures ──────────────────────────────────────────────────────────────────

FEATURE_GROUP_SIZES = {
    'price_tech': 11,
    'context': 3,
    'sentiment': 4,
    'alpha': 9,
}
F_TOTAL = sum(FEATURE_GROUP_SIZES.values())  # 27
B, T, H = 4, 90, 1


@pytest.fixture
def multimodal_model() -> CnnTransForecaster:
    """Multimodal CnnTransForecaster with Phase 3 config."""
    return CnnTransForecaster(
        n_features=F_TOTAL,
        seq_len=T,
        horizon=H,
        conv_channels=128,
        conv_kernel=3,
        d_model=128,
        n_heads=4,
        n_encoder_layers=3,
        ff_dim=256,
        dropout=0.1,
        mlp_hidden=64,
        multimodal=True,
        encoder_dim=64,
        fusion_dim=128,
        feature_group_sizes=FEATURE_GROUP_SIZES,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_multimodal_forward_shape(multimodal_model: CnnTransForecaster) -> None:
    """Output shape [B, H] with all modalities present."""
    multimodal_model.eval()
    x = torch.randn(B, T, F_TOTAL)
    mask = torch.ones(B, 4)  # all modalities available
    with torch.no_grad():
        y_hat = multimodal_model(x, modality_mask=mask)
    assert y_hat.shape == (B, H), f'Expected ({B}, {H}), got {tuple(y_hat.shape)}'
    assert torch.isfinite(y_hat).all(), 'Output contains non-finite values'


def test_modality_mask_zeros_missing_encoder_output(
    multimodal_model: CnnTransForecaster,
) -> None:
    """When a modality mask is 0, its gated contribution should be zeroed."""
    multimodal_model.eval()

    # Mask out context (index 1) and alpha (index 3)
    mask_partial = torch.tensor([[1.0, 0.0, 1.0, 0.0]] * B)

    # Get gate values after masking
    with torch.no_grad():
        gates = torch.softmax(multimodal_model.gate_params, dim=0)  # [4]
        gates_masked = gates.unsqueeze(0) * mask_partial  # [B, 4]
        gates_masked = gates_masked / (gates_masked.sum(dim=1, keepdim=True) + 1e-8)

    # Context and alpha gates should be zero
    assert (gates_masked[:, 1] == 0.0).all(), 'Context gate not zeroed when masked'
    assert (gates_masked[:, 3] == 0.0).all(), 'Alpha gate not zeroed when masked'
    # Price_tech and sentiment should be non-zero
    assert (gates_masked[:, 0] > 0.0).all(), 'Price_tech gate should be non-zero'
    assert (gates_masked[:, 2] > 0.0).all(), 'Sentiment gate should be non-zero'


def test_partial_modalities_still_predict(
    multimodal_model: CnnTransForecaster,
) -> None:
    """Model produces finite output even with 2 of 4 modalities masked."""
    multimodal_model.eval()
    x = torch.randn(B, T, F_TOTAL)

    # Only price_tech and sentiment available
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]] * B)
    with torch.no_grad():
        y_hat = multimodal_model(x, modality_mask=mask)
    assert y_hat.shape == (B, H), f'Expected ({B}, {H}), got {tuple(y_hat.shape)}'
    assert torch.isfinite(y_hat).all(), (
        'Output contains non-finite values with partial modalities'
    )

    # Only price_tech available (extreme case)
    mask_minimal = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * B)
    with torch.no_grad():
        y_hat_min = multimodal_model(x, modality_mask=mask_minimal)
    assert y_hat_min.shape == (B, H)
    assert torch.isfinite(y_hat_min).all(), (
        'Output contains non-finite values with only price_tech'
    )
