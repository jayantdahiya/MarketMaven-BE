"""Unit tests for Phase 2.5 evaluation metadata forwarding."""

from __future__ import annotations

import torch

from api.training import evaluate as evaluate_module


class _GraphAwareModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.asset_index = None
        self.graph_context = None

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        asset_index: torch.Tensor | None = None,
        graph_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.asset_index = asset_index
        self.graph_context = graph_context
        return torch.zeros(x.size(0), 1, dtype=x.dtype, device=x.device)


class _MaskAwareModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.modality_mask = None

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        modality_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.modality_mask = modality_mask
        return torch.zeros(x.size(0), 1, dtype=x.dtype, device=x.device)


def test_evaluate_checkpoint_passes_graph_context(monkeypatch) -> None:
    """Daily evaluation must pass graph metadata the same way training does."""
    monkeypatch.setattr(
        evaluate_module.checkpointing,
        'load_checkpoint',
        lambda checkpoint_path, model, device='cpu': None,
    )

    model = _GraphAwareModel()
    x = torch.randn(2, 5, 3)
    y = torch.tensor([[0.1], [0.2]], dtype=torch.float32)
    graph_context = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.7, 0.9]])
    meta = {
        'asset_id': ['AAPL', 'SPY'],
        'timestamp': ['2024-01-02', '2024-01-03'],
        'asset_index': torch.tensor([0, 1]),
        'graph_context': graph_context,
    }

    results = evaluate_module.evaluate_checkpoint(
        {'training': {'device': 'cpu'}, 'evaluation': {}},
        'unused.pt',
        [(x, y, meta)],
        scaler=None,  # type: ignore[arg-type]
        model=model,
    )

    assert results['summary']['mae'] >= 0.0
    assert model.asset_index is not None
    assert model.graph_context is not None
    assert torch.equal(model.asset_index.cpu(), torch.tensor([0, 1]))
    assert torch.equal(model.graph_context.cpu(), graph_context)


def test_evaluate_checkpoint_passes_modality_mask(monkeypatch) -> None:
    """Daily evaluation must pass modality masks for multimodal checkpoints."""
    monkeypatch.setattr(
        evaluate_module.checkpointing,
        'load_checkpoint',
        lambda checkpoint_path, model, device='cpu': None,
    )

    model = _MaskAwareModel()
    x = torch.randn(2, 5, 4)
    y = torch.tensor([[0.1], [0.2]], dtype=torch.float32)
    modality_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0]])
    meta = {
        'asset_id': ['AAPL', 'SPY'],
        'timestamp': ['2024-01-02', '2024-01-03'],
        'modality_mask': modality_mask,
    }

    results = evaluate_module.evaluate_checkpoint(
        {'training': {'device': 'cpu'}, 'evaluation': {}},
        'unused.pt',
        [(x, y, meta)],
        scaler=None,  # type: ignore[arg-type]
        model=model,
    )

    assert results['summary']['rmse'] >= 0.0
    assert model.modality_mask is not None
    assert torch.equal(model.modality_mask.cpu(), modality_mask)
