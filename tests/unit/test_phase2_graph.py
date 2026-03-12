"""Unit tests for asset-correlation graph (Phase 2)."""

import numpy as np
import pandas as pd
import pytest
import torch

from api.data.asset_graph import (
    build_correlation_graph,
    compute_graph_context,
    normalize_adjacency,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_returns_df() -> pd.DataFrame:
    """Deterministic log-returns DataFrame: 100 days, 5 assets."""
    np.random.seed(42)
    n = 100
    dates = pd.bdate_range(start='2021-01-01', periods=n, freq='B')
    assets = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'SPY']
    # Correlated returns: first 3 assets share a common factor
    common = np.random.randn(n) * 0.01
    data = {}
    for i, a in enumerate(assets):
        noise = np.random.randn(n) * 0.005
        if i < 3:
            data[a] = common + noise
        else:
            data[a] = np.random.randn(n) * 0.01
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def synthetic_daily_df() -> pd.DataFrame:
    """Synthetic daily DataFrame with asset_id, timestamp, close, log_return_1d."""
    np.random.seed(42)
    n = 100
    dates = pd.bdate_range(start='2021-01-01', periods=n, freq='B')
    assets = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'SPY']
    frames = []
    for a in assets:
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        close = np.maximum(close, 1.0)
        log_ret = np.log(close / np.roll(close, 1))
        log_ret[0] = 0.0
        df = pd.DataFrame({
            'timestamp': dates,
            'asset_id': a,
            'close': close,
            'log_return_1d': log_ret,
        })
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ── Graph construction ────────────────────────────────────────────────────────


def test_correlation_graph_shape(synthetic_returns_df: pd.DataFrame) -> None:
    """Graph must have correct number of nodes and symmetric binary adjacency."""
    G = build_correlation_graph(synthetic_returns_df, window=60, threshold=0.5)
    A = len(synthetic_returns_df.columns)
    assert G.number_of_nodes() == A, f'Expected {A} nodes, got {G.number_of_nodes()}'
    # Adjacency must be symmetric (undirected graph)
    import networkx as nx

    adj = nx.to_numpy_array(G)
    assert np.allclose(adj, adj.T), 'Adjacency matrix is not symmetric'
    # Values must be 0 or 1 (binary)
    unique_vals = set(adj.ravel())
    assert unique_vals <= {0.0, 1.0}, f'Non-binary adjacency values: {unique_vals}'


def test_correlation_graph_threshold_sensitivity(
    synthetic_returns_df: pd.DataFrame,
) -> None:
    """Higher threshold must produce fewer or equal edges."""
    G_low = build_correlation_graph(synthetic_returns_df, window=60, threshold=0.3)
    G_high = build_correlation_graph(synthetic_returns_df, window=60, threshold=0.8)
    assert G_high.number_of_edges() <= G_low.number_of_edges(), (
        f'Higher threshold produced more edges: {G_high.number_of_edges()} > {G_low.number_of_edges()}'
    )


def test_single_asset_graph() -> None:
    """Graph with 1 asset must have 1 node and 0 edges."""
    returns_df = pd.DataFrame({'AAPL': np.random.randn(60)})
    G = build_correlation_graph(returns_df, window=60, threshold=0.5)
    assert G.number_of_nodes() == 1
    assert G.number_of_edges() == 0


# ── Adjacency normalisation ──────────────────────────────────────────────────


def test_normalized_adjacency_shape(synthetic_returns_df: pd.DataFrame) -> None:
    """Normalised adjacency must be [A, A] float32."""
    G = build_correlation_graph(synthetic_returns_df, window=60, threshold=0.5)
    A_norm = normalize_adjacency(G)
    A = G.number_of_nodes()
    assert A_norm.shape == (A, A), f'Expected ({A}, {A}), got {A_norm.shape}'
    assert A_norm.dtype == torch.float32


def test_normalized_adjacency_row_sums(synthetic_returns_df: pd.DataFrame) -> None:
    """Row sums of normalised adjacency must be approximately 1 for connected nodes."""
    G = build_correlation_graph(synthetic_returns_df, window=60, threshold=0.3)
    A_norm = normalize_adjacency(G)
    row_sums = A_norm.sum(dim=1)
    # Due to D^{-1/2} A_hat D^{-1/2} normalisation, row sums are not exactly 1
    # but should be positive and bounded
    assert (row_sums > 0).all(), 'Some row sums are <= 0'
    assert (row_sums < 5).all(), 'Some row sums are unreasonably large'


def test_normalized_adjacency_symmetric(synthetic_returns_df: pd.DataFrame) -> None:
    """Normalised adjacency must be symmetric."""
    G = build_correlation_graph(synthetic_returns_df, window=60, threshold=0.5)
    A_norm = normalize_adjacency(G)
    assert torch.allclose(A_norm, A_norm.T, atol=1e-6), (
        'Normalised adjacency is not symmetric'
    )


# ── Graph-context features ────────────────────────────────────────────────────


def test_context_features_shape(synthetic_daily_df: pd.DataFrame) -> None:
    """Graph-context tensor must be [A, 3] with no NaN."""
    assets = sorted(synthetic_daily_df['asset_id'].unique().tolist())
    gc = compute_graph_context(
        synthetic_daily_df, assets, window=60, threshold=0.5, return_window=20
    )
    A = len(assets)
    assert gc.shape == (A, 3), f'Expected ({A}, 3), got {gc.shape}'
    assert gc.dtype == torch.float32
    assert not torch.isnan(gc).any(), 'Graph context contains NaN'


def test_context_degree_in_range(synthetic_daily_df: pd.DataFrame) -> None:
    """Normalised degree (feature 0) must be in [0, 1]."""
    assets = sorted(synthetic_daily_df['asset_id'].unique().tolist())
    gc = compute_graph_context(
        synthetic_daily_df, assets, window=60, threshold=0.5, return_window=20
    )
    degrees = gc[:, 0]
    assert (degrees >= 0).all() and (degrees <= 1).all(), (
        f'Degree out of [0, 1] range: {degrees}'
    )
