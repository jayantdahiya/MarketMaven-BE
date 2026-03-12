"""Asset-correlation graph builder for Phase 2 graph-context fusion.

Computes a rolling Pearson correlation graph over daily log-returns,
thresholds it to form an unweighted adjacency matrix, normalises via
the symmetric Laplacian trick, and derives per-asset context features
(degree, 20-day mean return, 20-day volatility).
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_correlation_graph(
    returns_df: pd.DataFrame,
    window: int = 60,
    threshold: float = 0.5,
) -> nx.Graph:
    """Build an unweighted correlation graph from asset log-returns.

    Args:
        returns_df: DataFrame with columns as asset IDs and rows as dates,
            containing daily log-return values.
        window: Number of trailing days for the rolling Pearson correlation.
        threshold: Minimum absolute correlation to form an edge (``|rho| >= threshold``).

    Returns:
        A ``networkx.Graph`` whose nodes are asset IDs and edges connect
        pairs with correlation above *threshold*.
    """
    if returns_df.shape[1] < 2:
        G = nx.Graph()
        G.add_nodes_from(returns_df.columns.tolist())
        return G

    # Use the last *window* rows for correlation
    tail = returns_df.tail(window)
    corr = tail.corr()

    G = nx.Graph()
    assets = returns_df.columns.tolist()
    G.add_nodes_from(assets)

    for i, a1 in enumerate(assets):
        for j in range(i + 1, len(assets)):
            a2 = assets[j]
            rho = corr.loc[a1, a2]
            if not np.isnan(rho) and abs(rho) >= threshold:
                G.add_edge(a1, a2)

    logger.info(
        'Correlation graph: %d nodes, %d edges (window=%d, threshold=%.2f)',
        G.number_of_nodes(),
        G.number_of_edges(),
        window,
        threshold,
    )
    return G


# ---------------------------------------------------------------------------
# Adjacency normalisation
# ---------------------------------------------------------------------------


def normalize_adjacency(G: nx.Graph) -> torch.Tensor:
    """Compute the symmetric normalised adjacency: D^{-1/2} (A + I) D^{-1/2}.

    Args:
        G: An unweighted ``networkx.Graph``.

    Returns:
        Tensor of shape ``[A, A]`` (float32) — the normalised adjacency.
    """
    A = len(G.nodes())
    if A == 0:
        return torch.zeros(0, 0)

    nodes = sorted(G.nodes())
    adj = nx.to_numpy_array(G, nodelist=nodes, dtype=np.float64)

    # A_hat = A + I (self-loops)
    a_hat = adj + np.eye(A)

    # Degree of A_hat
    d = a_hat.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, d ** (-0.5), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)

    # D^{-1/2} A_hat D^{-1/2}
    norm = D_inv_sqrt @ a_hat @ D_inv_sqrt

    return torch.tensor(norm, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Graph-context features
# ---------------------------------------------------------------------------


def compute_graph_context(
    df: pd.DataFrame,
    assets: list[str],
    window: int = 60,
    threshold: float = 0.5,
    return_window: int = 20,
) -> torch.Tensor:
    """Compute the per-asset graph-context feature tensor.

    Features (G=3):
        0. ``graph_degree``  — normalised node degree in [0, 1].
        1. ``graph_return_mean_20d``  — 20-day rolling mean log-return.
        2. ``graph_volatility_20d``  — 20-day rolling std of log-returns.

    Args:
        df: Full DataFrame with columns ``['asset_id', 'timestamp', 'log_return_1d']``
            (or whichever return column is present).
        assets: Ordered list of asset IDs defining the asset universe.
        window: Correlation window for graph construction.
        threshold: Correlation threshold for edges.
        return_window: Rolling window for return-mean and volatility features.

    Returns:
        Tensor of shape ``[A, 3]`` (float32).
    """
    # Build returns pivot: rows=dates, cols=asset_ids
    return_col = 'log_return_1d'
    if return_col not in df.columns:
        # Compute from close prices if not present
        pivot = df.pivot_table(index='timestamp', columns='asset_id', values='close')
        returns_df = np.log(pivot / pivot.shift(1)).dropna()
    else:
        returns_df = df.pivot_table(
            index='timestamp', columns='asset_id', values=return_col
        )
        returns_df = returns_df.dropna()

    # Ensure all requested assets are present; fill missing with 0
    for a in assets:
        if a not in returns_df.columns:
            returns_df[a] = 0.0
    returns_df = returns_df[assets]

    # Build graph
    G = build_correlation_graph(returns_df, window=window, threshold=threshold)

    # Feature 0: normalised degree
    A = len(assets)
    max_degree = max(A - 1, 1)
    degrees = np.array(
        [G.degree(a) / max_degree if G.has_node(a) else 0.0 for a in assets],
        dtype=np.float32,
    )

    # Feature 1 & 2: rolling return stats from the last *return_window* days
    tail = returns_df.tail(return_window)
    mean_returns = tail.mean().reindex(assets, fill_value=0.0).values.astype(np.float32)
    vol_returns = tail.std().reindex(assets, fill_value=0.0).values.astype(np.float32)
    # Replace NaN with 0
    mean_returns = np.nan_to_num(mean_returns, nan=0.0)
    vol_returns = np.nan_to_num(vol_returns, nan=0.0)

    context = np.stack([degrees, mean_returns, vol_returns], axis=-1)  # [A, 3]
    return torch.tensor(context, dtype=torch.float32)
