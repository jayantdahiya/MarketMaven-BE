"""Phase 2 — Mamba / SSM Forecaster with optional graph-context fusion.

Architecture:
    Input projection → 4× MambaBlock (residual) → LayerNorm → last-token pooling
    → optional graph fusion → MLP prediction head.

The MambaBlock dispatches to the CUDA-optimized ``mamba-ssm`` package when
available and falls back to a pure-PyTorch selective-scan implementation
(``MambaBlockFallback``) on CPU / MPS.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Backend dispatch: try CUDA mamba-ssm, fall back to pure-PyTorch
# ---------------------------------------------------------------------------

try:
    from mamba_ssm import Mamba as _CUDAMambaBlock  # type: ignore[import-untyped]

    _HAS_CUDA_MAMBA = True
except ImportError:
    _HAS_CUDA_MAMBA = False


# ---------------------------------------------------------------------------
# Pure-PyTorch S4D-style selective scan fallback
# ---------------------------------------------------------------------------


class MambaBlockFallback(nn.Module):
    """Simplified selective-state-space block using standard PyTorch ops.

    This reproduces the core Mamba data flow — input-dependent discretisation,
    a parallel scan via cumulative sums, and gated output — without relying on
    custom CUDA kernels.  It is ~5-10× slower than the fused CUDA version but
    is functionally equivalent for correctness testing and CPU/MPS inference.

    Args:
        d_model: Input / output dimension.
        d_state: SSM hidden-state dimension (N).
        d_conv: Width of the local depthwise convolution.
        expand: Expansion factor for the inner dimension.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Step 1: input projection → (x_branch, z_gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Step 2: depthwise causal conv1d
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # Step 4: SSM parameter projections
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # Learnable log-diagonal state matrix A and skip D
        self.A_log = nn.Parameter(
            torch.log(
                torch
                .arange(1, d_state + 1, dtype=torch.float32)
                .unsqueeze(0)
                .expand(self.d_inner, -1)
                .clone()
            )
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Step 7: output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            u: Input tensor of shape ``[B, T, d_model]``.

        Returns:
            Output tensor of shape ``[B, T, d_model]``.
        """
        B, T, _ = u.shape

        # 1. Project → x, z  each [B, T, d_inner]
        xz = self.in_proj(u)
        x, z = xz.chunk(2, dim=-1)

        # 2. Causal depthwise conv (transpose for Conv1d: [B, d_inner, T])
        x = x.transpose(1, 2).contiguous()
        x = self.conv1d(x)[:, :, :T]  # trim to causal length
        x = x.transpose(1, 2).contiguous()

        # 3. SiLU activation
        x = F.silu(x)

        # 4. SSM parameter projections from x → dt, B_ssm, C_ssm
        ssm_params = self.x_proj(x)  # [B, T, 2*N + 1]
        dt = F.softplus(ssm_params[..., :1].squeeze(-1))  # [B, T]
        B_ssm = ssm_params[..., 1 : 1 + self.d_state]  # [B, T, N]
        C_ssm = ssm_params[..., 1 + self.d_state :]  # [B, T, N]

        # 5. Selective scan (sequential — functional but slow)
        A = -torch.exp(self.A_log)  # [d_inner, N]
        y = self._selective_scan(x, dt, A, B_ssm, C_ssm)  # [B, T, d_inner]

        # 6. Gate: y = y * SiLU(z)
        y = y * F.silu(z)

        # 7. Output projection
        return self.out_proj(y)

    # ------------------------------------------------------------------
    # Sequential selective scan (pure-PyTorch, O(T) per sample)
    # ------------------------------------------------------------------

    def _selective_scan(
        self,
        x: torch.Tensor,  # [B, T, D]
        dt: torch.Tensor,  # [B, T]
        A: torch.Tensor,  # [D, N]
        B_ssm: torch.Tensor,  # [B, T, N]
        C_ssm: torch.Tensor,  # [B, T, N]
    ) -> torch.Tensor:
        """Run the discretised SSM recurrence step-by-step.

        Returns:
            y: [B, T, D]
        """
        B_batch, T, D = x.shape
        N = A.shape[1]
        device = x.device
        dtype = x.dtype

        # Discretise: dA = exp(A * dt), dB = dt * B_ssm
        dt_expanded = dt.unsqueeze(-1)  # [B, T, 1]
        dA = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt_expanded.unsqueeze(-1))
        # dA: [B, T, D, N]
        dB = dt_expanded.unsqueeze(-1) * B_ssm.unsqueeze(2)  # [B, T, D, N]

        h = torch.zeros(B_batch, D, N, device=device, dtype=dtype)
        ys = []
        for t in range(T):
            h = h * dA[:, t] + dB[:, t] * x[:, t, :].unsqueeze(-1)
            y_t = (h * C_ssm[:, t].unsqueeze(1)).sum(dim=-1)  # [B, D]
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # [B, T, D]
        # Skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y


# ---------------------------------------------------------------------------
# MambaBlock wrapper: auto-dispatches between CUDA and fallback
# ---------------------------------------------------------------------------


def _make_mamba_block(
    d_model: int,
    d_state: int,
    d_conv: int,
    expand: int,
    backend: str = 'auto',
) -> nn.Module:
    """Create a single MambaBlock, dispatching based on *backend*.

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension.
        d_conv: Convolution kernel width.
        expand: Expansion factor.
        backend: ``'auto'`` (try CUDA, fallback), ``'cuda'``, or
            ``'pure_pytorch'``.
    """
    use_cuda = False
    if backend == 'cuda':
        if not _HAS_CUDA_MAMBA:
            raise RuntimeError(
                "backend='cuda' requested but mamba-ssm is not installed"
            )
        use_cuda = True
    elif backend == 'auto':
        use_cuda = _HAS_CUDA_MAMBA and torch.cuda.is_available()
    # else: pure_pytorch

    if use_cuda:
        return _CUDAMambaBlock(  # type: ignore[return-value]
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    if backend == 'auto' and not _HAS_CUDA_MAMBA:
        warnings.warn(
            'mamba-ssm not available; using pure-PyTorch fallback. '
            'Training will be significantly slower.',
            RuntimeWarning,
            stacklevel=2,
        )
    return MambaBlockFallback(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
    )


# ---------------------------------------------------------------------------
# MambaForecaster — full model
# ---------------------------------------------------------------------------


class MambaForecaster(nn.Module):
    """Mamba-based multi-asset forecaster with optional graph-context fusion.

    Architecture
    ------------
    1. ``Linear(F, d_model)`` + dropout  →  ``[B, T, d_model]``
    2. *N* × ``MambaBlock`` with residual connections
    3. ``LayerNorm``
    4. Last-token pooling  →  ``[B, d_model]``
    5. (optional) Graph fusion: project ``graph_context[asset_index]``
       to ``graph_dim``, concatenate  →  ``[B, d_model + graph_dim]``
    6. MLP head  →  ``[B, H]``

    Args:
        input_dim: Number of input features *F*.
        d_model: Internal SSM / projection dimension.
        d_state: SSM state dimension *N*.
        d_conv: Local convolution width inside each MambaBlock.
        expand: Expansion factor for the inner MLP in each MambaBlock.
        num_layers: Number of stacked MambaBlocks.
        num_assets: Size of the asset universe *A* (used for graph fusion).
        graph_dim: Projected dimension for graph-context features.
        use_graph_context: Whether to enable the graph-fusion pathway.
        dropout: Dropout probability.
        forecast_horizons: Number of forecast horizons *H*.
        backend: Mamba backend — ``'auto'``, ``'cuda'``, or ``'pure_pytorch'``.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 4,
        num_assets: int = 1,
        graph_dim: int = 32,
        use_graph_context: bool = True,
        dropout: float = 0.1,
        forecast_horizons: int = 1,
        backend: str = 'auto',
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_graph_context = use_graph_context
        self.graph_dim = graph_dim
        self.forecast_horizons = forecast_horizons

        # Step 1 — input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.Dropout(dropout),
        )

        # Step 2 — Mamba block stack with residual connections
        self.mamba_layers = nn.ModuleList([
            _make_mamba_block(d_model, d_state, d_conv, expand, backend)
            for _ in range(num_layers)
        ])

        # Step 3 — layer norm
        self.norm = nn.LayerNorm(d_model)

        # Step 5 — optional graph fusion
        # G = 3 (graph_degree, graph_return_mean_20d, graph_volatility_20d)
        self._graph_context_dim = 3
        if use_graph_context:
            self.graph_proj = nn.Sequential(
                nn.Linear(self._graph_context_dim, graph_dim),
                nn.ReLU(),
            )
            head_input_dim = d_model + graph_dim
        else:
            self.graph_proj = None
            head_input_dim = d_model

        # Step 6 — MLP prediction head
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, forecast_horizons),
        )

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier-uniform for linear layers, zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        asset_index: torch.Tensor | None = None,
        graph_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Price / technical features — ``[B, T, F]``.
            asset_index: Integer asset IDs — ``[B]`` (int64).  Required when
                ``use_graph_context=True`` and *graph_context* is provided.
            graph_context: Per-asset graph features — ``[A, G]`` (float32).

        Returns:
            Predictions — ``[B, H]``.

        Raises:
            ValueError: If input is not 3-D.
        """
        if x.dim() != 3:
            raise ValueError(f'Input must be 3D [B, T, F], got {x.dim()}D')

        # Step 1 — input projection
        z = self.input_proj(x)  # [B, T, d_model]

        # Step 2 — Mamba blocks with residual
        for layer in self.mamba_layers:
            z = z + layer(z)

        # Step 3 — layer norm
        z = self.norm(z)  # [B, T, d_model]

        # Step 4 — last-token pooling
        z = z[:, -1, :]  # [B, d_model]

        # Step 5 — optional graph fusion
        if (
            self.use_graph_context
            and graph_context is not None
            and asset_index is not None
        ):
            gc = graph_context.to(x.device)
            gc = gc[asset_index]  # [B, G]
            assert self.graph_proj is not None  # always set when use_graph_context
            gc_proj = self.graph_proj(gc)  # [B, graph_dim]
            z = torch.cat([z, gc_proj], dim=-1)  # [B, d_model + graph_dim]
        elif self.use_graph_context and self.graph_proj is not None:
            # Graph enabled but no context provided — pad with zeros so head
            # dimension stays consistent.
            pad = torch.zeros(
                z.shape[0], self.graph_dim, device=z.device, dtype=z.dtype
            )
            z = torch.cat([z, pad], dim=-1)

        # Step 6 — MLP head
        return self.head(z)  # [B, H]
