"""
LOB model architectures: TLOBForecaster (dual spatial+temporal attention) and
LobCNNBaseline (ablation). Both accept structured book tensors [B, T, 4, L] and
auxiliary features [B, T, E], and produce class logits [B, 3].
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class _SpatialAttentionBlock(nn.Module):
    """Single spatial attention block: MultiheadAttention over the time-sequence dimension
    with LayerNorm and residual connection.

    Args:
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: [B, T, D]
        Returns:
            [B, T, D]
        """
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.drop(attn_out))


class TLOBForecaster(nn.Module):
    """Dual-attention LOB forecaster.

    Architecture:
        book [B,T,4,L] → flatten → book_proj [B,T,D]
        → SpatialAttn x2 → + pos_enc → TemporalTransformer x3
        → aux_fusion(cat([B,T,D], [B,T,E])) → last-step pool [B,D]
        → classification_head → logits [B,3]

    Args:
        levels: Number of LOB price levels (L).
        channels: Book channels (C=4: ask_px, ask_sz, bid_px, bid_sz).
        aux_dim: Auxiliary feature dimension (E).
        d_model: Hidden dimension throughout network.
        spatial_heads: Attention heads for spatial blocks.
        temporal_heads: Attention heads for temporal transformer.
        layers: Number of temporal transformer encoder layers.
        ff_dim: Feed-forward dimension in temporal transformer.
        dropout: Dropout rate.
        num_classes: Output classes (3 for down/flat/up).
    """

    def __init__(
        self,
        levels: int = 10,
        channels: int = 4,
        aux_dim: int = 4,
        d_model: int = 128,
        spatial_heads: int = 4,
        temporal_heads: int = 4,
        layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.1,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        if d_model % spatial_heads != 0:
            raise ValueError(
                f'd_model ({d_model}) must be divisible by spatial_heads ({spatial_heads})'
            )
        if d_model % temporal_heads != 0:
            raise ValueError(
                f'd_model ({d_model}) must be divisible by temporal_heads ({temporal_heads})'
            )

        self.levels = levels
        self.channels = channels
        self.aux_dim = aux_dim
        self.d_model = d_model
        self.num_classes = num_classes

        # Book projection: flatten [T, C*L] → [T, D]
        self.book_proj = nn.Linear(channels * levels, d_model)

        # Spatial attention blocks x2
        self.spatial_attn_block_1 = _SpatialAttentionBlock(
            d_model, spatial_heads, dropout
        )
        self.spatial_attn_block_2 = _SpatialAttentionBlock(
            d_model, spatial_heads, dropout
        )

        # Sinusoidal positional encoding (registered as buffer, not parameter)
        self._register_pos_encoding(max_len=512, d_model=d_model)

        # Temporal transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=temporal_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=layers
        )

        # Auxiliary fusion: cat([B,T,D], [B,T,E]) → [B,T,D]
        self.aux_fusion = nn.Sequential(
            nn.Linear(d_model + aux_dim, d_model),
            nn.GELU(),
        )

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _register_pos_encoding(self, max_len: int, d_model: int) -> None:
        """Compute and register sinusoidal positional encoding."""
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer('pos_encoding', pe)

    def _init_weights(self) -> None:
        """Xavier uniform for linear layers, zero bias; Xavier normal for attn projections."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_normal_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                if m.out_proj.weight is not None:
                    nn.init.xavier_normal_(m.out_proj.weight)

    def forward(self, x_book: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x_book: [B, T, C, L] structured book tensor.
            x_aux:  [B, T, E] auxiliary features.

        Returns:
            logits: [B, num_classes]
        """
        if x_book.dim() != 4:
            raise ValueError(f'x_book must be 4D [B,T,C,L], got {x_book.dim()}D')
        if x_aux.dim() != 3:
            raise ValueError(f'x_aux must be 3D [B,T,E], got {x_aux.dim()}D')

        B, T, C, L = x_book.shape
        if C != self.channels:
            raise ValueError(f'Expected {self.channels} channels, got {C}')
        if L != self.levels:
            raise ValueError(f'Expected {self.levels} levels, got {L}')

        # Flatten book channels: [B, T, C*L]
        z = x_book.reshape(B, T, C * L)

        # Book projection: [B, T, D]
        z = self.book_proj(z)

        # Spatial attention x2
        z = self.spatial_attn_block_1(z)
        z = self.spatial_attn_block_2(z)

        # Positional encoding
        z = z + self.pos_encoding[:, :T, :]  # type: ignore[operator]

        # Temporal transformer
        z = self.temporal_transformer(z)  # [B, T, D]

        # Auxiliary fusion
        z = self.aux_fusion(torch.cat([z, x_aux], dim=-1))  # [B, T, D]

        # Last-step pooling
        h = z[:, -1, :]  # [B, D]

        # Classification head
        logits = self.classification_head(h)  # [B, 3]
        return logits


class LobCNNBaseline(nn.Module):
    """Simplified 1D-conv ablation baseline for LOB classification.

    Architecture:
        book [B,T,4,L] → flatten [B,T,4*L] → permute [B,4*L,T]
        → Conv1D stack (3 layers, kernel=3) → global avg pool [B,D]
        → Linear head → logits [B,3]

    Same input/output contract as TLOBForecaster for direct comparison.

    Args:
        levels: Number of LOB price levels.
        channels: Book channels (fixed 4).
        aux_dim: Auxiliary feature dimension (E).
        d_model: Hidden channels in conv stack and head.
        dropout: Dropout rate.
        num_classes: Output classes.
    """

    def __init__(
        self,
        levels: int = 10,
        channels: int = 4,
        aux_dim: int = 4,
        d_model: int = 128,
        dropout: float = 0.1,
        num_classes: int = 3,
        **kwargs,  # absorb unused TLOBForecaster kwargs
    ) -> None:
        super().__init__()
        self.levels = levels
        self.channels = channels
        self.aux_dim = aux_dim
        self.d_model = d_model
        self.num_classes = num_classes

        in_channels = channels * levels
        self.conv_stack = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Aux fusion after global average pooling
        self.aux_pool_proj = nn.Linear(aux_dim, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_book: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x_book: [B, T, C, L]
            x_aux:  [B, T, E]

        Returns:
            logits: [B, num_classes]
        """
        if x_book.dim() != 4:
            raise ValueError(f'x_book must be 4D [B,T,C,L], got {x_book.dim()}D')

        B, T, C, L = x_book.shape

        # Flatten book: [B, T, C*L] → permute → [B, C*L, T] for Conv1D
        z = x_book.reshape(B, T, C * L).permute(0, 2, 1)  # [B, C*L, T]
        z = self.conv_stack(z)  # [B, D, T]

        # Global average pool over time
        z_pool = z.mean(dim=-1)  # [B, D]

        # Auxiliary: mean pool over time, project
        aux_pool = x_aux.mean(dim=1)  # [B, E]
        aux_feat = self.aux_pool_proj(aux_pool)  # [B, D]

        # Concatenate and classify
        h = torch.cat([z_pool, aux_feat], dim=-1)  # [B, 2D]
        return self.head(h)  # [B, 3]
