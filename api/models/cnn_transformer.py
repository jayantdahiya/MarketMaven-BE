"""
CNN-Transformer hybrid forecaster for daily log-return prediction.

Architecture (non-multimodal / Phase 1):
  [B, T, F]
    → 2× Conv1D(128, k=3, GELU, BN, padding='same')
    → Linear projection to d_model
    → Sinusoidal positional encoding
    → 3× TransformerEncoderLayer(nhead=4, ff=256, GELU)
    → Attention pooling (learnable query vector)
    → MLP(d_model → mlp_hidden → H)
    → [B, H]

Architecture (multimodal / Phase 3):
  [B, T, 27]
    → split into 4 groups by feature indices
    → 4× ModalityEncoder(Linear→GELU→Dropout) → each [B, T, encoder_dim]
    → Gated fusion with per-modality mask → [B, T, fusion_dim]
    → 2× Conv1D(conv_channels, k=3)
    → Positional encoding → Transformer → Attention pooling → MLP head
    → [B, H]
"""

import math

import torch
import torch.nn as nn


class ModalityEncoder(nn.Module):
    """Single-modality encoder: Linear → GELU → Dropout.

    Args:
        in_dim: Number of input features for this modality.
        out_dim: Latent dimension (encoder_dim).
        dropout: Dropout rate.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode modality features.

        Args:
            x: Input tensor [B, T, in_dim].

        Returns:
            Encoded tensor [B, T, out_dim].
        """
        return self.net(x)


class CnnTransForecaster(nn.Module):
    """CNN-Transformer hybrid for daily return forecasting.

    When ``multimodal=False`` (default), behaves identically to the Phase 1
    architecture.  When ``multimodal=True``, adds per-modality encoders and
    gated fusion before the CNN-Transformer backbone.

    Args:
        n_features: Number of input features (F). Default 8.
        seq_len: Input sequence length (T). Default 60.
        horizon: Forecast horizon (H). Default 1.
        conv_channels: Output channels for each Conv1d block. Default 128.
        conv_kernel: Kernel size for Conv1d layers. Default 3.
        d_model: Transformer model dimension. Default 128.
        n_heads: Number of attention heads. Default 4.
        n_encoder_layers: Number of TransformerEncoderLayer blocks. Default 3.
        ff_dim: Feed-forward dimension inside encoder. Default 256.
        dropout: Dropout rate applied throughout. Default 0.1.
        mlp_hidden: Hidden dim in the output MLP head. Default 64.
        multimodal: Enable multimodal encoders + gated fusion. Default False.
        encoder_dim: Per-modality encoder output dimension. Default 64.
        fusion_dim: Post-fusion projection dimension. Default 128.
        feature_group_sizes: Dict mapping group name → num features.
            Required when ``multimodal=True``.  Expected keys:
            ``price_tech``, ``context``, ``sentiment``, ``alpha``.
    """

    pos_enc: torch.Tensor  # sinusoidal PE buffer declared for type-checker

    # Canonical modality order — must match datasets._MODALITY_ORDER.
    _MODALITY_NAMES: list[str] = ['price_tech', 'context', 'sentiment', 'alpha']

    def __init__(
        self,
        n_features: int = 8,
        seq_len: int = 60,
        horizon: int = 1,
        conv_channels: int = 128,
        conv_kernel: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.1,
        mlp_hidden: int = 64,
        multimodal: bool = False,
        encoder_dim: int = 64,
        fusion_dim: int = 128,
        feature_group_sizes: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.horizon = horizon
        self.d_model = d_model
        self.multimodal = multimodal

        # ── Multimodal: per-modality encoders + gated fusion ───────────────
        if self.multimodal:
            if feature_group_sizes is None:
                raise ValueError('feature_group_sizes is required when multimodal=True')
            self.feature_group_sizes = feature_group_sizes
            self.encoder_dim = encoder_dim

            # Build cumulative offsets for slicing x along feature dim
            self._group_slices: dict[str, tuple[int, int]] = {}
            offset = 0
            for name in self._MODALITY_NAMES:
                size = feature_group_sizes[name]
                self._group_slices[name] = (offset, offset + size)
                offset += size

            # Per-modality encoders
            self.price_encoder = ModalityEncoder(
                feature_group_sizes['price_tech'], encoder_dim, dropout
            )
            self.context_encoder = ModalityEncoder(
                feature_group_sizes['context'], encoder_dim, dropout
            )
            self.sentiment_encoder = ModalityEncoder(
                feature_group_sizes['sentiment'], encoder_dim, dropout
            )
            self.alpha_encoder = ModalityEncoder(
                feature_group_sizes['alpha'], encoder_dim, dropout
            )
            self._encoders = [
                self.price_encoder,
                self.context_encoder,
                self.sentiment_encoder,
                self.alpha_encoder,
            ]

            # Learnable gate parameters — initialized to equal weights
            self.gate_params = nn.Parameter(torch.ones(4) * 0.25)

            # Projection from encoder_dim to fusion_dim after gated sum
            self.fusion_project = nn.Linear(encoder_dim, fusion_dim)

            # Conv blocks take fusion_dim as input channels
            conv_in_channels = fusion_dim
        else:
            # Non-multimodal: conv blocks take raw features
            conv_in_channels = n_features

        # ── Stage 1: Convolutional feature extraction ──────────────────────
        # padding='same' preserves sequence length T through both conv blocks.
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(
                conv_in_channels,
                conv_channels,
                kernel_size=conv_kernel,
                padding='same',
            ),
            nn.GELU(),
            nn.BatchNorm1d(conv_channels),
            nn.Dropout(dropout),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(
                conv_channels, conv_channels, kernel_size=conv_kernel, padding='same'
            ),
            nn.GELU(),
            nn.BatchNorm1d(conv_channels),
            nn.Dropout(dropout),
        )

        # ── Stage 2: Linear projection to d_model ──────────────────────────
        self.projection = nn.Linear(conv_channels, d_model)

        # ── Stage 3: Sinusoidal positional encoding (non-learnable buffer) ──
        pe = self._build_sinusoidal_pe(seq_len, d_model)  # [1, T, d_model]
        self.register_buffer('pos_enc', pe)

        self.pos_dropout = nn.Dropout(dropout)

        # ── Stage 4: Transformer encoder ────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,  # [B, T, d_model] convention
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # Learned attention-pooling query: [1, 1, d_model]
        self.attn_query = nn.Parameter(torch.empty(1, 1, d_model))

        # ── Stage 5: MLP head ───────────────────────────────────────────────
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, horizon),
        )

        self._init_weights()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
        """Build standard sinusoidal positional encoding.

        Returns:
            Tensor of shape [1, seq_len, d_model].
        """
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )  # [d_model/2]
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, T, d_model]

    def _init_weights(self) -> None:
        """Apply Kaiming init for Conv1d, Xavier for Linear, small normal for attn_query."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.attn_query, mean=0.0, std=0.02)
        # Re-initialize gate_params after _init_weights clobbers them
        if self.multimodal:
            nn.init.constant_(self.gate_params, 0.25)

    # ── Multimodal fusion ────────────────────────────────────────────────────

    def fuse_modalities(
        self,
        encoded: list[torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Gated fusion of per-modality encoded representations.

        Args:
            encoded: List of 4 tensors, each ``[B, T, encoder_dim]``.
            modality_mask: Binary mask ``[B, 4]`` (1.0 = available).

        Returns:
            Fused tensor ``[B, T, fusion_dim]``.
        """
        # Softmax-normalized gates: [4]
        gates = torch.softmax(self.gate_params, dim=0)  # [4]
        # Apply mask: zero out unavailable modalities per sample
        gates = gates.unsqueeze(0) * modality_mask  # [B, 4]
        # Re-normalize over available modalities
        gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)  # [B, 4]
        # Weighted sum of encoded modalities
        fused = sum(
            gates[:, i].unsqueeze(-1).unsqueeze(-1) * encoded[i] for i in range(4)
        )  # [B, T, encoder_dim]
        # Project to fusion_dim
        return self.fusion_project(fused)  # [B, T, fusion_dim]

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        modality_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, T, F].
            modality_mask: Optional binary mask [B, 4] for multimodal mode.
                Ignored when ``multimodal=False``.

        Returns:
            Predicted log returns of shape [B, H].

        Raises:
            ValueError: If x is not 3-dimensional.
        """
        if x.dim() != 3:
            raise ValueError(f'Input must be 3D [B, T, F], got shape {tuple(x.shape)}')

        if self.multimodal:
            # Default: all modalities available
            if modality_mask is None:
                modality_mask = torch.ones(x.size(0), 4, device=x.device)

            # Split x into modality groups using precomputed slices
            groups = []
            for name in self._MODALITY_NAMES:
                s, e = self._group_slices[name]
                groups.append(x[:, :, s:e])

            # Encode each modality
            encoded = [
                encoder(group)
                for encoder, group in zip(self._encoders, groups, strict=True)
            ]

            # Gated fusion → [B, T, fusion_dim]
            x = self.fuse_modalities(encoded, modality_mask)

        # Stage 1: conv blocks — operate on channels-first [B, C, T]
        x = x.permute(0, 2, 1)  # [B, F/fusion_dim, T]
        x = self.conv_block1(x)  # [B, conv_channels, T]
        x = self.conv_block2(x)  # [B, conv_channels, T]
        x = x.permute(0, 2, 1)  # [B, T, conv_channels]

        # Stage 2: project to d_model
        x = self.projection(x)  # [B, T, d_model]

        # Stage 3: sinusoidal positional encoding
        x = x + self.pos_enc.to(x.dtype)  # [B, T, d_model]  (broadcast over B)
        x = self.pos_dropout(x)

        # Stage 4: transformer encoder (batch_first=True → [B, T, d_model])
        x = self.encoder(x)  # [B, T, d_model]

        # Attention pooling: learned query attends to all T positions
        # query: [1, 1, d_model] → [B, 1, d_model]
        query = self.attn_query.expand(x.size(0), -1, -1)
        # scores: [B, 1, T]
        scores = torch.bmm(query, x.transpose(1, 2)) / math.sqrt(self.d_model)
        attn_weights = torch.softmax(scores, dim=-1)  # [B, 1, T]
        # context: [B, 1, d_model] → [B, d_model]
        context = torch.bmm(attn_weights, x).squeeze(1)

        # Stage 5: MLP head
        out = self.mlp_head(context)  # [B, H]
        return out

    def get_attention_weights(
        self,
        x: torch.Tensor,
        modality_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return attention pooling weights for interpretability.

        Args:
            x: Input tensor [B, T, F].
            modality_mask: Optional binary mask [B, 4] for multimodal mode.

        Returns:
            Attention weights [B, T].
        """
        with torch.no_grad():
            if x.dim() != 3:
                raise ValueError(
                    f'Input must be 3D [B, T, F], got shape {tuple(x.shape)}'
                )

            if self.multimodal:
                if modality_mask is None:
                    modality_mask = torch.ones(x.size(0), 4, device=x.device)
                groups = []
                for name in self._MODALITY_NAMES:
                    s, e = self._group_slices[name]
                    groups.append(x[:, :, s:e])
                encoded = [
                    encoder(group)
                    for encoder, group in zip(self._encoders, groups, strict=True)
                ]
                x = self.fuse_modalities(encoded, modality_mask)

            x = x.permute(0, 2, 1)
            x = self.conv_block1(x)
            x = self.conv_block2(x)
            x = x.permute(0, 2, 1)
            x = self.projection(x)
            x = x + self.pos_enc.to(x.dtype)
            x = self.encoder(x)
            query = self.attn_query.expand(x.size(0), -1, -1)
            scores = torch.bmm(query, x.transpose(1, 2)) / math.sqrt(self.d_model)
            attn_weights = torch.softmax(scores, dim=-1)  # [B, 1, T]
        return attn_weights.squeeze(1)  # [B, T]
