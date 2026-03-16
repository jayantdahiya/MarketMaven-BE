"""
LOBService: loads a trained LOB checkpoint and runs inference to produce LOBForecastResponse.

Inference flow:
  1. Load TLOBForecaster (or LobCNNBaseline) from checkpoint.
  2. Accept pre-processed (x_book, x_aux) tensors from the caller.
  3. Run model forward pass; apply softmax.
  4. Compute expected_mid_move_ticks as signed weighted sum.
  5. Return LOBForecastResponse.

Note: feature window assembly (fetching recent LOB events) is the caller's
responsibility. This service is stateless beyond the loaded model weights.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import torch

from api.models.factory import create_model
from api.schemas.lob import LOBForecastRequest, LOBForecastResponse
from api.training import checkpointing

logger = logging.getLogger(__name__)


class LOBService:
    """LOB forecast service: checkpoint loading + inference.

    Args:
        cfg: Full application config dict (reads model.*, lob.*).
        checkpoint_path: Optional path to a specific checkpoint .pt file.
                         If None, the service starts without a loaded model
                         (useful for testing / lazy loading).
    """

    def __init__(self, cfg: dict, checkpoint_path: str | None = None) -> None:
        self.cfg = cfg
        self.device = cfg.get('training', {}).get('device', 'cpu')
        model_cfg = cfg.get('model', {})
        self.model_type: str = model_cfg.get('type', 'tlob_forecaster')
        self._model: torch.nn.Module | None = None

        if checkpoint_path is not None:
            self._load_model(checkpoint_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self, checkpoint_path: str) -> None:
        """Instantiate model and load weights from checkpoint."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f'LOB checkpoint not found: {path.resolve()}')

        model_cfg = self.cfg.get('model', {})
        model = create_model(self.model_type, model_cfg)
        checkpointing.load_checkpoint(checkpoint_path, model, device=self.device)
        model.to(self.device)
        model.eval()
        self._model = model
        logger.info(
            'LOBService: loaded model %s from %s', self.model_type, checkpoint_path
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        req: LOBForecastRequest,
        x_book: torch.Tensor,
        x_aux: torch.Tensor,
    ) -> LOBForecastResponse:
        """Run inference and return a LOBForecastResponse.

        Args:
            req: LOBForecastRequest (provides asset_id, as_of_timestamp, model).
            x_book: Pre-processed book tensor [1, T, C, L] (single sample).
            x_aux:  Pre-processed auxiliary tensor [1, T, E].

        Returns:
            LOBForecastResponse with predicted_class, class_probabilities,
            expected_mid_move_ticks, confidence.

        Raises:
            RuntimeError: If no model has been loaded.
        """
        if self._model is None:
            raise RuntimeError(
                'LOBService: no model loaded. Call _load_model() first or '
                'pass checkpoint_path to __init__.'
            )

        x_book = x_book.to(self.device)
        x_aux = x_aux.to(self.device)

        with torch.no_grad():
            logits = self._model(x_book, x_aux)  # [1, 3]
            probs = torch.softmax(logits, dim=-1).squeeze(0)  # [3]

        p_down = float(probs[0].item())
        p_flat = float(probs[1].item())
        p_up = float(probs[2].item())

        # Normalize to exactly 1.0 to satisfy pydantic validator
        total = p_down + p_flat + p_up
        if total > 0:
            p_down, p_flat, p_up = p_down / total, p_flat / total, p_up / total

        predicted_class = int(probs.argmax().item())
        confidence = float(probs.max().item())

        # Expected mid-price move: p_up contributes +1 tick, p_down contributes -1 tick
        tick_size = self.cfg.get('lob', {}).get('tick_size', 0.01)
        horizon_ticks = req.horizon_events  # events ≈ ticks for normalized scale
        expected_mid_move_ticks = (p_up - p_down) * horizon_ticks * tick_size

        as_of = req.as_of_timestamp or datetime.now(tz=timezone.utc)

        return LOBForecastResponse(
            asset_id=req.asset_id,
            as_of_timestamp=as_of,
            predicted_class=predicted_class,
            class_probabilities=[p_down, p_flat, p_up],
            expected_mid_move_ticks=float(expected_mid_move_ticks),
            confidence=confidence,
        )
