"""
Model factory: create_model(model_type, model_cfg) → nn.Module.
"""
import torch.nn as nn

from api.models.lstm_baseline import LSTMBaseline

MODEL_REGISTRY: dict[str, type] = {
    "lstm_baseline": LSTMBaseline,
}


def create_model(model_type: str, model_cfg: dict) -> nn.Module:
    """Instantiate model from registry. Raises ValueError for unknown type."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"unsupported_model_type: {model_type}")
    cls = MODEL_REGISTRY[model_type]
    # Map config keys to constructor args
    kwargs = {
        "input_dim": model_cfg.get("input_dim", 8),
        "hidden_dim": model_cfg.get("hidden_dim", 128),
        "num_layers": model_cfg.get("num_layers", 2),
        "dropout": model_cfg.get("dropout", 0.2),
        "horizon": model_cfg.get("horizon", 1),
    }
    return cls(**kwargs)
