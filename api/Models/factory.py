"""
Model factory: create_model(model_type, model_cfg) → nn.Module.
"""

import torch.nn as nn

from api.models.cnn_transformer import CnnTransForecaster
from api.models.lstm_baseline import LSTMBaseline
from api.models.mamba_ssm import MambaForecaster

MODEL_REGISTRY: dict[str, type] = {
    'lstm_baseline': LSTMBaseline,
    'cnn_transformer': CnnTransForecaster,
    'mamba_ssm': MambaForecaster,
}

# Constructor kwargs extracted from model config for each registered model type.
_MODEL_KWARGS: dict[str, list[str]] = {
    'lstm_baseline': ['input_dim', 'hidden_dim', 'num_layers', 'dropout', 'horizon'],
    'cnn_transformer': [
        'n_features',
        'seq_len',
        'horizon',
        'conv_channels',
        'conv_kernel',
        'd_model',
        'n_heads',
        'n_encoder_layers',
        'ff_dim',
        'dropout',
        'mlp_hidden',
    ],
    'mamba_ssm': [
        'input_dim',
        'd_model',
        'd_state',
        'd_conv',
        'expand',
        'num_layers',
        'num_assets',
        'graph_dim',
        'use_graph_context',
        'dropout',
        'forecast_horizons',
    ],
}


def create_model(model_type: str, model_cfg: dict) -> nn.Module:
    """Instantiate model from registry.

    Args:
        model_type: Key into MODEL_REGISTRY (e.g. 'lstm_baseline', 'cnn_transformer').
        model_cfg: Dict of hyperparameters from config['model'].

    Returns:
        Instantiated nn.Module.

    Raises:
        ValueError: If model_type is not registered.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f'unsupported_model_type: {model_type}')
    cls = MODEL_REGISTRY[model_type]
    allowed_keys = _MODEL_KWARGS.get(model_type, [])
    kwargs = {k: model_cfg[k] for k in allowed_keys if k in model_cfg}
    return cls(**kwargs)
