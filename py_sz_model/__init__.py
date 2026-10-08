"""Core Sz/CO2 training and prediction package."""

from .config import (
    MODEL_NAMES,
    PREDICTION_FILE_NAMES,
    SZ_FEATURE_NAMES,
    build_models,
    load_sz_training_data,
)
from .models import ModelData, TrainingModel


def __getattr__(name: str):
    """Keep the public R helpers without importing the CLI before runpy."""
    if name in {"calculate_R_for_dataframe", "calculate_R_with_uncertainty"}:
        from . import r_calculation
        return getattr(r_calculation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MODEL_NAMES",
    "PREDICTION_FILE_NAMES",
    "SZ_FEATURE_NAMES",
    "ModelData",
    "TrainingModel",
    "build_models",
    "calculate_R_for_dataframe",
    "calculate_R_with_uncertainty",
    "load_sz_training_data",
]

__version__ = "0.2.1"
