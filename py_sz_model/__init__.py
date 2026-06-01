"""Core Sz/CO2 training and prediction package."""

from .config import (
    MODEL_NAMES,
    PREDICTION_FILE_NAMES,
    SZ_FEATURE_NAMES,
    build_models,
    load_sz_training_data,
)
from .models import ModelData, TrainingModel
from .r_calculation import calculate_R_for_dataframe, calculate_R_with_uncertainty

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

__version__ = "0.1.0"
