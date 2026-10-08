"""Configuration for Sz model training and Shilou/Jiaxian CO2 prediction."""

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from .models import ModelData, TrainingModel

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
TRAINING_DATA_PATH = DATA_DIR / "800kyr_amorphous_with_particlesize.CSV"
PREDICTION_SET_DIR = DATA_DIR / "prediction_set"
MODELS_DIR = REPO_ROOT / "models"
METRICS_DIR = REPO_ROOT / "metrics"
PREDICTIONS_DIR = REPO_ROOT / "predictions"
FROZEN_DIR = REPO_ROOT / "frozen"
FROZEN_MODELS_DIR = FROZEN_DIR / "models"

DATA_RANDOM_STATE = 42
MODEL_RANDOM_STATE = 42

MODEL_NAMES = ("GradientBoosting", "RandomForest", "Ridge")
PREDICTION_FILE_NAMES = (
    "Shilou_features_bulk.CSV",
    "Jiaxian_features_bulk.CSV",
)

SZ_FEATURE_NAMES = [
    "Xlf",
    "aFe",
    "aSi",
    "aAl",
    "fFe",
    "aFe/aSi",
    "aFe/aAl",
    "aSi/aAl",
    "aFe/fFe",
    "aSi/fFe",
    "aAl/fFe",
    "d18Oc",
    "D",
]


def load_sz_training_data(data_path: Path = TRAINING_DATA_PATH) -> ModelData:
    """Load the Sz training dataset."""
    data = pd.read_csv(data_path)
    return ModelData(
        data=data,
        dataset_name="sz",
        features_names=SZ_FEATURE_NAMES,
        target_name="Sz",
        random_state=DATA_RANDOM_STATE,
    )


def build_models(
    model_names: Iterable[str] = MODEL_NAMES,
    cv_folds: int = 5,
    n_jobs: int = -1,
) -> dict[str, TrainingModel]:
    """Build fresh model wrappers for the supported Sz regressors."""
    candidates = {
        "Ridge": TrainingModel(
            model=Ridge(random_state=MODEL_RANDOM_STATE),
            model_name="Ridge",
            hyper_param_grid={"model__alpha": [0.1, 1.0, 10.0, 100.0]},
            cv_folds=cv_folds,
            n_jobs=n_jobs,
        ),
        "RandomForest": TrainingModel(
            model=RandomForestRegressor(random_state=MODEL_RANDOM_STATE),
            model_name="RandomForest",
            hyper_param_grid={
                "model__n_estimators": [50, 100],
                "model__max_depth": [None, 10],
                "model__min_samples_split": [2, 5],
            },
            cv_folds=cv_folds,
            n_jobs=n_jobs,
        ),
        "GradientBoosting": TrainingModel(
            model=GradientBoostingRegressor(random_state=MODEL_RANDOM_STATE),
            model_name="GradientBoosting",
            hyper_param_grid={
                "model__n_estimators": [50, 100],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 5],
            },
            cv_folds=cv_folds,
            n_jobs=n_jobs,
        ),
    }

    selected = {}
    for name in model_names:
        if name not in candidates:
            supported = ", ".join(candidates)
            raise ValueError(f"Unsupported model '{name}'. Supported models: {supported}")
        selected[name] = candidates[name]
    return selected
