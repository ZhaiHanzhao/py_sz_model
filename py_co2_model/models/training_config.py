import os
from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from py_co2_model.models.models import (
    ModelData,
    TrainingModel,
    NGBRegressorWrapper,
)

DATA_RANDOM_STATE = 42
MODEL_RANDOM_STATE = 42
path = Path(__file__).parents[2] / "data/800kyr_amorphous_with_particlesize.CSV"
data = pd.read_csv(path)

TRAIN_SZ = ModelData(
    data=data,
    dataset_name="sz",
    features_names=[
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
    ],
    target_name="Sz",
    random_state=DATA_RANDOM_STATE,
)

MODELS = {
    "Ridge": TrainingModel(
        model=Ridge(random_state=MODEL_RANDOM_STATE),
        model_name="Ridge",
        hyper_param_grid={"model__alpha": [0.1, 1.0, 10.0, 100.0]},
    ),
    "RandomForest": TrainingModel(
        model=RandomForestRegressor(random_state=MODEL_RANDOM_STATE),
        model_name="RandomForest",
        hyper_param_grid={
            "model__n_estimators": [50, 100],
            "model__max_depth": [None, 10],
            "model__min_samples_split": [2, 5],
        },
    ),
    "GradientBoosting": TrainingModel(
        model=GradientBoostingRegressor(random_state=MODEL_RANDOM_STATE),
        model_name="GradientBoosting",
        hyper_param_grid={
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5],
        },
    ),
    "NGBoost": TrainingModel(
        model=NGBRegressorWrapper(random_state=MODEL_RANDOM_STATE, verbose=False),
        model_name="NGBoost",
        hyper_param_grid={
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.01, 0.05],
            "model__minibatch_frac": [0.5, 1.0],
            "model__Base": [
                DecisionTreeRegressor(max_depth=3),
                DecisionTreeRegressor(max_depth=5),
                DecisionTreeRegressor(max_depth=None),
            ],
        },
    ),
}
