"""Core model and data helpers for Sz training and prediction."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Optional, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainingModel:
    """A scikit-learn regressor wrapped with preprocessing and grid search."""

    model_name: str
    model: Optional[BaseEstimator] = None
    hyper_param_grid: dict[str, list] = field(default_factory=dict)
    cv_folds: int = 5
    scoring: str = "neg_mean_squared_error"
    n_jobs: int = -1
    trained_model: Optional[GridSearchCV] = None

    def train(
        self,
        features_names: list[str],
        x: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """Train the model with standard scaling and grid search."""
        logging.info("Training %s...", self.model_name)
        if self.model is None:
            raise ValueError("Set model before training.")
        if not self.hyper_param_grid:
            raise ValueError("Set hyper_param_grid before training.")

        preprocessor = ColumnTransformer(
            transformers=[("num", StandardScaler(), features_names)]
        )
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", self.model)]
        )
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=self.hyper_param_grid,
            cv=self.cv_folds,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )
        grid_search.fit(x, y)
        self.trained_model = grid_search

    def predict(self, x: pd.DataFrame) -> pd.Series:
        """Predict target values."""
        if self.trained_model is None:
            raise ValueError("No model available. Call train() or load_model() first.")
        return pd.Series(self.trained_model.predict(x), index=x.index, name="prediction")

    def predict_with_uncertainty(
        self,
        x: pd.DataFrame,
        x_uncertainty: pd.DataFrame,
        n_mc_samples: int,
    ) -> tuple[pd.Series, pd.Series]:
        """Predict with feature uncertainty using Monte Carlo sampling."""
        logging.info("Predicting with uncertainty using %s...", self.model_name)
        if self.trained_model is None:
            raise ValueError("No model available. Call train() or load_model() first.")
        if n_mc_samples <= 0:
            raise ValueError("n_mc_samples must be positive.")

        predictions = []
        x_values = x.to_numpy()
        x_std_values = x_uncertainty.to_numpy()
        for _ in range(n_mc_samples):
            x_sample = pd.DataFrame(
                np.random.normal(loc=x_values, scale=x_std_values),
                columns=x.columns,
                index=x.index,
            )
            predictions.append(self.trained_model.predict(x_sample))

        predictions_array = np.array(predictions)
        return (
            pd.Series(
                np.mean(predictions_array, axis=0),
                index=x.index,
                name="prediction",
            ),
            pd.Series(
                np.std(predictions_array, axis=0),
                index=x.index,
                name="uncertainty",
            ),
        )

    def evaluate(
        self,
        target_test: pd.Series,
        target_test_uncertainty: pd.Series,
        target_pred: pd.Series,
        target_pred_uncertainty: pd.Series,
        n_mc_samples: int = 1000,
    ) -> dict[str, float]:
        """Evaluate predictions while propagating target and prediction uncertainty."""
        logging.info("Evaluating %s...", self.model_name)
        y_true_mean = target_test.to_numpy()
        y_true_std = target_test_uncertainty.to_numpy()
        y_pred_mean = target_pred.to_numpy()
        y_pred_std = target_pred_uncertainty.to_numpy()

        y_true_samples = np.random.normal(
            loc=np.expand_dims(y_true_mean, axis=-1),
            scale=np.expand_dims(y_true_std, axis=-1),
            size=(len(y_true_mean), n_mc_samples),
        )
        y_pred_samples = np.random.normal(
            loc=np.expand_dims(y_pred_mean, axis=-1),
            scale=np.expand_dims(y_pred_std, axis=-1),
            size=(len(y_pred_mean), n_mc_samples),
        )

        rmse_list = []
        r2_list = []
        for idx in range(n_mc_samples):
            rmse_list.append(
                np.sqrt(mean_squared_error(y_true_samples[:, idx], y_pred_samples[:, idx]))
            )
            r2_list.append(r2_score(y_true_samples[:, idx], y_pred_samples[:, idx]))

        return {
            "rmse_mean": float(np.mean(rmse_list)),
            "rmse_std": float(np.std(rmse_list)),
            "r2_mean": float(np.mean(r2_list)),
            "r2_std": float(np.std(r2_list)),
        }

    def save_model(self, save_path: Path) -> None:
        """Save the trained model."""
        if self.trained_model is None:
            raise ValueError("No model available. Call train() or load_model() first.")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.trained_model, save_path)
        logging.info("Saved %s to %s", self.model_name, save_path)

    def load_model(self, load_path: Path) -> None:
        """Load a pretrained model."""
        self.trained_model = joblib.load(load_path)
        logging.info("Loaded %s from %s", self.model_name, load_path)


class ModelData:
    """Data container for training or prediction with uncertainty columns."""

    def __init__(
        self,
        data: pd.DataFrame,
        features_names: list[str],
        dataset_name: str = "dataset",
        target_name: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.features_names = features_names
        self.target_name = target_name
        self.random_state = random_state

        required_columns = set(self.features_names)
        if self.target_name:
            required_columns.add(self.target_name)

        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing columns: {sorted(missing_columns)}")

        if data[list(required_columns)].isnull().to_numpy().any():
            initial_rows = len(data)
            data = data.dropna(subset=list(required_columns)).copy()
            logging.warning("Dropped %s rows because of NA values", initial_rows - len(data))
        else:
            data = data.copy()

        self.data = data
        self.features: Optional[pd.DataFrame] = None
        self.features_uncertainty: Optional[pd.DataFrame] = None
        self.target: Optional[pd.Series] = None
        self.target_uncertainty: Optional[pd.Series] = None
        self.features_train: Optional[pd.DataFrame] = None
        self.features_train_uncertainty: Optional[pd.DataFrame] = None
        self.target_train: Optional[pd.Series] = None
        self.target_train_uncertainty: Optional[pd.Series] = None
        self.features_test: Optional[pd.DataFrame] = None
        self.features_test_uncertainty: Optional[pd.DataFrame] = None
        self.target_test: Optional[pd.Series] = None
        self.target_test_uncertainty: Optional[pd.Series] = None
        self.features_train_mc: Optional[pd.DataFrame] = None
        self.target_train_mc: Optional[pd.Series] = None
        self.train_set: Optional[pd.DataFrame] = None
        self.test_set: Optional[pd.DataFrame] = None

    def _ensure_uncertainty_columns(self) -> None:
        cols_to_check = self.features_names.copy()
        if self.target_name:
            cols_to_check.append(self.target_name)

        for col in cols_to_check:
            col_std = f"{col}_std"
            if col_std not in self.data.columns:
                logging.warning("Standard deviation column %s not found, assuming 0", col_std)
                self.data[col_std] = np.zeros_like(self.data[col], dtype=float)

    def _generate_mc_samples(
        self,
        df: pd.DataFrame,
        n_mc_samples: int,
    ) -> tuple[pd.DataFrame, pd.Series]:
        if self.target_name is None:
            raise ValueError("target_name is required for Monte Carlo training samples.")

        features_data = {}
        for feature in self.features_names:
            means = df[feature].to_numpy().repeat(n_mc_samples)
            stds = df[f"{feature}_std"].to_numpy().repeat(n_mc_samples)
            features_data[feature] = np.random.normal(means, stds)

        target_means = df[self.target_name].to_numpy().repeat(n_mc_samples)
        target_stds = df[f"{self.target_name}_std"].to_numpy().repeat(n_mc_samples)
        return (
            pd.DataFrame(features_data),
            pd.Series(
                np.random.normal(target_means, target_stds),
                name=self.target_name,
            ),
        )

    def pre_process(
        self,
        test_size: float = 0.0,
        n_mc_samples: int = 0,
        split_strategy: str = "random",
        test_prefix: str = "93L",
    ) -> None:
        """Prepare features, uncertainties, and optional train/test splits."""
        self._ensure_uncertainty_columns()
        self.features = self.data[self.features_names]
        self.features_uncertainty = self.data[[f"{col}_std" for col in self.features_names]]

        if self.target_name:
            self.target = self.data[self.target_name]
            self.target_uncertainty = self.data[f"{self.target_name}_std"]

        has_test_set = True
        if split_strategy == "random" and test_size <= 0:
            train_set = self.data.copy()
            test_set = None
            has_test_set = False
        elif split_strategy == "prefix":
            if "Sample_ID" not in self.data.columns:
                raise ValueError("Column 'Sample_ID' is required for split_strategy='prefix'.")
            prefixes = self.data["Sample_ID"].str.split("_").str[0]
            test_mask = prefixes == test_prefix
            if not test_mask.any():
                raise ValueError(
                    f"No samples found with prefix '{test_prefix}'. "
                    f"Available prefixes: {sorted(prefixes.unique().tolist())}"
                )
            train_set = cast(pd.DataFrame, self.data[~test_mask].copy())
            test_set = cast(pd.DataFrame, self.data[test_mask].copy())
        elif split_strategy == "random":
            train_set, test_set = train_test_split(
                self.data,
                test_size=test_size,
                random_state=self.random_state,
            )
            train_set = cast(pd.DataFrame, train_set)
            test_set = cast(pd.DataFrame, test_set)
        else:
            raise ValueError(
                f"Unknown split_strategy '{split_strategy}'. Choose 'random' or 'prefix'."
            )

        self.train_set = train_set
        self.test_set = test_set
        self.features_train = cast(pd.DataFrame, train_set[self.features_names])
        self.features_train_uncertainty = cast(
            pd.DataFrame,
            train_set[[f"{col}_std" for col in self.features_names]],
        )

        if self.target_name:
            self.target_train = cast(pd.Series, train_set[self.target_name])
            self.target_train_uncertainty = cast(
                pd.Series,
                train_set[f"{self.target_name}_std"],
            )

        if n_mc_samples > 0 and self.target_name:
            self.features_train_mc, self.target_train_mc = self._generate_mc_samples(
                train_set,
                n_mc_samples,
            )
        else:
            self.features_train_mc = None
            self.target_train_mc = None

        if not has_test_set or test_set is None:
            self.features_test = None
            self.features_test_uncertainty = None
            self.target_test = None
            self.target_test_uncertainty = None
            return

        self.features_test = cast(pd.DataFrame, test_set[self.features_names])
        self.features_test_uncertainty = cast(
            pd.DataFrame,
            test_set[[f"{col}_std" for col in self.features_names]],
        )
        if self.target_name:
            self.target_test = cast(pd.Series, test_set[self.target_name])
            self.target_test_uncertainty = cast(
                pd.Series,
                test_set[f"{self.target_name}_std"],
            )
