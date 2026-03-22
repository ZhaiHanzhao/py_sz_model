from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from typing import cast

import joblib
from ngboost import NGBRegressor
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro, kstest
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox


class NGBRegressorWrapper(NGBRegressor):
    """
    Wrapper for NGBRegressor to fix sklearn pipeline compatibility issue.
    Sets __sklearn_is_fitted__ to True after fitting.
    """

    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        return self

    def __sklearn_is_fitted__(self):
        return True


@dataclass
class TrainingModel:
    """
    Class for training a model using GridSearchCV.

    Attributes:
        model_name (str): Name of the model.
        model (BaseEstimator): The model to be trained.
        hyper_param_grid (Dict[str, list]): Grid of hyperparameters to search over.
        cv_folds (Optional[int]): Number of folds for cross-validation.
        scoring (Optional[str]): Scoring metric for model selection.
        n_jobs (Optional[int]): Number of jobs to run in parallel.
        trained_model (Optional[GridSearchCV]): The trained model.
    """

    model_name: str
    model: Optional[BaseEstimator] = None
    hyper_param_grid: Dict[str, list] = field(default_factory=dict)
    cv_folds: Optional[int] = 5
    scoring: Optional[str] = "neg_mean_squared_error"
    n_jobs: Optional[int] = -1
    trained_model: Optional[GridSearchCV] = None

    def train(
        self,
        features_names: List[str],
        x: pd.DataFrame,
        y: pd.Series,
        x_uncertainty: Optional[pd.DataFrame] = None,
        y_uncertainty: Optional[pd.Series] = None,
    ) -> None:
        """
        Train the model using GridSearchCV.

        Args:
            features_names: List of feature names.
            x: Training data features.
            y: Training data labels.

        Returns:
            The fitted GridSearchCV object.
        """
        logging.info("Training %s...", self.model_name)

        preprocessor = ColumnTransformer(
            transformers=[("num", StandardScaler(), features_names)]
        )

        if self.model is None:
            raise ValueError("Set model before training.")

        if not self.hyper_param_grid:
            raise ValueError("Set hyper_param_grid before training.")

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
        """
        Make predictions using the trained model.

        Args:
            x: Test data features.

        Returns:
            The predicted values.
        """
        if self.trained_model is None:
            raise ValueError("No model available. Call train() or load_model() first.")
        return self.trained_model.predict(x)

    def predict_with_uncertainty(
        self, x: pd.DataFrame, x_uncertainty: pd.DataFrame, n_mc_samples: int
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Make predictions with uncertainty using the trained model with Monte Carlo simulation.

        Args:
            x: Test data features.
            x_uncertainty: Uncertainty of test data features.
            n_mc_samples: Number of Monte Carlo samples.

        Returns:
            A tuple containing:
            - The predicted values (mean of MC samples).
            - The uncertainty of the predictions (std of MC samples).
        """
        logging.info(
            "Predicting with uncertainty on test set using %s...", self.model_name
        )
        if self.trained_model is None:
            raise ValueError("No model available. Call train() or load_model() first.")

        best_pipeline = self.trained_model.best_estimator_
        if hasattr(best_pipeline, "named_steps"):
            model_step = best_pipeline.named_steps["model"]
            if hasattr(model_step, "pred_dist"):
                logging.info("Using pred_dist for uncertainty prediction.")

                preprocessor = best_pipeline.named_steps["preprocessor"]
                x_transformed = preprocessor.transform(x)
                dist = model_step.pred_dist(x_transformed)

                mean_prediction = pd.Series(dist.loc, index=x.index, name="prediction")
                std_prediction = pd.Series(
                    dist.scale, index=x.index, name="uncertainty"
                )
                return mean_prediction, std_prediction

        predictions = []
        for _ in range(n_mc_samples):

            x_sample = pd.DataFrame(
                np.random.normal(x, x_uncertainty), columns=x.columns, index=x.index
            )
            predictions.append(self.trained_model.predict(x_sample))

        predictions_array = np.array(predictions)
        mean_prediction = pd.Series(
            np.mean(predictions_array, axis=0), index=x.index, name="prediction"
        )
        std_prediction = pd.Series(
            np.std(predictions_array, axis=0), index=x.index, name="uncertainty"
        )

        return mean_prediction, std_prediction

    def evaluate(
        self,
        target_test: pd.Series,
        target_test_uncertainty: pd.Series,
        target_pred: pd.Series,
        target_pred_uncertainty: pd.Series,
        n_mc_samples: int = 10000,
    ) -> Dict[str, float]:
        """
        Evaluate the model on the test set using Monte Carlo simulation.

        Args:
            target_test: Test data labels (mean and std columns).
            target_pred: Predicted labels (mean and std columns).
            n_mc_samples: Number of Monte Carlo samples.

        Returns:
            A dictionary containing the evaluation metrics (mean and std of RMSE and R2).
        """
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

        for i in range(n_mc_samples):
            rmse_list.append(
                np.sqrt(mean_squared_error(y_true_samples[:, i], y_pred_samples[:, i]))
            )
            r2_list.append(r2_score(y_true_samples[:, i], y_pred_samples[:, i]))

        return {
            "rmse_mean": float(np.mean(rmse_list)),
            "rmse_std": float(np.std(rmse_list)),
            "r2_mean": float(np.mean(r2_list)),
            "r2_std": float(np.std(r2_list)),
        }

    def evaluate_residuals(
        self,
        target_test: pd.Series,
        target_pred: pd.Series,
    ) -> Optional[Dict[str, float]]:
        """
        Calculates residuals and performs statistical tests for normality and randomness.

        Args:
            target_test: True target values.
            target_pred: Predicted target values.

        Returns:
            A dictionary containing the test results, or None if data is insufficient.
        """
        logging.info("Evaluating residuals for %s...", self.model_name)

        valid_data = pd.concat([target_test, target_pred], axis=1).dropna()

        if len(valid_data) < 4:
            logging.warning("Insufficient data for residual analysis.")
            return None

        y_true_clean = valid_data.iloc[:, 0]
        y_pred_clean = valid_data.iloc[:, 1]

        residuals = y_pred_clean - y_true_clean

        shapiro_stat, shapiro_p = shapiro(residuals)

        ks_stat, ks_p = kstest(
            residuals, "norm", args=(residuals.mean(), residuals.std())
        )

        dw_stat = durbin_watson(residuals)

        n_obs = len(residuals)
        lags = min(10, n_obs // 5)
        lb_stat, lb_pvalue = np.nan, np.nan
        if lags > 0:
            ljung_box_result = acorr_ljungbox(residuals, lags=[lags], return_df=True)
            lb_stat = ljung_box_result["lb_stat"].iloc[0]
            lb_pvalue = ljung_box_result["lb_pvalue"].iloc[0]

        return {
            "n_samples": len(residuals),
            "shapiro_statistic": float(shapiro_stat),
            "shapiro_p_value": float(shapiro_p),
            "ks_statistic": float(ks_stat),
            "ks_p_value": float(ks_p),
            "durbin_watson_statistic": float(dw_stat),
            "ljung_box_statistic": float(lb_stat),
            "ljung_box_p_value": float(lb_pvalue),
            "ljung_box_lags": float(lags),
        }

    def save_model(self, save_path: Path) -> None:
        """
        Save the trained model.

        Args:
            save_path: Path to save the model file.
        """
        if self.trained_model is None:
            raise ValueError("No model available. Call train() or load_model() first.")
        joblib.dump(self.trained_model, save_path)
        logging.info("Saved %s to %s", self.model_name, save_path)

    def load_model(self, load_path: Path) -> None:
        """
        Load pretrained model.

        Args:
            load_path: Path to load the model.
        """
        self.trained_model = joblib.load(load_path)
        logging.info("Loaded %s from %s", self.model_name, load_path)


@dataclass
class ModelData:
    """
    Class for data used to train a model or make predictions.
    """

    data: pd.DataFrame
    dataset_name: str = "dataset"
    features_names: List[str] = field(default_factory=list)
    target_name: Optional[str] = None
    random_state: Optional[int] = None

    features: Optional[pd.DataFrame] = field(default_factory=pd.DataFrame)
    features_uncertainty: Optional[pd.DataFrame] = field(default_factory=pd.DataFrame)
    target: Optional[pd.Series] = field(default_factory=pd.Series)
    target_uncertainty: Optional[pd.Series] = field(default_factory=pd.Series)

    features_train: Optional[pd.DataFrame] = field(default_factory=pd.DataFrame)
    features_train_uncertainty: Optional[pd.DataFrame] = field(
        default_factory=pd.DataFrame
    )
    target_train: Optional[pd.Series] = field(default_factory=pd.Series)
    target_train_uncertainty: Optional[pd.Series] = field(default_factory=pd.Series)
    features_test: Optional[pd.DataFrame] = field(default_factory=pd.DataFrame)
    features_test_uncertainty: Optional[pd.DataFrame] = field(
        default_factory=pd.DataFrame
    )
    target_test: Optional[pd.Series] = field(default_factory=pd.Series)
    target_test_uncertainty: Optional[pd.Series] = field(default_factory=pd.Series)
    features_train_mc: Optional[pd.DataFrame] = field(default_factory=pd.DataFrame)
    target_train_mc: Optional[pd.Series] = field(default_factory=pd.Series)
    train_set: Optional[pd.DataFrame] = field(default_factory=pd.DataFrame)
    test_set: Optional[pd.DataFrame] = field(default_factory=pd.DataFrame)

    def __init__(
        self,
        data: pd.DataFrame,
        features_names: List[str],
        dataset_name: str = "dataset",
        target_name: Optional[str] = None,
        random_state: Optional[int] = None,
    ):
        self.dataset_name = dataset_name
        self.features_names = features_names
        self.target_name = target_name
        self.random_state = random_state

        required_columns = set(self.features_names)
        if self.target_name:
            required_columns.add(self.target_name)

        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")

        if data[list(required_columns)].isnull().to_numpy().any():
            initial_rows = len(data)
            data.dropna(subset=list(required_columns), inplace=True)
            logging.warning(
                "Dropped %s rows because of NA values", initial_rows - len(data)
            )

        self.data = data

    def _generate_mc_samples(
        self, df: pd.DataFrame, n_mc_samples: int
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate Monte Carlo samples using vectorized operations.
        """
        features_data = {}
        for feature in self.features_names:
            means = df[feature].to_numpy().repeat(n_mc_samples)
            stds = df[feature + "_std"].to_numpy().repeat(n_mc_samples)
            features_data[feature] = np.random.normal(means, stds)

        features_mc = pd.DataFrame(features_data)

        target_means = df[self.target_name].to_numpy().repeat(n_mc_samples)
        target_stds = df[self.target_name + "_std"].to_numpy().repeat(n_mc_samples)
        target_mc = pd.Series(
            np.random.normal(target_means, target_stds), name=self.target_name
        )

        return features_mc, target_mc

    def pre_process(
        self,
        test_size: float = 0.0,
        n_mc_samples: int = 0,
        split_strategy: str = "random",
        test_prefix: str = "93L",
    ) -> None:
        """
        Preprocess the data.

        Args:
            test_size: Proportion of data to use for testing (used only for
                ``split_strategy="random"``). If 0 and strategy is "random",
                no splitting is performed.
            n_mc_samples: Number of Monte Carlo samples to generate for training data.
            split_strategy: How to split train/test data.
                ``"random"`` – random split via ``train_test_split`` (default).
                ``"prefix"`` – split by the ``Sample_ID`` prefix (part before
                the first ``"_"``). All samples whose prefix matches
                ``test_prefix`` go to the test set; the rest become training
                data.
            test_prefix: The ``Sample_ID`` prefix to use as the test set when
                ``split_strategy="prefix"``. Defaults to ``"93L"``.
        """

        if self.data is None:
            raise ValueError("Data is not loaded. Call load_data() first.")

        cols_to_check = self.features_names.copy()
        if self.target_name:
            cols_to_check.append(self.target_name)

        for col in cols_to_check:
            col_std = col + "_std"
            if col_std not in self.data.columns:
                logging.warning(
                    "Standard deviation column %s not found, assuming 0", col_std
                )
                self.data[col_std] = np.zeros_like(self.data[col])

        self.features = self.data[self.features_names]
        self.features_uncertainty = self.data[
            [col + "_std" for col in self.features_names]
        ]

        if self.target_name:
            self.target = self.data[self.target_name]
            self.target_uncertainty = self.data[self.target_name + "_std"]

        if split_strategy == "random" and test_size <= 0:
            return

        if split_strategy == "prefix":
            if "Sample_ID" not in self.data.columns:
                raise ValueError(
                    "Column 'Sample_ID' is required for split_strategy='prefix'."
                )
            prefixes = self.data["Sample_ID"].str.split("_").str[0]
            test_mask = prefixes == test_prefix
            if not test_mask.any():
                raise ValueError(
                    f"No samples found with prefix '{test_prefix}'. "
                    f"Available prefixes: {sorted(prefixes.unique().tolist())}"
                )
            logging.info(
                "Prefix split: using %d samples with prefix '%s' as test set, "
                "%d samples for training.",
                test_mask.sum(),
                test_prefix,
                (~test_mask).sum(),
            )
            train_set = cast(pd.DataFrame, self.data[~test_mask].copy())
            test_set = cast(pd.DataFrame, self.data[test_mask].copy())
        elif split_strategy == "random":
            (
                train_set,
                test_set,
            ) = train_test_split(
                self.data,
                test_size=test_size,
                random_state=self.random_state,
            )
            train_set = cast(pd.DataFrame, train_set)
            test_set = cast(pd.DataFrame, test_set)
        else:
            raise ValueError(
                f"Unknown split_strategy '{split_strategy}'. "
                "Choose 'random' or 'prefix'."
            )

        self.train_set = train_set
        self.test_set = test_set

        if n_mc_samples > 0 and self.target_name:
            self.features_train_mc, self.target_train_mc = self._generate_mc_samples(
                train_set, n_mc_samples
            )
        else:
            self.features_train_mc = None
            self.target_train_mc = None

        self.features_train = cast(pd.DataFrame, train_set[self.features_names])
        self.features_train_uncertainty = cast(
            pd.DataFrame, train_set[[col + "_std" for col in self.features_names]]
        )

        if self.target_name:
            self.target_train = cast(pd.Series, train_set[self.target_name])
            self.target_train_uncertainty = cast(
                pd.Series, train_set[self.target_name + "_std"]
            )
            self.target_test = cast(pd.Series, test_set[self.target_name])
            self.target_test_uncertainty = cast(
                pd.Series, test_set[self.target_name + "_std"]
            )

        self.features_test = cast(pd.DataFrame, test_set[self.features_names])
        self.features_test_uncertainty = cast(
            pd.DataFrame, test_set[[col + "_std" for col in self.features_names]]
        )
