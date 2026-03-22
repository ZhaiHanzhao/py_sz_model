import logging
import os
from pathlib import Path
import sys

import pandas as pd

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from py_co2_model.models.training_config import (
    MODELS,
    TRAIN_SZ,
)
from py_co2_model.plots.plotting import (
    plot_prediction_performance,
    plot_residual_analysis,
    plot_residual_vs_uncertainty,
)

MODEL_TO_TRAIN = ["GradientBoosting", "RandomForest", "Ridge"]

DATASETS = [TRAIN_SZ]
if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING"))

    for DATA in DATASETS:
        logging.info("Processing dataset: %s", DATA.dataset_name)

        DATA.pre_process(test_size=0.25, n_mc_samples=1000, split_strategy="random")

        for model_name, model in MODELS.items():
            if model_name not in MODEL_TO_TRAIN:
                continue
            if DATA.features_train is None or DATA.target_train is None:
                raise ValueError(
                    "Features or target is None. Please preprocess data first."
                )

            if DATA.features_train_mc is not None and DATA.target_train_mc is not None:
                logging.info(
                    "Using Monte Carlo expanded data for %s (n=%d)",
                    model_name,
                    len(DATA.features_train_mc),
                )
                x_train = DATA.features_train_mc
                y_train = DATA.target_train_mc
            else:
                logging.warning(
                    "MC data not available for %s, using means.", model_name
                )
                x_train = DATA.features_train
                y_train = DATA.target_train

            model.train(
                features_names=DATA.features_names,
                x=x_train,
                y=y_train,
                x_uncertainty=DATA.features_train_uncertainty,
                y_uncertainty=DATA.target_train_uncertainty,
            )

            model.save_model(
                Path(__file__).parents[2]
                / "models/"
                / f"{model_name}_{DATA.dataset_name}.joblib"
            )

            if DATA.features_test is not None:
                predictions = model.predict(DATA.features_test)

                if DATA.features_test_uncertainty is not None:
                    predictions_with_uncertainty = model.predict_with_uncertainty(
                        DATA.features_test,
                        DATA.features_test_uncertainty,
                        10000,
                    )

                    if DATA.test_set is not None:
                        DATA.test_set["prediction"] = predictions_with_uncertainty[0]
                        DATA.test_set["prediction_uncertainty"] = (
                            predictions_with_uncertainty[1]
                        )
                        predictions_path = (
                            Path(__file__).parents[2]
                            / "predictions/"
                            / f"{model_name}_{DATA.dataset_name}.csv"
                        )
                        predictions_path.parent.mkdir(exist_ok=True)
                        DATA.test_set.to_csv(
                            predictions_path,
                            index=False,
                        )
                        logging.info(
                            "Saved predictions combined with true values to %s",
                            predictions_path,
                        )

                    if (
                        DATA.target_test is not None
                        and DATA.target_test_uncertainty is not None
                        and predictions_with_uncertainty is not None
                    ):
                        metrics = model.evaluate(
                            target_test=DATA.target_test,
                            target_test_uncertainty=DATA.target_test_uncertainty,
                            target_pred=predictions_with_uncertainty[0],
                            target_pred_uncertainty=predictions_with_uncertainty[1],
                            n_mc_samples=1000,
                        )

                        metrics_path = (
                            Path(__file__).parents[2]
                            / f"metrics/{model_name}_{DATA.dataset_name}.csv"
                        )
                        metrics_path.parent.mkdir(exist_ok=True)
                        pd.DataFrame(metrics, index=[model_name]).to_csv(
                            metrics_path,
                            index=True,
                            index_label="model",
                        )
                        logging.info("Saved metrics to %s", metrics_path)

                        plot_path = (
                            Path(__file__).parents[2]
                            / f"plots/{model_name}_{DATA.dataset_name}_performance.png"
                        )

                        axis_limits = None
                        residual_limits = None
                        if DATA.dataset_name.lower() == "sz":
                            axis_limits = (200, 1200)
                            residual_limits = (-400, 400)
                        elif DATA.dataset_name.lower() == "co2":
                            axis_limits = (100, 350)
                            residual_limits = (-150, 150)

                        plot_prediction_performance(
                            target_true=DATA.target_test,
                            target_pred=predictions_with_uncertainty[0],
                            target_true_std=DATA.target_test_uncertainty,
                            target_pred_std=predictions_with_uncertainty[1],
                            metrics=metrics,
                            save_path=plot_path,
                            axis_limits=axis_limits,
                            residual_limits=residual_limits,
                        )
                        logging.info("Saved performance plot to %s", plot_path)

                        if DATA.target_test_uncertainty is not None:
                            res_unc_plot_path = (
                                Path(__file__).parents[2]
                                / f"plots/{model_name}_{DATA.dataset_name}_residual_vs_uncertainty.png"
                            )
                            plot_residual_vs_uncertainty(
                                target_true_std=DATA.target_test_uncertainty,
                                target_pred=predictions_with_uncertainty[0],
                                target_true=DATA.target_test,
                                save_path=res_unc_plot_path,
                            )
                            logging.info(
                                "Saved residual vs uncertainty plot to %s",
                                res_unc_plot_path,
                            )

                        residual_metrics = model.evaluate_residuals(
                            target_test=DATA.target_test,
                            target_pred=predictions_with_uncertainty[0],
                        )
                        if residual_metrics:

                            res_metrics_path = (
                                Path(__file__).parents[2]
                                / f"metrics/{model_name}_{DATA.dataset_name}_residuals.csv"
                            )
                            pd.DataFrame(residual_metrics, index=[model_name]).to_csv(
                                res_metrics_path, index=True, index_label="model"
                            )
                            logging.info(
                                "Saved residual metrics to %s", res_metrics_path
                            )

                            res_plot_path = (
                                Path(__file__).parents[2]
                                / f"plots/{model_name}_{DATA.dataset_name}_residuals.png"
                            )
                            plot_residual_analysis(
                                target_true=DATA.target_test,
                                target_pred=predictions_with_uncertainty[0],
                                save_path=res_plot_path,
                            )
                            logging.info(
                                "Saved residual analysis plot to %s", res_plot_path
                            )
                else:
                    logging.warning(
                        "Test features uncertainty is None. Skipping prediction with uncertainty."
                    )
            else:
                logging.warning("Test features is None. Skipping prediction.")
