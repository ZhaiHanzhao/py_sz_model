import logging
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from py_co2_model.models.models import TrainingModel, ModelData
from py_co2_model.models.training_config import MODELS, TRAIN_SZ
from py_co2_model.models.r_calculation import calculate_R_with_uncertainty

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_prediction_files(data_dir: Path) -> List[Path]:
    return list(data_dir.glob("*.CSV")) + list(data_dir.glob("*.csv"))


def run_predictions():
    root_dir = Path(__file__).parents[2]
    data_dir = root_dir / "data/prediction_set"
    models_dir = root_dir / "models"
    output_dir = root_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = get_prediction_files(data_dir)
    if not input_files:
        logging.warning(f"No CSV files found in {data_dir}")
        return

    model_names = ["GradientBoosting", "RandomForest", "Ridge"]

    feature_names = TRAIN_SZ.features_names

    for input_file in input_files:
        logging.info(f"Processing input file: {input_file.name}")
        try:
            df = pd.read_csv(input_file)
        except Exception as e:
            logging.error(f"Failed to read {input_file}: {e}")
            continue

        if "R" not in df.columns or "R_std" not in df.columns:
            for idx, row in df.iterrows():
                try:
                    res = calculate_R_with_uncertainty(
                        d13Cc=float(row["d13Cc"]),
                        d13Cc_std=float(row["d13Cc_std"]),
                        d13Co=float(row["d13Co"]),
                        d13Co_std=float(row["d13Co_std"]),
                        d13Ca=float(row["d13Ca"]),
                        d13Ca_std=float(row["d13Ca_std"]),
                        T=float(row["Temperature"]),
                        T_std=float(row["Temperature_std"]),
                        num_simulations=100000,
                        ci_level=90,
                    )
                    df.at[idx, "R"] = res["R_mean"]
                    df.at[idx, "R_std"] = res["R_std"]
                    df.at[idx, "R_90_low"] = res["R_ci_lower"]
                    df.at[idx, "R_90_high"] = res["R_ci_upper"]
                except Exception as e:
                    logging.error(
                        f"Error calculating R for row {idx} in {input_file.name}: {e}"
                    )
                    df.at[idx, "R"] = np.nan
                    df.at[idx, "R_std"] = np.nan
                    df.at[idx, "R_90_low"] = np.nan
                    df.at[idx, "R_90_high"] = np.nan

        df.to_csv(input_file, index=False)

        try:

            data_model = ModelData(
                data=df, features_names=feature_names, dataset_name=input_file.stem
            )

            data_model.pre_process(test_size=0)

            x = data_model.features
            x_uncertainty = data_model.features_uncertainty

        except ValueError as e:
            logging.error(f"Data preparation failed for {input_file.name}: {e}")
            continue

        for model_name in model_names:
            if model_name not in MODELS:
                logging.warning(
                    f"Model {model_name} not defined in configuration. Skipping."
                )
                continue

            model_path = models_dir / f"{model_name}_sz.joblib"

            if not model_path.exists():
                logging.warning(f"Model file not found: {model_path}. Skipping.")
                continue

            wrapper = MODELS[model_name]
            try:
                wrapper.load_model(model_path)
            except Exception as e:
                logging.error(
                    f"Failed to load model {model_name} from {model_path}: {e}"
                )
                continue

            logging.info(f"Predicting with {model_name}...")
            try:

                R_mean = df["R"]
                R_std = df["R_std"]

                mean_pred, std_pred = wrapper.predict_with_uncertainty(
                    x=x, x_uncertainty=x_uncertainty, n_mc_samples=1000
                )

                CO2_mean = R_mean * mean_pred
                CO2_std = np.sqrt((R_mean * std_pred) ** 2 + (mean_pred * R_std) ** 2)

                Sz_ci_lower = mean_pred - 1.645 * std_pred
                Sz_ci_upper = mean_pred + 1.645 * std_pred
                CO2_ci_lower = CO2_mean - 1.645 * CO2_std
                CO2_ci_upper = CO2_mean + 1.645 * CO2_std

                result_df = df.copy()
                result_df["Sz_mean"] = mean_pred
                result_df["Sz_std"] = std_pred
                result_df["Sz_90_low"] = Sz_ci_lower
                result_df["Sz_90_high"] = Sz_ci_upper
                result_df["CO2_mean"] = CO2_mean
                result_df["CO2_std"] = CO2_std
                result_df["CO2_90_low"] = CO2_ci_lower
                result_df["CO2_90_high"] = CO2_ci_upper

                output_filename = f"{model_name}_{input_file.stem}.csv"
                output_path = output_dir / output_filename

                result_df.to_csv(output_path, index=False)
                logging.info(f"Saved prediction to {output_path}")

            except Exception as e:
                logging.error(
                    f"Prediction failed for {model_name} on {input_file.name}: {e}"
                )


if __name__ == "__main__":
    run_predictions()
