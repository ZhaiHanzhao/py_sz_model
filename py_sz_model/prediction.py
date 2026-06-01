"""Predict Shilou/Jiaxian Sz and CO2 from trained Sz models."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    MODEL_NAMES,
    MODELS_DIR,
    PREDICTION_FILE_NAMES,
    PREDICTION_SET_DIR,
    PREDICTIONS_DIR,
    SZ_FEATURE_NAMES,
    build_models,
)
from .models import ModelData
from .r_calculation import calculate_R_for_dataframe


def default_prediction_files(data_dir: Path) -> list[Path]:
    """Return only the Shilou and Jiaxian prediction inputs."""
    return [data_dir / filename for filename in PREDICTION_FILE_NAMES]


def _prepare_prediction_dataframe(
    input_file: Path,
    recalculate_r: bool,
    r_simulations: int,
    r_ci_level: int,
    decomp_corr_mean: float,
    decomp_corr_std: float,
    write_r: bool,
) -> pd.DataFrame:
    df = pd.read_csv(input_file)
    needs_r = recalculate_r or not {"R", "R_std"}.issubset(df.columns)
    if needs_r:
        logging.info("Calculating R for %s...", input_file.name)
        df = calculate_R_for_dataframe(
            df,
            num_simulations=r_simulations,
            ci_level=r_ci_level,
            decomp_corr_mean=decomp_corr_mean,
            decomp_corr_std=decomp_corr_std,
        )
        if write_r:
            df.to_csv(input_file, index=False)
            logging.info("Updated R columns in %s", input_file)
    return df


def predict_file(
    input_file: Path,
    models_dir: Path = MODELS_DIR,
    output_dir: Path = PREDICTIONS_DIR,
    model_names: list[str] | tuple[str, ...] = MODEL_NAMES,
    n_mc_samples: int = 1000,
    recalculate_r: bool = False,
    r_simulations: int = 100000,
    r_ci_level: int = 90,
    decomp_corr_mean: float = 0.0,
    decomp_corr_std: float = 0.0,
    write_r: bool = False,
) -> list[Path]:
    """Predict Sz and CO2 for one Shilou/Jiaxian feature file."""
    df = _prepare_prediction_dataframe(
        input_file=input_file,
        recalculate_r=recalculate_r,
        r_simulations=r_simulations,
        r_ci_level=r_ci_level,
        decomp_corr_mean=decomp_corr_mean,
        decomp_corr_std=decomp_corr_std,
        write_r=write_r,
    )

    data = ModelData(
        data=df,
        features_names=SZ_FEATURE_NAMES,
        dataset_name=input_file.stem,
    )
    data.pre_process(test_size=0)
    if data.features is None or data.features_uncertainty is None:
        raise ValueError(f"Failed to prepare features for {input_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    models = build_models(model_names=model_names)
    for model_name, wrapper in models.items():
        model_path = models_dir / f"{model_name}_sz.joblib"
        if not model_path.exists():
            logging.warning("Model file not found: %s. Skipping.", model_path)
            continue

        wrapper.load_model(model_path)
        mean_pred, std_pred = wrapper.predict_with_uncertainty(
            x=data.features,
            x_uncertainty=data.features_uncertainty,
            n_mc_samples=n_mc_samples,
        )

        r_mean = df["R"].astype(float)
        r_std = df["R_std"].astype(float)
        co2_mean = r_mean * mean_pred
        co2_std = np.sqrt((r_mean * std_pred) ** 2 + (mean_pred * r_std) ** 2)

        result_df = df.copy()
        result_df["Sz_mean"] = mean_pred
        result_df["Sz_std"] = std_pred
        result_df["Sz_90_low"] = mean_pred - 1.645 * std_pred
        result_df["Sz_90_high"] = mean_pred + 1.645 * std_pred
        result_df["CO2_mean"] = co2_mean
        result_df["CO2_std"] = co2_std
        result_df["CO2_90_low"] = co2_mean - 1.645 * co2_std
        result_df["CO2_90_high"] = co2_mean + 1.645 * co2_std

        output_path = output_dir / f"{model_name}_{input_file.stem}.csv"
        result_df.to_csv(output_path, index=False)
        outputs.append(output_path)
        logging.info("Saved prediction to %s", output_path)

    return outputs


def run_predictions(
    input_files: list[Path] | None = None,
    data_dir: Path = PREDICTION_SET_DIR,
    models_dir: Path = MODELS_DIR,
    output_dir: Path = PREDICTIONS_DIR,
    model_names: list[str] | tuple[str, ...] = MODEL_NAMES,
    n_mc_samples: int = 1000,
    recalculate_r: bool = False,
    r_simulations: int = 100000,
    r_ci_level: int = 90,
    decomp_corr_mean: float = 0.0,
    decomp_corr_std: float = 0.0,
    write_r: bool = False,
) -> list[Path]:
    """Run predictions for the default Shilou and Jiaxian files."""
    files = input_files if input_files is not None else default_prediction_files(data_dir)
    outputs = []
    for input_file in files:
        if not input_file.exists():
            logging.warning("Input file not found: %s. Skipping.", input_file)
            continue
        outputs.extend(
            predict_file(
                input_file=input_file,
                models_dir=models_dir,
                output_dir=output_dir,
                model_names=model_names,
                n_mc_samples=n_mc_samples,
                recalculate_r=recalculate_r,
                r_simulations=r_simulations,
                r_ci_level=r_ci_level,
                decomp_corr_mean=decomp_corr_mean,
                decomp_corr_std=decomp_corr_std,
                write_r=write_r,
            )
        )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Shilou/Jiaxian Sz and CO2.")
    parser.add_argument("--input", nargs="*", type=Path, help="Input CSV files.")
    parser.add_argument("--data-dir", type=Path, default=PREDICTION_SET_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    parser.add_argument("--mc-samples", type=int, default=1000)
    parser.add_argument("--recalculate-r", action="store_true")
    parser.add_argument("--r-simulations", type=int, default=100000)
    parser.add_argument("--r-ci-level", type=int, default=90)
    parser.add_argument("--decomp-corr-mean", type=float, default=0.0)
    parser.add_argument("--decomp-corr-std", type=float, default=0.0)
    parser.add_argument("--write-r", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    outputs = run_predictions(
        input_files=args.input,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        model_names=args.models,
        n_mc_samples=args.mc_samples,
        recalculate_r=args.recalculate_r,
        r_simulations=args.r_simulations,
        r_ci_level=args.r_ci_level,
        decomp_corr_mean=args.decomp_corr_mean,
        decomp_corr_std=args.decomp_corr_std,
        write_r=args.write_r,
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
