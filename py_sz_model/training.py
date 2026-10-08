"""Train Sz models without plotting side effects."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .config import FROZEN_DIR, METRICS_DIR, MODEL_NAMES, MODELS_DIR, PREDICTIONS_DIR, TRAINING_DATA_PATH, build_models
from .models import ModelData
from .config import load_sz_training_data
from .randomness import MC_RANDOM_SEED, generator


def _training_inputs(data: ModelData) -> tuple[pd.DataFrame, pd.Series]:
    if data.features_train is None or data.target_train is None:
        raise ValueError("Features or target is None. Please preprocess data first.")
    if data.features_train_mc is not None and data.target_train_mc is not None:
        logging.info("Using Monte Carlo expanded training data (n=%d).", len(data.features_train_mc))
        return data.features_train_mc, data.target_train_mc
    return data.features_train, data.target_train


def train_sz_models(
    data_path: Path = TRAINING_DATA_PATH,
    models_dir: Path = MODELS_DIR,
    metrics_dir: Path = METRICS_DIR,
    model_names: list[str] | tuple[str, ...] = MODEL_NAMES,
    test_size: float = 0.25,
    n_mc_samples: int = 1000,
    cv_folds: int = 5,
    n_jobs: int = -1,
    split_strategy: str = "random",
    test_prefix: str = "93L",
    predictions_dir: Path = PREDICTIONS_DIR,
    seed: int = MC_RANDOM_SEED,
) -> list[Path]:
    """Train selected Sz models and return saved model paths."""
    if n_mc_samples <= 0:
        raise ValueError("n_mc_samples must be positive.")
    for directory in (models_dir, metrics_dir, predictions_dir):
        if directory.resolve().is_relative_to(FROZEN_DIR.resolve()):
            raise ValueError("New training outputs cannot overwrite the frozen manuscript files.")
    rng = generator(seed)
    data = load_sz_training_data(data_path)
    data.pre_process(
        test_size=test_size,
        n_mc_samples=n_mc_samples,
        split_strategy=split_strategy,
        test_prefix=test_prefix,
        mc_random_state=rng,
    )
    x_train, y_train = _training_inputs(data)

    saved_paths = []
    models = build_models(model_names=model_names, cv_folds=cv_folds, n_jobs=n_jobs)
    for model_name, model in models.items():
        model.train(features_names=data.features_names, x=x_train, y=y_train)

        model_path = models_dir / f"{model_name}_{data.dataset_name}.joblib"
        model.save_model(model_path)
        saved_paths.append(model_path)

        if (
            data.features_test is None
            or data.features_test_uncertainty is None
            or data.target_test is None
            or data.target_test_uncertainty is None
        ):
            continue

        mean_pred, std_pred = model.predict_with_uncertainty(
            data.features_test,
            data.features_test_uncertainty,
            n_mc_samples,
            random_state=rng,
        )
        metrics = model.evaluate(
            target_test=data.target_test,
            target_test_uncertainty=data.target_test_uncertainty,
            target_pred=mean_pred,
            target_pred_uncertainty=std_pred,
            n_mc_samples=n_mc_samples,
            random_state=rng,
        )
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / f"{model_name}_{data.dataset_name}.csv"
        pd.DataFrame(metrics, index=[model_name]).to_csv(
            metrics_path,
            index=True,
            index_label="model",
        )
        logging.info("Saved metrics to %s", metrics_path)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        held_out = data.test_set.copy()
        held_out["prediction"] = mean_pred
        held_out["prediction_uncertainty"] = std_pred
        held_out.to_csv(predictions_dir / f"{model_name}_{data.dataset_name}.csv", index=False)

    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "training_run.json").write_text(json.dumps({
        "seed": seed, "mc_samples": n_mc_samples, "cv_folds": cv_folds,
        "cv_scheme": "ordinary KFold on sample-contiguous Monte Carlo rows",
        "train_ids": data.train_set["Sample_ID"].tolist(),
        "test_ids": [] if data.test_set is None else data.test_set["Sample_ID"].tolist(),
        "best_parameters": {name: model.trained_model.best_params_ for name, model in models.items()},
        "note": "Seed-42 manuscript protocol. Calibration labels and elemental ratios are fixed archived inputs.",
    }, indent=2) + "\n", encoding="utf-8")

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sz models.")
    parser.add_argument("--data", type=Path, default=TRAINING_DATA_PATH)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--mc-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=MC_RANDOM_SEED)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--split-strategy", choices=["random", "prefix"], default="random")
    parser.add_argument("--test-prefix", default="93L")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    saved_paths = train_sz_models(
        data_path=args.data,
        models_dir=args.models_dir,
        metrics_dir=args.metrics_dir,
        model_names=args.models,
        test_size=args.test_size,
        n_mc_samples=args.mc_samples,
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
        split_strategy=args.split_strategy,
        test_prefix=args.test_prefix,
        predictions_dir=args.predictions_dir,
        seed=args.seed,
    )
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
