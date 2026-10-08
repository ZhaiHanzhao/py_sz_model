"""Replay all adopted draws and optionally refit the selected estimators."""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from .config import FROZEN_DIR, MODEL_NAMES, REPO_ROOT, load_sz_training_data
from .manuscript import check_release
from .models import TrainingModel
from .prediction import run_predictions


def reproduce(output_dir: Path, refit: bool = False) -> dict:
    """Verify the full seed-42 calculation, preserving data and frozen outputs.

    --refit fits each recorded best estimator to freshly regenerated augmented
    training data. Full grid selection is available separately in training.py.
    """
    check_release()
    output_dir = output_dir.resolve()
    for protected in (REPO_ROOT / 'data', FROZEN_DIR):
        if output_dir.is_relative_to(protected.resolve()):
            raise ValueError('Reproduction outputs cannot overwrite data/ or frozen/.')
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    data = load_sz_training_data()
    data.pre_process(test_size=.25, n_mc_samples=1000, mc_random_state=rng)
    report = {'seed': 42, 'generator': 'PCG64', 'training_rows': len(data.features_train_mc),
              'refitted_selected_estimators': refit, 'test25': {}}
    for name in MODEL_NAMES:
        logging.info('Replaying %s%s', name, ' with refit' if refit else '')
        fitted = joblib.load(FROZEN_DIR / 'models' / f'{name}_sz.joblib')
        original = fitted.best_estimator_
        if refit:
            estimator = clone(original).fit(data.features_train_mc, data.target_train_mc)
            expected = original.predict(data.features)
            actual = estimator.predict(data.features)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)
            fitted.best_estimator_ = estimator
        wrapper = TrainingModel(name)
        wrapper.trained_model = fitted
        mean, sd = wrapper.predict_with_uncertainty(data.features_test, data.features_test_uncertainty, 1000, random_state=rng)
        metrics = wrapper.evaluate(data.target_test, data.target_test_uncertainty, mean, sd, 1000, random_state=rng)
        saved = pd.read_csv(FROZEN_DIR / 'test_predictions' / f'{name}_sz.csv')
        assert saved.Sample_ID.tolist() == data.test_set.Sample_ID.tolist()
        np.testing.assert_allclose(mean, saved.prediction, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(sd, saved.prediction_uncertainty, rtol=1e-12, atol=1e-9)
        stored_metrics = pd.read_csv(FROZEN_DIR / 'metrics' / f'{name}_sz.csv').iloc[0]
        np.testing.assert_allclose(list(metrics.values()), [stored_metrics[k] for k in metrics], rtol=1e-12, atol=1e-9)
        report['test25'][name] = {'n': len(saved), 'max_prediction_difference': float(np.max(np.abs(mean.to_numpy()-saved.prediction.to_numpy())))}
        (output_dir / 'models').mkdir(exist_ok=True)
        joblib.dump(fitted, output_dir / 'models' / f'{name}_sz.joblib', compress=3)
    # Each stream starts at seed 42 and processes Shilou before Jiaxian.
    paths = run_predictions(models_dir=output_dir / 'models', output_dir=output_dir / 'predictions', recalculate_r=True, random_state=42)
    for path, site in zip(paths, ('Shilou', 'Jiaxian')):
        actual = pd.read_csv(path)
        expected = pd.read_csv(FROZEN_DIR / 'reconstructions' / f'{site}_CO2.csv')
        assert actual['Sample ID'].tolist() == expected['Sample ID'].tolist()
        columns = ['R', 'R_std', 'Sz_mean', 'Sz_std', 'CO2_mean', 'CO2_std']
        np.testing.assert_allclose(actual[columns], expected[columns], rtol=1e-12, atol=1e-9)
    report.update({'status': 'PASS', 'reconstruction_samples': 92, 'R_and_prediction_streams_independent': True})
    (output_dir / 'reproduction_verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'outputs/reproduction')
    parser.add_argument('--refit', action='store_true', help='Also refit the three selected estimators on regenerated training draws.')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    print(json.dumps(reproduce(args.output_dir, args.refit), indent=2))


if __name__ == '__main__':
    main()
