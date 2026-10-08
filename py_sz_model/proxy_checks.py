"""Cross-period prediction of each measured proxy from the other six."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from .config import DATA_DIR, FROZEN_DIR, REPO_ROOT

CORE = ['Xlf', 'aFe', 'aSi', 'aAl', 'fFe', 'd18Oc', 'D']
PARAMETERS = {'n_estimators': 50, 'learning_rate': .05, 'max_depth': 3, 'random_state': 42}


def run_checks(output_dir: Path) -> dict:
    """Reproduce the adopted raw-scale checks; no S(z), R or CO2 fit."""
    if output_dir.resolve().is_relative_to(FROZEN_DIR.resolve()) or output_dir.resolve().is_relative_to(DATA_DIR.resolve()):
        raise ValueError('Proxy diagnostics must write outside data/ and frozen/.')
    data = pd.read_csv(DATA_DIR / 'proxy_measurements.csv')
    frames = []
    metrics = []
    for direction, source, destination in [('Q_to_RC', 'Q', 'RC'), ('RC_to_Q', 'RC', 'Q')]:
        train = data.loc[data.era == source]
        test = data.loc[data.era == destination]
        if set(train.sample_id) & set(test.sample_id):
            raise ValueError('Training and testing sample IDs overlap.')
        for target in CORE:
            predictors = [name for name in CORE if name != target]
            scaler = StandardScaler().fit(train[predictors].to_numpy(float))
            model = GradientBoostingRegressor(**PARAMETERS)
            model.fit(scaler.transform(train[predictors].to_numpy(float)), train[target].to_numpy(float))
            prediction = model.predict(scaler.transform(test[predictors].to_numpy(float)))
            frame = test[['sample_id','profile','age','era']].copy()
            frame['model_id'] = f'{direction}__raw__{target}'
            frame['direction'] = direction
            frame['scale'] = 'raw'
            frame['target'] = target
            frame['actual_raw'] = test[target].to_numpy(float)
            frame['predicted_raw'] = prediction
            frames.append(frame)
            for scope, group in [('pooled', frame), *list(frame.groupby('profile'))]:
                actual = group.actual_raw.to_numpy()
                predicted = group.predicted_raw.to_numpy()
                residual = predicted-actual
                record = {'direction': direction, 'scale': 'raw', 'target': target,
                    'model': 'GB_Sparams', 'profile': scope, 'n': len(group),
                    'raw_bias': float(residual.mean()), 'raw_RMSE': float(np.sqrt(np.mean(residual**2))),
                    'raw_predictive_R2': float(1-np.sum(residual**2)/np.sum((actual-actual.mean())**2)),
                    'raw_r': float(np.corrcoef(actual, predicted)[0,1]),
                    'nonpositive_predictions': int((predicted <= 0).sum()) if target != 'd18Oc' else 0}
                if target != 'd18Oc' and (predicted > 0).all():
                    a, b = np.log(actual), np.log(predicted)
                    error = b-a
                    record.update(log_predictive_R2=float(1-np.sum(error**2)/np.sum((a-a.mean())**2)),
                        log_r=float(np.corrcoef(a,b)[0,1]), geometric_bias_pct=float(100*np.expm1(error.mean())),
                        RMS_log=float(np.sqrt(np.mean(error**2))))
                metrics.append(record)
    predictions = pd.concat(frames, ignore_index=True)
    scores = pd.DataFrame(metrics)
    for produced, filename, keys in [
        (predictions, 'gb_predictions.csv', ['model_id','sample_id']),
        (scores, 'gb_metrics.csv', ['direction','scale','target','profile']),
    ]:
        archived = pd.read_csv(FROZEN_DIR / 'proxy_checks' / filename)
        left = produced.sort_values(keys).reset_index(drop=True)
        right = archived.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(left[right.columns], right, check_dtype=False, rtol=1e-10, atol=1e-10)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / 'gb_predictions.csv', index=False)
    scores.to_csv(output_dir / 'gb_metrics.csv', index=False)
    return {'new_proxy_fits': 14, 'new_Sz_fits': 0, 'prediction_rows': len(predictions), 'matches_archived_checks': True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'outputs/proxy_checks')
    print(run_checks(parser.parse_args().output_dir))


if __name__ == '__main__':
    main()
