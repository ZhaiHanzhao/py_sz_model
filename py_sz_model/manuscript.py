"""Check the frozen release and rebuild the manuscript's CO2 statistics."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, FROZEN_DIR, MODEL_NAMES, REPO_ROOT, SZ_FEATURE_NAMES


def load_reconstructions() -> pd.DataFrame:
    """Return the 92 archived manuscript estimates with their section labels."""
    return pd.concat([
        pd.read_csv(FROZEN_DIR / 'reconstructions' / f'{site}_CO2.csv').assign(site=site)
        for site in ('Jiaxian', 'Shilou')
    ], ignore_index=True)


def check_release() -> dict:
    """Check file integrity, membership, fitted models and propagated errors."""
    manifest = json.loads((FROZEN_DIR / 'manifest.json').read_text(encoding='utf-8'))
    for record in manifest['files']:
        path = REPO_ROOT / record['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f"Frozen release file has changed: {record['path']}")
    q = pd.read_csv(DATA_DIR / '800kyr_amorphous_with_particlesize.CSV')
    if len(q) != 99 or not q.Sample_ID.is_unique:
        raise ValueError('Expected 99 unique calibration samples.')
    training, testing = train_test_split(q, test_size=.25, random_state=42)
    split = pd.read_csv(DATA_DIR / 'calibration_split.csv')
    for label, data in [('train', training), ('test', testing)]:
        if split.loc[split.split == label, 'Sample_ID'].tolist() != data.Sample_ID.tolist():
            raise ValueError(f'Incorrect {label} membership or order.')
    for name in MODEL_NAMES:
        saved = pd.read_csv(FROZEN_DIR / 'test_predictions' / f'{name}_sz.csv')
        if saved.Sample_ID.tolist() != testing.Sample_ID.tolist():
            raise ValueError(f'{name}: archived predictions do not use the 25 test samples.')
        fitted = joblib.load(FROZEN_DIR / 'models' / f'{name}_sz.joblib')
        pipeline = fitted.best_estimator_
        if list(fitted.feature_names_in_) != SZ_FEATURE_NAMES:
            raise ValueError(f'{name}: incorrect feature order.')
        if pipeline.named_steps['preprocessor'].named_transformers_['num'].n_samples_seen_ != 74000:
            raise ValueError(f'{name}: expected a fit to 74,000 augmented rows.')
        if not np.isfinite(fitted.predict(testing[SZ_FEATURE_NAMES])).all():
            raise ValueError(f'{name}: nonfinite predictions.')
    data = load_reconstructions()
    membership = pd.read_csv(DATA_DIR / 'manuscript_samples.csv')
    if len(data) != 92 or not data['Age (Ma)'].between(4, 6.5).all():
        raise ValueError('Expected 92 reconstructions within 4.0–6.5 Ma.')
    for site, expected_n in [('Jiaxian', 60), ('Shilou', 32)]:
        inputs = pd.read_csv(DATA_DIR / 'prediction_set' / f'{site}_features_bulk.CSV')
        results = data.loc[data.site == site]
        ids = inputs['Sample ID']
        if len(inputs) != expected_n or not ids.is_unique or not inputs['Age(Ma)'].between(4, 6.5).all():
            raise ValueError(f'{site}: incorrect release input sample set.')
        if set(ids) != set(results['Sample ID']) or set(ids) != set(membership.loc[membership.site == site, 'sample_id']):
            raise ValueError(f'{site}: input/result membership mismatch.')
        if set(ids) & set(manifest['excluded_sample_ids']):
            raise ValueError('Excluded samples are present in the release.')
    np.testing.assert_allclose(data.CO2_mean, data.Sz_mean * data.R, rtol=1e-12, atol=1e-10)
    propagated = np.hypot(data.R * data.Sz_std, data.Sz_mean * data.R_std)
    np.testing.assert_allclose(data.CO2_std, propagated, rtol=1e-12, atol=1e-10)
    return {'status': 'PASS', 'checked_files': len(manifest['files']),
            'calibration': 99, 'train': 74, 'test': 25,
            'Jiaxian': 60, 'Shilou': 32, 'manuscript_samples': 92}


def interval_summary(data: pd.DataFrame, label: str, lower: float, upper: float, inclusive_upper: bool = False) -> dict:
    ages = data['Age (Ma)']
    selected = data.loc[(ages >= lower) & ((ages <= upper) if inclusive_upper else (ages < upper))]
    if len(selected) < 2:
        raise ValueError(f'{label}: fewer than two observations.')
    from_s = selected.R * selected.Sz_std
    from_r = selected.Sz_mean * selected.R_std
    n = len(selected)
    return {'interval': label, 'lower_Ma': lower, 'upper_Ma': upper,
            'upper_inclusive': inclusive_upper, 'n': n,
            'Jiaxian_n': int((selected.site == 'Jiaxian').sum()),
            'Shilou_n': int((selected.site == 'Shilou').sum()),
            'mean_CO2_ppm': float(selected.CO2_mean.mean()),
            'sample_SD_ppm': float(selected.CO2_mean.std(ddof=1)),
            'RMS_uncertainty_from_S_ppm': float(np.sqrt(np.mean(from_s**2))),
            'RMS_uncertainty_from_R_ppm': float(np.sqrt(np.mean(from_r**2))),
            'RMS_single_sample_uncertainty_ppm': float(np.sqrt(np.mean(selected.CO2_std**2))),
            'mean_uncertainty_independent_errors_ppm': float(np.sqrt(np.sum(selected.CO2_std**2)) / n)}


def cross_section_ratios(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Interpolate Jiaxian to Shilou ages, then compare the two ratios."""
    j = data.loc[data.site == 'Jiaxian'].sort_values('Age (Ma)')
    s = data.loc[data.site == 'Shilou'].sort_values('Age (Ma)')
    age = s['Age (Ma)'].to_numpy()
    if age.min() < j['Age (Ma)'].min() or age.max() > j['Age (Ma)'].max():
        raise ValueError('Shilou ages require extrapolation beyond Jiaxian.')
    interpolate = lambda y: np.interp(age, j['Age (Ma)'], y)
    j_inv = interpolate(1 / j.R)
    j_inv_sd = interpolate(j.R_std / j.R**2)
    j_s = interpolate(j.Sz_mean)
    j_s_sd = interpolate(j.Sz_std)
    s_inv = 1 / s.R.to_numpy()
    s_inv_sd = s.R_std.to_numpy() / s.R.to_numpy()**2
    x = j_inv / s_inv
    y = j_s / s.Sz_mean.to_numpy()
    x_sd = x * np.hypot(j_inv_sd / j_inv, s_inv_sd / s_inv)
    y_sd = y * np.hypot(j_s_sd / j_s, s.Sz_std.to_numpy() / s.Sz_mean.to_numpy())
    combined = np.hypot(x_sd, y_sd)
    pairs = pd.DataFrame({'Sample ID': s['Sample ID'].to_numpy(), 'Age (Ma)': age,
        'ratio_1R': x, 'ratio_Sz': y, 'ratio_1R_std': x_sd, 'ratio_Sz_std': y_sd,
        'combined_1sigma': combined, 'within_combined_1sigma': np.abs(y-x) <= combined})
    fit = linregress(x, y)
    count = int(pairs.within_combined_1sigma.sum())
    return pairs, {'n': len(pairs), 'within_combined_1sigma': count,
        'within_percent': 100 * count / len(pairs), 'r_squared': float(fit.rvalue**2),
        'p_value': float(fit.pvalue), 'slope': float(fit.slope), 'intercept': float(fit.intercept),
        'fit': 'OLS; displayed uncertainties are not fitted',
        'agreement': 'abs(y-x) <= hypot(sigma_x,sigma_y), uncorrelated errors'}


def moving_average(data: pd.DataFrame) -> pd.DataFrame:
    """The manuscript's centered 10-point mean and independent-error bands."""
    ordered = data.sort_values('Age (Ma)').reset_index(drop=True)
    values = ordered.CO2_mean
    rolling = values.rolling(10, min_periods=1, center=True)
    n = rolling.count()
    propagated = (ordered.CO2_std**2).rolling(10, min_periods=1, center=True).sum() / n**2
    sampling = rolling.var(ddof=1).fillna(0) / n
    sem = np.sqrt(propagated + sampling)
    average = rolling.mean()
    return pd.DataFrame({'age_Ma': ordered['Age (Ma)'], 'n': n.astype(int),
        'mean_CO2_ppm': average, 'combined_SE_ppm': sem,
        'lower_1SE_ppm': average-sem, 'upper_1SE_ppm': average+sem,
        'lower_1_96SE_ppm': average-1.96*sem, 'upper_1_96SE_ppm': average+1.96*sem})


def test_set_metrics() -> pd.DataFrame:
    """Recompute central-estimate metrics and CO2 agreement from test25."""
    rows = []
    for model in MODEL_NAMES:
        df = pd.read_csv(FROZEN_DIR / 'test_predictions' / f'{model}_sz.csv')
        co2 = df.prediction * df.R
        co2_sd = np.hypot(df.prediction_uncertainty * df.R, df.prediction * df.R_std)
        for target, truth, prediction, uncertainty in [
            ('Sz', df.Sz, df.prediction, df.prediction_uncertainty),
            ('CO2', df.CO2_ice, co2, co2_sd),
        ]:
            residual = prediction-truth
            rows.append({'model': model, 'target': target, 'n': len(df),
                'RMSE_ppm': float(np.sqrt(np.mean(residual**2))),
                'predictive_R2': float(1 - np.sum(residual**2) / np.sum((truth-truth.mean())**2)),
                'mean_residual_ppm': float(residual.mean()), 'residual_sample_SD_ppm': float(residual.std(ddof=1)),
                'within_prediction_1sigma': int((np.abs(residual) <= uncertainty).sum())})
    return pd.DataFrame(rows)


def build(output_dir: Path, run_proxy_checks: bool = True) -> dict:
    """Rebuild statistics without fitting a new S(z) reconstruction model."""
    report = check_release()
    if output_dir.resolve().is_relative_to(FROZEN_DIR.resolve()) or output_dir.resolve().is_relative_to(DATA_DIR.resolve()):
        raise ValueError('Write generated statistics outside data/ and frozen/.')
    data = load_reconstructions()
    stages = pd.DataFrame([interval_summary(data, *args) for args in [
        ('overall', 4.0, 6.5, True), ('pre_drawdown', 6.0, 6.5, True),
        ('low_CO2', 5.3, 6.0, False), ('early_Pliocene', 4.0, 5.3, False)]])
    local = pd.DataFrame([interval_summary(data, 'pre_100kyr', 6.1, 6.2),
                          interval_summary(data, 'post_100kyr', 5.8, 5.9)])
    pairs, ratio = cross_section_ratios(data)
    reference_pairs = pd.read_csv(FROZEN_DIR / 'statistics/figure2c_pairs.csv')
    pd.testing.assert_frame_equal(pairs, reference_pairs[pairs.columns], check_dtype=False, rtol=1e-12, atol=1e-12)
    reference_ratio = json.loads((FROZEN_DIR / 'statistics/figure2c_validation.json').read_text(encoding='utf-8'))
    for field in ['n','within_combined_1sigma','r_squared','p_value']:
        np.testing.assert_allclose(ratio[field], reference_ratio[field], rtol=1e-12)
    output_dir.mkdir(parents=True, exist_ok=True)
    stages.to_csv(output_dir / 'stage_summary.csv', index=False)
    local.to_csv(output_dir / 'drawdown_windows.csv', index=False)
    pairs.to_csv(output_dir / 'figure2c_pairs.csv', index=False)
    moving_average(data).to_csv(output_dir / 'moving_average_10point.csv', index=False)
    test_set_metrics().to_csv(output_dir / 'test25_metrics.csv', index=False)
    report.update({'stage_n': stages.n.tolist(), 'stage_means_ppm': stages.mean_CO2_ppm.tolist(),
        'local_n': local.n.tolist(), 'local_means_ppm': local.mean_CO2_ppm.tolist(),
        'drawdown_ppm': float(local.mean_CO2_ppm.iloc[0]-local.mean_CO2_ppm.iloc[1]),
        'figure2c': ratio, 'new_Sz_fits': 0})
    if run_proxy_checks:
        from .proxy_checks import run_checks
        report['proxy_checks'] = run_checks(output_dir / 'proxy_checks')
    (output_dir / 'verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'build'], nargs='?', default='check')
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'outputs/manuscript')
    parser.add_argument('--skip-proxy-checks', action='store_true', help='Skip refitting the 14 measured-proxy diagnostic models.')
    args = parser.parse_args()
    report = check_release() if args.command == 'check' else build(args.output_dir, not args.skip_proxy_checks)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
