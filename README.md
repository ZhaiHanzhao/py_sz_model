# py_sz_model

Code and data accompanying **Atmospheric CO2 drawdown shaped the 6 Ma Earth-system transition**. Version 0.3.0 adopts the seed-42 models and results calculated on 8 October 2026. Model fitting, uncertainty propagation and manuscript statistics are reproducible from the included fixed calibration inputs.

The release contains the 99 Quaternary calibration samples, the original 74/25 training/test split, and the 92 Red Clay samples used for the 6.5–4.0 Ma reconstruction. The fitted Gradient Boosting, Random Forest and Ridge models and their archived test predictions are included. Gradient Boosting supplies the final CO2 reconstruction.

## Install from the repository

Use Python 3.13 for the recorded reproduction environment. Run the following from a checkout of this repository:

```shell
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` in a POSIX shell), then install:

```shell
python -m pip install -r requirements-reproduction.txt
python -m pip install -e .
```

The editable installation uses the adjacent `data/` and `frozen/` directories in this checkout. Keep that directory structure. The recorded runtime is Python 3.13.7, NumPy 2.2.6, pandas 2.3.2, SciPy 1.16.3, scikit-learn 1.7.2 and joblib 1.5.2. Optional `python -m pip install -e ".[fast]"` adds numba for the R calculation; it does not change the formulas or draw the random inputs.

## Reproduce the manuscript statistics

```shell
python -m py_sz_model.manuscript check
python -m py_sz_model.manuscript build
```

`check` verifies the release hashes, the sample sets, the 74/25 split, the fitted models, and CO2 = S(z) × R with its stored first-order uncertainty. `build` recalculates the following in `outputs/manuscript/`:

- Equal-weight stage means, sample SDs, and the independent-error uncertainty diagnostic used in Supplementary Table S1.
- The adjacent 100-kyr means: 333.4082 ppm at 6.1 ≤ age < 6.2 Ma (n = 6) and 213.6141 ppm at 5.8 ≤ age < 5.9 Ma (n = 4). Their difference is 119.7942 ppm, reported as approximately 120 ppm.
- The 32 Fig. 2C cross-section pairs, the OLS fit (R² = 0.299299), and 20/32 agreement within the combined 1σ uncertainty.
- The centered 10-point moving average and its uncertainty bands.
- Central-estimate performance and residual statistics from the archived 25-sample test predictions.
- Fourteen fixed Gradient Boosting diagnostic models that predict each of seven measured proxies from the other six, across the Quaternary and Red Clay datasets. Predictions and metrics are checked against the archived raw-scale results.

This command does not refit the S(z) models. Use `--skip-proxy-checks` for statistics without rerunning the measured-proxy diagnostics, or `--output-dir PATH` to choose a generated-output directory.

Expected stage counts are 30, 27 and 35; means are 309.5149, 235.3659 and 325.9390 ppm. Stage intervals are 6.0 ≤ age ≤ 6.5, 5.3 ≤ age < 6.0 and 4.0 ≤ age < 5.3 Ma. The Gradient Boosting test CO2 reconstruction agrees within its displayed 1σ uncertainty for 19/25 samples (76%); mean residual is 10.4139 ppm and residual sample SD is 40.6934 ppm.

## Data and recorded model outputs

| Location | Contents |
| --- | --- |
| `data/800kyr_amorphous_with_particlesize.CSV` | 99 Quaternary calibration samples, unchanged from the manuscript input table |
| `data/calibration_split.csv` | The fixed 74 training and 25 test sample IDs, including their split order |
| `data/prediction_set/Jiaxian_features_bulk.CSV` | 60 Jiaxian samples within 4.0–6.5 Ma |
| `data/prediction_set/Shilou_features_bulk.CSV` | 32 Shilou samples within 4.0–6.5 Ma |
| `data/manuscript_samples.csv` | The exact 92-sample reconstruction membership |
| `data/proxy_measurements.csv` | The same 99 + 92 samples, containing the seven measured proxies used in the cross-period checks |
| `frozen/models/` | The three adopted seed-42 fitted model files, including their complete grid searches |
| `frozen/test_predictions/` | Seed-42 predictions for the same 25 held-out calibration samples |
| `frozen/reconstructions/` | The 60 + 32 final Gradient Boosting CO2 reconstructions used in the paper |
| `frozen/metrics/` | Archived Monte Carlo performance summaries; distinct from metrics of central estimates |
| `frozen/proxy_checks/` | Archived raw-scale measured-proxy diagnostics |
| `frozen/statistics/` | Archived Fig. 2C pairs and fit statistics |
| `frozen/manifest.json` | File hashes, membership, hyperparameters, runtime and provenance notes |

No samples outside the paper's reconstruction interval are included in the Red Clay input tables. `2-SL01_2420` and `2-SL01_2450` were removed following the author's identification of sample problems. Both were already absent from the final reconstruction. Their removal changes no paper result. Retained values and row order are preserved.

The full datasets remain separately archived on Figshare: [Quaternary calibration](https://doi.org/10.6084/m9.figshare.31829857.v1) and [Red Clay reconstruction](https://doi.org/10.6084/m9.figshare.31829581.v3). These historical Figshare archives contain broader data coverage than this manuscript-specific code release.

## Reproduce model fits and all propagated results

The adopted calculations use seed `42`: the split and estimators use scikit-learn's `random_state=42`, and Monte Carlo draws use NumPy `default_rng(42)` (PCG64). The calibration table, including its previously derived elemental ratios and S(z) labels, is treated as fixed input. This release supersedes the older result snapshots in v0.2.x, whose model Monte Carlo seeds were not recorded.

```shell
python -m py_sz_model.reproduce --refit
```

This regenerates the 74,000 augmented training rows, refits each selected estimator, replays the 25-sample predictions and Monte Carlo performance summaries in the recorded model order, recalculates R, and regenerates all 92 Gradient Boosting reconstructions. It compares the results with `frozen/` to floating-point tolerance and saves a verification report under `outputs/reproduction/`. Omit `--refit` to replay the propagation using the frozen estimators. This verification refits the selected hyperparameters; the training command below reruns the full grid search.

Training and held-out evaluation share one seed-42 stream, with models ordered Gradient Boosting, Random Forest, Ridge. Red Clay R propagation and S(z) prediction each use a separate seed-42 stream, processing Shilou before Jiaxian. Recalculating R therefore does not change the S(z) draws. Use both default input files in that order to reproduce the paper exactly; processing one section alone starts a new stream and may yield slightly different Monte Carlo summaries.

To reproduce the paper's 1,000 prediction draws per sample:

```shell
python -m py_sz_model.prediction --seed 42
python -m py_sz_model.prediction --seed 42 --recalculate-r
```

Outputs go to `predictions/`. The default model is the frozen Gradient Boosting model. Specify `--models RandomForest Ridge` to evaluate the comparison models. The saved R and R_std columns are used unless `--recalculate-r` is passed.

To rerun the original model-selection procedure with a recorded seed:

```shell
python -m py_sz_model.training --seed 42
```

This uses the original split seed 42, 13 predictors, 1,000 independent normal draws for each of 74 training samples, and ordinary 5-fold cross-validation on the sample-contiguous augmented rows. It retains the original three hyperparameter grids. It does not substitute sample-grouped CV or refit on all 99 samples. Model files and `training_run.json` go to `models/`, performance summaries to `metrics/`, and new held-out predictions to `predictions/`.

To predict using a newly trained model, explicitly pass `--models-dir models`. Without this option, prediction continues to use the frozen manuscript model.

## R and elemental-ratio calculations

```shell
python -m py_sz_model.r_calculation data/prediction_set/Shilou_features_bulk.CSV --output outputs/Shilou_with_r.csv --seed 42
```

R uses 100,000 normal draws by default. The decomposition correction has mean 0‰ and SD 0‰, matching the adopted reconstruction. The same defaults apply to the function, standalone CLI and prediction CLI. The fractionation term is ε = A − BT, with A = 11.98 ± 0.13 and B = 0.12 ± 0.01. The finite R distribution is retained without clipping negative values.

Quaternary elemental-ratio means and SDs come from 1,000 normal numerator/denominator draws. Red Clay inputs use direct ratios and first-order propagation of independent numerator/denominator uncertainties:

```shell
python -m py_sz_model.ratios data/800kyr_amorphous_with_particlesize.CSV --method monte-carlo --output outputs/calibration_ratios.csv
python -m py_sz_model.ratios data/prediction_set/Shilou_features_bulk.CSV --method first-order --output outputs/Shilou_ratios.csv
```

These commands produce separate files. The archived input ratios remain the values used by the manuscript's fitted models.

## Interpretation of uncertainties and diagnostics

S(z) uncertainties perturb the specified input features through a fixed fitted model. Missing reported input SDs are held at zero, following the original workflow. CO2 uncertainty is sqrt((R × SD_S)² + (S × SD_R)²), assuming zero covariance. Stored `*_90_low/high` reconstruction columns use the normal approximation mean ± 1.645 SD; R interval columns are Monte Carlo percentiles.

Stage SD describes dispersion among sample estimates. The independent-error uncertainty of a stage mean is a separate diagnostic; it omits shared errors and is not a total confidence interval. Moving-average bands combine sum(single-sample variances)/n² with the within-window sample variance/n and use 1 and 1.96 combined standard errors.

Cross-period checks assess persistence of relationships among measured proxies. They use seven basic measurements and exclude all derived ratios, S(z), R, CO2, temperature, age and profile from the predictors. They support parts of the transfer argument but do not directly validate absolute S(z). These checks were exploratory and are not a fresh confirmatory test set.

## Verify changes

```shell
python -m unittest discover -s tests -v
python -m py_sz_model.manuscript build
```

Use [CITATION.cff](CITATION.cff) to cite this software version. This repository covers the S(z)/CO2 model and its supporting diagnostics; external SST processing and the complete manuscript figure-layout project are separate.
