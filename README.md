# py_sz_model

`py_sz_model` provides a compact Python workflow for training Sz regressors and estimating CO2 for the Shilou and Jiaxian records.

The package trains Sz models from the included calibration dataset, applies the trained models to Shilou and Jiaxian feature tables, and propagates uncertainty into both Sz and CO2 estimates.

## Data

The repository includes the data needed to run the workflow:

- `data/800kyr_amorphous_with_particlesize.CSV`: calibration data for Sz model training
- `data/prediction_set/Shilou_features_bulk.CSV`: Shilou prediction features
- `data/prediction_set/Jiaxian_features_bulk.CSV`: Jiaxian prediction features

## Installation

```shell
python -m venv .venv
```

Activate the virtual environment using the convention for your shell, then install the package:

```shell
python -m pip install -e .
```

To accelerate Monte Carlo R calculations, install the optional `fast` extra, which adds `numba`:

```shell
python -m pip install -e ".[fast]"
```

## Train Models

```shell
python -m py_sz_model.training
```

This trains three Sz regressors:

- `GradientBoosting`
- `RandomForest`
- `Ridge`

Trained models are saved to `models/`, and test-set metrics are saved to `metrics/`.

## Predict Shilou and Jiaxian CO2

After training, run:

```shell
python -m py_sz_model.prediction
```

Prediction results are written to `predictions/`. Each output file contains the original input columns plus:

- `Sz_mean`, `Sz_std`, `Sz_90_low`, `Sz_90_high`
- `CO2_mean`, `CO2_std`, `CO2_90_low`, `CO2_90_high`

If an input table does not contain `R` and `R_std`, the prediction workflow calculates them in memory. To force recalculation and write the R columns back to the input CSV files:

```shell
python -m py_sz_model.prediction --recalculate-r --write-r
```

## Command-Line Entry Points

After installation, the same workflow can be run with:

```shell
sz-train
sz-predict
sz-calc-r data/prediction_set/Shilou_features_bulk.CSV
```
