"""
This module contains statistical analysis functions.
"""

import pickle
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from joblib import Parallel, delayed
from scipy import stats

CACHE_DIR = Path(__file__).parents[3] / "stats" / "cache"


def pearson_correlation(
    x: np.ndarray,
    y: np.ndarray,
    x_err: np.ndarray,
    y_err: np.ndarray,
    n_simulations: int = 1000,
    confidence_level: float = 0.95,
) -> tuple[float, tuple[float, float], float]:
    """
    Calculates the Pearson correlation coefficient between two variables,
    taking into account their uncertainties.

    :param x: Array of x values.
    :type x: np.ndarray
    :param y: Array of y values.
    :type y: np.ndarray
    :param x_err: Array of x uncertainties.
    :type x_err: np.ndarray
    :param y_err: Array of y uncertainties.
    :type y_err: np.ndarray
    :param n_simulations: Number of Monte Carlo simulations. Defaults to 1000.
    :type n_simulations: int
    :param confidence_level: Confidence level for the confidence interval. Defaults to 0.95.
    :type confidence_level: float
    :return: Tuple containing the Pearson correlation coefficient, confidence interval, and p-value.
    :rtype: tuple[float, tuple[float, float], float]
    """

    x_sim = np.random.normal(loc=x, scale=x_err, size=(n_simulations, len(x)))
    y_sim = np.random.normal(loc=y, scale=y_err, size=(n_simulations, len(y)))

    x_mean = np.mean(x_sim, axis=1, keepdims=True)
    y_mean = np.mean(y_sim, axis=1, keepdims=True)

    x_centered = x_sim - x_mean
    y_centered = y_sim - y_mean

    numerator = np.sum(x_centered * y_centered, axis=1)

    denominator = np.sqrt(np.sum(x_centered**2, axis=1)) * np.sqrt(
        np.sum(y_centered**2, axis=1)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        correlations = numerator / denominator

    correlations = correlations[~np.isnan(correlations)]

    if len(correlations) == 0:
        return np.nan, (np.nan, np.nan), np.nan

    mean_correlation = float(np.mean(correlations))

    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = float(np.percentile(correlations, lower_percentile))
    ci_upper = float(np.percentile(correlations, upper_percentile))

    t_statistic = mean_correlation * np.sqrt((len(x) - 2) / (1 - mean_correlation**2))

    p_value = stats.t.sf(np.abs(t_statistic), len(x) - 2) * 2

    return mean_correlation, (ci_lower, ci_upper), float(p_value)


def monte_carlo_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    x_err: np.ndarray,
    y_err: np.ndarray,
    n_simulations: int = 1000,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Performs linear regression using Monte Carlo sampling to account for uncertainties in x and y.

    :param x: Array of x values.
    :type x: np.ndarray
    :param y: Array of y values.
    :type y: np.ndarray
    :param x_err: Array of x uncertainties.
    :type x_err: np.ndarray
    :param y_err: Array of y uncertainties.
    :type y_err: np.ndarray
    :param n_simulations: Number of Monte Carlo simulations. Defaults to 1000.
    :type n_simulations: int
    :param confidence_level: Confidence level for the confidence interval. Defaults to 0.95.
    :type confidence_level: float
    :return: Dictionary containing regression statistics and plotting data.
    :rtype: Dict[str, Any]
    """

    x_sim = np.random.normal(loc=x, scale=x_err, size=(n_simulations, len(x)))
    y_sim = np.random.normal(loc=y, scale=y_err, size=(n_simulations, len(y)))

    slopes = []
    intercepts = []
    r_values = []

    for i in range(n_simulations):
        slope, intercept, r_value, _, _ = stats.linregress(x_sim[i], y_sim[i])
        slopes.append(slope)
        intercepts.append(intercept)
        r_values.append(r_value)

    slopes = np.array(slopes)
    intercepts = np.array(intercepts)
    r_values = np.array(r_values)

    slope_mean = np.mean(slopes)
    slope_std = np.std(slopes)
    intercept_mean = np.mean(intercepts)
    intercept_std = np.std(intercepts)
    r_mean = np.mean(r_values)
    r_std = np.std(r_values)

    r_squared_values = r_values**2
    r_squared_mean = np.mean(r_squared_values)
    r_squared_std = np.std(r_squared_values)

    x_pred = np.linspace(x.min(), x.max(), 100)
    y_preds = []

    for i in range(n_simulations):
        y_preds.append(slopes[i] * x_pred + intercepts[i])

    y_preds = np.array(y_preds)
    y_pred_mean = np.mean(y_preds, axis=0)

    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    y_pred_lower = np.percentile(y_preds, lower_percentile, axis=0)
    y_pred_upper = np.percentile(y_preds, upper_percentile, axis=0)

    return {
        "slope_mean": slope_mean,
        "slope_std": slope_std,
        "intercept_mean": intercept_mean,
        "intercept_std": intercept_std,
        "r_mean": r_mean,
        "r_std": r_std,
        "r_squared_mean": r_squared_mean,
        "r_squared_std": r_squared_std,
        "x_pred": x_pred,
        "y_pred_mean": y_pred_mean,
        "y_pred_lower": y_pred_lower,
        "y_pred_upper": y_pred_upper,
    }


def _calculate_simex_slope_intercept(
    x_curr: np.ndarray,
    y_curr: np.ndarray,
    x_err_curr: np.ndarray,
    y_err_curr: np.ndarray | None,
    lambdas: list[float],
    n_simulations: int,
) -> tuple[float, float]:
    """
    Helper function to calculate SIMEX slope and intercept for a single dataset.
    """

    weights = None
    if y_err_curr is not None:

        if np.all(y_err_curr == 0):
            weights = None
        else:
            y_err_safe = np.where(y_err_curr == 0, 1e-10, y_err_curr)
            weights = 1 / (y_err_safe**2)

    slope_means = []
    intercept_means = []

    try:
        if weights is not None:
            naive_coeffs = np.polyfit(x_curr, y_curr, 1, w=weights)
            naive_slope, naive_intercept = naive_coeffs[0], naive_coeffs[1]
        else:
            naive_slope, naive_intercept, _, _, _ = stats.linregress(x_curr, y_curr)
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    slope_means.append(naive_slope)
    intercept_means.append(naive_intercept)
    all_lambdas = [0.0] + lambdas

    for lam in lambdas:
        slopes = []
        intercepts = []

        noise = np.random.normal(
            loc=0,
            scale=np.sqrt(lam) * x_err_curr,
            size=(n_simulations, len(x_curr)),
        )
        x_sims = x_curr + noise

        for i in range(n_simulations):
            try:
                if weights is not None:
                    coeffs = np.polyfit(x_sims[i], y_curr, 1, w=weights)
                    slopes.append(coeffs[0])
                    intercepts.append(coeffs[1])
                else:
                    slope, intercept, _, _, _ = stats.linregress(x_sims[i], y_curr)
                    slopes.append(slope)
                    intercepts.append(intercept)
            except np.linalg.LinAlgError:
                slopes.append(np.nan)
                intercepts.append(np.nan)

        slope_means.append(np.mean(slopes))
        intercept_means.append(np.mean(intercepts))

    slope_poly = np.polyfit(all_lambdas, slope_means, 2)
    intercept_poly = np.polyfit(all_lambdas, intercept_means, 2)

    simex_slope = float(np.polyval(slope_poly, -1))
    simex_intercept = float(np.polyval(intercept_poly, -1))
    return simex_slope, simex_intercept


def simex_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    x_err: np.ndarray,
    y_err: np.ndarray,
    n_simulations: int = 1000,
    lambdas: list[float] | None = None,
    n_bootstraps: int = 100,
) -> Dict[str, Any]:
    """
    Performs linear regression using the SIMEX (Simulation Extrapolation)
    method to account for measurement error in x.

    :param x: Array of x values.
    :type x: np.ndarray
    :param y: Array of y values.
    :type y: np.ndarray
    :param x_err: Array of x uncertainties.
    :type x_err: np.ndarray
    :param y_err: Array of y uncertainties.
    :type y_err: np.ndarray
    :param n_simulations: Number of simulations per lambda. Defaults to 1000.
    :type n_simulations: int
    :param lambdas: List of lambda values for simulation. Defaults to [0.5, 1.0, 1.5, 2.0].
    :type lambdas: list[float] | None
    :param n_bootstraps: Number of bootstrap samples for uncertainty estimation. Defaults to 100.
    :type n_bootstraps: int
    :return: Dictionary containing regression statistics and plotting data.
    :rtype: Dict[str, Any]
    """
    if lambdas is None:
        lambdas = [0.5, 1.0, 1.5, 2.0]

    if len(x) < 2:
        return {}

    x_orig = x
    y_orig = y
    x_err_orig = x_err
    y_err_orig = y_err

    simex_slope, simex_intercept = _calculate_simex_slope_intercept(
        x_orig, y_orig, x_err_orig, y_err_orig, lambdas, n_simulations
    )

    boot_slopes = []
    boot_intercepts = []

    if n_bootstraps > 0:
        indices = np.arange(len(x_orig))

        def _bootstrap_iteration(seed):
            np.random.seed(seed)
            resample_idx = np.random.choice(indices, size=len(indices), replace=True)
            x_boot = x_orig[resample_idx]
            y_boot = y_orig[resample_idx]
            x_err_boot = x_err_orig[resample_idx]
            y_err_boot = y_err_orig[resample_idx] if y_err_orig is not None else None
            return _calculate_simex_slope_intercept(
                x_boot, y_boot, x_err_boot, y_err_boot, lambdas, n_simulations
            )

        seeds = np.random.randint(0, 1000000, size=n_bootstraps)

        results = Parallel(n_jobs=-1)(
            delayed(_bootstrap_iteration)(seed) for seed in seeds
        )

        boot_slopes, boot_intercepts = zip(*results)
        boot_slopes = list(boot_slopes)
        boot_intercepts = list(boot_intercepts)

    if n_bootstraps > 1:
        slope_std = np.std(boot_slopes)
        intercept_std = np.std(boot_intercepts)
    else:
        slope_std = 0.0
        intercept_std = 0.0

    y_pred_simex = simex_slope * x_orig + simex_intercept
    ss_res = np.sum((y_orig - y_pred_simex) ** 2)
    ss_tot = np.sum((y_orig - np.mean(y_orig)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    x_pred = np.linspace(x_orig.min(), x_orig.max(), 100)
    y_pred_mean = simex_slope * x_pred + simex_intercept

    y_pred_lower = y_pred_mean
    y_pred_upper = y_pred_mean

    if n_bootstraps > 1:

        y_preds_boot = np.array(
            [s * x_pred + i for s, i in zip(boot_slopes, boot_intercepts)]
        )
        y_pred_lower = np.percentile(y_preds_boot, 2.5, axis=0)
        y_pred_upper = np.percentile(y_preds_boot, 97.5, axis=0)

    return {
        "slope_mean": simex_slope,
        "slope_std": slope_std,
        "intercept_mean": simex_intercept,
        "intercept_std": intercept_std,
        "r_mean": np.sqrt(r_squared) if r_squared >= 0 else 0.0,
        "r_std": 0.0,
        "r_squared_mean": r_squared,
        "r_squared_std": 0.0,
        "x_pred": x_pred,
        "y_pred_mean": y_pred_mean,
        "y_pred_lower": y_pred_lower,
        "y_pred_upper": y_pred_upper,
    }


def scientific_notation(num: float, sig_fig: int = 2) -> str:
    """
    Convert a number to a Latex string in scientific notation with a
    specified number of significant figures.

    :param num: The number to convert.
    :type num: float
    :param sig_fig: The number of significant figures to use.
    :type sig_fig: int
    :return: The number in scientific notation with the specified number of significant figures.
    :rtype: str
    """
    if np.isnan(num):
        return "NaN"
    elif num == 0:
        return "0"
    elif num < 0:
        return "-" + scientific_notation(-num, sig_fig)
    else:
        order = int(np.floor(np.log10(num)))
        mantissa = num / 10**order
        notation_text = (
            R"$\bf{" + f"{mantissa:.{sig_fig}f} \\times 10^{ {order}} " + R"}$"
        )
        return notation_text


def calculate_correlation_stats(
    x: np.ndarray,
    y: np.ndarray,
    x_err: np.ndarray,
    y_err: np.ndarray,
    regression_method: str = "simex",
    consider_uncertainty: bool = False,
) -> dict[str, Any]:
    """
    Perform linear regression using the specified method and calculate correlation statistics.

    :param x: Array of x values.
    :type x: np.ndarray
    :param y: Array of y values.
    :type y: np.ndarray
    :param x_err: Array of x uncertainties.
    :type x_err: np.ndarray
    :param y_err: Array of y uncertainties.
    :type y_err: np.ndarray
    :param regression_method: Method for linear regression. Defaults to "simex".
    :type regression_method: str
    :param consider_uncertainty: Whether to consider uncertainties in the\
        linear regression. Defaults to False.
    :type consider_uncertainty: bool
    :return: Dictionary containing linear regression results and correlation statistics.
    :rtype: dict[str, Any]
    """
    if not consider_uncertainty:
        x_err = np.zeros_like(x)
        y_err = np.zeros_like(y)

    inputs = {
        "x": x,
        "y": y,
        "x_err": x_err,
        "y_err": y_err,
        "regression_method": regression_method,
        "consider_uncertainty": consider_uncertainty,
    }

    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    input_hash = joblib.hash(inputs)
    cache_file = CACHE_DIR / f"{input_hash}.pkl"

    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)

            if "inputs" in cached_data:

                inputs_match = True
                cached_inputs = cached_data["inputs"]
                for key, value in inputs.items():
                    if isinstance(value, np.ndarray):
                        if not np.array_equal(value, cached_inputs.get(key)):
                            inputs_match = False
                            break
                    else:
                        if value != cached_inputs.get(key):
                            inputs_match = False
                            break

                if inputs_match:
                    print("Using cached result.")
                    return cached_data["result"]

        except (EOFError, pickle.UnpicklingError, FileNotFoundError):

            pass

    if regression_method == "simex":
        linear_regression_result = simex_linear_regression(x, y, x_err, y_err)
    elif regression_method == "mc":
        linear_regression_result = monte_carlo_linear_regression(x, y, x_err, y_err)
    else:
        raise ValueError("Invalid regression method")

    pearson_corr = pearson_correlation(x, y, x_err, y_err)

    result = {
        "linear_regression_result": linear_regression_result,
        "pearson_corr": pearson_corr,
    }

    try:
        with open(cache_file, "wb") as f:
            pickle.dump({"inputs": inputs, "result": result}, f)
    except (OSError, pickle.PicklingError):

        pass

    return result


def average_with_uncertainty(
    values: np.ndarray,
    errors: np.ndarray,
) -> tuple[float, float]:
    """
    Calculates the average of an array of values with propagation of
    uncertainties from standard deviations of each value and the average
    calculation.

    :param values: Array of values.
    :type values: np.ndarray
    :param errors: Array of uncertainties.
    :type errors: np.ndarray
    :return: Tuple containing the average value and its standard deviation.
    :rtype: tuple[float, float]
    """
    if len(values) == 0:
        return np.nan, np.nan

    mean_val = np.mean(values)

    n = len(values)
    prop_uncertainty = (1 / n) * np.sqrt(np.sum(errors**2))

    if n > 1:
        sem = np.std(values, ddof=1) / np.sqrt(n)
    else:
        sem = 0.0

    total_uncertainty = np.sqrt(prop_uncertainty**2 + sem**2) * np.sqrt(n)

    return float(mean_val), float(total_uncertainty)
