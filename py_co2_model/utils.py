"""
Utils for the py_co2_model package.
"""

from pathlib import Path
from typing import Tuple
from scipy.interpolate import make_interp_spline

import numpy as np

import pandas as pd


def cal_ratios_with_uncertainty(
    a: np.ndarray,
    b: np.ndarray,
    a_std: np.ndarray,
    b_std: np.ndarray,
    n_mc_samples: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate ratios with uncertainties using Monte Carlo Simulation.

    Args:
        a: Array of values.
        b: Array of values.
        a_std: Array of uncertainties.
        b_std: Array of uncertainties.
        n_mc_samples: Number of Monte Carlo samples.

    Returns:
        Tuple of arrays containing the ratio of a / b and its uncertainty.
    """

    a = np.asarray(a)
    b = np.asarray(b)
    a_std = np.asarray(a_std)
    b_std = np.asarray(b_std)

    a_samples = np.random.normal(
        loc=np.expand_dims(a, axis=-1),
        scale=np.expand_dims(a_std, axis=-1),
        size=a.shape + (n_mc_samples,),
    )
    b_samples = np.random.normal(
        loc=np.expand_dims(b, axis=-1),
        scale=np.expand_dims(b_std, axis=-1),
        size=b.shape + (n_mc_samples,),
    )

    ratios_mc = a_samples / b_samples

    ratio_mean = np.mean(ratios_mc, axis=-1)
    ratio_std = np.std(ratios_mc, axis=-1)

    return ratio_mean, ratio_std


def _process_input_data(data_path: Path) -> None:
    """
    Calculate ratios with uncertainties for the input data.
    """

    data = pd.read_csv(data_path)

    ratios = ["aFe/aSi", "aFe/aAl", "aSi/aAl", "aFe/fFe", "aSi/fFe", "aAl/fFe"]
    for r in ratios:
        numerator, denominator = r.split("/")
        data[r], data[r + "_std"] = cal_ratios_with_uncertainty(
            data[numerator].to_numpy(),
            data[denominator].to_numpy(),
            data[numerator + "_std"].to_numpy(),
            data[denominator + "_std"].to_numpy(),
        )

    ratio_cols = [r for r in ratios]
    std_cols = [r + "_std" for r in ratios]
    other_cols = [col for col in data.columns if col not in ratio_cols + std_cols]
    data = data[other_cols + ratio_cols + std_cols]

    data.to_csv(
        data_path.with_name("800kyr_amorphous_with_ratios.CSV"),
        index=False,
    )


def interpolate_co2_icecore(
    target_age: np.ndarray, co2_data: pd.DataFrame, n_mc_samples: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate CO2 icecore data to target age using Monte Carlo Linear Interpolation.

    Args:
        target_age: Target age for interpolation.
        co2_data: DataFrame containing CO2 icecore data.
                  Must have columns: 'Gasage (yr BP)', 'CO2 (ppmv)', 'sigma mean CO2 (ppmv)'
        n_mc_samples: Number of Monte Carlo samples to generate.

    Returns:
        Tuple of arrays containing the interpolated CO2 mean and standard deviation.
    """

    x = co2_data["Gasage (yr BP)"].to_numpy()
    y = co2_data["CO2 (ppmv)"].to_numpy()
    sigma = co2_data["sigma mean CO2 (ppmv)"].to_numpy()

    interpolated_samples = np.zeros((n_mc_samples, len(target_age)))

    for i in range(n_mc_samples):

        y_sample = np.random.normal(y, sigma)

        interpolated_samples[i, :] = make_interp_spline(x, y_sample)(target_age)

    y_pred = np.mean(interpolated_samples, axis=0)
    y_std = np.std(interpolated_samples, axis=0)

    return y_pred, y_std


if __name__ == "__main__":

    co2_ice_data = pd.read_csv(Path(__file__).parent.parent / "data/CO2_icecore.csv")

    features_data = pd.read_csv(
        Path(__file__).parent.parent / "data/800kyr_amorphous_with_ratios.CSV"
    )

    features_data.sort_values(by="Age(Ma)", inplace=True)

    age = features_data["Age(Ma)"] * 1e6

    co2_mean, co2_std = interpolate_co2_icecore(age, co2_ice_data)

    features_data["CO2"] = co2_mean
    features_data["CO2_std"] = co2_std

    features_data.sort_index(inplace=True)

    features_data.to_csv(
        Path(__file__).parent.parent / "data/800kyr_amorphous_with_co2.CSV",
        index=False,
    )
