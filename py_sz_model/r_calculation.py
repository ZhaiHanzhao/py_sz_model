"""R calculation with uncertainty propagation."""

import argparse
from pathlib import Path
import time
from typing import Callable

import numpy as np
import pandas as pd

try:
    import numba
except ImportError:
    numba = None


def _jit(*args, **kwargs) -> Callable:
    if numba is not None:
        return numba.jit(*args, **kwargs)
    if args and callable(args[0]):
        return args[0]

    def decorator(func):
        return func

    return decorator


@_jit(nopython=True)
def core_monte_carlo_calculation_corrected(
    d13Cc_samples,
    d13Co_samples,
    d13Ca_samples,
    T_samples,
    A_samples,
    B_samples,
    decomp_corr_samples,
):
    """Perform the core R calculation for sampled inputs."""
    d13Cr_samples = d13Co_samples + decomp_corr_samples
    epsilon_samples = A_samples - B_samples * T_samples
    d13Cs_samples = d13Cc_samples - epsilon_samples
    numerator_samples = d13Cs_samples - 1.0044 * d13Cr_samples - 4.4
    denominator_samples = d13Ca_samples - d13Cs_samples
    return numerator_samples / denominator_samples


def calculate_R_with_uncertainty(
    d13Cc: float,
    d13Cc_std: float,
    d13Co: float,
    d13Co_std: float,
    d13Ca: float,
    d13Ca_std: float,
    T: float,
    T_std: float,
    decomp_corr_mean: float = -1.0,
    decomp_corr_std: float = 0.5,
    num_simulations: int = 10000,
    ci_level: int = 90,
) -> dict:
    """Calculate R and uncertainty with Monte Carlo simulation."""
    if num_simulations <= 0:
        raise ValueError("Number of simulations must be positive.")
    if not 0 < ci_level < 100:
        raise ValueError("Confidence interval level must be between 0 and 100.")

    A_mean, A_std = 11.98, 0.13
    B_mean, B_std = 0.12, 0.01

    R_samples = core_monte_carlo_calculation_corrected(
        np.random.normal(d13Cc, d13Cc_std, num_simulations),
        np.random.normal(d13Co, d13Co_std, num_simulations),
        np.random.normal(d13Ca, d13Ca_std, num_simulations),
        np.random.normal(T, T_std, num_simulations),
        np.random.normal(A_mean, A_std, num_simulations),
        np.random.normal(B_mean, B_std, num_simulations),
        np.random.normal(decomp_corr_mean, decomp_corr_std, num_simulations),
    )

    R_samples_clean = R_samples[np.isfinite(R_samples)]
    num_finite = len(R_samples_clean)
    if num_finite == 0:
        raise ValueError("All simulations resulted in non-finite R values.")

    lower_percentile = (100 - ci_level) / 2
    upper_percentile = 100 - lower_percentile
    return {
        "R_mean": float(np.mean(R_samples_clean)),
        "R_std": float(np.std(R_samples_clean)),
        "R_ci_lower": float(np.percentile(R_samples_clean, lower_percentile)),
        "R_ci_upper": float(np.percentile(R_samples_clean, upper_percentile)),
        "ci_level": ci_level,
        "num_simulations_used": num_finite,
    }


CSV_COLUMN_MAP = {
    "d13Cc": ["d13Cc", "d13C"],
    "d13Cc_std": ["d13Cc_std"],
    "d13Co": ["d13Co"],
    "d13Co_std": ["d13Co_std"],
    "d13Ca": ["d13Ca"],
    "d13Ca_std": ["d13Ca_std"],
    "T": ["Temperature", "temp"],
    "T_std": ["Temperature_std", "temp.se"],
}

OPTIONAL_STD_FIELDS = {"d13Cc_std", "d13Co_std", "d13Ca_std", "T_std"}


def get_column_name(df: pd.DataFrame, field_name: str) -> str | None:
    for column_name in CSV_COLUMN_MAP[field_name]:
        if column_name in df.columns:
            return column_name
    return None


def calculate_R_for_dataframe(
    df: pd.DataFrame,
    num_simulations: int = 10000,
    ci_level: int = 90,
    decomp_corr_mean: float = -1.0,
    decomp_corr_std: float = 0.5,
) -> pd.DataFrame:
    """Calculate R columns for every row in a dataframe."""
    missing_columns = []
    resolved_columns: dict[str, str | None] = {}
    for field_name in CSV_COLUMN_MAP:
        column_name = get_column_name(df, field_name)
        resolved_columns[field_name] = column_name
        if column_name is None and field_name not in OPTIONAL_STD_FIELDS:
            missing_columns.append("/".join(CSV_COLUMN_MAP[field_name]))

    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))

    result_df = df.copy()
    ci_low_col = f"R_{ci_level}_low"
    ci_high_col = f"R_{ci_level}_high"
    result_df["R"] = np.nan
    result_df["R_std"] = np.nan
    result_df[ci_low_col] = np.nan
    result_df[ci_high_col] = np.nan

    for idx, row in result_df.iterrows():
        try:
            results = calculate_R_with_uncertainty(
                d13Cc=float(row[resolved_columns["d13Cc"]]),
                d13Cc_std=(
                    float(row[resolved_columns["d13Cc_std"]])
                    if resolved_columns["d13Cc_std"] is not None
                    else 0.0
                ),
                d13Co=float(row[resolved_columns["d13Co"]]),
                d13Co_std=(
                    float(row[resolved_columns["d13Co_std"]])
                    if resolved_columns["d13Co_std"] is not None
                    else 0.0
                ),
                d13Ca=float(row[resolved_columns["d13Ca"]]),
                d13Ca_std=(
                    float(row[resolved_columns["d13Ca_std"]])
                    if resolved_columns["d13Ca_std"] is not None
                    else 0.0
                ),
                T=float(row[resolved_columns["T"]]),
                T_std=(
                    float(row[resolved_columns["T_std"]])
                    if resolved_columns["T_std"] is not None
                    else 0.0
                ),
                decomp_corr_mean=decomp_corr_mean,
                decomp_corr_std=decomp_corr_std,
                num_simulations=num_simulations,
                ci_level=ci_level,
            )
            result_df.at[idx, "R"] = results["R_mean"]
            result_df.at[idx, "R_std"] = results["R_std"]
            result_df.at[idx, ci_low_col] = results["R_ci_lower"]
            result_df.at[idx, ci_high_col] = results["R_ci_upper"]
        except Exception as exc:
            print(f"Warning: failed to calculate R for row {idx}: {exc}")

    return result_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate R and uncertainty for all rows in a CSV file."
    )
    parser.add_argument("input_csv", type=Path, help="Path to the input CSV file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to the output CSV file. Defaults to '<input_stem>_with_r.csv'.",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=10000,
        help="Number of Monte Carlo simulations per row.",
    )
    parser.add_argument(
        "--ci-level",
        type=int,
        default=90,
        help="Confidence interval level to use for R.",
    )
    parser.add_argument("--decomp-corr-mean", type=float, default=-1.0)
    parser.add_argument("--decomp-corr-std", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = (
        args.output
        if args.output
        else args.input_csv.with_name(f"{args.input_csv.stem}_with_r.csv")
    )

    start_time = time.time()
    result_df = calculate_R_for_dataframe(
        df=pd.read_csv(args.input_csv),
        num_simulations=args.num_simulations,
        ci_level=args.ci_level,
        decomp_corr_mean=args.decomp_corr_mean,
        decomp_corr_std=args.decomp_corr_std,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print(f"Processed {len(result_df)} rows from {args.input_csv}")
    print(f"Saved output to {output_path}")
    print(f"Execution time: {time.time() - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
