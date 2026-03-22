import numpy as np
import numba
import time
import scipy.stats as st


@numba.jit(nopython=True)
def core_monte_carlo_calculation_corrected(
    d13Cc_samples,
    d13Co_samples,
    d13Ca_samples,
    T_samples,
    A_samples,
    B_samples,
    decomp_corr_samples,
):
    """Performs the core R calculation for sampled inputs, including correction."""

    d13Cr_samples = d13Co_samples + decomp_corr_samples

    epsilon_samples = A_samples - B_samples * T_samples

    d13Cs_samples = d13Cc_samples - epsilon_samples

    numerator_samples = d13Cs_samples - 1.0044 * d13Cr_samples - 4.4

    denominator_samples = d13Ca_samples - d13Cs_samples

    R_samples = numerator_samples / denominator_samples
    return R_samples


def calculate_R_with_uncertainty(
    d13Cc: float,
    d13Cc_std: float,
    d13Co: float,
    d13Co_std: float,
    d13Ca: float,
    d13Ca_std: float,
    T: float,
    T_std: float,
    decomp_corr_mean=-1.0,
    decomp_corr_std=0.5,
    num_simulations=10000,
    ci_level=90,
) -> dict:
    """
    Calculates the R value and its uncertainty using Monte Carlo simulation,
    including a correction for SOM decomposition to estimate d13Cr.

    Args:
        d13Cc (float): Carbon isotopic composition of pedogenic carbonate (permil).
        d13Cc_std (float): Standard deviation of d13Cc (permil).
        d13Co (float): Carbon isotopic composition of soil organic matter (d13C_SOM) (permil).
        d13Co_std (float): Standard deviation of d13Co (permil).
        d13Ca (float): Carbon isotopic composition of atmospheric CO2 (permil).
        d13Ca_std (float): Standard deviation of d13Ca (permil).
        T (float): Temperature (°C).
        T_std (float): Standard deviation of Temperature (°C).
        decomp_corr_mean (float, optional): Mean value of the d13Cr decomposition
                                             correction offset. Defaults to -1.0.
        decomp_corr_std (float, optional): Standard deviation of the decomposition
                                            correction offset. Defaults to 0.5.
        num_simulations (int, optional): Number of Monte Carlo iterations.
                                         Defaults to 10000.
        ci_level (int, optional): Desired confidence interval level (e.g., 90 for 90%).
                                  Defaults to 90.

    Returns:
        dict: A dictionary containing results.
    """
    if num_simulations <= 0:
        raise ValueError("Number of simulations must be positive.")
    if not 0 < ci_level < 100:
        raise ValueError("Confidence interval level must be between 0 and 100.")

    A_mean, A_std = 11.98, 0.13
    B_mean, B_std = 0.12, 0.01

    d13Cc_samples = np.random.normal(d13Cc, d13Cc_std, num_simulations)
    d13Co_samples = np.random.normal(d13Co, d13Co_std, num_simulations)
    d13Ca_samples = np.random.normal(d13Ca, d13Ca_std, num_simulations)
    T_samples = np.random.normal(T, T_std, num_simulations)
    A_samples = np.random.normal(A_mean, A_std, num_simulations)
    B_samples = np.random.normal(B_mean, B_std, num_simulations)

    decomp_corr_samples = np.random.normal(
        decomp_corr_mean, decomp_corr_std, num_simulations
    )

    R_samples = core_monte_carlo_calculation_corrected(
        d13Cc_samples,
        d13Co_samples,
        d13Ca_samples,
        T_samples,
        A_samples,
        B_samples,
        decomp_corr_samples,
    )

    R_samples_clean = R_samples[np.isfinite(R_samples)]
    num_finite = len(R_samples_clean)

    if num_finite < num_simulations:
        print(
            f"Warning: Removed {num_simulations - num_finite} non-finite results from R calculation."
        )
        if num_finite == 0:
            raise ValueError(
                "All simulations resulted in non-finite R values. Check inputs and correction factors."
            )

    R_mean = np.mean(R_samples_clean)
    R_std = np.std(R_samples_clean)

    lower_percentile = (100 - ci_level) / 2
    upper_percentile = 100 - lower_percentile

    R_ci_lower = np.percentile(R_samples_clean, lower_percentile)
    R_ci_upper = np.percentile(R_samples_clean, upper_percentile)

    return {
        "R_mean": R_mean,
        "R_std": R_std,
        "R_ci_lower": R_ci_lower,
        "R_ci_upper": R_ci_upper,
        "ci_level": ci_level,
        "num_simulations_used": num_finite,
    }


if __name__ == "__main__":

    d13Cc_val = -5.1
    d13Cc_s = 0.3

    d13Co_val = -22.1
    d13Co_s = 0.5

    d13Ca_val = -6.1755
    d13Ca_s = 0.1

    T_val = 19.72783817
    T_s = 3.186068903

    print("Running corrected calculation with Numba...")
    start_time = time.time()
    results_corrected = calculate_R_with_uncertainty(
        d13Cc=d13Cc_val,
        d13Cc_std=d13Cc_s,
        d13Co=d13Co_val,
        d13Co_std=d13Co_s,
        d13Ca=d13Ca_val,
        d13Ca_std=d13Ca_s,
        T=T_val,
        T_std=T_s,
        num_simulations=10000,
        ci_level=90,
    )
    end_time = time.time()

    print(f"Corrected calculation execution time: {end_time - start_time:.4f} seconds")
    print(
        f"\nCalculation Results (with d13Cr correction, based on {results_corrected['num_simulations_used']} valid simulations):"
    )
    print(f"Mean R value: {results_corrected['R_mean']:.4f}")
    print(f"Standard Deviation of R: {results_corrected['R_std']:.4f}")
    print(f"{results_corrected['ci_level']}% Confidence Interval for R:")
    print(
        f"  Lower bound (-{results_corrected['ci_level']}% CI): {results_corrected['R_ci_lower']:.4f}"
    )
    print(
        f"  Upper bound (+{results_corrected['ci_level']}% CI): {results_corrected['R_ci_upper']:.4f}"
    )
