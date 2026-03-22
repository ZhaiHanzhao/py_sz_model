"""
Module for generating plots for the Paleo CO2 model project.
Includes performance plots, residual analysis, and combined model comparisons.
"""

import copy
import json
import string
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.axes import Axes


def add_panel_labels(axes: np.ndarray | Axes) -> None:
    """
    Add panel labels (a., b., c., ...) to axes in row-major order.
    Position: Top right.
    """

    if isinstance(axes, (np.ndarray, list)):
        flat_axes = np.array(axes).flatten()
    else:
        flat_axes = np.array([axes])

    labels = string.ascii_lowercase

    for i, ax in enumerate(flat_axes):
        if i >= len(labels):
            break
        label_text = f"{labels[i]}."
        ax.text(
            0.95,
            0.85,
            label_text,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
            ha="right",
            bbox=dict(facecolor="none", edgecolor="none", pad=1),
        )


def load_and_apply_plot_config(config_path: Path) -> Dict:
    """
    Loads a JSON config file, applies standard settings to matplotlib.rcParams,
    and returns the full config dictionary including custom styles.
    """
    config = {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        rc_config = copy.deepcopy(config)

        custom_keys = [
            "scatter",
            "errorbar",
            "line",
            "residual_line",
            "k_dashed_line",
            "textbox",
            "savefig",
        ]
        for key in custom_keys:
            if key in rc_config:
                del rc_config[key]

        flat_config = {}
        for section, settings in rc_config.items():
            for key, value in settings.items():
                flat_config[f"{section}.{key}"] = value

        plt.rcParams.update(flat_config)
        return config
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: Could not apply plot styles from '{config_path}': {e}")
        return config


def _monte_carlo_linregress(
    x: np.ndarray,
    y: np.ndarray,
    x_std: Optional[np.ndarray] = None,
    y_std: Optional[np.ndarray] = None,
    n_samples: int = 10000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform linear regression using Monte Carlo simulation to account for uncertainties.

    Args:
        x: Independent variable values.
        y: Dependent variable values.
        x_std: Standard deviation of x values.
        y_std: Standard deviation of y values.
        n_samples: Number of MC samples.

    Returns:
        Tuple containing:
        - slopes: Array of slope samples.
        - intercepts: Array of intercept samples.
        - r_squareds: Array of R^2 samples.
        - p_values: Array of p-value samples.
    """
    slopes = np.zeros(n_samples)
    intercepts = np.zeros(n_samples)
    r_squareds = np.zeros(n_samples)
    p_values = np.zeros(n_samples)

    for i in range(n_samples):

        x_sample = x
        if x_std is not None:
            x_sample = np.random.normal(x, x_std)

        y_sample = y
        if y_std is not None:
            y_sample = np.random.normal(y, y_std)

        res = cast(Any, linregress(x_sample, y_sample))
        slopes[i] = res.slope
        intercepts[i] = res.intercept
        r_squareds[i] = res.rvalue**2
        p_values[i] = res.pvalue

    return slopes, intercepts, r_squareds, p_values


def _fill_performance_axes(
    ax_scatter,
    ax_resid,
    target_true: pd.Series,
    target_pred: pd.Series,
    target_true_std: Optional[pd.Series] = None,
    target_pred_std: Optional[pd.Series] = None,
    metrics: Optional[Dict[str, float]] = None,
    plot_config: Optional[Dict] = None,
    xlabel: str = "True Values",
    ylabel: str = "Predicted Values",
    use_mc_regression: bool = True,
    axis_limits: Optional[Tuple[float, float]] = None,
    residual_limits: Optional[Tuple[float, float]] = None,
    sample_ids: Optional[pd.Series] = None,
    show_legend: bool = True,
) -> None:
    """
    Helper function to fill performance plots on given axes.
    """
    if plot_config is None:
        plot_config = {}

    section_colors = {
        "ZJC": ("#cdaa2d", "Zhaojiachuan"),
        "FX": ("#4f834b", "Fuxian"),
        "93L": ("#5a6fe0", "Luochuan"),
    }

    all_values = pd.concat([target_true, target_pred])
    min_val = all_values.min()
    max_val = all_values.max()
    range_val = max_val - min_val
    buffer = range_val * 0.05
    plot_min = min_val - buffer
    plot_max = max_val + buffer

    if axis_limits:
        plot_min, plot_max = axis_limits

    if sample_ids is not None:

        df_plot = pd.DataFrame(
            {"true": target_true, "pred": target_pred, "sample_id": sample_ids}
        )

        df_plot["section"] = df_plot["sample_id"].apply(
            lambda x: x.split("_")[0] if isinstance(x, str) else "Unknown"
        )

        present_sections = df_plot["section"].unique()

        ordered_sections = [s for s in ["ZJC", "FX", "93L"] if s in present_sections]

        for s in present_sections:
            if s not in ordered_sections:
                ordered_sections.append(s)

        for section in ordered_sections:
            mask = df_plot["section"] == section
            subset = df_plot[mask]

            color, label = section_colors.get(section, ("gray", section))

            indices = subset.index

            xerr_subset = (
                target_true_std.loc[indices] if target_true_std is not None else None
            )
            yerr_subset = (
                target_pred_std.loc[indices] if target_pred_std is not None else None
            )

            ax_scatter.errorbar(
                subset["true"],
                subset["pred"],
                xerr=xerr_subset,
                yerr=yerr_subset,
                label=label,
                markerfacecolor=mcolors.to_rgba(color, alpha=1.0),
                markeredgecolor=mcolors.to_rgba(color, alpha=1.0),
                ecolor=mcolors.to_rgba(color, alpha=0.4),
                **{
                    k: v
                    for k, v in plot_config.get("errorbar", {}).items()
                    if k != "alpha"
                },
            )
    else:

        ax_scatter.errorbar(
            target_true,
            target_pred,
            xerr=target_true_std if target_true_std is not None else None,
            yerr=target_pred_std if target_pred_std is not None else None,
            label="Data",
            markerfacecolor=mcolors.to_rgba("#226e9c", alpha=1.0),
            markeredgecolor=mcolors.to_rgba("#226e9c", alpha=1.0),
            ecolor=mcolors.to_rgba("#226e9c", alpha=0.4),
            **{
                k: v for k, v in plot_config.get("errorbar", {}).items() if k != "alpha"
            },
        )

    x_range = np.linspace(plot_min, plot_max, 100)

    if use_mc_regression:

        x_arr = target_true.to_numpy()
        y_arr = target_pred.to_numpy()
        x_std_arr = target_true_std.to_numpy() if target_true_std is not None else None
        y_std_arr = target_pred_std.to_numpy() if target_pred_std is not None else None

        mc_slopes, mc_intercepts, mc_r2, mc_p_values = _monte_carlo_linregress(
            x_arr, y_arr, x_std_arr, y_std_arr
        )

        mean_slope = np.mean(mc_slopes)
        mean_intercept = np.mean(mc_intercepts)
        mean_r2 = np.mean(mc_r2)
        std_r2 = np.std(mc_r2)
        mean_p_value = np.mean(mc_p_values)

        y_mc = np.outer(mc_slopes, x_range) + mc_intercepts[:, np.newaxis]

        y_mean = np.mean(y_mc, axis=0)
        y_lower = np.percentile(y_mc, 2.5, axis=0)
        y_upper = np.percentile(y_mc, 97.5, axis=0)

        ax_scatter.plot(
            x_range,
            y_mean,
            label=None,
            **plot_config.get("line", {}),
        )

        ax_scatter.plot(
            x_range,
            y_lower,
            linestyle="--",
            color=plot_config.get("line", {}).get("color", "red"),
            alpha=0.5,
            linewidth=1,
            label=None,
        )
        ax_scatter.plot(
            x_range,
            y_upper,
            linestyle="--",
            color=plot_config.get("line", {}).get("color", "red"),
            alpha=0.5,
            linewidth=1,
        )

        annotation_text = rf"$R^2$ (fit) = {mean_r2:.3f}"
        if std_r2 > 1e-6:
            annotation_text += rf" $\pm$ {std_r2:.3f}"
        annotation_text += "\n"
        if mean_p_value < 0.001:
            _exp = int(np.floor(np.log10(abs(mean_p_value))))
            _man = mean_p_value / 10**_exp
            annotation_text += rf"p = {_man:.2f} $\times$ 10$^{{{_exp}}}$"
        else:
            annotation_text += f"p = {mean_p_value:.3f}"

    else:

        res = cast(Any, linregress(target_true, target_pred))
        slope = res.slope
        intercept = res.intercept
        r_value = res.rvalue
        p_value = res.pvalue

        y_range = slope * x_range + intercept
        ax_scatter.plot(
            x_range,
            y_range,
            label=None,
            **plot_config.get("line", {}),
        )

        annotation_text = f"$R^2$ (fit) = {r_value**2:.3f}\n"
        if p_value < 0.001:
            _exp = int(np.floor(np.log10(abs(p_value))))
            _man = p_value / 10**_exp
            annotation_text += rf"p = {_man:.2f} $\times$ 10$^{{{_exp}}}$"
        else:
            annotation_text += f"p = {p_value:.3f}"

    ax_scatter.plot(
        [plot_min, plot_max],
        [plot_min, plot_max],
        label=None,
        **plot_config.get("k_dashed_line", {}),
    )

    ax_scatter.text(
        0.05,
        0.95,
        annotation_text,
        transform=ax_scatter.transAxes,
        verticalalignment="top",
        bbox=plot_config.get("textbox", {}),
    )

    ax_scatter.set_ylabel(ylabel)
    if show_legend:
        ax_scatter.legend(loc="lower right")

    ax_scatter.set_xlim(plot_min, plot_max)
    ax_scatter.set_ylim(plot_min, plot_max)
    ax_scatter.set_box_aspect(1)

    ax_scatter.tick_params(axis="x", top=True, bottom=True, which="both", rotation=45)
    ax_scatter.tick_params(axis="y", left=True, right=True, which="both", rotation=45)

    residuals = target_pred - target_true

    residual_err: Optional[pd.Series] = None
    if target_true_std is not None and target_pred_std is not None:

        t_std = target_true_std.fillna(0)
        p_std = target_pred_std.fillna(0)
        residual_err = np.sqrt(t_std**2 + p_std**2)
    elif target_true_std is not None:
        residual_err = target_true_std
    elif target_pred_std is not None:
        residual_err = target_pred_std

    ax_resid.errorbar(
        target_true,
        residuals,
        yerr=residual_err,
        xerr=target_true_std if target_true_std is not None else None,
        label="Data",
        markerfacecolor=mcolors.to_rgba("#226e9c", alpha=1.0),
        markeredgecolor=mcolors.to_rgba("#226e9c", alpha=1.0),
        ecolor=mcolors.to_rgba("#226e9c", alpha=0.4),
        **{k: v for k, v in plot_config.get("errorbar", {}).items() if k != "alpha"},
    )

    if sample_ids is not None:

        ax_resid.cla()
        ax_resid.axhline(0, **plot_config.get("residual_line", {}))

        df_resid = pd.DataFrame(
            {"true": target_true, "resid": residuals, "sample_id": sample_ids}
        )
        df_resid["section"] = df_resid["sample_id"].apply(
            lambda x: x.split("_")[0] if isinstance(x, str) else "Unknown"
        )

        present_sections = df_resid["section"].unique()
        ordered_sections = [s for s in ["ZJC", "FX", "93L"] if s in present_sections]
        for s in present_sections:
            if s not in ordered_sections:
                ordered_sections.append(s)

        for section in ordered_sections:
            mask = df_resid["section"] == section
            subset = df_resid[mask]

            color, _ = section_colors.get(section, ("gray", section))

            indices = subset.index

            yerr_subset = (
                residual_err.loc[indices] if residual_err is not None else None
            )
            xerr_subset = (
                target_true_std.loc[indices] if target_true_std is not None else None
            )

            ax_resid.errorbar(
                subset["true"],
                subset["resid"],
                yerr=yerr_subset,
                xerr=xerr_subset,
                markerfacecolor=mcolors.to_rgba(color, alpha=1.0),
                markeredgecolor=mcolors.to_rgba(color, alpha=1.0),
                ecolor=mcolors.to_rgba(color, alpha=0.4),
                **{
                    k: v
                    for k, v in plot_config.get("errorbar", {}).items()
                    if k != "alpha"
                },
            )

    ax_resid.axhline(0, **plot_config.get("residual_line", {}))
    ax_resid.set_ylabel("Residuals (Pred - True)")
    ax_resid.set_xlabel(xlabel)
    if residual_limits:
        ax_resid.set_ylim(residual_limits)

    ax_resid.set_box_aspect(1 / 3)

    ax_resid.tick_params(axis="x", top=True, bottom=True, which="both", rotation=45)
    ax_resid.tick_params(axis="y", left=True, right=True, which="both", rotation=45)

    ax_scatter.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax_scatter.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax_resid.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax_resid.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    ax_scatter.minorticks_on()
    ax_resid.minorticks_on()


def plot_prediction_performance(
    target_true: pd.Series,
    target_pred: pd.Series,
    target_true_std: Optional[pd.Series] = None,
    target_pred_std: Optional[pd.Series] = None,
    metrics: Optional[Dict[str, float]] = None,
    save_path: Optional[Path] = None,
    xlabel: str = "True Values",
    ylabel: str = "Predicted Values",
    use_mc_regression: bool = True,
    axis_limits: Optional[Tuple[float, float]] = None,
    residual_limits: Optional[Tuple[float, float]] = None,
    sample_ids: Optional[pd.Series] = None,
) -> None:
    """
    Create plots for test_set with predictions:
    1. prediction values vs. true values, add a linear regression line with equation
       and R^2 annotated
    2. residual vs. true values

    Args:
        target_true: True target values.
        target_pred: Predicted target values.
        target_true_std: Uncertainty of true target values.
        target_pred_std: Uncertainty of predicted target values.
        metrics: Dictionary containing evaluation metrics (rmse_mean, rmse_std, r2_mean, r2_std).
        save_path: Path to save the plot.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        ylabel: Label for y-axis.
        use_mc_regression: Whether to use Monte Carlo regression with 95% CI.
        axis_limits: Optional tuple (min, max) for plot axes.
        residual_limits: Optional tuple (min, max) for residual axes.
    """

    config_path = Path(__file__).parent / "plot_config.json"
    plot_config = load_and_apply_plot_config(config_path)

    _, axes = plt.subplots(
        2,
        1,
        figsize=(5, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )

    _fill_performance_axes(
        axes[0],
        axes[1],
        target_true,
        target_pred,
        target_true_std,
        target_pred_std,
        metrics,
        plot_config,
        xlabel=xlabel,
        ylabel=ylabel,
        use_mc_regression=use_mc_regression,
        axis_limits=axis_limits,
        residual_limits=residual_limits,
        sample_ids=sample_ids,
    )

    add_panel_labels(axes)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=plot_config.get("savefig", {}).get("dpi", 300))
        plt.close()
    else:
        plt.show()


def plot_residual_vs_uncertainty(
    target_true_std: pd.Series,
    target_pred: pd.Series,
    target_true: pd.Series,
    save_path: Optional[Path] = None,
) -> None:
    """
    Create plot for residual vs true uncertainty.

    Args:
        target_true_std: Uncertainty of true target values.
        target_pred: Predicted target values.
        target_true: True target values.
        save_path: Path to save the plot.
    """

    config_path = Path(__file__).parent / "plot_config.json"
    plot_config = load_and_apply_plot_config(config_path)

    residuals = target_pred - target_true

    plt.figure(figsize=(6, 6))
    scatter_cfg = plot_config.get("scatter", {})
    scatter_cfg = {k: v for k, v in scatter_cfg.items() if k != "alpha"}
    plt.scatter(target_true_std, residuals, alpha=1.0, **scatter_cfg)
    plt.axhline(0, **plot_config.get("residual_line", {}))
    plt.xlabel("True Values Std (Uncertainty)")
    plt.ylabel("Residuals (Pred - True)")
    plt.title("Residuals vs True Uncertainty")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=plot_config.get("savefig", {}).get("dpi", 300))
        plt.close()
    else:
        plt.show()


def plot_residual_analysis(
    target_true: pd.Series,
    target_pred: pd.Series,
    save_path: Optional[Path] = None,
) -> None:
    """
    Create residual analysis plots:
    1. Histogram of residuals with fitted normal distribution.
    2. Q-Q plot of residuals.
    3. Residuals vs. Index (Run plot).

    Args:
        target_true: True target values.
        target_pred: Predicted target values.
        save_path: Path to save the plot.
    """

    config_path = Path(__file__).parent / "plot_config.json"
    plot_config = load_and_apply_plot_config(config_path)

    residuals = target_pred - target_true

    _, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    ax.hist(residuals, bins=20, density=True, alpha=0.6, color="g", label="Residuals")

    mu, std = stats.norm.fit(residuals)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax.plot(
        x,
        p,
        label=rf"Normal Fit\n$\mu={mu:.2f}, \sigma={std:.2f}$",
        **plot_config.get("line", {}),
    )
    ax.set_title("Residuals Histogram")
    ax.set_xlabel("Residuals")
    ax.legend()

    ax = axes[1]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot")

    ax = axes[2]
    ax.plot(residuals.values, "o-", alpha=0.5)
    ax.axhline(0, **plot_config.get("residual_line", {}))
    ax.set_title("Residuals vs Index")
    ax.set_xlabel("Index")
    ax.set_ylabel("Residuals")

    add_panel_labels(axes)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=plot_config.get("savefig", {}).get("dpi", 300))
        plt.close()
    else:
        plt.show()


def plot_combined_by_target(
    results: Dict[str, Dict[str, Dict]],
    target_type: str,
    save_path: Path,
    use_mc_regression: bool = True,
    axis_limits: Optional[Tuple[float, float]] = None,
    residual_limits: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Generate a combined plot for a specific target (Sz or CO2) with all models.
    """

    models_with_target = [
        model for model, data in results.items() if target_type in data
    ]
    if not models_with_target:
        return

    if target_type.lower() == "sz":
        xlabel = "Ice core calculated S(z) (ppmv)"
        ylabel = "Predicted S(z) (ppmv)"
    else:
        xlabel = R"Ice core $\bf{CO_2}$ (ppmv)"
        ylabel = R"Predicted $\bf{CO_2}$ (ppmv)"

    n_models = len(models_with_target)

    _, axes = plt.subplots(
        2,
        n_models,
        figsize=(4 * n_models, 6),
        sharex="col",
        sharey="row",
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )

    if n_models == 1:
        axes = axes.reshape(2, 1)

    config_path = Path(__file__).parent / "plot_config.json"
    plot_config = load_and_apply_plot_config(config_path)

    for i, model_name in enumerate(models_with_target):
        data = results[model_name][target_type]
        ax_scatter = axes[0, i]
        ax_resid = axes[1, i]

        _fill_performance_axes(
            ax_scatter,
            ax_resid,
            data["target_true"],
            data["target_pred"],
            data["target_true_std"],
            data["target_pred_std"],
            data["metrics"],
            plot_config,
            xlabel=xlabel,
            ylabel=ylabel,
            use_mc_regression=use_mc_regression,
            axis_limits=axis_limits,
            residual_limits=residual_limits,
            sample_ids=data.get("sample_ids"),
            show_legend=(i == 0),
        )
        ax_scatter.set_title(f"{model_name}")

        if i > 0:
            ax_scatter.set_ylabel("")
            ax_resid.set_ylabel("")

    add_panel_labels(axes)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=plot_config.get("savefig", {}).get("dpi", 300))
    plt.close()


def plot_combined_by_model(
    data_dict: Dict[str, Dict],
    save_path: Path,
    use_mc_regression: bool = True,
    axis_limits_sz: Optional[Tuple[float, float]] = None,
    axis_limits_co2: Optional[Tuple[float, float]] = None,
    residual_limits_sz: Optional[Tuple[float, float]] = None,
    residual_limits_co2: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Generate a combined plot for a specific model with Sz and CO2 side-by-side.
    """
    targets = ["sz", "co2"]

    present_targets = [t for t in targets if t in data_dict]
    if not present_targets:
        return

    n_cols = len(present_targets)
    _, axes = plt.subplots(
        2,
        n_cols,
        figsize=(4 * n_cols, 6),
        sharex="col",
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )

    if n_cols == 1:
        axes = axes.reshape(2, 1)

    config_path = Path(__file__).parent / "plot_config.json"
    plot_config = load_and_apply_plot_config(config_path)

    for i, target in enumerate(present_targets):
        data = data_dict[target]
        ax_scatter = axes[0, i]
        ax_resid = axes[1, i]

        if target.lower() == "sz":
            xlabel = "Ice core calculated S(z) (ppmv)"
            ylabel = "Predicted S(z) (ppmv)"
        else:
            xlabel = R"Ice core $\bf{CO_2}$ (ppmv)"
            ylabel = R"Predicted $\bf{CO_2}$ (ppmv)"

        _fill_performance_axes(
            ax_scatter,
            ax_resid,
            data["target_true"],
            data["target_pred"],
            data["target_true_std"],
            data["target_pred_std"],
            data["metrics"],
            plot_config,
            xlabel=xlabel,
            ylabel=ylabel,
            use_mc_regression=use_mc_regression,
            axis_limits=axis_limits_sz if target.lower() == "sz" else axis_limits_co2,
            residual_limits=(
                residual_limits_sz if target.lower() == "sz" else residual_limits_co2
            ),
            sample_ids=data.get("sample_ids"),
            show_legend=(i == 0),
        )
        ax_scatter.set_title("S(z)" if target.lower() == "sz" else R"$\bf{CO_2}$")

        if i > 0:
            ax_scatter.set_ylabel("")
            ax_resid.set_ylabel("")

    add_panel_labels(axes)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=plot_config.get("savefig", {}).get("dpi", 300))
    plt.close()


def plot_residual_kde_comparison(
    results: Dict[str, Dict[str, Dict]],
    save_path: Path,
) -> None:
    """
    Generate a two-panel (1×2) KDE plot comparing standardised residuals of
    every model against a standard-normal reference curve.

    Panel layout
    ------------
    Left  – S(z) residuals     Right – CO₂ residuals

    Each model's residuals are z-scored (subtract mean, divide by std) so
    that all curves live on the same scale and can be compared directly
    with the N(0,1) reference.

    Args:
        results: Nested dict ``results[model_name][target_type]`` where each
                 leaf contains at least ``target_true`` and ``target_pred``.
        save_path: Destination file for the saved figure.
    """

    config_path = Path(__file__).parent / "plot_config.json"
    plot_config = load_and_apply_plot_config(config_path)

    target_types = ["sz", "co2"]
    target_labels = {
        "sz": "S(z)",
        "co2": r"$\mathrm{CO_2}$",
    }

    model_colors = [
        "#226e9c",
        "#e45e32",
        "#5a6fe0",
        "#4f834b",
        "#cdaa2d",
        "#c44e52",
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4.5),
        constrained_layout=True,
    )

    for col_idx, target in enumerate(target_types):
        ax = axes[col_idx]

        models_with_target = [m for m, d in results.items() if target in d]

        for m_idx, model_name in enumerate(models_with_target):
            data = results[model_name][target]
            residuals = data["target_pred"] - data["target_true"]

            res_mean = residuals.mean()
            res_std = residuals.std()
            if res_std > 0:
                z_residuals = (residuals - res_mean) / res_std
            else:
                z_residuals = residuals - res_mean

            color = model_colors[m_idx % len(model_colors)]
            sns.kdeplot(
                z_residuals,
                ax=ax,
                label=model_name,
                color=color,
                fill=False,
                linewidth=2.5,
            )

        x_ref = np.linspace(-4, 4, 300)
        ax.plot(
            x_ref,
            stats.norm.pdf(x_ref),
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="N(0, 1)",
        )

        ax.set_title(target_labels.get(target, target))
        ax.set_xlabel("Standardised Residual")
        if col_idx == 0:
            ax.set_ylabel("Density")
        else:
            ax.set_ylabel("")
        if col_idx == 0:
            ax.legend(fontsize=8)
        ax.minorticks_on()
        ax.tick_params(axis="x", top=True, bottom=True, which="both")
        ax.tick_params(axis="y", left=True, right=True, which="both")

    for i, ax in enumerate(axes.flatten()):
        ax.text(
            0.95,
            0.95,
            f"{string.ascii_lowercase[i]}.",
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
            ha="right",
            bbox=dict(facecolor="none", edgecolor="none", pad=1),
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=plot_config.get("savefig", {}).get("dpi", 300))
    plt.close()


def main():
    """
    Main function to generate plots from saved predictions and metrics.
    """

    use_mc_regression = False

    sz_range = (200, 1200)
    co2_range = (100, 350)
    sz_residual_range = (-400, 400)
    co2_residual_range = (-150, 150)

    base_dir = Path(__file__).parents[2]
    predictions_dir = base_dir / "predictions"
    metrics_dir = base_dir / "metrics"
    plots_dir = base_dir / "figures"

    if not predictions_dir.exists():
        print(f"Predictions directory not found: {predictions_dir}")
        return

    results: Dict[str, Dict[str, Dict]] = {}

    for pred_file in predictions_dir.glob("*.csv"):
        print(f"Processing {pred_file.name}...")

        filename_stem = pred_file.stem
        parts = filename_stem.split("_")
        if len(parts) < 2:
            print(f"Skipping {pred_file.name}: Invalid filename format.")
            continue

        model_name = "_".join(parts[:-1])

        try:
            df_pred = pd.read_csv(pred_file)
        except (OSError, pd.errors.ParserError) as e:
            print(f"Error reading {pred_file}: {e}")
            continue

        target_col = None
        target_std_col = None
        target_type = None

        if "Sz" in df_pred.columns:
            target_col = "Sz"
            target_std_col = "Sz_std"
            target_type = "sz"
        elif "CO2" in df_pred.columns:
            target_col = "CO2"
            target_std_col = "CO2_std"
            target_type = "co2"

        if target_col is None:

            if "CO2_ice" in df_pred.columns:
                target_col = "CO2_ice"
                target_type = "co2"
                if "CO2_ice_std" in df_pred.columns:
                    target_std_col = "CO2_ice_std"

        if target_col is None:
            print(
                f"Skipping {pred_file.name}: Could not identify target column (Sz or CO2)."
            )
            continue
        if target_col not in df_pred.columns:
            print(f"Skipping {pred_file.name}: Target column {target_col} not found.")
            continue

        target_true = cast(pd.Series, df_pred[target_col])
        target_pred = cast(pd.Series, df_pred["prediction"])

        target_true_std = None
        if target_std_col and target_std_col in df_pred.columns:
            target_true_std = cast(pd.Series, df_pred[target_std_col])

        target_pred_std = None
        if "prediction_uncertainty" in df_pred.columns:
            target_pred_std = cast(pd.Series, df_pred["prediction_uncertainty"])

        sample_ids = None
        if "Sample_ID" in df_pred.columns:
            sample_ids = cast(pd.Series, df_pred["Sample_ID"])

        metrics_file = metrics_dir / f"{filename_stem}.csv"
        metrics = None
        if metrics_file.exists():
            try:
                df_metrics = pd.read_csv(metrics_file)
                if not df_metrics.empty:
                    metrics = df_metrics.iloc[0].to_dict()
            except (OSError, pd.errors.ParserError) as e:
                print(f"Error reading metrics {metrics_file}: {e}")
        else:

            pass

        if model_name not in results:
            results[model_name] = {}

        t_key = target_type.lower() if target_type else "unknown"
        results[model_name][t_key] = {
            "target_true": target_true,
            "target_pred": target_pred,
            "target_true_std": target_true_std,
            "target_pred_std": target_pred_std,
            "metrics": metrics,
            "sample_ids": sample_ids,
        }

        if t_key == "sz":
            xlabel = "Ice core calculated S(z) (ppmv)"
            ylabel = "Predicted S(z) (ppmv)"
        else:
            xlabel = R"Ice core $\bf{CO_2}$ (ppmv)"
            ylabel = R"Predicted $\bf{CO_2}$ (ppmv)"

        plot_path = plots_dir / f"{filename_stem}_performance.png"
        plot_prediction_performance(
            target_true=target_true,
            target_pred=target_pred,
            target_true_std=target_true_std,
            target_pred_std=target_pred_std,
            metrics=metrics,
            save_path=plot_path,
            xlabel=xlabel,
            ylabel=ylabel,
            use_mc_regression=use_mc_regression,
            axis_limits=(
                sz_range if t_key == "sz" else (co2_range if t_key == "co2" else None)
            ),
            residual_limits=(
                sz_residual_range
                if t_key == "sz"
                else (co2_residual_range if t_key == "co2" else None)
            ),
            sample_ids=sample_ids,
        )
        print(f"Saved {plot_path}")

        if target_true_std is not None:
            res_unc_plot_path = (
                plots_dir / f"{filename_stem}_residual_vs_uncertainty.png"
            )
            plot_residual_vs_uncertainty(
                target_true_std=target_true_std,
                target_pred=target_pred,
                target_true=target_true,
                save_path=res_unc_plot_path,
            )
            print(f"Saved {res_unc_plot_path}")

        res_plot_path = plots_dir / f"{filename_stem}_residuals.png"
        plot_residual_analysis(
            target_true=target_true,
            target_pred=target_pred,
            save_path=res_plot_path,
        )
        print(f"Saved {res_plot_path}")

        if t_key == "sz" and "R" in df_pred.columns and "CO2_ice" in df_pred.columns:
            print(f"Generating derived CO2 performance for {model_name}...")

            r_ratio = df_pred["R"]
            co2_pred = target_pred * r_ratio
            co2_true = df_pred["CO2_ice"]

            co2_pred_std = None
            if target_pred_std is not None and "R_std" in df_pred.columns:
                r_std = df_pred["R_std"]

                term1 = (target_pred_std / target_pred.replace(0, np.nan)) ** 2
                term2 = (r_std / r_ratio.replace(0, np.nan)) ** 2
                co2_pred_std = np.abs(co2_pred) * np.sqrt(term1 + term2)
                co2_pred_std = co2_pred_std.fillna(0)

            co2_true_std = None

            if "CO2_ice_std" in df_pred.columns:
                co2_true_std = df_pred["CO2_ice_std"]

            derived_metrics = {}

            derived_metrics["rmse_mean"] = np.sqrt(
                mean_squared_error(co2_true, co2_pred)
            )
            derived_metrics["r2_mean"] = r2_score(co2_true, co2_pred)

            derived_metrics["rmse_std"] = 0
            derived_metrics["r2_std"] = 0

            derived_plot_path = (
                plots_dir / f"{model_name}_sz_derived_co2_performance.png"
            )
            plot_prediction_performance(
                target_true=cast(pd.Series, co2_true),
                target_pred=cast(pd.Series, co2_pred),
                target_true_std=(
                    cast(pd.Series, co2_true_std) if co2_true_std is not None else None
                ),
                target_pred_std=(
                    cast(pd.Series, co2_pred_std) if co2_pred_std is not None else None
                ),
                metrics=derived_metrics,
                save_path=derived_plot_path,
                xlabel=R"Ice core $\bf{CO_2}$ (ppmv)",
                ylabel=R"Predicted $\bf{CO_2}$ (ppmv)",
                use_mc_regression=use_mc_regression,
                axis_limits=co2_range,
                residual_limits=co2_residual_range,
                sample_ids=sample_ids,
            )
            print(f"Saved {derived_plot_path}")

            if "co2" not in results[model_name]:
                results[model_name]["co2"] = {
                    "target_true": co2_true,
                    "target_pred": co2_pred,
                    "target_true_std": co2_true_std,
                    "target_pred_std": co2_pred_std,
                    "metrics": derived_metrics,
                    "sample_ids": sample_ids,
                }

    print("Generating combined plots...")

    plot_combined_by_target(
        results,
        "sz",
        plots_dir / "combined_performance_sz.png",
        use_mc_regression,
        axis_limits=sz_range,
        residual_limits=sz_residual_range,
    )

    plot_combined_by_target(
        results,
        "co2",
        plots_dir / "combined_performance_co2.png",
        use_mc_regression,
        axis_limits=co2_range,
        residual_limits=co2_residual_range,
    )

    for model_name, data in results.items():
        plot_combined_by_model(
            data,
            plots_dir / f"{model_name}_combined_performance.png",
            use_mc_regression,
            axis_limits_sz=sz_range,
            axis_limits_co2=co2_range,
            residual_limits_sz=sz_residual_range,
            residual_limits_co2=co2_residual_range,
        )
        print(f"Saved combined plot for {model_name}")

    plot_residual_kde_comparison(
        results,
        plots_dir / "combined_residual_kde.png",
    )
    print("Saved combined residual KDE plot")


if __name__ == "__main__":
    main()
