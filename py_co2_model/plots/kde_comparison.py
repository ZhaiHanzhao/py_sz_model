import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def plot_kde_curve(data_dict, title, output_path):
    """
    Plots KDE curves for multiple arrays on the same figure.

    Args:
        data_dict: Dictionary where key is the label and value is the np.ndarray or Series of data.
        title: Title of the plot.
        output_path: Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    for label, data in data_dict.items():
        if data is not None and len(data) > 0:
            sns.kdeplot(data, label=label, fill=True, alpha=0.3)

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    data_dir = os.path.join(base_dir, "data")
    pred_dir = os.path.join(base_dir, "predictions")
    output_dir = os.path.join(base_dir, "plots", "kde_comparisons")

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "800kyr_amorphous_with_particlesize.CSV")
    if not os.path.exists(train_path):
        print(f"Error: Training data not found at {train_path}")
        return
    train_df = pd.read_csv(train_path)

    features = [
        "Xlf",
        "aFe",
        "aSi",
        "aAl",
        "fFe",
        "aFe/aSi",
        "aFe/aAl",
        "aSi/aAl",
        "aFe/fFe",
        "aSi/fFe",
        "aAl/fFe",
        "d18Oc",
        "D",
    ]
    target = "Sz"

    models = ["GradientBoosting", "Ridge", "RandomForest"]
    sites = ["Jiaxian", "Shilou"]

    dfs = {}
    for model in models:
        dfs[model] = {}
        for site in sites:
            filename = f"{model}_{site}_features_bulk.csv"
            path = os.path.join(pred_dir, filename)
            if os.path.exists(path):
                dfs[model][site] = pd.read_csv(path)
            else:
                print(f"Warning: Prediction file not found: {path}")

    print("Plotting features...")
    for feature in features:
        data_dict = {}

        if feature in train_df.columns:
            data_dict["Training Data"] = train_df[feature].dropna()

        if "GradientBoosting" in dfs and "Jiaxian" in dfs["GradientBoosting"]:
            df = dfs["GradientBoosting"]["Jiaxian"]
            if feature in df.columns:
                data_dict["Jiaxian Data"] = df[feature].dropna()

        if "GradientBoosting" in dfs and "Shilou" in dfs["GradientBoosting"]:
            df = dfs["GradientBoosting"]["Shilou"]
            if feature in df.columns:
                data_dict["Shilou Data"] = df[feature].dropna()

        safe_feature_name = feature.replace("/", "_")
        output_path = os.path.join(output_dir, f"feature_kde_{safe_feature_name}.png")
        plot_kde_curve(data_dict, f"Feature KDE Comparison: {feature}", output_path)

    print("Plotting targets...")
    for model in models:
        data_dict = {}

        if target in train_df.columns:
            data_dict[f"Training {target}"] = train_df[target].dropna()

        prediction_col = f"Sz_mean"

        for site in sites:
            if model in dfs and site in dfs[model]:
                df = dfs[model][site]
                if prediction_col in df.columns:
                    data_dict[f"{site} Prediction"] = df[prediction_col].dropna()
                else:
                    print(
                        f"Warning: {prediction_col} not found in {model} {site} file."
                    )

        output_path = os.path.join(output_dir, f"target_kde_{model}_{target}.png")
        plot_kde_curve(
            data_dict, f"{model} Target ({target}) KDE Comparison", output_path
        )


if __name__ == "__main__":
    main()
