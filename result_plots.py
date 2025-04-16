import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("experiment_results.csv")
datasets = ["MNIST", "FashionMNIST"]
os.makedirs("Results", exist_ok=True)

for dataset in datasets:
    df_dataset = df[df["dataset"] == dataset]

    # Line Plot: Accuracy vs. Noise STD by Configuration
    plt.figure(figsize=(10, 6))
    for config in df["config_name"].unique():
        subset = df_dataset[df_dataset["config_name"] == config]
        grouped = (
            subset.groupby("noise_std")
            .agg({"clean_acc": ["mean", "std"], "noisy_acc": ["mean", "std"]})
            .reset_index()
        )
        grouped.columns = [
            "noise_std",
            "clean_acc_mean",
            "clean_acc_std",
            "noisy_acc_mean",
            "noisy_acc_std",
        ]

        plt.errorbar(
            grouped["noise_std"],
            grouped["clean_acc_mean"],
            yerr=grouped["clean_acc_std"],
            label=f"{config} (Clean)",
            marker="o",
            capsize=4,
        )
        plt.errorbar(
            grouped["noise_std"],
            grouped["noisy_acc_mean"],
            yerr=grouped["noisy_acc_std"],
            label=f"{config} (Noisy)",
            marker="s",
            linestyle="--",
            capsize=4,
        )

    plt.xlabel("Noise Standard Deviation")
    plt.ylabel("Accuracy (%)")
    plt.title("Mean Accuracy vs. Noise STD by Configuration")
    plt.legend()
    plt.savefig(os.path.join("Results", f"lineplot_accuracy_vs_noise_{dataset}.png"))
    plt.close()

    # Grouped Bar Chart: Configurations at Fixed Noise STD

    noise_level = 0.4
    subset = df_dataset[df_dataset["noise_std"] == noise_level]
    agg_data = (
        subset.groupby("config_name")[["clean_acc", "noisy_acc"]].mean().reset_index()
    )
    agg_data_melted = pd.melt(
        agg_data,
        id_vars="config_name",
        value_vars=["clean_acc", "noisy_acc"],
        var_name="Condition",
        value_name="Accuracy",
    )
    plt.figure(figsize=(8, 6))
    sns.barplot(x="config_name", y="Accuracy", hue="Condition", data=agg_data_melted)
    plt.title(f"Mean Accuracy at Noise STD = {noise_level}")
    plt.xlabel("Configuration")
    plt.ylabel("Accuracy (%)")
    plt.savefig(os.path.join("Results", f"barchart_accuracy_fixed_noise_{dataset}.png"))
    plt.close()

    # Heatmap: Accuracy vs Inhibition Strength and Noise STD

    heatmap_data = df_dataset[df_dataset["config_name"] == "All Biological"]
    heatmap_pivot = heatmap_data.pivot_table(
        index="inhibition_strength",
        columns="noise_std",
        values="noisy_acc",
        aggfunc="mean",
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_pivot, annot=True, fmt=".1f", cmap="viridis")
    plt.title("Heatmap: Noisy Accuracy (All Biological)")
    plt.xlabel("Noise STD")
    plt.ylabel("Inhibition Strength")
    plt.savefig(
        os.path.join("Results", f"heatmap_noisy_accuracy_all_biological_{dataset}.png")
    )
    plt.close()

    # Box Plot: Distribution of Noisy Accuracies by Configuration

    plt.figure(figsize=(8, 6))
    sns.boxplot(x="config_name", y="noisy_acc", data=df)
    plt.title("Distribution of Noisy Accuracies by Configuration")
    plt.xlabel("Configuration")
    plt.ylabel("Noisy Accuracy (%)")
    plt.savefig(
        os.path.join("Results", f"boxplot_noisy_accuracy_by_config_{dataset}.png")
    )
    plt.close()
