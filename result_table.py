import pandas as pd

csv_path = "experiment_results.csv"

df = pd.read_csv(csv_path)

noise_levels = [0.3, 0.4, 0.5]
output_path = "experiment_summary.xlsx"

with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
    workbook = writer.book

    for n in noise_levels:
        df_filtered = df[df["noise_std"] == n].copy()

        stats = (
            df_filtered.groupby(["dataset", "config_name"])
            .agg(
                clean_acc_mean=("clean_acc", "mean"),
                clean_acc_std=("clean_acc", "std"),
                noisy_acc_mean=("noisy_acc", "mean"),
                noisy_acc_std=("noisy_acc", "std"),
                runtime_mean=("runtime_sec", "mean"),
                runtime_std=("runtime_sec", "std"),
            )
            .reset_index()
        )
        stats["Clean Acc (%)"] = stats.apply(
            lambda r: f"{r.clean_acc_mean:.2f} ± {r.clean_acc_std:.2f}", axis=1
        )
        stats["Noisy Acc (%)"] = stats.apply(
            lambda r: f"{r.noisy_acc_mean:.2f} ± {r.noisy_acc_std:.2f}", axis=1
        )
        stats["Runtime (s)"] = stats.apply(
            lambda r: f"{r.runtime_mean:.1f} ± {r.runtime_std:.1f}", axis=1
        )

        table = stats[
            [
                "dataset",
                "config_name",
                "Clean Acc (%)",
                "Noisy Acc (%)",
                "Runtime (s)",
            ]
        ]
        table.columns = [
            "Dataset",
            "Configuration",
            "Clean Acc (%)",
            "Noisy Acc (%)",
            "Runtime (s)",
        ]

        sheet_name = f"Noise_{str(n).replace('.', '')}"
        table.to_excel(writer, sheet_name=sheet_name, index=False)

        worksheet = writer.sheets[sheet_name]
        max_row, max_col = table.shape
        worksheet.add_table
