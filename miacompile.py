import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

def summarize_mia_results(mia_dir="MIA", metrics=["attack_acc", "auc"]):
    summary_data = {metric: [] for metric in metrics}

    for metric in metrics:
        # Find all matching CSV files
        csv_files = glob.glob(os.path.join(mia_dir, f"*_{metric}_all.csv"))

        for csv_path in csv_files:
            algo = os.path.basename(csv_path).replace(f"_{metric}_all.csv", "")
            df = pd.read_csv(csv_path)

            if metric not in df.columns or len(df[metric]) == 0:
                continue

            avg = df[metric].mean()
            std = df[metric].std()
            summary_data[metric].append({
                "algorithm": algo,
                "mean": round(avg, 4),
                "std": round(std, 4)
            })

    return summary_data

def plot_summary(summary_data, output_dir="MIA"):
    os.makedirs(output_dir, exist_ok=True)

    for metric, records in summary_data.items():
        df = pd.DataFrame(records).sort_values("mean", ascending=False)
        df.to_csv(os.path.join(output_dir, f"{metric}_summary.csv"), index=False)

        # Bar plot with error bars
        plt.figure(figsize=(10, 5))
        plt.bar(df["algorithm"], df["mean"], yerr=df["std"], capsize=5, color='cornflowerblue')
        plt.xlabel("Algorithm")
        plt.ylabel(f"{metric.replace('_', ' ').title()} (Avg ± Std)")
        plt.title(f"MIA {metric.replace('_', ' ').title()} Comparison")
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_compare_bar.png"))
        plt.close()

if __name__ == "__main__":
    mia_dir = "MIA"  # Folder containing *_attack_acc_all.csv and *_auc_all.csv
    metrics = ["attack_acc", "auc"]

    results = summarize_mia_results(mia_dir, metrics)
    plot_summary(results, output_dir=mia_dir)
    print(f"✅ Summary CSVs and plots saved in {mia_dir}/")
