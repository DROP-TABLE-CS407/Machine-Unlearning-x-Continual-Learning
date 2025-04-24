import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

# def summarize_mia_results(mia_dir="MIA", metrics=["attack_acc", "auc"]):
#     summary_data = {metric: [] for metric in metrics}

#     for metric in metrics:
#         # Find all matching CSV files
#         csv_files = glob.glob(os.path.join(mia_dir, f"*_{metric}_all.csv"))

#         for csv_path in csv_files:
#             algo = os.path.basename(csv_path).replace(f"_{metric}_all.csv", "")
#             df = pd.read_csv(csv_path)

#             if metric not in df.columns or len(df[metric]) == 0:
#                 continue

#             avg = df[metric].mean()
#             std = df[metric].std()
#             summary_data[metric].append({
#                 "algorithm": algo,
#                 "mean": round(avg, 4),
#                 "std": round(std, 4)
#             })

#     return summary_data

# def plot_summary(summary_data, output_dir="MIA"):
#     os.makedirs(output_dir, exist_ok=True)

#     for metric, records in summary_data.items():
#         df = pd.DataFrame(records).sort_values("mean", ascending=False)
#         df.to_csv(os.path.join(output_dir, f"{metric}_summary.csv"), index=False)

#         # Bar plot with error bars
#         plt.figure(figsize=(10, 5))
#         plt.bar(df["algorithm"], df["mean"], yerr=df["std"], capsize=5, color='cornflowerblue')
#         plt.xlabel("Algorithm")
#         plt.ylabel(f"{metric.replace('_', ' ').title()} (Avg ± Std)")
#         plt.title(f"MIA {metric.replace('_', ' ').title()} Comparison")
#         plt.ylim(0, 1)
#         plt.grid(axis='y', linestyle='--', alpha=0.6)
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, f"{metric}_compare_bar.png"))
#         plt.close()


import os
import pandas as pd
import matplotlib.pyplot as plt

def load_avg_mia_results(folder_path: str):
    algo_names, pre_vals, post_vals = [], [], []

    for file in os.listdir(folder_path):
        if file.endswith("_mia_results.csv"):
            algo = file.replace("_mia_results.csv", "")
            df = pd.read_csv(os.path.join(folder_path, file))

            # Sanity check: filter just in case
            df_pre = df[df["stage"] == "pre"]
            df_post = df[df["stage"] == "post"]

            if not df_pre.empty and not df_post.empty:
                pre_avg = df_pre["attack_acc"].mean()
                post_avg = df_post["attack_acc"].mean()
                algo_names.append(algo)
                pre_vals.append(pre_avg)
                post_vals.append(post_avg)

    return algo_names, pre_vals, post_vals

def plot_avg_attack_acc(algo_names, pre_vals, post_vals):
    x = np.arange(len(algo_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, pre_vals, width, label="Pre-unlearning")
    bars2 = ax.bar(x + width/2, post_vals, width, label="Post-unlearning")

    ax.set_ylabel("Avg MIA Attack Accuracy")
    ax.set_title("Avg Pre vs Post MIA Attack Accuracy per Algorithm")
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names, rotation=45)
    ax.axhline(0.5, linestyle='--', color='gray', label='Random Guess (0.5)')
    ax.legend()
    ax.grid(True, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.savefig("MIA5/bars.png")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    mia_dir = "MIA3"  # Folder containing *_attack_acc_all.csv and *_auc_all.csv
    metrics = ["attack_acc", "auc"]

    # results = summarize_mia_results(mia_dir, metrics)
    # plot_summary(results, output_dir=mia_dir)
    # print(f"Summary CSVs and plots saved in {mia_dir}/")

    # Usage
    folder_path = "./MIA5"  # ⬅️ change to your actual path
    algos, pre, post = load_avg_mia_results(folder_path)
    plot_avg_attack_acc(algos, pre, post)