import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- config ---
OUT_DIR = "./mia_outputs_algo_compare"
os.makedirs(OUT_DIR, exist_ok=True)

algorithms = ["neggem", "negagem", "RL-GEM", "RL-AGEM", "ALT-NEGGEM", "neggrad"]
datasets = ["train", "buff"]
tasks = list(range(1, 9))

# --- load everything ---
records = []
for algo in algorithms:
    for dset in datasets:
        for task_id in tasks:
            fname = f"task{task_id}_postunlearn_{algo}_mia_results{'_buff' if dset=='buff' else ''}.csv"
            if not os.path.exists(fname):
                print(f"⚠️ File missing: {fname}")
                continue
            df = pd.read_csv(fname)
            df["algorithm"] = algo
            df["dataset"] = "Training Set" if dset == "train" else "Buffer"
            df["task"] = task_id
            df["target_iter"] = 21 - task_id


            # Only average over numeric columns EXCEPT 'iter'
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != "iter"]

            df_grouped = df.groupby("iter")[numeric_cols].mean().reset_index()


            # Find target
            df_target = df_grouped[df_grouped["iter"] == df_grouped["target_iter"].iloc[0]]

            if df_target.empty:
                print(f"⚠️ No matching iter {21-task_id} after averaging in {fname}")
                continue

            auc = df_target.iloc[0]["auc"]
            records.append({
                "Task": task_id,
                "Algorithm": algo,
                "Dataset": "Training Set" if dset == "train" else "Buffer",
                "Final AUC": auc
            })

df_all = pd.DataFrame(records)
print(df_all.head())

# --- save raw table ---
df_all.to_csv(os.path.join(OUT_DIR, "final_mia_auc_across_algorithms.csv"), index=False)

# --- plot per dataset ---
for dset in ["Training Set", "Buffer"]:
    subset = df_all[df_all["Dataset"] == dset]
    pivot = subset.pivot(index="Task", columns="Algorithm", values="Final AUC")

    pivot.plot(kind="bar", figsize=(14, 6), width=0.8)
    plt.title(f"Final MIA AUC after Unlearning per Task ({dset})")
    plt.ylabel("Final AUC")
    plt.xlabel("Task")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"final_auc_barplot_{dset.replace(' ', '_').lower()}.png"))
    plt.close()

print(f"✅ Per-task bar plots saved to {OUT_DIR}")

# --- mean and std across tasks ---
agg_auc = (
    df_all.groupby(["Algorithm", "Dataset"])
          .agg(mean_final_auc=("Final AUC", "mean"),
               std_final_auc=("Final AUC", "std"))
          .reset_index()
)

# Pivot to nice tables
mean_table = agg_auc.pivot(index="Algorithm", columns="Dataset", values="mean_final_auc")
std_table = agg_auc.pivot(index="Algorithm", columns="Dataset", values="std_final_auc")

mean_table = mean_table[["Training Set", "Buffer"]]  # Reorder nicely
std_table = std_table[["Training Set", "Buffer"]]

# Save tables
mean_table.to_csv(os.path.join(OUT_DIR, "mean_final_auc_per_algorithm.csv"))
std_table.to_csv(os.path.join(OUT_DIR, "std_final_auc_per_algorithm.csv"))

print(mean_table)
print(std_table)

# --- plot mean AUC comparison ---
mean_table.plot(kind="bar", figsize=(10, 6), capsize=3)
plt.title("Mean Final MIA AUC per Algorithm after Unlearning")
plt.ylabel("Mean Final AUC")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "mean_final_auc_barplot.png"))
plt.close()

print(f"✅ Mean and std tables + barplot saved to {OUT_DIR}")
