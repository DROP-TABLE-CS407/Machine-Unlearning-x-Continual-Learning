import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Parameters
ALGOS = ["neggem", "negagem", "RL-GEM", "RL-AGEM", "ALT-NEGGEM", "neggrad"]
TASKS = list(range(1, 9))  # Tasks 1–8
LEARN_ITERS = 10
OUT_DIR = "./mia_outputs_learning_true_auc_iters"
os.makedirs(OUT_DIR, exist_ok=True)

def load_and_label(task, alg):
    df_train = pd.read_csv(f"task{task}_postlearn_{alg}_mia_results.csv")
    df_buff  = pd.read_csv(f"task{task}_postlearn_{alg}_mia_results_buff.csv")
    df_train["dataset"] = "Train"
    df_buff["dataset"]  = "Buffer"
    df_train["task"] = task
    df_buff["task"] = task
    df_train["Algorithm"] = alg
    df_buff["Algorithm"] = alg
    return pd.concat([df_train, df_buff])

# --- Load all ---
dfs = []
for alg in ALGOS:
    for task in TASKS:
        try:
            dfs.append(load_and_label(task, alg))
        except Exception as e:
            print(f"⚠️ Failed to load task {task} for {alg}: {e}")

df_all = pd.concat(dfs, ignore_index=True)

# --- Normalize run_id ---
df_all["run_id"] = (df_all.groupby(["Algorithm", "task", "dataset"])
                          .cumcount() // (LEARN_ITERS + 1))

# Keep only learning phase
df_all = df_all[df_all["iter"] <= LEARN_ITERS]

# --- Aggregate ---
agg = (df_all.groupby(["Algorithm", "dataset", "iter"])
             .agg(auc_mean=("auc", "mean"),
                  auc_std=("auc", "std"))
             .reset_index())

# --- Plot AUC vs Iterations (all on one graph) ---
plt.figure(figsize=(14,8))

for alg in ALGOS:
    for dataset in ["Train", "Buffer"]:
        subset = agg[(agg["Algorithm"] == alg) & (agg["dataset"] == dataset)]
        linestyle = "-" if dataset == "Train" else "--"
        plt.plot(subset["iter"], subset["auc_mean"], linestyle=linestyle, marker="o", label=f"{alg} ({dataset})")

plt.title(f"True MIA AUC during Learning Phase (All Algorithms)", fontsize=16)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Mean AUC", fontsize=14)
plt.grid(alpha=0.3)
plt.xticks(range(0, LEARN_ITERS+1))
plt.ylim(0.4, 1.0)
plt.legend(ncol=2, fontsize=10)
plt.tight_layout()

save_path = os.path.join(OUT_DIR, "true_auc_learning_all_algos.png")
plt.savefig(save_path)
plt.close()

print(f"✅ One single AUC Iteration plot saved to {save_path}")
