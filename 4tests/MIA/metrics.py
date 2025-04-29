import os
import pandas as pd
import numpy as np

# ------------------------
# Settings
# ------------------------
OUTPUT_DIR = "./plot"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Algorithm names and their folders
algorithms = ["neggem", "negagem", "RL-GEM", "RL-AGEM", "ALT-NEGGEM", "neggrad"]
data_dirs = [f"./accuracies/{algo}" for algo in algorithms]  # assuming each algo folder inside "./plot/"

# New Task Sequence (learn 0-10, forget 10-1)
TASK_COUNT = 11
TASK_SEQUENCE = [0,1,2,3,4,5,6,7,8,9,10,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1]

# ------------------------
# Helper function
# ------------------------
def process_csv(csv_path):
    df = pd.read_csv(csv_path, header=None)
    df = df.iloc[:, 1:1+TASK_COUNT]  # Only task columns
    df.columns = [f"Task {i}" for i in range(TASK_COUNT)]
    df = df.iloc[1:1+len(TASK_SEQUENCE)].reset_index(drop=True)

    retain_set = set()
    forget_set = set(df.columns)
    retain_avgs = []
    forget_avgs = []

    for t, task_index in enumerate(TASK_SEQUENCE):
        task_name = f"Task {abs(task_index)}"

        if task_index >= 0:
            retain_set.add(task_name)
            forget_set.discard(task_name)
        else:
            forget_set.add(task_name)
            retain_set.discard(task_name)

        row = df.iloc[t]
        if retain_set:
            retain_avg = row[list(retain_set)].mean()
        else:
            retain_avg = np.nan
        forget_avg = row[list(forget_set)].mean()

        retain_avgs.append(retain_avg)
        forget_avgs.append(forget_avg)

    return np.nanmean(retain_avgs), np.nanmean(forget_avgs)

# ------------------------
# Collect data
# ------------------------
records = []

for algo, folder in zip(algorithms, data_dirs):
    # Find the single CSV
    files = [f for f in os.listdir(folder) if f.endswith("TASK_ACCURACIES_AVERAGED.csv")]
    if len(files) != 1:
        print(f"❌ Error: Expected 1 CSV in {folder}, found {len(files)}")
        continue

    csv_path = os.path.join(folder, files[0])
    retain_mean, forget_mean = process_csv(csv_path)

    records.append({
        "Algorithm": algo,
        "Mean Retain Accuracy": round(retain_mean * 100, 2),
        "Mean Forget Accuracy": round(forget_mean * 100, 2)
    })

# ------------------------
# Save results
# ------------------------
df_results = pd.DataFrame(records)
results_path = os.path.join(OUTPUT_DIR, "retain_forget_summary.csv")
df_results.to_csv(results_path, index=False)

print(f"✅ Results saved to {results_path}")
print(df_results)
