#!/usr/bin/env python3
import os, numpy as np, pandas as pd
import glob

# --------------------------------------------------------
# paths / labels
# --------------------------------------------------------
BASE_DIR = "./"                               # adjust if needed
CONFIGS   = {                                 # folder names
    "Random"          : "none1",
    "Most Memorized"  : "most1",
    "Least Memorized" : "least1"
}
OUT_DIR  = "./memory_buffer_summary_final"
os.makedirs(OUT_DIR, exist_ok=True)

TASKS        = list(range(11))  # tasks 0-10
ROWS_PER_RUN = 11               # 11 iterations per run
RUNS         = 3

# --------------------------------------------------------
def load_matrix(folder):
    """return a (33×11) DataFrame of accuracies for tasks 0-10"""
    csv = [f for f in os.listdir(folder) if
           "ALL_TASK_ACCURACIES.csv" in f and "AVERAGED" not in f and "TRAIN" not in f][0]
    df  = pd.read_csv(os.path.join(folder, csv), index_col=0)
    df.columns = df.columns.astype(int)             # make columns int
    return df.loc[:, TASKS]                         # keep tasks 0-10

def split_runs(df):
    """split 33×11 DataFrame into list of 11×11 DataFrames (runs)"""
    return [df.iloc[i*ROWS_PER_RUN:(i+1)*ROWS_PER_RUN].reset_index(drop=True)
            for i in range(RUNS)]

def run_metrics(run_df):
    """compute final accuracy & forgetting for one 11-iteration run"""
    # final accuracy is last row mean over tasks
    final_acc = run_df.iloc[-1].mean()

    # forgetting: for each task t, drop = acc_at_learn - acc_final
    drops = []
    for t in TASKS:
        acc_learn = run_df.loc[t, t]       # accuracy immediately after learning task t
        acc_final = run_df.iloc[-1, t]     # accuracy at end of run
        drops.append(acc_learn - acc_final)
    avg_forget = np.mean(drops)
    return final_acc, avg_forget

# --------------------------------------------------------
summary_rows = []

for label, folder in CONFIGS.items():
    df  = load_matrix(folder)
    runs = split_runs(df)

    finals, forgets = [], []
    for r in runs:
        f_acc, f_drop = run_metrics(r)
        finals.append(f_acc)
        forgets.append(f_drop)

    summary_rows.append({
        "Buffer Config"           : label,
        "Final Avg Acc (%)"       : f"{np.mean(finals)*100:.2f} ± {np.std(finals)*100:.2f}",
        "Avg Forgetting (%)"      : f"{np.mean(forgets)*100:.2f} ± {np.std(forgets)*100:.2f}"
    })

summary = pd.DataFrame(summary_rows)
csv_path = os.path.join(OUT_DIR, "buffer_config_comparison.csv")
summary.to_csv(csv_path, index=False)
print(summary)
print(f"\n✅ Saved summary table to {csv_path}")

# --- Config ---
TASKS = list(range(1, 9))
OUT_DIR = "./buffer_analysis_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Path mapping based on your actual folders
CONFIGS = {
    "Random": "./none/",
    "Least Memorized": "./least/",
    "Most Memorized": "./most/"
}

# --- Helper Functions ---
def find_accuracy_file(folder_path):
    """Find the ALL_TASK_ACCURACIES.csv file inside a folder."""
    files = glob.glob(os.path.join(folder_path, "*ALL_TASK_ACCURACIES.csv"))
    if not files:
        raise FileNotFoundError(f"No ALL_TASK_ACCURACIES.csv file found in {folder_path}")
    return files[0]

def load_matrix(path):
    return pd.read_csv(path, index_col=0)

def split_runs(df):
    """Split into 3 separate runs (each 11 rows)."""
    return [df.iloc[i*11:(i+1)*11].reset_index(drop=True) for i in range(3)]

# --- Storage ---
summary_final = []
summary_forgot = []

# --- Main Loop ---
for buffer_type, folder in CONFIGS.items():
    file_path = find_accuracy_file(folder)
    df = load_matrix(file_path)
    df.columns = df.columns.astype(int)             # make columns int
    runs = split_runs(df)

    # Collect per-task data
    all_final_acc = {task: [] for task in TASKS}
    all_forgetting = {task: [] for task in TASKS}

    for run in runs:
        for task in TASKS:
            acc_learn  = run.loc[task, task]
            acc_final  = run.iloc[-1, task]
            forgetting = acc_learn - acc_final

            all_final_acc[task].append(acc_final * 100)
            all_forgetting[task].append(forgetting * 100)

    # Aggregate mean/std per task
    for task in TASKS:
        summary_final.append({
            "Buffer": buffer_type,
            "Task": task,
            "Mean Final Accuracy": np.mean(all_final_acc[task]),
            "Std Final Accuracy": np.std(all_final_acc[task])
        })
        summary_forgot.append({
            "Buffer": buffer_type,
            "Task": task,
            "Mean Forgetting": np.mean(all_forgetting[task]),
            "Std Forgetting": np.std(all_forgetting[task])
        })

# --- Save results ---
df_summary_final  = pd.DataFrame(summary_final)
df_summary_forgot = pd.DataFrame(summary_forgot)

df_summary_final.to_csv(os.path.join(OUT_DIR, "appendix_final_acc_per_task.csv"), index=False)
df_summary_forgot.to_csv(os.path.join(OUT_DIR, "appendix_forgetting_per_task.csv"), index=False)

print("✅ Appendix tables written:")
print(f"   • {os.path.join(OUT_DIR, 'appendix_final_acc_per_task.csv')}")
print(f"   • {os.path.join(OUT_DIR, 'appendix_forgetting_per_task.csv')}")