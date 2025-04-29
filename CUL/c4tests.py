import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Settings
# ------------------------
C4_DIR = "4tests/C4"
A4_DIR = "4tests/A4"
ALGORITHMS = ["NegGEM", "NegAGEM", "RL-GEM", "RL-AGEM", "ALT-NEGGEM", "NegGrad+"]
OUT_DIR = "./cul_analysis_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

TASKS_FIRST10 = list(range(0, 10))
TASKS_LAST5   = list(range(15, 20))

# ------------------------
# Helper to load correct CSV
# ------------------------
def load_algo_df(folder, algo):
    pattern = os.path.join(folder, algo, "*ALL_TASK_ACCURACIES_AVERAGED.csv")
    files = glob.glob(pattern)
    if len(files) != 1:
        raise ValueError(f"❌ Expected exactly one file for {algo} in {folder}, but found {len(files)}: {files}")
    return pd.read_csv(files[0], index_col=0)

# ------------------------
# Extract final accuracies
# ------------------------
def extract_final_acc(df, tasks, final_iter):
    final_row = df.loc[final_iter]
    return final_row[tasks].mean(), final_row[tasks].std()

# ------------------------
# Collect results
# ------------------------
records = []

for algo in ALGORITHMS:
    # Load files
    df_c4 = load_algo_df(C4_DIR, algo)
    df_a4 = load_algo_df(A4_DIR, algo)
    
    # C4
    final_iter_c4 = df_c4.index.max()  # Should be 24
    c4_first10_mean, c4_first10_std = extract_final_acc(df_c4, TASKS_FIRST10, final_iter_c4)
    c4_last5_mean,  c4_last5_std  = extract_final_acc(df_c4, TASKS_LAST5, final_iter_c4)
    
    # A4
    final_iter_a4 = 19  # Only up to learning, before forgetting starts
    a4_first10_mean, a4_first10_std = extract_final_acc(df_a4, TASKS_FIRST10, final_iter_a4)
    a4_last5_mean,  a4_last5_std  = extract_final_acc(df_a4, TASKS_LAST5, final_iter_a4)

    records.append({
        "Algorithm": algo,
        "C4 First10 Mean": round(c4_first10_mean*100, 2),
        "C4 First10 Std" : round(c4_first10_std*100, 2),
        "C4 Last5 Mean"  : round(c4_last5_mean*100, 2),
        "C4 Last5 Std"   : round(c4_last5_std*100, 2),
        "A4 First10 Mean": round(a4_first10_mean*100, 2),
        "A4 First10 Std" : round(a4_first10_std*100, 2),
        "A4 Last5 Mean"  : round(a4_last5_mean*100, 2),
        "A4 Last5 Std"   : round(a4_last5_std*100, 2),
    })

# ------------------------
# Make final dataframe
# ------------------------
df_results = pd.DataFrame(records)

# Save table
results_path = os.path.join(OUT_DIR, "cul_final_accuracy_summary.csv")
df_results.to_csv(results_path, index=False)

print(f"✅ Saved results table: {results_path}")

# ------------------------
# Single plot with error bars
# ------------------------
fig, ax = plt.subplots(figsize=(14,7))
width = 0.2
x = np.arange(len(ALGORITHMS))

# Plot 4 bars per algorithm
ax.bar(x - 1.5*width, df_results["C First5 Mean"], width, 
       yerr=df_results["C First5 Std"], capsize=5, label="Cv2 (First 5)", color="skyblue")

ax.bar(x - 0.5*width, df_results["A First5 Mean"], width, 
       yerr=df_results["A First5 Std"], capsize=5, label="A (First 5)", color="lightcoral")

ax.bar(x + 0.5*width, df_results["C Last10 Mean"], width, 
       yerr=df_results["C Last10 Std"], capsize=5, label="Cv2 (Last 10)", color="lightgreen")

ax.bar(x + 1.5*width, df_results["A Last10 Mean"], width, 
       yerr=df_results["A Last10 Std"], capsize=5, label="A (Last 10)", color="lightsalmon")

# Labels
ax.set_xticks(x)
ax.set_xticklabels(df_results["Algorithm"], rotation=45)
ax.set_ylabel("Final Accuracy (%)")
ax.set_title("Final Accuracy Comparison Across Buffer Configurations (First 5 and Last 10 Tasks)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "full_accuracy_comparison.png"))
plt.close()

print(f"✅ Saved single combined plot to {OUT_DIR}/full_accuracy_comparison.png")
