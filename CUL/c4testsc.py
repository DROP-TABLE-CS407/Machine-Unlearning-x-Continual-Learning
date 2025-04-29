# import os
# import glob
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # ------------------------
# # Settings
# # ------------------------
# C4_DIR = "4tests/C4v2"
# A4_DIR = "4tests/A4"
# ALGORITHMS = ["NegGEM", "NegAGEM", "RL-GEM", "RL-AGEM", "ALT-NEGGEM", "NegGrad+"]
# OUT_DIR = "./cul_analysis_outputs"
# # os.makedirs(OUT_DIR, exist_ok=True)

# # TASKS_FIRST5 = list(range(0, 5))      # Tasks 0 to 4
# # TASKS_LAST10 = list(range(10, 20))    # Tasks 10 to 19

# # # ------------------------
# # # Helper to load correct CSV
# # # ------------------------
# # def load_algo_df(folder, algo):
# #     pattern = os.path.join(folder, algo, "*ALL_TASK_ACCURACIES_AVERAGED.csv")
# #     files = glob.glob(pattern)
# #     if len(files) != 1:
# #         raise ValueError(f"❌ Expected exactly one file for {algo} in {folder}, but found {len(files)}: {files}")
# #     return pd.read_csv(files[0], index_col=0)

# # # ------------------------
# # # Extract final accuracies
# # # ------------------------
# # def extract_final_acc(df, tasks, final_iter):
# #     final_row = df.loc[final_iter]
# #     return final_row[tasks].mean(), final_row[tasks].std()

# # # ------------------------
# # # Collect results
# # # ------------------------
# # records = []

# # for algo in ALGORITHMS:
# #     # Load files
# #     df_c4 = load_algo_df(C4_DIR, algo)
# #     df_a4 = load_algo_df(A4_DIR, algo)
    
# #     # C4
# #     final_iter_c4 = df_c4.index.max()  # Should be 24
# #     c4_first5_mean, c4_first5_std = extract_final_acc(df_c4, TASKS_FIRST5, final_iter_c4)
# #     c4_last10_mean, c4_last10_std = extract_final_acc(df_c4, TASKS_LAST10, final_iter_c4)
    
# #     # A4
# #     final_iter_a4 = 19  # Only up to learning, before forgetting starts
# #     a4_first5_mean, a4_first5_std = extract_final_acc(df_a4, TASKS_FIRST5, final_iter_a4)
# #     a4_last10_mean, a4_last10_std = extract_final_acc(df_a4, TASKS_LAST10, final_iter_a4)

# #     records.append({
# #         "Algorithm": algo,
# #         "C First5 Mean": round(c4_first5_mean*100, 2),
# #         "C First5 Std" : round(c4_first5_std*100, 2),
# #         "C Last10 Mean"  : round(c4_last10_mean*100, 2),
# #         "C Last10 Std"   : round(c4_last10_std*100, 2),
# #         "A First5 Mean": round(a4_first5_mean*100, 2),
# #         "A First5 Std" : round(a4_first5_std*100, 2),
# #         "A Last10 Mean"  : round(a4_last10_mean*100, 2),
# #         "A Last10 Std"   : round(a4_last10_std*100, 2),
# #     })

# # # ------------------------
# # # Make final dataframe
# # # ------------------------
# # df_results = pd.DataFrame(records)

# # # Save table

# # results_path = os.path.join(OUT_DIR, "cul_final_accuracy_summary_first5_last10.csv")
# # df_results.to_csv(results_path, index=False)


# # print(f"✅ Saved results table: {results_path}")

# # ------------------------
# # Single plot with error bars
# # ------------------------
# fig, ax = plt.subplots(figsize=(14,7))
# width = 0.2
# x = np.arange(len(ALGORITHMS))

# # Plot 4 bars per algorithm
# ax.bar(x - 1.5*width, df_results["C First5 Mean"], width, 
#        yerr=df_results["C First5 Std"], capsize=5, label="C (First 5)", color="skyblue")

# ax.bar(x - 0.5*width, df_results["A First5 Mean"], width, 
#        yerr=df_results["A First5 Std"], capsize=5, label="A (First 5)", color="lightcoral")

# ax.bar(x + 0.5*width, df_results["C Last10 Mean"], width, 
#        yerr=df_results["C Last10 Std"], capsize=5, label="C (Last 10)", color="lightgreen")

# ax.bar(x + 1.5*width, df_results["A Last10 Mean"], width, 
#        yerr=df_results["A Last10 Std"], capsize=5, label="A (Last 10)", color="lightsalmon")

# # Labels
# ax.set_xticks(x)
# ax.set_xticklabels(df_results["Algorithm"], rotation=45)
# ax.set_ylabel("Final Accuracy (%)")
# ax.set_title("Final Accuracy Comparison Across Buffer Configurations (First 5 and Last 10 Tasks)")
# ax.legend()
# ax.grid(alpha=0.3)

# plt.tight_layout()
# plt.savefig(os.path.join(OUT_DIR, "full_accuracy_comparison_first5_last10.png"))
# plt.close()

# print(f"✅ Saved single combined plot to {OUT_DIR}/full_accuracy_comparison_first5_last10.png")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Settings ---
CSV_PATH = "./cul_analysis_outputs/cul_final_accuracy_summary_first5_last10.csv"
OUT_DIR = "./cul_analysis_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Load saved results ---
df_results = pd.read_csv(CSV_PATH)

# --- Plotting ---
fig, ax = plt.subplots(figsize=(14,7))
width = 0.2
x = np.arange(len(df_results))

# Plot 4 bars per algorithm
ax.bar(x - 1.5*width, df_results["C First5 Mean"], width, 
       yerr=df_results["C First5 Std"], capsize=5, label="C (First 5)", color="skyblue")

ax.bar(x - 0.5*width, df_results["A First5 Mean"], width, 
       yerr=df_results["A First5 Std"], capsize=5, label="A (First 5)", color="lightcoral")

ax.bar(x + 0.5*width, df_results["C Last10 Mean"], width, 
       yerr=df_results["C Last10 Std"], capsize=5, label="C (Last 10)", color="lightgreen")

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
plt.savefig(os.path.join(OUT_DIR, "full_accuracy_comparison_first5_last10_FROM_CSV.png"))
plt.close()

print(f"✅ Replotted from CSV and saved to {OUT_DIR}/full_accuracy_comparison_first5_last10_FROM_CSV.png")
