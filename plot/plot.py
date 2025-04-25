import os
import pandas as pd
import matplotlib.pyplot as plt

# Define your directory and CSV filter
DATA_DIR = "./plot"  # current directory
OUTPUT_DIR = "./plot"
TASK_COUNT = 20
TASK_SEQUENCE = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1
][:38]  # Trimmed to match 38 data rows

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Loop through CSVs
for file in os.listdir(DATA_DIR):
    if file.endswith(".csv"):
        try:
            # Load file and ignore first column (index/time)
            df = pd.read_csv(os.path.join(DATA_DIR, file), header=None)
            df = df.iloc[:, 1:1 + TASK_COUNT]  # Columns 1–20 → Task 1–20
            df.columns = [f"Task {i+1}" for i in range(TASK_COUNT)]

            # Drop the first row (initialization), reset index
            df = df.iloc[1:39].reset_index(drop=True)

            # Initialize retain/forget logic
            retain_set = set()
            forget_set = set(df.columns)
            avg_retain_filtered = []
            avg_forget_filtered = []
            filtered_steps = []

            for t, task_index in enumerate(TASK_SEQUENCE):
                task_name = f"Task {abs(task_index) + 1}"

                if task_index >= 0:
                    retain_set.add(task_name)
                    forget_set.discard(task_name)
                else:
                    forget_set.add(task_name)
                    retain_set.discard(task_name)

                if forget_set:
                    row = df.iloc[t]
                    retain_avg = row[list(retain_set)].mean() if retain_set else None
                    forget_avg = row[list(forget_set)].mean()
                    avg_retain_filtered.append(retain_avg)
                    avg_forget_filtered.append(forget_avg)
                    filtered_steps.append(t)

            # Plot
            plt.figure(figsize=(10, 5))
            plt.plot(filtered_steps, avg_retain_filtered, label="Retain Set Average", marker='o')
            plt.plot(filtered_steps, avg_forget_filtered, label="Forget Set Average", marker='o')
            plt.title(f"Retain vs Forget Accuracy")
            plt.xlabel("Task Sequence Step")
            plt.ylabel("Average Accuracy")
            plt.ylim(0.0, 1.0)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            # Save plot
            base_name = os.path.splitext(file)[0]
            plot_path = os.path.join(OUTPUT_DIR, f"{base_name}.png")
            plt.savefig(plot_path, dpi=600, transparent=True)
            plt.close()
            print(f"✅ Saved: {plot_path}")

        except Exception as e:
            print(f"❌ Failed to process {file}: {e}")