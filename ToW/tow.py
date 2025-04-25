"""
Compute Tug-of-War (ToW) scores for four continual-unlearning runs that share
an identical task curriculum.

Input CSVs (all eight files must be present in the working directory)
--------------------------------------------------------------------
* **neggemtrain.csv**   - training-set accuracies for *NegGEM*
* **neggemtest.csv**    -   test-set accuracies for *NegGEM*
* **rlgemtrain.csv**    - training-set accuracies for *RL-GEM*
* **rlgemtest.csv**     -   test-set accuracies for *RL-GEM*
* **negagemtrain.csv**  - training-set accuracies for *NegAGEM*
* **negagemtest.csv**   -   test-set accuracies for *NegAGEM*
* **neggradtrain.csv**  - training-set accuracies for *NegGrad+*
* **neggradtest.csv**   -   test-set accuracies for *NegGrad+*

Rows 0-19  : model still “original”          » ToW = 1.0
Rows 20-38 : unlearning iterations commence  » ToW computed vs. baseline (row 19)

Columns
-------
Task-ID columns must be integers 0-19.  Any extra columns (e.g. “avg”) are
ignored; the mean over task columns is used as the overall accuracy.
"""

from __future__ import annotations

import os
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------
#  Task curriculum (edit here only if your curriculum differs)
# -------------------------------------------------------------
TASK_SEQUENCE: List[int] = [
     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10,11,12,13,14,15,16,17,18,19,
    -19,-18,-17,-16,-15,-14,-13,-12,
    -11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1
]

# -------------------------------------------------------------
#  CSV-file mapping (label ➜ {test, train})
# -------------------------------------------------------------
CSV_FILES: Dict[str, Dict[str, str]] = {
    "NegGEM":   {"test": "neggemtest.csv",   "train": "neggemtrain.csv"},
    "RL-GEM":   {"test": "rlgemtest.csv",    "train": "rlgemtrain.csv"},
    "NegAGEM":  {"test": "negagemtest.csv",  "train": "negagemtrain.csv"},
    "RL-AGEM":  {"test": "rlagemtest.csv",   "train": "rlagemtrain.csv"},
    "ALT-NEGGEM": {"test": "altneggemtest.csv", "train": "altneggemtrain.csv"},
    "NegGrad+ (alpha = 0.50)": {"test": "neggradtest050.csv",  "train": "neggradtrain050.csv"},
    "NegGrad+ (alpha = 0.95)": {"test": "neggradtest095.csv",  "train": "neggradtrain095.csv"},
}

# ————————————————————————————————————————————————————————————————
#  Utility helpers
# ————————————————————————————————————————————————————————————————

def _split_sets(iter_idx: int) -> Tuple[Set[int], Set[int]]:
    """Return (S, R) after iteration *iter_idx*.

    S - set of *forgotten* tasks (should be empty until unlearning begins)
    R - set of *retain*   tasks (learned but not forgotten)
    """
    learned, forgotten = set(), set()
    for idx in range(iter_idx + 1):
        tid = TASK_SEQUENCE[idx]
        (learned if tid >= 0 else forgotten).add(abs(tid))
    return forgotten, learned - forgotten


def _mean_abs_diff(cur: pd.Series, base: pd.Series, cols: Set[int]) -> float:
    """Mean |cur - base| over *cols* (0.0 if *cols* is empty)."""
    return 0.0 if not cols else float(np.abs(cur[list(cols)] - base[list(cols)]).mean())

# ————————————————————————————————————————————————————————————————
#  Core computation
# ————————————————————————————————————————————————————————————————

def calculate_tow_single(train_csv: str, test_csv: str) -> List[float]:
    """Return a complete ToW trajectory for one run."""

    if not (os.path.isfile(train_csv) and os.path.isfile(test_csv)):
        raise FileNotFoundError(f"Missing file(s): {train_csv} / {test_csv}")

    df_tr = pd.read_csv(train_csv, index_col=0)  # training accuracies
    df_te = pd.read_csv(test_csv,  index_col=0)  # test accuracies

    if df_tr.shape != df_te.shape:
        raise ValueError(f"Shape mismatch between {train_csv} and {test_csv}")

    task_cols = [c for c in df_tr.columns if str(c).lstrip("-").isdigit()]
    if len(task_cols) < 20:
        raise ValueError(
            f"Expected at least 20 task columns (0-19). Found only: {task_cols}"
        )

    tow: List[float] = []

    for i in range(len(df_tr)):
        if i <= 18:  # original-model window - score fixed at 1.0
            tow.append(1.0)
            continue

        S, R = _split_sets(i)
        row_tr, row_te = df_tr.iloc[i], df_te.iloc[i]
        base_tr, base_te = df_tr.iloc[38 - i], df_te.iloc[38 - i]

        da_s = _mean_abs_diff(row_tr, base_tr, S)
        da_r = _mean_abs_diff(row_tr, base_tr, R)

        retain_cols = [t for t in R if str(t) in row_te.index]
        da_test = (
            abs(row_te[retain_cols].mean() - base_te[retain_cols].mean()) if retain_cols else 0.0
        )

        tow.append((1 - da_s) * (1 - da_r) * (1 - da_test))

    return tow


def calculate_tow_all(csv_files: Dict[str, Dict[str, str]]) -> Dict[str, List[float]]:
    """Compute ToW trajectories for every configured run."""
    return {label: calculate_tow_single(p["train"], p["test"]) for label, p in csv_files.items()}

# ————————————————————————————————————————————————————————————————
#  Plotting
# ————————————————————————————————————————————————————————————————

def plot_tow(tow_scores: Dict[str, List[float]]) -> None:
    plt.figure(figsize=(12, 6))
    for label, scores in tow_scores.items():
        # Only show unlearning iterations (20 onwards)
        plt.plot(range(20, len(scores)), scores[20:], label=label, linewidth=2)

    plt.axhline(1.0, color="k", ls="--", lw=1, label="Ideal (ToW = 1)")
    plt.xlabel("Iteration")
    plt.ylabel("ToW score")
    plt.title("Continual-Unlearning Tug-of-War - Four Methods")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("tow_scores_all_methods.png", dpi=600, transparent=True)
    plt.show()

# ————————————————————————————————————————————————————————————————
#  Main entry point
# ————————————————————————————————————————————————————————————————

def main() -> None:
    print("Computing Tug-of-War scores …")
    tow_scores = calculate_tow_all(CSV_FILES)

    if tow_scores:
        plot_tow(tow_scores)

        # Export per-run CSVs (optional, comment out if not needed)
        for label, scores in tow_scores.items():
            out = f"ToW_scores_{label.replace(' ', '_').replace('+', 'plus')}.csv"
            pd.DataFrame({"ToW": scores}).to_csv(out, index_label="iter")
            print(f"  saved {out}")


if __name__ == "__main__":
    main()
