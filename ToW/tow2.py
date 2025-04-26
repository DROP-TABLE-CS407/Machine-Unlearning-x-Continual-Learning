"""
Compute Tug‑of‑War (ToW) curves for **six** continual‑unlearning methods from
previously saved score files and visualise them on a single plot.

Expected input files (produced by *tow_scores_six_methods.py*)
-------------------------------------------------------------
* `ToW_scores_NegGEM.csv`
* `ToW_scores_RL-GEM.csv`
* `ToW_scores_NegAGEM.csv`
* `ToW_scores_NegGradplus.csv`   ← note the “plus” in the filename

Each CSV contains one column `ToW` indexed by iteration (0–38).

Only iterations 20–38 are plotted, because that is where unlearning happens and
ToW values depart from the ideal 1·0 baseline.
"""

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────────
# Mapping *display‑label* → filename on disk
TOW_CSVS: Dict[str, str] = {
    "NegGEM":   "ToW_scores_NegGEM.csv",
    "RL-GEM":   "ToW_scores_RL-GEM.csv",
    "NegAGEM":  "ToW_scores_NegAGEM.csv",
    "RL-AGEM":  "ToW_scores_RL-AGEM.csv",
    "ALT-NEGGEM": "ToW_scores_ALT-NEGGEM.csv",
    "NegGrad+ (alpha = 0.5)": "ToW_scores_NegGradplus_(alpha_=_0.50).csv",  # “+” was replaced by “plus”
    "NegGrad+ (alpha = 0.95)": "ToW_scores_NegGradplus_(alpha_=_0.95).csv",  # “+” was replaced by “plus”
}

# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_tow_scores(csv_map: Dict[str, str]) -> Dict[str, List[float]]:
    """Load ToW trajectories from the given mapping of labels → CSV path."""
    scores: Dict[str, List[float]] = {}
    for label, path in csv_map.items():
        if not os.path.isfile(path):
            print(f"⚠️  {path} not found – skipping {label}.")
            continue
        df = pd.read_csv(path, index_col=0)
        if "ToW" not in df.columns:
            print(f"⚠️  {path} lacks a 'ToW' column – skipping {label}.")
            continue
        scores[label] = df["ToW"].tolist()
        print(f"  loaded {path}  ({len(df)} iterations)")
    return scores

# ──────────────────────────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_tow(tow_scores: Dict[str, List[float]]) -> None:
    if not tow_scores:
        print("No valid ToW data loaded – nothing to plot.")
        return

    plt.figure(figsize=(12, 6))
    for label, scores in tow_scores.items():
        # plot only iterations 20 → end
        plt.plot(range(20, len(scores)), scores[20:], label=label, linewidth=2)

    # ideal baseline (ToW = 1)
    plt.axhline(1.0, color="k", ls="--", lw=1, label="Ideal (ToW = 1)")

    # integer ticks on the x‑axis
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

    plt.xlabel("Iteration")
    plt.ylabel("ToW score")
    plt.title("Continual‑Unlearning Tug‑of‑War – Six Methods")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("tow_scores_six_methods_plot.png", dpi=600, transparent=True)
    plt.show()

# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Reading pre‑computed Tug‑of‑War scores …")
    tow_scores = load_tow_scores(TOW_CSVS)

    print("Plotting …")
    plot_tow(tow_scores)


if __name__ == "__main__":
    main()
