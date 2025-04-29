#!/usr/bin/env python3
"""
first_learn_ma_only.py
──────────────────────
One shared figure showing *only* the 3-point moving-average of the first-learn
accuracies for each algorithm (tasks 0‥19).

Y-axis ticks are fixed at 0.10 steps.
"""

from __future__ import annotations
import pathlib, itertools
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, ticker

# ─────────── configuration ───────────
TASK_SEQUENCE: List[int] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
   10,11,12,13,14,15,16,17,18,19
]

CSV_FILES: Dict[str, str] = {
    "GEM" : "GEM1.csv",
    "AGEM": "AGEM.csv",
    # add / remove as needed
}

MOV_AVG_K  = 5
OUT_DIR    = pathlib.Path("plots_first_learn")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ─────────── helpers ───────────
def first_row_indices() -> List[int]:
    """Row index where each task is first learned."""
    idx = {}
    for r, tid in enumerate(TASK_SEQUENCE):
        if tid >= 0 and tid not in idx:
            idx[tid] = r
    return [idx[t] for t in range(len(idx))]

FIRST_IDX = first_row_indices()

def first_learn_vector(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path, index_col=0)
    acc = [float(df.iloc[r, t]) for t, r in enumerate(FIRST_IDX)]
    return np.asarray(acc, dtype=float)

def moving_avg(x: np.ndarray, k: int = MOV_AVG_K) -> np.ndarray:
    if k <= 1:
        return x
    pad = (k - 1) // 2
    xp  = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, np.ones(k)/k, mode="valid")[: len(x)]

COLOR = itertools.cycle(cm.get_cmap("tab10", 10).colors)

# ─────────── plotting ───────────
def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x       = np.arange(len(FIRST_IDX))

    for label, path in CSV_FILES.items():
        y   = first_learn_vector(path)
        col = next(COLOR)

        # -- raw markers (optional) ----------------------------------------
        # ax.plot(x, y, "-o", lw=.8, ms=3, alpha=.25, color=col)

        # moving average only
        ax.plot(x, moving_avg(y), lw=2.5, color=col, label=label)

    ax.set_xlabel("Task ID")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.10))
    ax.grid(alpha=.3, linestyle="--", linewidth=.5)
    ax.set_title("First-learn accuracy (5-pt moving average)")
    ax.legend(fontsize=9, ncols=2)
    fig.tight_layout()

    out = OUT_DIR / "first_learn_moving_average.png"
    fig.savefig(out, dpi=600, transparent=True)
    print("saved", out)


if __name__ == "__main__":
    main()
