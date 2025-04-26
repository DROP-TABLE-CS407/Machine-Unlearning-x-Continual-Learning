"""
Plot CLU_mix(t) for several continual-unlearning runs that share one task
sequence.  Anchors are taken as the accuracy recorded the *first time* a task
is finished (row where that task is first positive in TASK_SEQUENCE).

Files, directory layout and TASK_SEQUENCE follow the Tug-of-War script.
"""

from __future__ import annotations
import os
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
#  Task curriculum (edit only if different)
# ──────────────────────────────────────────────────────────────────────────────
TASK_SEQUENCE: List[int] = [
     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10,11,12,13,14,15,16,17,18,19,
    -19,-18,-17,-16,-15,-14,-13,-12,
    -11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1
]

# ──────────────────────────────────────────────────────────────────────────────
#  CSV mapping  (update to match your filenames)
# ──────────────────────────────────────────────────────────────────────────────
CSV_FILES: Dict[str, Dict[str, str]] = {
    "NegGEM":   {"test": "neggemtest.csv",   "train": "neggemtrain.csv"},
    "RL-GEM":   {"test": "rlgemtest.csv",    "train": "rlgemtrain.csv"},
    "NegAGEM":  {"test": "negagemtest.csv",  "train": "negagemtrain.csv"},
    "RL-AGEM":  {"test": "rlagemtest.csv",   "train": "rlagemtrain.csv"},
    "ALT-NEGGEM": {"test": "altneggemtest.csv", "train": "altneggemtrain.csv"},
    "NegGrad+ α=0.50": {"test": "neggradtest050.csv",  "train": "neggradtrain050.csv"},
    # "NegGrad+ α=0.95": {"test": "neggradtest095.csv",  "train": "neggradtrain095.csv"},
}

# ──────────────────────────────────────────────────────────────────────────────
#  Hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────
BETA        = 0.2   # weight of backward-transfer bonus (0–1)
RANDOM_GUESS = 0.20  # accuracy of random guess (5-way tasks ⇒ 0.2)

# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _split_sets(iter_idx: int) -> Tuple[Set[int], Set[int]]:
    """Return (forget_set S, retain_set R) after TASK_SEQUENCE[iter_idx]."""
    learned, forgotten = set(), set()
    for idx in range(iter_idx + 1):
        tid = TASK_SEQUENCE[idx]
        (learned if tid >= 0 else forgotten).add(abs(tid))
    return forgotten, learned - forgotten


def _compute_anchors(df_train: pd.DataFrame) -> pd.Series:
    """Return per-task anchor = accuracy when task first learned."""
    anchors = {}
    seen: Set[int] = set()
    for row_idx, tid in enumerate(TASK_SEQUENCE):
        if tid >= 0 and tid not in seen:          # first positive occurrence
            anchors[tid] = df_train.iloc[row_idx][str(tid)]
            seen.add(tid)
        if len(seen) == 20:
            break
    return pd.Series(anchors)


def _mean(series: pd.Series, tasks: Set[int]) -> float:
    return 0.0 if not tasks else float(series[[str(t) for t in tasks]].mean())

# ──────────────────────────────────────────────────────────────────────────────
#  CLU_mix computation for a single run
# ──────────────────────────────────────────────────────────────────────────────
def clu_mix_single(train_csv: str) -> List[float]:
    if not os.path.isfile(train_csv):
        raise FileNotFoundError(train_csv)
    df_tr = pd.read_csv(train_csv, index_col=0)
    task_cols = [c for c in df_tr.columns if str(c).isdigit()]
    if len(task_cols) < 20:
        raise ValueError(f"{train_csv}: need task columns 0-19")

    anchors = _compute_anchors(df_tr)

    clu: List[float] = []
    for i in range(len(df_tr)):
        S, R = _split_sets(i)

        # ── Retention / residual losses ────────────────────────────────────
        if R:
            anchor_R = anchors[list(R)].to_numpy(dtype=float)
            row_R    = df_tr.iloc[i][[str(t) for t in R]].to_numpy(dtype=float)

            diff     = anchor_R - row_R
            diff[np.isnan(diff)] = 0.0          # ← clear the NaNs
            L_ret = float(np.maximum(0.0, diff).mean())

            gain     = row_R - anchor_R
            gain[np.isnan(gain)] = 0.0          # ← clear the NaNs
            B = float(np.maximum(0.0, gain).mean())
        else:
            L_ret = B = 0.0 
        if S:
            row_S = df_tr.iloc[i][[str(t) for t in S]].to_numpy(dtype=float)
            diff  = row_S - RANDOM_GUESS
            diff[np.isnan(diff)] = 0.0          # ← clear the NaNs
            L_res = float(np.maximum(0.0, diff).mean())
        else:
            L_res = 0.0

        P = (1.0 - L_ret) * (1.0 - L_res)     # multiplicative penalty

        # ── Positive backward transfer bonus ───────────────────────────────
        # B = 0.0 if not R else float(
        #     np.maximum(0.0, df_tr.iloc[i][[str(t) for t in R]] - anchors[list(R)]).mean()
        # )

        clu_score = (P + BETA * B) / (1.0 + BETA)  # keeps range [0,1]
        clu.append(clu_score)

    return clu

# ──────────────────────────────────────────────────────────────────────────────
#  Wrapper for all runs
# ──────────────────────────────────────────────────────────────────────────────
def clu_mix_all(csv_map: Dict[str, Dict[str, str]]) -> Dict[str, List[float]]:
    return {lbl: clu_mix_single(paths["train"]) for lbl, paths in csv_map.items()}

# ──────────────────────────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────────────────────────
def plot_clu(clu_scores: Dict[str, List[float]]) -> None:
    plt.figure(figsize=(12, 6))
    for label, scores in clu_scores.items():
        plt.plot(range(len(scores)), scores, label=label, lw=2)

    plt.axhline(1.0, color="k", ls="--", lw=1)
    plt.xlabel("Iteration")
    plt.ylabel("CLU$_{\\text{mix}}$ score")
    plt.title(f"Continual-Unlearning CLU$_{{\\text{{mix}}}}$   ($\\beta={BETA}$)")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("CLU_mix_all_methods.png", dpi=600, transparent=True)
    plt.show()

# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Computing CLU_mix trajectories …")
    clu_scores = clu_mix_all(CSV_FILES)
    plot_clu(clu_scores)

    # optional CSV export
    for label, scores in clu_scores.items():
        fname = f"CLU_mix_{label.replace(' ', '_').replace('+','plus')}.csv"
        pd.DataFrame({"CLU_mix": scores}).to_csv(fname, index_label="iter")
        print(" saved", fname)


if __name__ == "__main__":
    main()
