"""
Make four separate plots per method:
  • CLU_mix(t)           ->  <label>_CLU_mix.png
  • 1 - L_ret(t)         ->  <label>_keep.png
  • 1 - L_res(t)         ->  <label>_forget.png
  • B(t)                 ->  <label>_bonus.png

Each task may flip between retain / forget later in TASK_SEQUENCE;
_state_sets() tracks only the most-recent sign.
"""

from __future__ import annotations
import os, pathlib
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ───────────────────────── configuration ─────────────────────────────────────
TASK_SEQUENCE: List[int] = [
     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10,11,12,13,14,15,16,17,18,19,
    -19,-18,-17,-16,-15,-14,-13,-12,
    -11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1
]      # as before
CSV_FILES: Dict[str, Dict[str, str]] = {
    "NegGEM":   {"test": "neggemtest.csv",   "train": "neggemtrain.csv"},
    "RL-GEM":   {"test": "rlgemtest.csv",    "train": "rlgemtrain.csv"},
    "NegAGEM":  {"test": "negagemtest.csv",  "train": "negagemtrain.csv"},
    "RL-AGEM":  {"test": "rlagemtest.csv",   "train": "rlagemtrain.csv"},
    "ALT-NEGGEM": {"test": "altneggemtest.csv", "train": "altneggemtrain.csv"},
    "NegGrad+ α=0.50": {"test": "neggradtest050.csv",  "train": "neggradtrain050.csv"},
    # "NegGrad+ α=0.95": {"test": "neggradtest095.csv",  "train": "neggradtrain095.csv"},
}        # as before
BETA          = 0.0
RANDOM_GUESS  = 0.20
OUT_DIR       = pathlib.Path("plots")
OUT_DIR.mkdir(exist_ok=True)

# ───────────────────────── helpers ───────────────────────────────────────────
# ───────────────────────── Sets that can flip sign ───────────────────────────
def _state_sets(i: int) -> Tuple[Set[int], Set[int]]:
    """For iteration i return (forget S, retain R) using the *last* sign seen."""
    state = {}                                  # tid → last sign (±1)
    for k in range(i + 1):
        tid = TASK_SEQUENCE[k]
        state[abs(tid)] = 1 if tid >= 0 else -1
    S = {t for t, s in state.items() if s < 0}
    R = {t for t, s in state.items() if s > 0}
    return S, R

# ───────────────────────── anchor helper (unchanged) ─────────────────────────
def _anchors(df: pd.DataFrame) -> pd.Series:
    seen, anchor = set(), {}
    for r, tid in enumerate(TASK_SEQUENCE):
        if tid >= 0 and tid not in seen:
            anchor[tid] = df.iloc[r][str(tid)]
            seen.add(tid)
        if len(seen) == 20:
            break
    return pd.Series(anchor)

# ───────────────────────── per-run component time-series ─────────────────────
def one_run(train_csv: str) -> Dict[str, List[float]]:
    df   = pd.read_csv(train_csv, index_col=0)
    anch = _anchors(df)

    hist = dict(clu=[], keep=[], forget=[], bonus=[])

    for i in range(len(df)):
        S, R = _state_sets(i)

        # retention + bonus
        if R:
            row_R, anc_R = df.iloc[i][[str(t) for t in R]].to_numpy(float), anch[list(R)].to_numpy(float)
            L_ret = np.maximum(0.0, anc_R - row_R).mean()
            B     = np.maximum(0.0, row_R - anc_R).mean()
        else:
            L_ret = B = 0.0

        # residual forgetting
        if S:
            row_S  = df.iloc[i][[str(t) for t in S]].to_numpy(float)
            L_res  = np.maximum(0.0, row_S - RANDOM_GUESS).mean()
        else:
            L_res = 0.0

        P   = (1.0 - L_ret) * (1.0 - L_res)
        clu = (P + BETA * B) / (1.0 + BETA)

        hist["clu"].append(clu)
        hist["keep"].append(1.0 - L_ret)
        hist["forget"].append(1.0 - L_res)
        hist["bonus"].append(B)

    return hist

def all_runs(csv_map: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, List[float]]]:
    return {lbl: one_run(p["train"]) for lbl, p in csv_map.items()}

# ───────────────────────── plotting – four PNGs total ────────────────────────
def save_four_plots(all_scores: Dict[str, Dict[str, List[float]]]) -> None:
    comp_names = dict(clu   ="CLU$_{\\text{mix}}$",
                      keep  ="$1-L_{\\text{ret}}$",
                      forget="$1-L_{\\text{res}}$",
                      bonus ="$B$")

    for key, title in comp_names.items():
        plt.figure(figsize=(10, 5))
        for label, series in all_scores.items():
            plt.plot(series[key], lw=2, label=label)
        plt.ylim(0, 1)
        plt.xlabel("Iteration")
        plt.ylabel(title)
        plt.title(f"{title}  (β={BETA})")
        plt.grid(alpha=.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        out_png = f"{key}.png"
        plt.savefig(out_png, dpi=600, transparent=True)
        plt.close()
        print("saved", out_png)

# ───────────────────────── main ──────────────────────────────────────────────
def main() -> None:
    scores = all_runs(CSV_FILES)

    # CSV dump for numeric inspection
    for label, comp in scores.items():
        pd.DataFrame(comp).to_csv(f"{label.replace(' ','_')}_components.csv",
                                  index_label="iter")

    save_four_plots(scores)

if __name__ == "__main__":
    main()