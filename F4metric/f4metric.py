#!/usr/bin/env python3
"""
compare_relearning.py
─────────────────────
For each 30×20 accuracy CSV that follows task-sequence
[ 0-9 , –9…–5 , 10-19 , 5-9 ] this script

1.  finds the first row where each task 5-9 is *learned*
    (first positive occurrence in the sequence);
2.  finds the row where the same task is *re-learned*
    (first positive after its negative row);
3.  plots the two accuracies side-by-side.

It also emits a grand‐average plot across all CSVs so you
can see overall relearning behaviour.
"""

import glob, os, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------ configuration --------------------------------------------------
TASK_SEQ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
           -9,-8,-7,-6,-5, 10,11,12,13,14,15,16,17,18,19,
            5, 6, 7, 8, 9]                              # 30 rows
FORGET   = {5, 6, 7, 8, 9}

# Pre-compute (row index → task id, sign)
row2tid  = [abs(t) for t in TASK_SEQ]
row2sign = [ 1 if t>=0 else -1 for t in TASK_SEQ]

def two_events(col: int) -> tuple[int,int]:
    """row indices of first learn and first relearn for *task col*"""
    first, forget, second = None, None, None
    for r, tid in enumerate(row2tid):
        if tid != col:        # not this task
            continue
        if row2sign[r] > 0 and first is None:
            first = r
        elif row2sign[r] < 0:         # negative row
            forget = r
        elif row2sign[r] > 0 and forget is not None:
            second = r
            break
    return first, second

# ------------ per-file processing -------------------------------------------
stats   = {t: [] for t in FORGET}     # for global mean plot
files   = sorted(glob.glob("*.csv"))

for csv in files:
    df = pd.read_csv(csv, index_col=0)        # 30×20
    before, after = [], []                    # one value per task

    for t in FORGET:
        r1, r2 = two_events(t)
        before.append(df.loc[r1, str(t)])
        after .append(df.loc[r2, str(t)])
        stats[t].append( (df.loc[r1, str(t)], df.loc[r2, str(t)]) )

    # ── bar chart for this run ───────────────────────────────────────────
    width = 0.35
    idx   = np.arange(len(FORGET))

    plt.figure(figsize=(6,3))
    plt.bar(idx - width/2, before, width, label="initial")
    plt.bar(idx + width/2, after,  width, label="re-learn")
    plt.xticks(idx, [f"{t}" for t in FORGET])
    plt.ylabel("accuracy")
    plt.ylim(0,1)
    plt.title(f"{csv} – tasks 5-9")
    plt.legend()
    out = pathlib.Path(csv).stem + "_relearn.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print("saved", out)

# ------------ aggregate plot (per-CSV summary WITH σ bars) -------------------
labels = []
mean_init,  std_init  = [], []
mean_fgt,   std_fgt   = [], []
mean_rel,   std_rel   = [], []

for csv in files:
    df = pd.read_csv(csv, index_col=0)

    # helper: first / second time task appears positively
    def first_and_second(t: int) -> tuple[int, int]:
        pos = [i for i, x in enumerate(TASK_SEQ) if x ==  t]
        return pos[0], pos[1]                     # learn-row, re-learn-row

    # ❶ initial learning
    init_vals = np.array([df.loc[first_and_second(t)[0], str(t)] for t in FORGET], float)
    mean_init.append(init_vals.mean())
    std_init .append(init_vals.std(ddof=1))

    # ❷ just after last unlearn (row with “-5”)
    r_fgt = max(i for i, x in enumerate(TASK_SEQ) if x == -5)
    f_vals = df.loc[r_fgt, [str(t) for t in FORGET]].to_numpy(float)
    mean_fgt.append(f_vals.mean())
    std_fgt .append(f_vals.std(ddof=1))

    # ❸ re-learn
    rel_vals = np.array([df.loc[first_and_second(t)[1], str(t)] for t in FORGET], float)
    mean_rel.append(rel_vals.mean())
    std_rel .append(rel_vals.std(ddof=1))

    labels.append(pathlib.Path(csv).stem)

# --------------------- grouped bar-chart -------------------------------------
width = 0.25
idx   = np.arange(len(labels))

plt.figure(figsize=(1.6*len(labels)+2, 4.6))
plt.bar(idx - width, mean_init, width,
        yerr=std_init, capsize=4, label="initial")
plt.bar(idx         , mean_fgt , width,
        yerr=std_fgt, capsize=4, label="after forget")
plt.bar(idx + width, mean_rel , width,
        yerr=std_rel, capsize=4, label="re-learn")

plt.xticks(idx, labels, rotation=30, ha="right", fontsize=8)
plt.ylabel("mean accuracy  (tasks 5–9)")
plt.ylim(0, 1)
plt.title("Initial vs Forgotten vs Re-learn accuracy (±1 σ)")
plt.legend()
plt.tight_layout()
plt.savefig("aggregate_per_run.png", dpi=300)
plt.close()
print("saved aggregate_per_run.png")