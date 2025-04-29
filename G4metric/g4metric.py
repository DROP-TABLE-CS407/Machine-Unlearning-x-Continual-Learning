"""
compare_A4_vs_G4.py
───────────────────
For every CSV pair
    <Algorithm>_A4.csv   and   <Algorithm>_G4.csv
in the current directory:

  • take the last row of each file
  • average its last five columns
  • compute Δ = mean_G4 − mean_A4

All Δ-values are shown in a single bar chart.
"""

from __future__ import annotations
import pathlib, re, sys
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

DIR = pathlib.Path(".")                         # change if needed
CSV_RE = re.compile(r"(?P<alg>.+)_(?P<phase>A4|G4)\.csv$", re.I)

# ───────────────────── collect file paths ──────────────────────
pairs: Dict[str, Dict[str, pathlib.Path]] = defaultdict(dict)

for p in DIR.glob("*.csv"):
    m = CSV_RE.match(p.name)
    if m:
        alg, phase = m["alg"], m["phase"].upper()
        pairs[alg][phase] = p

# keep only algorithms that have both files
pairs = {alg: d for alg, d in pairs.items() if {"A4", "G4"} <= d.keys()}
if not pairs:
    sys.exit("No matching A4 / G4 csv pairs found.")

# ───────────────────── compute means & deltas ──────────────────
results: Dict[str, Tuple[float, float, float]] = {}   # alg → (mean_A4, mean_G4, Δ)

for alg, d in pairs.items():
    mean_vals = {}
    for phase in ("A4", "G4"):
        df      = pd.read_csv(d[phase], index_col=0)
        lastrow = df.iloc[-1, -5:]                    # last 5 columns of last row
        mean_vals[phase] = float(lastrow.mean())
    delta = mean_vals["G4"] - mean_vals["A4"]
    results[alg] = (mean_vals["A4"], mean_vals["G4"], delta)

# ───────────────────── plotting ────────────────────────────────
algs   = list(results.keys())
deltas = [results[a][2] for a in algs]

fig, ax = plt.subplots(figsize=(0.9*len(algs)+2, 4))
bars = ax.bar(algs, deltas, color="steelblue", alpha=.8)

ax.set_ylabel(r"$\;\bar{a}^{\mathrm{G4}} - \bar{a}^{\mathrm{A4}}$")
ax.set_title("Average accuracy difference on tasks 15–19\n(last row, last five columns)")
ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.grid(axis="y", alpha=.3, ls="--")

for bar, d in zip(bars, deltas):
    ax.annotate(f"{d:+.3f}",
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8)

fig.tight_layout()
fig.savefig("A4_vs_G4_delta_barchart.png", dpi=300, transparent=True)
print("saved  A4_vs_G4_delta_barchart.png")