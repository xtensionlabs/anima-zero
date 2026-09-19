"""Figure for the reservoir battery: one row per (mode, task), one column per anatomical readout.

x = dynamical regime (alpha for rate mode, gain for spiking), y = task score.
Blue = real connectome read out from the anatomical population; orange = real connectome read
out from a size-matched random population; grey band = degree-preserving nulls (min to max, line
= mean) read out from the anatomical population. Also writes a CSV twin of every point.

    python experiments/plot_03.py [results/03_reservoir_battery.json]
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "03_reservoir_battery.json"
data = json.loads(SRC.read_text())

# palette (dataviz reference, light mode)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
REAL, RANDOM, NULL = "#2a78d6", "#eb6834", "#898781"

ANATOMICAL = ["descending", "mushroom_body_output", "central_complex"]
TITLES = {"descending": "Descending neurons", "mushroom_body_output": "Mushroom body output (MBON)",
          "central_complex": "Central complex"}
YLABEL = {"memory_capacity": "memory capacity (sum R²)", "perceptual_decision": "accuracy",
          "context_decision": "accuracy"}

# group runs: (mode, task) -> network -> x -> result
groups: dict = defaultdict(lambda: defaultdict(dict))
for r in data["runs"]:
    x = r.get("alpha", r.get("gain"))
    groups[(r["mode"], r["task"])][r["network"]][x] = r["result"]
MODE_ORDER = {"rate": 0, "rate_raw": 1, "spiking": 2}
rows = sorted(groups, key=lambda k: (MODE_ORDER[k[0]], ["memory_capacity", "perceptual_decision", "context_decision"].index(k[1])))
sizes = data["readouts"]

fig, axes = plt.subplots(len(rows), len(ANATOMICAL), figsize=(4.2 * len(ANATOMICAL), 3.1 * len(rows)),
                         squeeze=False, facecolor=SURFACE)
csv_rows = []
for i, (mode, task) in enumerate(rows):
    nets = groups[(mode, task)]
    xs = sorted(nets["real"])
    for j, ro in enumerate(ANATOMICAL):
        ax = axes[i, j]
        ax.set_facecolor(SURFACE)
        rnd = f"random_{sizes[ro]}"
        real = [nets["real"][x][ro]["score"] for x in xs]
        rand = [nets["real"][x][rnd]["score"] for x in xs]
        null_names = [n for n in nets if n.startswith("null")]
        if null_names:
            N = np.array([[nets[n][x][ro]["score"] if x in nets[n] else np.nan for x in xs] for n in null_names])
            ax.fill_between(xs, np.nanmin(N, 0), np.nanmax(N, 0), color=NULL, alpha=0.18, linewidth=0)
            ax.plot(xs, np.nanmean(N, 0), color=NULL, linewidth=2, label=f"nulls, mean of {len(null_names)} ({ro.replace('_', ' ')})")
            for n, row in zip(null_names, N):
                for x, v in zip(xs, row):
                    csv_rows.append([mode, task, n, x, ro, v])
        ax.plot(xs, rand, color=RANDOM, linewidth=2, marker="o", markersize=5, label=f"real, random {sizes[ro]} neurons")
        ax.plot(xs, real, color=REAL, linewidth=2, marker="o", markersize=5, label=f"real, {ro.replace('_', ' ')}")
        for x, v, w in zip(xs, real, rand):
            csv_rows += [[mode, task, "real", x, ro, v], [mode, task, "real", x, rnd, w]]
        if task != "memory_capacity":
            ax.axhline(0.5, color=AXIS, linewidth=1)
            ax.set_ylim(0.3, 1.02)
        if i == 0:
            ax.set_title(f"{TITLES[ro]} (n = {sizes[ro]})", fontsize=11, color=INK, loc="left")
        if j == 0:
            ax.set_ylabel(f"{task.replace('_', ' ')}\n{YLABEL[task]}", color=INK2, fontsize=9)
        ax.set_xlabel({"rate": "spectral radius α (conn2res scaling)", "rate_raw": "gain on per-neuron-normalised weights",
                       "spiking": "synaptic gain (spiking)"}[mode], color=INK2, fontsize=9)
        if mode == "rate_raw":
            ax.set_xscale("log")
        ax.grid(True, color=GRID, linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8)
        if mode in ("rate", "rate_raw"):
            ax.axvline(1.0, color=AXIS, linewidth=1)
        if i == 0 and j == len(ANATOMICAL) - 1:
            ax.legend(fontsize=7.5, frameon=False, loc="best", labelcolor=INK2)

fig.suptitle("MaleCNS as a reservoir: real connectome vs degree-preserving nulls, by readout population",
             fontsize=12, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
png = SRC.with_suffix(".png")
fig.savefig(png, dpi=150, facecolor=SURFACE)
with open(SRC.with_suffix(".csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["mode", "task", "network", "regime", "readout", "score"])
    w.writerows(csv_rows)
print(f"wrote {png} and {SRC.with_suffix('.csv')}")
