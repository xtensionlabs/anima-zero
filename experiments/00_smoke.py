"""Phase 0 smoke test: can this laptop run the whole male fruit-fly CNS, and what is in it?

Loads the MaleCNS v1.0 spiking model (flybrain), prints structural facts, times a step,
measures resting activity, and reproduces the looming -> giant-fiber escape reflex.
Writes results/00_smoke.json.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

from flybrain import FlyBrain
from flybrain.data import DATA

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)
report: dict = {}

t0 = time.perf_counter()
brain = FlyBrain(device="cpu", sensory_input=False, seed=0)
report["load_seconds"] = round(time.perf_counter() - t0, 2)
n = brain.n
nnz = len(brain.weights)
print(f"loaded {n:,} neurons, {nnz:,} connections in {report['load_seconds']} s from {DATA}")

# ---- structure -----------------------------------------------------------------------------
W = sparse.csc_matrix((brain.weights, brain.indices, brain.indptr), shape=(n, n))  # rows post, cols pre
out_deg = np.diff(W.indptr)                       # per presynaptic neuron
in_deg = np.bincount(W.indices, minlength=n)      # per postsynaptic neuron
density = nnz / (n * n)
report["structure"] = {
    "neurons": int(n),
    "connections": int(nnz),
    "density": density,
    "mean_out_degree": float(out_deg.mean()),
    "median_out_degree": float(np.median(out_deg)),
    "max_out_degree": int(out_deg.max()),
    "mean_in_degree": float(in_deg.mean()),
    "max_in_degree": int(in_deg.max()),
    "fraction_inhibitory_connections": float((brain.weights < 0).mean()),
    "cell_types": int(len(set(brain.cell_type)) - (1 if "" in set(brain.cell_type) else 0)),
    "superclasses": dict(Counter(brain.superclass.tolist()).most_common()),
}
print(f"density {density:.2e}  mean out-degree {out_deg.mean():.0f}  max {out_deg.max():,}")
print(f"{report['structure']['cell_types']:,} named cell types")
for k, v in report["structure"]["superclasses"].items():
    print(f"  {k:28s} {v:7,}")

# ---- speed ---------------------------------------------------------------------------------
t0 = time.perf_counter()
brain.step()
report["first_step_seconds_incl_jit"] = round(time.perf_counter() - t0, 2)
times = []
fired_counts = []
for _ in range(100):
    t0 = time.perf_counter()
    fired = brain.step()
    times.append(time.perf_counter() - t0)
    fired_counts.append(len(fired))
report["step_ms_median"] = round(1000 * float(np.median(times)), 2)
report["realtime_factor"] = round(brain.dt / float(np.median(times)), 2)   # >1 = faster than real fly
print(f"step: {report['step_ms_median']} ms (dt = {brain.dt*1000:.0f} ms biological) -> "
      f"{report['realtime_factor']}x real time on CPU")

# ---- resting activity ------------------------------------------------------------------------
brain.reset(seed=1)
for _ in range(25):          # settle 0.5 s
    brain.step()
counts = np.zeros(n)
steps = 100                  # 2 s
for _ in range(steps):
    counts[brain.step()] += 1
rate = counts / (steps * brain.dt)
active_frac = float(np.mean(fired_counts)) / n
report["rest"] = {
    "mean_rate_hz": float(rate.mean()),
    "median_rate_hz": float(np.median(rate)),
    "fraction_silent": float((counts == 0).mean()),
    "fraction_neurons_spiking_per_step": active_frac,
    "descending_mean_rate_hz": float(rate[brain.cells(["descending_neuron"])].mean()),
}
print(f"rest: mean {rate.mean():.2f} Hz, {100*(counts==0).mean():.0f}% silent, "
      f"{100*active_frac:.2f}% of neurons spike per 20 ms step, "
      f"descending neurons {report['rest']['descending_mean_rate_hz']:.2f} Hz")

# ---- reflex: looming on the left eye -> left giant fiber (DNp01) ------------------------------
def escape_test(side: str, amount: float = 0.8, steps: int = 50, seed: int = 2) -> dict:
    brain.reset(seed=seed)
    loom = brain.cells(["LC4", "LPLC2"], side=side)
    gf = {s: brain.cells(["DNp01"], side=s) for s in "LR"}
    hits = {s: 0 for s in "LR"}
    for _ in range(steps):
        fired = set(brain.step(inject=[(loom, amount)]).tolist())
        for s in "LR":
            hits[s] += len(fired & set(gf[s].tolist()))
    return {"stim_side": side, "n_loom_neurons": int(len(loom)), "DNp01_spikes": hits}

report["reflex"] = {
    "loom_left": escape_test("L"),
    "loom_right": escape_test("R"),
    "no_stim": escape_test("L", amount=0.0),
}
for k, v in report["reflex"].items():
    print(f"{k:11s} DNp01 spikes L={v['DNp01_spikes']['L']:3d} R={v['DNp01_spikes']['R']:3d}  "
          f"({v['n_loom_neurons']} looming neurons driven)")

(OUT / "00_smoke.json").write_text(json.dumps(report, indent=2))
print(f"wrote {OUT / '00_smoke.json'}")
