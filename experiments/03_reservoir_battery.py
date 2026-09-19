"""Phase 2a: the conn2res protocol on the full synaptic fly connectome (hypothesis H6).

For each network (real MaleCNS + degree-preserving nulls), each dynamical regime (alpha for the
rate reservoir, gain for the spiking one) and each task, drive visual projection neurons with the
task input and fit a ridge readout from several anatomical populations at once.

    python experiments/03_reservoir_battery.py --quick            # 2-3 min pipeline check
    python experiments/03_reservoir_battery.py                    # full run (hours, background)
    python experiments/03_reservoir_battery.py --mode spiking

Results are appended to results/03_reservoir_battery.json after every run, so a partial run is
still usable. Plot with experiments/plot_03.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from flybrain import FlyBrain  # noqa: E402
from anima.reservoir import (TASKS, input_map, readout_sets, run_rate, run_spiking,  # noqa: E402
                             scramble, spectral_radius)

p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true")
p.add_argument("--mode", choices=["rate", "rate_raw", "spiking", "all"], default="all")
p.add_argument("--nulls", type=int, default=8)
p.add_argument("--alphas", type=float, nargs="*", default=[0.3, 0.6, 0.9, 1.0, 1.1, 1.3, 1.6],
               help="spectral radii for rate mode (conn2res protocol)")
p.add_argument("--raw-gains", type=float, nargs="*", default=[0.5, 1.0, 2.0, 4.0, 8.0],
               help="multipliers on the per-neuron-normalised matrix for rate_raw mode")
p.add_argument("--gains", type=float, nargs="*", default=[1.0, 2.0, 3.0, 4.0, 5.0], help="spiking synaptic gain")
p.add_argument("--tasks", nargs="*", default=["memory_capacity", "perceptual_decision", "context_decision"])
p.add_argument("--spiking-tasks", nargs="*", default=["memory_capacity"])
p.add_argument("--decision-alphas", type=float, nargs="*", default=[0.6, 0.9, 1.1, 1.4])
p.add_argument("--decision-nulls", type=int, default=5)
p.add_argument("--out", default=str(ROOT / "results" / "03_reservoir_battery.json"))
args = p.parse_args()

if args.quick:
    args.nulls, args.decision_nulls = 1, 1
    args.alphas, args.raw_gains, args.decision_alphas, args.gains = [0.9, 1.1], [1.0], [1.0], [3.0]
    TASK_KW = {"memory_capacity": dict(steps=400, k_max=20, washout=50),
               "perceptual_decision": dict(n_trials=40), "context_decision": dict(n_trials=40)}
else:
    TASK_KW = {"memory_capacity": dict(steps=1200, k_max=60, washout=100),
               "perceptual_decision": dict(n_trials=160), "context_decision": dict(n_trials=160)}

OUT = Path(args.out)
OUT.parent.mkdir(exist_ok=True)
results = json.loads(OUT.read_text()) if OUT.exists() and not args.quick else {"runs": [], "networks": {}}


def save():
    OUT.write_text(json.dumps(results, indent=1))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- brain, matrices, readouts ---------------------------------------------------------------------
brain = FlyBrain(device="cpu", sensory_input=False, seed=0)
n = brain.n
W_real = sparse.csc_matrix((brain.weights, brain.indices, brain.indptr), shape=(n, n))
readouts = readout_sets(brain)
pool = brain.cells(["visual_projection"])
log(f"{n:,} neurons; readouts: " + ", ".join(f"{k}={len(v)}" for k, v in readouts.items()))
results["readouts"] = {k: int(len(v)) for k, v in readouts.items()}
results["config"] = vars(args) | {"task_kw": TASK_KW}

networks = [("real", W_real)] + [(f"null_{s}", scramble(W_real, seed=s)) for s in range(1, args.nulls + 1)]


def already(**key) -> bool:
    return any(all(r.get(k) == v for k, v in key.items()) for r in results["runs"])


def fmt(r: dict) -> str:
    return " ".join(f"{k.replace('mushroom_body_output', 'mbon').replace('central_complex', 'cx').replace('descending', 'dn')}"
                    f"={v['score']:.3f}" for k, v in r.items() if k != "_state")


# ---- rate modes ----------------------------------------------------------------------------------------
csr_cache: dict = {}
if args.mode in ("rate", "rate_raw", "all"):
    for name, Wc in networks:
        csr_cache[name] = Wc.tocsr()
        if name not in results["networks"] or "rho" not in results["networks"][name]:
            t0 = time.perf_counter()
            rho = spectral_radius(csr_cache[name])
            results["networks"][name] = {"rho": rho, "nnz": int(csr_cache[name].nnz)}
            log(f"{name}: spectral radius {rho:.4f} ({time.perf_counter() - t0:.0f} s)")
            save()

for mode in [m for m in ("rate", "rate_raw") if args.mode in (m, "all")]:
    for task_name in args.tasks:
        if mode == "rate":
            regimes = args.alphas if task_name == "memory_capacity" else args.decision_alphas
        else:
            regimes = args.raw_gains
            if task_name != "memory_capacity":
                regimes = [g for g in regimes if g in (1.0, 4.0)]      # keep the raw decision sweep short
        nets = networks if task_name == "memory_capacity" else networks[: 1 + args.decision_nulls]
        task = TASKS[task_name](**TASK_KW[task_name])
        C = task.inputs.shape[1]
        imap = input_map(pool, C, per_channel=600 if C == 1 else 150)
        for name, _ in nets:
            rho = results["networks"][name]["rho"] if mode == "rate" else 1.0
            for alpha in regimes:
                if already(mode=mode, task=task_name, network=name, alpha=alpha):
                    continue
                r = run_rate(csr_cache[name], alpha, rho, task, imap, readouts)
                results["runs"].append({"mode": mode, "task": task_name, "network": name, "alpha": alpha, "result": r})
                save()
                log(f"{mode:8s} {task_name:20s} {name:7s} a={alpha:<4} {fmt(r)}  |x|={r['_state']['mean_abs_activity']:.3f} "
                    f"sat={r['_state']['saturated_fraction']:.2f} {r['_state']['ms_per_step']} ms/step")

# ---- spiking mode ------------------------------------------------------------------------------------
if args.mode in ("spiking", "all"):
    orig_idx, orig_w = brain.indices, brain.weights
    for task_name in args.spiking_tasks:
        task = TASKS[task_name](**TASK_KW[task_name])
        C = task.inputs.shape[1]
        imap = input_map(pool, C, per_channel=600 if C == 1 else 150)
        for name, Wc in networks:
            brain.indices, brain.weights = (orig_idx, orig_w) if name == "real" else (Wc.indices, Wc.data)
            for gain in args.gains:
                if already(mode="spiking", task=task_name, network=name, gain=gain):
                    continue
                r = run_spiking(brain, gain, task, imap, readouts)
                results["runs"].append({"mode": "spiking", "task": task_name, "network": name, "gain": gain, "result": r})
                save()
                log(f"spiking  {task_name:20s} {name:7s} g={gain:<4} {fmt(r)}  rate={r['_state']['mean_rate_hz']:.2f}Hz "
                    f"{r['_state']['ms_per_step']} ms/step")
    brain.indices, brain.weights = orig_idx, orig_w
    brain.gain = 3.0

log(f"done: {len(results['runs'])} runs in {OUT}")
