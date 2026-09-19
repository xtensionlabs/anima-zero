"""Does the wiring matter? Real connectome vs degree-preserving scramble.

fly.ai reported that, as a generic reservoir, the real wiring carried no more information
than scrambled wiring. Here we ask the opposite question at the level of a specific circuit:
does the looming (LC4/LPLC2) -> giant fiber (DNp01) escape reflex survive scrambling?

Scramble: permute the postsynaptic targets of all edges. Every neuron keeps its exact
out-degree, the multiset of in-degrees is preserved, weights and signs stay attached to
their presynaptic neuron. Only *who talks to whom* is destroyed.
Writes results/01_scramble_control.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flybrain import FlyBrain

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)


def reflex(brain: FlyBrain, side: str, amount: float = 0.8, steps: int = 50, seed: int = 2) -> dict:
    brain.reset(seed=seed)
    loom = brain.cells(["LC4", "LPLC2"], side=side)
    gf = {s: set(brain.cells(["DNp01"], side=s).tolist()) for s in "LR"}
    dn_all = set(brain.cells(["descending_neuron"]).tolist())
    hits = {s: 0 for s in "LR"}
    dn_spikes = 0
    for _ in range(steps):
        fired = set(brain.step(inject=[(loom, amount)]).tolist())
        for s in "LR":
            hits[s] += len(fired & gf[s])
        dn_spikes += len(fired & dn_all)
    return {"DNp01_L": hits["L"], "DNp01_R": hits["R"],
            "all_DN_spikes_per_s": dn_spikes / (steps * brain.dt) / len(dn_all)}


def lateralization(r_left: dict, r_right: dict) -> float:
    """1 = perfectly ipsilateral, 0 = no side preference, nan = nothing fired."""
    same = r_left["DNp01_L"] + r_right["DNp01_R"]
    other = r_left["DNp01_R"] + r_right["DNp01_L"]
    return float("nan") if same + other == 0 else (same - other) / (same + other)


brain = FlyBrain(device="cpu", sensory_input=False, seed=0)
report = {}

real = {"L": reflex(brain, "L"), "R": reflex(brain, "R")}
report["real"] = {**real, "lateralization": lateralization(real["L"], real["R"])}
print(f"real      loomL->DNp01 L/R {real['L']['DNp01_L']}/{real['L']['DNp01_R']}   "
      f"loomR->DNp01 L/R {real['R']['DNp01_L']}/{real['R']['DNp01_R']}   "
      f"lateralization {report['real']['lateralization']:.2f}")

rng = np.random.default_rng(0)
original = brain.indices.copy()
for trial in range(3):
    brain.indices = original[rng.permutation(len(original))]
    scr = {"L": reflex(brain, "L"), "R": reflex(brain, "R")}
    lat = lateralization(scr["L"], scr["R"])
    report[f"scrambled_{trial}"] = {**scr, "lateralization": lat}
    print(f"scramble{trial} loomL->DNp01 L/R {scr['L']['DNp01_L']}/{scr['L']['DNp01_R']}   "
          f"loomR->DNp01 L/R {scr['R']['DNp01_L']}/{scr['R']['DNp01_R']}   lateralization {lat:.2f}   "
          f"all-DN rate {scr['L']['all_DN_spikes_per_s']:.2f} Hz")
brain.indices = original

(OUT / "01_scramble_control.json").write_text(json.dumps(report, indent=2))
print(f"wrote {OUT / '01_scramble_control.json'}")
