"""Offline test of the BrainLab tool layer, and the ground truth for hypothesis H4.

Traces the two textbook pathways an LLM should be able to rediscover on its own:
  looming detectors LC4 / LPLC2  ->  giant fiber DNp01  (escape)
  small-target detector LC10a    ->  DNa02              (steering toward a target)
Then stimulates LC4 on the left and lets the lab report which descending neurons change.
Writes results/02_pathways.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anima import BrainLab  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)

t0 = time.perf_counter()
lab = BrainLab()
print(f"BrainLab ready in {time.perf_counter() - t0:.1f} s: {lab.overview()['cell_types']:,} types, "
      f"{lab.overview()['descending_neuron_types']} descending types")

report = {"overview": lab.overview()}
report["search_DNp"] = lab.search_types("DNp", limit=8)
print("search 'DNp':", [t["type"] for t in report["search_DNp"]["types"]])

report["describe_LC4"] = lab.describe_type("LC4")
d = report["describe_LC4"]
print(f"LC4: {d['neurons']} neurons {d['sides']} {d['sign']}; downstream:",
      [(x["type"], x["input_fraction"]) for x in d["strongest_downstream"][:5]])

for src, dst in [("LC4", "DNp01"), ("LPLC2", "DNp01"), ("LC10a", "DNa02")]:
    r = lab.trace_paths(src, dst)
    report[f"path_{src}_{dst}"] = r
    print(f"{src} -> {dst}: direct {r['direct_input_fraction']} ({r['direct_sign']}); "
          f"best routes: {[p['route'] for p in r['paths'][:3]]}")

r = lab.stimulate(["LC4"], side="L", amount=0.8, steps=50, watch=["DNp01", "DNp02", "DNa02"])
report["stim_LC4_L"] = r
print(f"stimulate LC4 L ({r['wall_seconds']} s wall): watched {r['watched']}")
print("most changed descending:", [(x["type"], x["side"], x["delta_hz"]) for x in r["most_changed_descending_neurons"][:6]])

lab.set_scrambled(True)
r2 = lab.stimulate(["LC4"], side="L", amount=0.8, steps=50, watch=["DNp01"])
report["stim_LC4_L_scrambled"] = r2
print("scrambled control, DNp01:", r2["watched"])
lab.set_scrambled(False)

(OUT / "02_pathways.json").write_text(json.dumps(report, indent=2))
print(f"wrote {OUT / '02_pathways.json'}")
