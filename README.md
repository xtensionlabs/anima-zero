# anima-zero

An exploration of bio-digital AI: a language model fused with the complete wiring diagram of a
male fruit fly's nervous system (MaleCNS v1.0, HHMI Janelia FlyEM + Google Research, *Cell* 2026).

The fly does robust real-time behaviour with 166,700 neurons and a few microwatts. Modern AI gets
there by brute force. This project asks what actually changes when a real brain's wiring becomes a
component of an AI system, and measures it instead of assuming it.

Read `docs/00_thesis.md` first, then `docs/01_landscape.md` for what already exists.

## Layout

```
anima/          the bridge: BrainLab (connectome as tools) + agent (Claude as scientist/cortex)
experiments/    numbered, reproducible scripts; each writes results/NN_*.json
results/        outputs of the experiments (committed so the numbers in the docs are traceable)
docs/           thesis, landscape survey, and later findings
```

## Setup

Python 3.12 on a laptop CPU is enough for everything in phase 0 and 1.

```sh
pip install -r requirements.txt
flybrain download          # 260 MB prebuilt MaleCNS brain files into ~/fly-data (once)
python experiments/00_smoke.py
```

`flybrain build` instead rebuilds those files from the raw Janelia release (1.1 GB) if you want
to audit the pipeline from electron-microscopy tables to matrix.

## The bridge

```python
from anima import BrainLab
lab = BrainLab()                              # ~5 s
lab.trace_paths("LC4", "DNp01")               # anatomy: strongest routes looming -> escape
lab.stimulate(["LC4"], side="L", watch=["DNp01"])   # physiology: 1 s of drive vs baseline
lab.set_scrambled(True)                       # degree-preserving control wiring
```

With `ANTHROPIC_API_KEY` set:

```sh
python -m anima.agent "How does the fly escape a looming object? Prove it by experiment."
```

Claude gets these tools over the live simulation and is instructed to trace anatomy, test by
stimulation, and run the scrambled control before claiming a mechanism.

## Results so far

| Experiment | Finding |
|---|---|
| `00_smoke` | Whole CNS runs at 1.0x real time on a 6-core laptop; 3.9% of neurons spike per 20 ms; looming left -> left DNp01 48 spikes vs 1 right |
| `01_scramble_control` | Same network with permuted targets: lateralization 0.96 -> ~0 while global activity is unchanged. The reflex is in the wiring, not the statistics |
| `02_pathways` | BrainLab traces LC4/LPLC2 -> DNp01 and LC10a -> DNa02 from anatomy alone; stimulation confirms them; control abolishes them |
| `03_agent_run.log` | Claude (claude-opus-5), given only the six tools, rediscovered the escape circuit unaided in 22 tool calls: anatomy, left/right mirror, LC4 vs LPLC2 dissection, scrambled control (all deltas 0), then correctly flagged that the giant-fiber-to-jump-motor link is electrical and absent from the model. Hypothesis H4 supported |
| `04_comparison.md` | Courtship-song circuit, same question to claude-fable-5-1 and claude-opus-4-8. Both found pC1 -> pIP10 -> TN1a with controls. Opus 4.8 followed the textbook pheromone route, whose physiology failed. Fable 5.1 read the connectome, found the hub's real dominant input (ascending leg-contact neuron AN08B074), proved it with a control (+15 Hz real, +0.3 Hz scrambled), and graded the strength of each link |

## Credits

Connectome: MaleCNS v1.0, CC-BY, Berg et al. 2026 (*Cell*). Simulation layer: `flybrain` by
alextitonis (MIT), neuron model after Fly64 by Jessica Paquette. See `docs/01_landscape.md` for
the research this builds on.
