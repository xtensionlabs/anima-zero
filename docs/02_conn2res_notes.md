# Reading notes: conn2res (Suárez et al., *Nature Communications* 2024)

Paper: "Connectome-based reservoir computing with the conn2res toolbox", Suárez, Mihalik,
Milisav, Marshall, Li, Vértes, Lajoie, Misic. Nat Commun 15 (2024). DOI 10.1038/s41467-024-44900-4.
Open access (PMC10803782). Code: github.com/netneurolab/conn2res, BSD-3. Tutorial data on Zenodo
10.5281/zenodo.10205004.

## What the paper does

conn2res treats a connectome as the fixed recurrent layer of a reservoir computer:

```
task input -> input nodes -> connectome reservoir (fixed weights, chosen dynamics) -> readout nodes -> linear model
```

Five stages: fetch task, set connectivity matrix, simulate dynamics, fit the linear readout,
score. Design choices at each stage:

- Connectome: binary or weighted, directed or undirected, any species or imaging modality.
- Dynamics: discrete-time echo-state units (tanh, sigmoid, ReLU, leaky ReLU), a continuous-time
  spiking model, or memristive devices for physical reservoirs.
- Global regime: the matrix is scaled so its spectral radius is alpha. "Stable if alpha < 1,
  chaotic if alpha > 1, critical when alpha is about 1."
- Input and readout node sets are chosen anatomically (sensory in, motor out; or one intrinsic
  network as readout at a time).
- Tasks: memory capacity, plus the 20+ NeuroGym cognitive tasks (perceptual decision making,
  context-dependent decision making, delayed match, ...).
- Control: "randomly rewired null connectomes with preserved density and degree sequence."

## Key results, with numbers

1. Regime is task-dependent. Perceptual decision making (needs memory) degrades from stable to
   chaotic dynamics with tanh units. Context-dependent decision making (needs separability)
   improves with chaotic dynamics.
2. Human connectome vs rewired nulls on memory capacity: "At criticality (alpha = 1), empirical
   networks perform significantly better than rewired nulls (p = 0.002)."
3. Regional specialisation: with visual cortex as input, readouts from different intrinsic
   networks differ strongly (F = 1143.5, p < 0.002); default mode and somatomotor read out best.
4. Cross-species memory capacity, empirical vs rewired nulls: mouse p < 0.002, rat p < 0.002,
   macaque p < 0.002, **fruit fly p = 0.11 (not significant)**, at their respective peak alpha.

## Why this matters for anima-zero

**It is the methodology we were missing.** Our phase-0 finding was that the fly wiring carries
specific programs (escape reflex survives only with real wiring) while, as a generic reservoir,
fly.ai found no advantage over scrambled wiring. conn2res gives that second observation a
rigorous form: a task battery, a tunable dynamical regime, anatomical input and readout sets, and
degree-preserving nulls with significance tests. Three things follow.

1. **Their fruit fly is not our fruit fly.** conn2res used a mesoscale, region-level fly network
   derived from FlyCircuit light-microscopy data (Chiang et al. 2011, *Curr. Biol.*; brain regions
   as nodes, on the order of 50; exact count to confirm in their Zenodo data.zip), the only
   species in their table where topology gave no significant benefit. MaleCNS is synaptic resolution:
   166,700 nodes, 25 M signed edges, sides, and 11,862 cell types. Re-running their memory-
   capacity and NeuroGym protocol on the real synaptic connectome is a direct, publishable
   extension, and it answers whether the null result was about flies or about resolution.

2. **Their alpha is our gain.** flybrain's `gain` scales synaptic drive exactly as conn2res's
   spectral radius does. fly.ai calibrated gain by hand for a quiet resting state; conn2res says
   to sweep it and expect the useful regime near criticality, and to expect the best regime to
   differ by task. Our stimulation experiments should report where on that curve they sit.

3. **Readout-set heterogeneity is a fly question too.** Their strongest result is that *where you
   read out* matters more than the global topology test. The fly analogue is cheap: same visual
   input, readouts from descending neurons vs mushroom body output neurons vs central complex vs
   ventral nerve cord intrinsic neurons. If the descending-neuron bottleneck is the best readout
   for motor-shaped tasks and the mushroom body for memory-shaped tasks, that is functional
   specialisation demonstrated at synaptic resolution, and it tells the LLM-cortex design where to
   plug in.

## Caveats the authors state, and ours

- Evidence for "topology matters" rests entirely on the rewired null; it says the arrangement
  beats a random arrangement with the same degrees, not why.
- No spatial embedding, conduction delays, or wiring cost.
- Exploratory by design; results vary with task and regime.
- Ours: conn2res uses rate (echo-state) units; flybrain is spiking. To make the comparison clean,
  phase 2b should add a rate-mode reservoir over the same MaleCNS matrix, then run both.

## Concrete changes to the plan

- New hypothesis **H6**: at synaptic resolution, the fly connectome beats degree-preserving nulls
  on memory capacity near criticality (reversing conn2res's p = 0.11), and the effect is carried by
  specific sub-circuits, so it disappears when readouts are restricted to random neuron sets.
- New experiment `03_reservoir_battery.py`: memory capacity + two NeuroGym tasks; inputs at
  visual projection neurons; readouts at descending neurons, MBONs, central complex, and random
  sets; alpha (gain) sweep; 20 rewired nulls; both spiking and rate modes.
- Adopt conn2res's reporting conventions (alpha on the x-axis, nulls as distributions, one panel
  per readout set) so results are comparable with the paper's figures.
