# Landscape: what already exists (checked 2026-09-20)

Purpose: know what has been done before we build, so anima-zero extends the field instead of
repeating it. Every claim links to its source. "Unverified" is marked where it applies.

## 1. The data

### MaleCNS v1.0 (the Google / Janelia release we are using)
- Announcement: Google Research blog, 2026-09-03
  (https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/).
  Paper: Berg et al., *Cell* 2026, DOI 10.1016/j.cell.2026.08.015.
- Coverage: the complete central nervous system of one adult male: central brain, both optic
  lobes, and the ventral nerve cord (https://www.janelia.org/project-team/flyem/male-cns-connectome).
- Size: 166,691 neurons, 11,691 cell types, ~125 million synapses (https://male-cns.janelia.org/).
- Versions: v0.9 (Oct 2025), v1.0 (2026-06-08, recommended). neuPrint names `male-cns:v0.9`, `male-cns:v1.0`.
- License: CC-BY (https://male-cns.janelia.org/download/).
- No-login download (Apache Feather) from the public bucket
  `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`:
  - `body-annotations-male-cns-v1.0-minconf-0.5.feather` (14 MB): class, type, side per neuron
  - `body-neurotransmitters-male-cns-v1.0.feather` (41 MB): predicted transmitter per neuron
  - `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (1.0 GB): body-to-body synapse counts
  - `syn-points-...` (12 GB) and `syn-partners-...` (6.3 GB) for synapse-level work
- Also available: neo4j dump, SWC skeletons, EM segmentation.

### FlyWire FAFB v783 (female brain, 2024)
- 139,255 neurons, ~50 M synapses; 2.7 M connections at the 5-synapse threshold
  (https://www.nature.com/articles/s41586-024-07968-y).
- No-login CSVs: `https://storage.googleapis.com/flywire-data/codex/data/fafb/783/{neurons,classification,connections}.csv.gz`
  (connections.csv.gz 48 MB).
- License CC BY-NC 4.0 (https://flywire.ai/guidelines). Useful as a second individual, and the
  male/female pair enables dimorphism comparisons.

### Hemibrain v1.2.1 (2020)
- ~25k neurons, ~20 M synapses, partial brain. `gs://hemibrain/v1.2/`. CC-BY.

### Tooling already on this machine
- `flybrain` 0.1.0 (MIT, github.com/alextitonis/fly.ai): MaleCNS v1.0 as a leaky
  integrate-and-fire network, numba CPU or CuPy GPU. Prebuilt brain files (260 MB) are in
  `~/fly-data`. Its `build` step reproduces them from the Janelia bucket, so the pipeline is
  auditable end to end.
- numpy, scipy, networkx, numba, jax (CPU), matplotlib, anthropic SDK 1.7.

## 2. Whole-brain simulations

| Work | What | Key result | Code |
|---|---|---|---|
| Shiu et al. 2024, *Nature* | LIF model of 127k FlyWire neurons in Brian2; weights = synapse count x transmitter sign x one scalar | Sugar input -> proboscis extension; 91% of 164 predictions matched experiment | github.com/philshiu/Drosophila_brain_model (MIT) |
| Li et al. 2026, bioRxiv | Fit 138k-neuron model to whole-brain calcium imaging by gradient descent | Held-out functional connectivity r = 0.42 (ceiling 0.47) | not stated |
| Sandia, arXiv 2508.16792 | Full 140k LIF on 12 Loihi 2 chips | 3 to 350x faster than Brian2 CPU; needed 9-bit weights, fan-in cap | none |
| fly.ai (flybrain) | MaleCNS LIF as a frozen reservoir; games, "Flytalk" | Looming -> DNp01 +17 to +25 spikes/s ipsilateral. **Negative:** as a reservoir, real wiring carried no more information than scrambled wiring (0.83 vs 0.87 bits) | MIT |

Caveats shared by all of these: no plasticity, no neuromodulation, graded (non-spiking) neurons
such as the lamina are modelled as spiking and fail to relay, and none is validated against
whole-brain recordings except Li et al.

## 3. Connectome-constrained deep networks

- **flyvis** (Lappalainen et al. 2024, *Nature*): 64 optic-lobe cell types on a hexagonal lattice,
  wiring fixed from the connectome, per-type gains learned on an optic-flow task. Predicted
  direction selectivity of T4/T5 and tuning from 26 studies with no neural data in training.
  github.com/TuragaLab/flyvis (MIT, `pip install flyvis`).
- **Beiran & Litwin-Kumar 2025, *Nat. Neurosci.***: theory showing a connectome alone often
  under-determines task dynamics; partial recordings restore predictability. This is the formal
  reason "connectome-only" models need controls.

## 4. Embodied simulations

- **flybody** (DeepMind + Janelia, *Nature* 2025): MuJoCo fly body, 59-D action space, RL
  imitation of real flight and walking. github.com/TuragaLab/flybody (Apache-2.0). No pretrained
  policies published.
- **NeuroMechFly v2 / FlyGym** (EPFL, *Nat. Methods* 2024): MuJoCo body with ommatidia vision,
  olfaction, terrain. github.com/NeLy-EPFL/flygym. ~2x real time on CPU.
- **Eon Systems** (blog, Mar 2026): Shiu-style brain + flyvis eyes + NeuroMechFly body;
  descending neurons trigger *pre-trained* imitation controllers. No peer review, no code.
- **FlyGM** (arXiv 2602.17997): ~3,000 FlyWire neurons as a graph-network policy for flybody.
  Heading error 5.57 deg vs 7.77 deg for degree-preserving rewiring vs 11.33 deg random.
  Claims a topology advantage. A100 hardware.

## 5. Connectome-inspired machine learning (the efficiency-relevant work)

- **Neural Circuit Policies** (Lechner, Hasani et al., *Nat. Mach. Intell.* 2020): C. elegans-
  inspired wiring, 19 neurons and 253 synapses steer a car end to end, more robust than CNN/LSTM
  baselines. github.com/mlech26l/ncps.
- **FlyHash** (Dasgupta, Stevens, Navlakha, *Science* 2017): the mushroom body's projection
  neuron -> Kenyon cell random sparse expansion (~50 -> 2,000 dims, ~95% sparse via winner-take-
  all) is a locality-sensitive hash that beats classical LSH.
- **FlyModel** continual learning (Shen, Dasgupta, Navlakha, *PNAS* 2021): sparse expansion plus
  local plasticity only at the output layer. Split-MNIST after 5 tasks: 0.86 vs 0.77 (BI-R),
  0.69 (GEM), 0.58 (EWC). Almost no catastrophic forgetting.
- **FLYNN** (arXiv 2607.00025): the whole FlyWire connectome (99.97% sparse) as an RNN mask,
  trained by DAgger for robot navigation. Same in-distribution accuracy as a CNN (~93%) but far
  more robust: 44% success under vision loss vs 4 to 17%; 42% vs 0 to 17% on out-of-distribution
  textures. No code.
- **Negative control** (Dhiman, arXiv 2604.04033): apparent topology advantages on an edge-
  decoding task vanish under fair initialisation and degree-preserving rewiring (loss gap 0.184 ->
  -0.002). github.com/nalin-dhiman/Connectome-Constrained-Neural-Networks.

Reading: whether the *topology* helps (FlyGM, FLYNN say yes; Dhiman says no) is genuinely open.
Whether the *coding principle* (sparse expansion + local learning) helps is well supported.

### Connectome-based reservoir computing (the methodology)

- **conn2res** (Suárez et al., *Nat. Commun.* 2024; github.com/netneurolab/conn2res, BSD-3):
  a connectome becomes the fixed reservoir; dynamics (echo-state, spiking, memristive), spectral
  radius alpha, anatomical input/readout sets, NeuroGym tasks, and degree-preserving rewired
  nulls. Human connectome beats nulls on memory capacity at criticality (p = 0.002); mouse, rat,
  macaque too; **fruit fly did not (p = 0.11)**, using a region-level FlyCircuit network. Readout
  location (which intrinsic network) mattered more than global topology. Full notes in
  `02_conn2res_notes.md`. Suggested by the user as the frame for phase 2.

## 6. LLMs and connectomes

- No published work fuses an LLM with a fly whole-brain model. Closest:
  Neuro-LIFT (arXiv 2501.19259, LLM plans, spiking net flies a drone), BrainAgent
  (arXiv 2607.22082, LLM reasons over human brain graphs), SpikingBrain-7B (arXiv 2509.05276,
  spiking activations inside an LLM, 69% sparsity).
- The natural interface, used by Eon and FlyGM, is the ~1,300 descending neurons: the brain's
  compressed command channel to the body.

## 7. The efficiency argument, honestly

- A fly brain runs on roughly 5 to 10 microwatts; a chemical synapse costs ~1 fJ per bit
  (Laughlin et al. 1998; Niven & Laughlin 2008, *J. Exp. Biol.*). Mean firing rates are a few Hz.
- Why so few neurons suffice: energy as a selective pressure produced sparse coding and minimal
  wiring; behaviour is mostly innate, compressed through a genomic bottleneck, so *evolution did the
  training* (Zador 2019, *Nat. Commun.*).
- Counterarguments to keep in view:
  1. The fly's competence is narrow and hardwired. Its only flexible learner is the ~2,000-cell
     mushroom body.
  2. Biological energy efficiency does not transfer to a digital replica: our CPU LIF costs orders
     of magnitude more than 10 uW. What can transfer is architecture, not substrate.
  3. Whole-brain simulation is memory-bandwidth bound, not FLOP bound (State of Brain Emulation
     Report 2025, arXiv 2510.15745).
  4. Topology advantages can be initialisation artefacts (Dhiman).
