# anima-zero: a bio-digital architecture built on the male fruit-fly connectome

*Working thesis, 2026-09-20. This is the idea after one day of grounding it in data. It will change.*

## The idea as stated

Fuse a large language model with the complete wiring diagram of a male fruit fly's nervous
system (MaleCNS v1.0, Google Research + HHMI Janelia, Cell 2026). Motivation: a fly does
complex, robust, real-time behaviour with 166,700 neurons and a few microwatts, while modern
AI reaches competence by brute force. Not to solve that overnight, but to find out what
actually changes when a real brain's wiring becomes a component of an AI system, and what can
be built from it.

## What we established today (all reproducible from `experiments/`)

| Fact | Value | Source |
|---|---|---|
| Neurons simulated | 166,700 | `results/00_smoke.json` |
| Connections | 25.1 M (synapse-count weighted, signed by transmitter) | same |
| Density | 0.09% of possible pairs | same |
| Named cell types | 11,862 | same |
| Descending neurons (brain -> body command channel) | 1,314 | same |
| Step cost on this laptop (6-core CPU, no GPU) | 19.8 ms per 20 ms of biological time: **real time** | same |
| Neurons spiking per 20 ms step at rest | 3.9% | same |
| Looming on the left eye -> left giant fiber DNp01 | 48 spikes vs 1 on the right | same |
| Same test after degree-preserving scramble (3 trials) | lateralization -0.14, -0.17, 0.11 vs 0.96 real | `results/01_scramble_control.json` |
| LLM (claude-opus-5) asked to find the escape circuit with six tools, no hints | Recovered LC4/LPLC2 -> DNp01/02/04/11, ran mirror, dissection and scrambled control; found DNp01 needs both streams (LC4 alone +24 Hz, LPLC2 alone +20 Hz, both +47 Hz); flagged the missing electrical synapse | `results/03_agent_run.log` |

The last row is the important one. A scrambled network has identical neuron count, identical
degree sequence, identical signs, identical global firing rate (1.2 Hz on descending neurons in
both cases), and **no reflex**. Combined with fly.ai's finding that as a *generic* reservoir the
real wiring carries no more information than scrambled wiring, the picture is:

> The connectome is not a better random network. It is a library of specific evolved programs
> (looming -> escape, target -> steer, sugar -> extend proboscis). Its value to AI is the
> programs and the principles behind them, not its statistics.

That reframes the project.

## Refining the motivation: why the fly is efficient, and what transfers

Why 166k neurons suffice (see `01_landscape.md` section 7 for sources):

1. **Evolution did the training.** Behaviour is mostly innate, compressed through a genomic
   bottleneck. The fly is a pretrained, frozen model with one small plastic module.
2. **Extreme sparsity in wiring and activity.** 0.09% connectivity; ~4% of neurons active per
   20 ms; mean rates of a few Hz. Computation is event-driven.
3. **A compressed action interface.** 1,314 descending neurons are the whole command channel
   from brain to body: a fixed vocabulary of "action tokens".
4. **One flexible learner, built for few-shot.** The mushroom body: ~50 input channels expand
   to ~2,000 Kenyon cells, ~5% active, with plasticity only at the single output layer, gated by
   dopamine (reward/punishment). This is the only place the fly learns anything new, and it is
   provably good (FlyHash beats LSH; FlyModel beats EWC/GEM on continual learning).
5. **Explicit, interpretable state.** The central complex holds heading as a ring attractor: a
   physical working-memory variable you can read out.

What does *not* transfer: substrate energy efficiency. A digital LIF replica costs orders of
magnitude more than 10 uW. The fly's watts come from chemistry, not architecture. So the honest
goal is: **take the architectural principles (1 to 5) and the actual circuits, and measure what
they buy an AI system in compute, data efficiency, robustness, and interpretability.**

## Where the LLM fits

An LLM is the opposite animal: a general, slow, expensive, dense sequence model with no body
and no reflexes. The fly is a specific, fast, cheap, sparse sensorimotor machine with no
language and almost no learning. The interesting fusion is complementary, not a merge:

```
                 goals, plans, explanations, novel situations
   +----------+  ------------------------------------------>  +-----------------+
   |   LLM    |   drive: named cell types (like neuromodulation)|  Fly connectome |
   | (cortex) |  <------------------------------------------   |  (spinal brain) |
   +----------+   sense: descending-neuron activity, state      +-----------------+
                  (1,314-dim action vocabulary, ~ms latency)          |   ^
                                                                      v   |
                                                                   body / world
```

Three roles the LLM can play, in increasing ambition:

- **Scientist.** The LLM designs and runs experiments on the brain (stimulate types, read out
  types, trace pathways) and reports mechanisms. Immediate, useful, and tests whether an LLM can
  reason about a 166k-node circuit through tools. *Nobody has published this.*
- **Cortex.** In an embodied task, the LLM sets slow goals by driving cell populations; the fly
  circuits handle fast reflexes; descending activity returns as a compact state. Measures
  whether a frozen biological policy plus a small language planner beats a planner alone on
  latency, robustness, and tokens spent.
- **Student.** The LLM (or a trainable network beside it) learns *from* the connectome's
  principles: sparse expansion + local plasticity as a memory/adapter module; the descending
  neuron set as an action tokenizer; connectome topology as a wiring prior.

## Hypotheses worth testing (each is a runnable experiment on this laptop)

H1. **Topology matters for circuits, not for reservoirs.** Partly shown today. Extend: for each
    of the ~10 known sensorimotor pathways, real vs scrambled. Expect all to vanish.

H2. **Mushroom-body learning beats a dense learner per parameter and per example.** Implement
    Kenyon-cell expansion + dopamine-gated output plasticity using the *real* mushroom body
    wiring from MaleCNS (KC, MBON, PN types are annotated). Compare with a same-parameter MLP on
    few-shot and continual classification. This is the most direct "what can change" for AI.

H3. **The connectome as a wiring mask helps out-of-distribution robustness, not in-distribution
    accuracy.** FLYNN claims this at whole-brain scale on a GPU; Dhiman finds no effect on a
    small task. Test at sub-circuit scale (central complex, ~3k neurons) on CPU, with Dhiman's
    fair-initialisation and degree-preserving controls built in from the start.

H4. **An LLM can discover circuit mechanisms through a tool interface** with fewer, better-
    targeted simulations than a brute-force sweep. Measure: can it recover the known
    LC4/LPLC2 -> DNp01 and LC10a -> DNa02 pathways without being told?

H5. **LLM-as-cortex over fly-as-reflex reduces LLM calls per unit of task success** in a simple
    embodied environment, versus the LLM controlling the body directly.

H6. **At synaptic resolution the fly connectome beats degree-preserving nulls as a reservoir near
    criticality, and the advantage is carried by specific sub-circuits.** conn2res (Suárez et al.
    2024) found no significant advantage for a region-level fly network (p = 0.11) while human,
    mouse, rat, and macaque all showed one. MaleCNS lets us rerun their protocol (memory capacity,
    NeuroGym tasks, gain sweep, rewired nulls) with 166,700 nodes and anatomically chosen readouts:
    descending neurons vs mushroom body output vs central complex vs random sets. See
    `02_conn2res_notes.md`.

## Build plan

Phase 0 (today): ground truth. Done: smoke test, structural stats, reflex reproduction,
scramble control, landscape survey, this document.

Phase 1 (done 2026-09-20): **the bridge** (`anima/`). A tool layer over the brain (find types,
describe connectivity, trace pathways, stimulate and observe) and a Claude agent that uses it.
Deliverable met: asked "how does the fly escape a looming object?", the agent traced the mechanism
by experiment with controls, in 22 tool calls and about six minutes. H4 supported on one circuit;
the next test is a circuit the model was not primed with (e.g. courtship song, grooming).

Phase 2a: **reservoir battery** (`experiments/03_reservoir_battery.py`), following conn2res:
memory capacity and two NeuroGym tasks, visual projection neurons as input, readouts from
descending neurons / MBONs / central complex / random sets, gain sweep across the critical
point, 20 rewired nulls, spiking and rate modes. Tests H6 and locates where the LLM should plug in.

Phase 2b: **mushroom-body module** (`experiments/04_mushroom_body.py`). Extract the real
PN -> KC -> MBON wiring, implement the plasticity rule, benchmark vs MLP. Tests H2.

Phase 3: **embodied loop**. A small 2-D world (own code, or FlyGym later), fly circuits for
reflexes, LLM for goals. Tests H5.

Phase 4: **topology prior**. Central-complex mask vs controls on a heading-integration task.
Tests H3.

## Constraints and honesty

- Hardware: 6-core laptop CPU, 15 GB RAM, no CUDA. Whole-brain LIF runs in real time; anything
  requiring gradients through 166k units must be done on sub-circuits or a rented GPU.
- The LIF model's tonic drive, gain, and noise are hand-calibrated, not measured. Sensory-to-
  brain relays (lamina) do not work in a spiking model; sensory encoders drive projection
  neurons directly, as Eon and fly.ai do. Every result inherits these caveats.
- Data: MaleCNS is CC-BY. FlyWire is CC BY-NC. Anything we publish uses MaleCNS primarily.
- We will report negative results. The scramble control already shows how easy it is to fool
  yourself with "the connectome did it."
