"""BrainLab: the fly connectome as a set of tools an LLM (or a person) can call.

Wraps flybrain.FlyBrain (MaleCNS v1.0 as a leaky integrate-and-fire network) with the
primitives a neuroscientist uses:

  * search_types(query)              find cell types by name fragment
  * describe_type(type)              size, side, superclass, sign, strongest partners
  * trace_paths(src, dst)            strongest multi-hop routes between two cell types
  * stimulate(types, side, amount)   drive a population, compare firing to baseline,
                                     report which descending neurons (motor commands) change
  * set_scrambled(True/False)        degree-preserving control wiring for "is it the topology?"

Everything returns plain dicts so the same layer serves an API tool, a notebook, or a test.
Connectivity is summarised at the *cell-type* level: 11.8k types instead of 166.7k neurons,
which is the level at which the literature (and an LLM) reasons.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from flybrain import FlyBrain

UNNAMED = ""


@dataclass
class BrainLab:
    brain: FlyBrain = field(default_factory=lambda: FlyBrain(device="cpu", sensory_input=False, seed=0))
    beam: int = 20          # strongest outgoing type-edges expanded per node in trace_paths

    def __post_init__(self) -> None:
        b = self.brain
        self.n = b.n
        types, self.type_of = np.unique(b.cell_type, return_inverse=True)
        self.types: list[str] = types.tolist()
        self.type_index = {t: i for i, t in enumerate(self.types)}
        T = len(self.types)
        self.type_size = np.bincount(self.type_of, minlength=T)

        # W: rows = postsynaptic neuron, cols = presynaptic neuron, weight = signed fraction of
        # the postsynaptic neuron's total input. P: one-hot neuron -> type.
        self._W = sparse.csc_matrix((b.weights, b.indices, b.indptr), shape=(self.n, self.n)).tocsr()
        self._original_indices = b.indices.copy()
        self.scrambled = False
        P = sparse.csr_matrix((np.ones(self.n, np.float32), (np.arange(self.n), self.type_of)), shape=(self.n, T))
        self._build_type_graph(P)
        # sign of each neuron: fraction of its outgoing weight that is inhibitory
        Wc = self._W.tocsc()
        neg = np.zeros(self.n, np.float32)
        tot = np.zeros(self.n, np.float32)
        np.add.at(neg, np.repeat(np.arange(self.n), np.diff(Wc.indptr)), np.abs(Wc.data) * (Wc.data < 0))
        np.add.at(tot, np.repeat(np.arange(self.n), np.diff(Wc.indptr)), np.abs(Wc.data))
        self.inhib_frac = np.where(tot > 0, neg / np.maximum(tot, 1e-9), np.nan)
        self.descending_types = sorted(set(b.cell_type[b.cells(["descending_neuron"])].tolist()) - {UNNAMED})

    def _build_type_graph(self, P) -> None:
        """Type-level graphs. T_signed[post_type, pre_type] = mean (over neurons of post_type)
        signed fraction of input received from pre_type. T_abs = same with |weights|."""
        Wt = P.T @ self._W          # (T, n): summed over postsynaptic neurons of each type
        inv = sparse.diags(1.0 / np.maximum(self.type_size, 1))
        self.T_signed = (inv @ (Wt @ P)).tocsr().astype(np.float32)
        self.T_abs = (inv @ (abs(Wt) @ P)).tocsr().astype(np.float32)
        self.T_abs_out = self.T_abs.T.tocsr()   # [pre_type, post_type] for forward search

    # ---- lookups -------------------------------------------------------------------------------

    def _neurons(self, t: str, side: str | None = None) -> np.ndarray:
        idx = self.brain.cells([t], side=side or None)
        return idx

    def _side_counts(self, t: str) -> dict[str, int]:
        idx = self._neurons(t)
        sides = self.brain.side[idx]
        return {s: int((sides == s).sum()) for s in np.unique(sides)}

    def _superclasses(self, t: str) -> list[str]:
        idx = self._neurons(t)
        return sorted(set(self.brain.superclass[idx].tolist()))

    def search_types(self, query: str, limit: int = 25) -> dict:
        q = query.lower()
        hits = [t for t in self.types if t != UNNAMED and q in t.lower()]
        # superclass names are searchable too (e.g. "descending_neuron")
        supers = sorted({s for s in set(self.brain.superclass.tolist()) if q in s.lower()})
        hits.sort(key=lambda t: (not t.lower().startswith(q), -self.type_size[self.type_index[t]], t))
        out = [{"type": t, "neurons": int(self.type_size[self.type_index[t]]),
                "superclass": self._superclasses(t)} for t in hits[:limit]]
        return {"query": query, "matches": len(hits), "types": out, "matching_superclasses": supers}

    def describe_type(self, t: str, top: int = 10) -> dict:
        if t not in self.type_index:
            near = self.search_types(t, limit=8)["types"]
            return {"error": f"no cell type named {t!r}", "did_you_mean": [x["type"] for x in near]}
        i = self.type_index[t]
        idx = self._neurons(t)
        inhib = float(np.nanmean(self.inhib_frac[idx])) if len(idx) else float("nan")
        sign = "inhibitory" if inhib > 0.5 else "excitatory"
        # downstream: types whose input fraction from t is largest
        col = self.T_abs[:, i].toarray().ravel()
        col_s = self.T_signed[:, i].toarray().ravel()
        down = np.argsort(-col)[:top + 1]
        # upstream: types that contribute most of t's input
        row = self.T_abs[i].toarray().ravel()
        row_s = self.T_signed[i].toarray().ravel()
        up = np.argsort(-row)[:top + 1]

        def fmt(js, a, s):
            return [{"type": self.types[j], "neurons": int(self.type_size[j]),
                     "input_fraction": round(float(a[j]), 4), "net_sign": "+" if s[j] >= 0 else "-",
                     "superclass": self._superclasses(self.types[j])}
                    for j in js if a[j] > 0 and self.types[j] != UNNAMED][:top]

        return {
            "type": t, "neurons": int(len(idx)), "sides": self._side_counts(t),
            "superclass": self._superclasses(t),
            "sign": sign, "inhibitory_fraction_of_output": round(inhib, 3),
            "total_input_fraction_accounted_by_named_types": round(float(row.sum()), 3),
            "strongest_downstream": fmt(down, col, col_s),
            "strongest_upstream": fmt(up, row, row_s),
            "note": "input_fraction = mean share of a postsynaptic neuron's total synaptic input that comes "
                    "from the named type (0..1). Downstream entries are how much of *their* input comes from this type.",
        }

    def trace_paths(self, src: str, dst: str, max_hops: int = 3, top: int = 5, min_fraction: float = 0.005) -> dict:
        for t in (src, dst):
            if t not in self.type_index:
                return {"error": f"no cell type named {t!r}", "did_you_mean": [x["type"] for x in self.search_types(t, 8)["types"]]}
        s, d = self.type_index[src], self.type_index[dst]
        found: list[tuple[float, list[int]]] = []
        frontier = [(1.0, [s])]
        for hop in range(max_hops):
            nxt = []
            for score, path in frontier:
                row = self.T_abs_out[path[-1]]
                if row.nnz == 0:
                    continue
                order = np.argsort(-row.data)[:self.beam]
                for k in order:
                    j, w = row.indices[k], row.data[k]
                    if w < min_fraction or j in path or self.types[j] == UNNAMED:
                        continue
                    sc = score * float(w)
                    if j == d:
                        found.append((sc, path + [j]))
                    elif hop < max_hops - 1:
                        nxt.append((sc, path + [j]))
            nxt.sort(key=lambda x: -x[0])
            frontier = nxt[: self.beam * 10]
        found.sort(key=lambda x: -x[0])
        paths = []
        for sc, p in found[:top]:
            hops = []
            for a, b in zip(p[:-1], p[1:]):
                hops.append({"from": self.types[a], "to": self.types[b],
                             "input_fraction": round(float(self.T_abs[b, a]), 4),
                             "sign": "+" if self.T_signed[b, a] >= 0 else "-"})
            paths.append({"hops": len(hops), "score": round(sc, 6), "route": " -> ".join(self.types[i] for i in p), "steps": hops})
        direct = float(self.T_abs[d, s])
        return {"from": src, "to": dst, "direct_input_fraction": round(direct, 4),
                "direct_sign": "+" if self.T_signed[d, s] >= 0 else "-",
                "paths": paths, "searched_hops": max_hops,
                "note": "score = product of input fractions along the route; higher = stronger. Beam search, "
                        f"top {self.beam} partners per hop, so weak routes can be missed."}

    # ---- experiments ---------------------------------------------------------------------------

    def _rates(self, stim: list[tuple[np.ndarray, float]], steps: int, seed: int) -> np.ndarray:
        b = self.brain
        b.reset(seed=seed)
        for _ in range(10):           # 200 ms settle
            b.step()
        counts = np.zeros(self.n, np.float32)
        for _ in range(steps):
            counts[b.step(inject=stim)] += 1
        return counts / (steps * b.dt)

    def stimulate(self, types: list[str], side: str | None = None, amount: float = 0.8, steps: int = 50,
                  watch: list[str] | None = None, seed: int = 7, top_changed: int = 12) -> dict:
        """Drive `types` (on `side` if given) every step for `steps` x 20 ms, and compare firing
        rates with a baseline run (same seed, no drive). Reports the watched types by side and
        the descending-neuron types whose rate changed most (the fly's motor output)."""
        unknown = [t for t in types if t not in self.type_index and t not in set(self.brain.superclass.tolist())]
        if unknown:
            return {"error": f"unknown types {unknown}", "did_you_mean": [x["type"] for t in unknown for x in self.search_types(t, 4)["types"]]}
        idx = self.brain.cells(types, side=side or None)
        if len(idx) == 0:
            return {"error": f"no neurons of {types} on side {side!r}", "sides_available": {t: self._side_counts(t) for t in types}}
        amount = float(np.clip(amount, 0.0, 1.0))
        t0 = time.perf_counter()
        base = self._rates([], steps, seed)
        stim = self._rates([(idx, amount)], steps, seed)
        wall = time.perf_counter() - t0
        delta = stim - base

        def by_side(t: str) -> dict:
            out = {}
            for s in ("L", "R"):
                j = self._neurons(t, s)
                if len(j):
                    out[s] = {"n": int(len(j)), "baseline_hz": round(float(base[j].mean()), 2),
                              "stimulated_hz": round(float(stim[j].mean()), 2), "delta_hz": round(float(delta[j].mean()), 2)}
            return out

        watched = {t: by_side(t) for t in (watch or []) if t in self.type_index}
        # descending neurons: aggregate delta by type and side
        dn = self.brain.cells(["descending_neuron"])
        rows = []
        for t in self.descending_types:
            for s in ("L", "R"):
                j = self._neurons(t, s)
                if len(j):
                    rows.append((t, s, int(len(j)), float(base[j].mean()), float(stim[j].mean())))
        rows.sort(key=lambda r: -abs(r[4] - r[3]))
        changed = [{"type": t, "side": s, "n": n, "baseline_hz": round(b0, 2), "stimulated_hz": round(b1, 2),
                    "delta_hz": round(b1 - b0, 2)} for t, s, n, b0, b1 in rows[:top_changed] if abs(b1 - b0) >= 0.5]
        return {
            "stimulated": {"types": types, "side": side, "neurons": int(len(idx)), "amount": amount,
                           "duration_s": round(steps * self.brain.dt, 2)},
            "wiring": "scrambled (control)" if self.scrambled else "real connectome",
            "watched": watched,
            "most_changed_descending_neurons": changed,
            "whole_brain": {"baseline_mean_hz": round(float(base.mean()), 3), "stimulated_mean_hz": round(float(stim.mean()), 3),
                            "neurons_with_delta_over_5hz": int((np.abs(delta) > 5).sum())},
            "wall_seconds": round(wall, 1),
        }

    def set_scrambled(self, enabled: bool) -> dict:
        """Degree-preserving control: permute all postsynaptic targets. Same neurons, same
        out-degrees, same in-degree multiset, same signs; destroys who-talks-to-whom."""
        b = self.brain
        if enabled and not self.scrambled:
            rng = np.random.default_rng(0)
            b.indices = self._original_indices[rng.permutation(len(self._original_indices))]
        elif not enabled and self.scrambled:
            b.indices = self._original_indices
        self.scrambled = enabled
        return {"wiring": "scrambled (control)" if enabled else "real connectome",
                "note": "trace_paths/describe_type still report the REAL connectome; only stimulate() uses the control wiring."}

    def overview(self) -> dict:
        b = self.brain
        sup = {s: int(c) for s, c in zip(*np.unique(b.superclass, return_counts=True))}
        return {"neurons": int(self.n), "connections": int(len(b.weights)), "cell_types": len(self.types) - 1,
                "descending_neuron_types": len(self.descending_types), "superclasses": dict(sorted(sup.items(), key=lambda kv: -kv[1])),
                "model": "leaky integrate-and-fire, 20 ms steps, weights = synapse count x transmitter sign, "
                         "normalised per postsynaptic neuron; tonic drive and noise hand-calibrated (flybrain 0.1.0)",
                "dataset": "MaleCNS v1.0 (HHMI Janelia FlyEM, Google Research, Cambridge, MRC LMB; CC-BY)"}
