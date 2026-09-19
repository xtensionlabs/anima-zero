"""Connectome-based reservoir computing on MaleCNS, following conn2res (Suárez et al. 2024).

    task input -> input neurons -> connectome reservoir (fixed) -> anatomical readout set -> ridge

Two reservoir modes over the same 166,700 x 166,700 signed matrix:
  * RateReservoir : echo-state units, x <- tanh(alpha/rho * W x + W_in u). alpha is the spectral
                    radius (conn2res: stable < 1, critical ~ 1, chaotic > 1).
  * spiking       : flybrain.FlyBrain (leaky integrate-and-fire), regime set by `gain`.

Null model: degree-preserving rewiring. Permuting the CSC `indices` array keeps every neuron's
exact out-degree and in-degree, weights and signs stay attached to their presynaptic neuron, and
we re-normalise each postsynaptic neuron's total |input| to 1 as in the real matrix. What is
destroyed is only who connects to whom.

Tasks (native re-implementations of the conn2res / NeuroGym trio):
  * memory_capacity          : reconstruct u(t-k) for k = 1..k_max (Jaeger 2001); score = sum of R^2
  * perceptual_decision      : which of two noisy channels is stronger; needs temporal integration
  * context_decision         : two modalities, a context cue picks the relevant one; needs gating
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numba
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigs

# ---- fast sparse matvec ----------------------------------------------------------------------


@numba.njit(parallel=True, fastmath=True, cache=True)
def csr_matvec(indptr, indices, data, x):
    n = len(indptr) - 1
    out = np.empty(n, np.float32)
    for i in numba.prange(n):
        s = np.float32(0.0)
        for e in range(indptr[i], indptr[i + 1]):
            s += data[e] * x[indices[e]]
        out[i] = s
    return out


def spectral_radius(W_csr: sparse.csr_matrix, tol: float = 1e-3, seed: int = 0) -> float:
    """Largest |eigenvalue| via ARPACK on a fast matvec; power iteration as fallback."""
    n = W_csr.shape[0]
    op = LinearOperator((n, n), matvec=lambda v: csr_matvec(W_csr.indptr, W_csr.indices, W_csr.data,
                                                           np.asarray(v, np.float32).ravel()).astype(np.float64),
                        dtype=np.float64)
    rng = np.random.default_rng(seed)
    try:
        vals = eigs(op, k=1, which="LM", tol=tol, maxiter=3000, v0=rng.standard_normal(n), return_eigenvectors=False)
        return float(abs(vals[0]))
    except Exception:
        v = rng.standard_normal(n).astype(np.float32)
        v /= np.linalg.norm(v)
        growth = 1.0
        for _ in range(300):
            w = csr_matvec(W_csr.indptr, W_csr.indices, W_csr.data, v)
            growth = float(np.linalg.norm(w))
            v = w / max(growth, 1e-12)
        return growth


def scramble(W_csc: sparse.csc_matrix, seed: int, renormalize: bool = True) -> sparse.csc_matrix:
    """Degree-preserving null: permute postsynaptic targets, keep everything else."""
    rng = np.random.default_rng(seed)
    idx = W_csc.indices[rng.permutation(len(W_csc.indices))]
    data = W_csc.data.copy()
    if renormalize:
        incoming = np.bincount(idx, weights=np.abs(data), minlength=W_csc.shape[0])
        data = (data / np.maximum(incoming[idx], 1e-9)).astype(np.float32)
    return sparse.csc_matrix((data, idx, W_csc.indptr.copy()), shape=W_csc.shape)


# ---- tasks -----------------------------------------------------------------------------------


@dataclass
class Task:
    name: str
    inputs: np.ndarray          # (T, C) in [-1, 1]
    eval_idx: np.ndarray        # time steps whose reservoir state is used for the readout
    targets: np.ndarray         # (len(eval_idx), K)
    train: np.ndarray           # boolean mask over eval_idx
    kind: str                   # "regression" or "classification"
    meta: dict = field(default_factory=dict)


def memory_capacity(steps: int = 1200, k_max: int = 30, washout: int = 100, seed: int = 0, hold: int = 1) -> Task:
    """`hold` > 1 keeps each random value for `hold` steps (spiking mode: 20 ms steps are too
    fast for spike traces to carry per-step memory). Delays k are then counted in held blocks
    and the readout is taken at the last step of each block."""
    rng = np.random.default_rng(seed)
    n_blocks = steps // hold
    vals = rng.uniform(-1, 1, n_blocks).astype(np.float32)
    u = np.repeat(vals, hold)[:steps]
    b = np.arange(washout // hold + k_max, n_blocks)
    t = b * hold + hold - 1
    targets = np.stack([vals[b - k] for k in range(1, k_max + 1)], axis=1)
    train = np.zeros(len(t), bool)
    train[: int(0.7 * len(t))] = True
    return Task("memory_capacity", u[:, None], t, targets, train, "regression", {"k_max": k_max, "hold": hold})


def perceptual_decision(n_trials: int = 200, fix: int = 3, stim: int = 12, delay: int = 3, noise: float = 0.6,
                        seed: int = 0) -> Task:
    """Two channels: c1 = 0.5 + coh/2 + noise, c2 = 0.5 - coh/2 + noise during the stimulus;
    decide sign(coh) `delay` steps after it ends. Per-step noise is larger than the coherence, so
    a single step cannot solve it: the reservoir has to integrate."""
    rng = np.random.default_rng(seed)
    L = fix + stim + delay + 1
    X = np.zeros((n_trials * L, 2), np.float32)
    eval_idx, labels = [], []
    for i in range(n_trials):
        coh = rng.choice([0.1, 0.2, 0.4, 0.8]) * rng.choice([-1, 1])
        t0 = i * L
        s = slice(t0 + fix, t0 + fix + stim)
        X[s, 0] = 0.5 + coh / 2 + rng.normal(0, noise, stim)
        X[s, 1] = 0.5 - coh / 2 + rng.normal(0, noise, stim)
        eval_idx.append(t0 + L - 1)
        labels.append(1.0 if coh > 0 else -1.0)
    X = np.clip(X, -1, 1)
    train = np.zeros(n_trials, bool)
    train[: int(0.7 * n_trials)] = True
    return Task("perceptual_decision", X, np.array(eval_idx), np.array(labels, np.float32)[:, None], train,
                "classification", {"trial_len": L})


def context_decision(n_trials: int = 200, fix: int = 3, stim: int = 12, delay: int = 3, noise: float = 0.4,
                     seed: int = 0) -> Task:
    """Mante-style: modality A (2 channels) and modality B (2 channels) each carry an independent
    noisy coherence; two context channels (one-hot, whole trial) say which modality counts.
    Solvable only by gating one modality with the context: a linear readout of the raw inputs
    cannot do it, so any accuracy above chance is the reservoir's nonlinearity at work."""
    rng = np.random.default_rng(seed)
    L = fix + stim + delay + 1
    X = np.zeros((n_trials * L, 6), np.float32)
    eval_idx, labels = [], []
    for i in range(n_trials):
        t0 = i * L
        ctx = rng.integers(2)
        cohs = [rng.choice([0.2, 0.4, 0.8]) * rng.choice([-1, 1]) for _ in range(2)]
        s = slice(t0 + fix, t0 + fix + stim)
        for m, coh in enumerate(cohs):
            X[s, 2 * m] = 0.5 + coh / 2 + rng.normal(0, noise, stim)
            X[s, 2 * m + 1] = 0.5 - coh / 2 + rng.normal(0, noise, stim)
        X[t0: t0 + L, 4 + ctx] = 1.0
        eval_idx.append(t0 + L - 1)
        labels.append(1.0 if cohs[ctx] > 0 else -1.0)
    X = np.clip(X, -1, 1)
    train = np.zeros(n_trials, bool)
    train[: int(0.7 * n_trials)] = True
    return Task("context_decision", X, np.array(eval_idx), np.array(labels, np.float32)[:, None], train,
                "classification", {"trial_len": L})


TASKS = {"memory_capacity": memory_capacity, "perceptual_decision": perceptual_decision,
         "context_decision": context_decision}


# ---- input map and readout sets --------------------------------------------------------------


def input_map(pool: np.ndarray, n_channels: int, per_channel: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Disjoint random neuron subsets of `pool`, one per input channel, with positive weights in [0.5, 1]."""
    rng = np.random.default_rng(seed)
    chosen = rng.choice(pool, size=n_channels * per_channel, replace=False)
    return [(chosen[c * per_channel:(c + 1) * per_channel],
             rng.uniform(0.5, 1.0, per_channel).astype(np.float32)) for c in range(n_channels)]


CX_PREFIXES = ("EPG", "PEG", "PEN", "PFL", "PFN", "PFR", "PFG", "FB", "FC", "FR", "FS", "ER", "ExR", "hDelta",
               "vDelta", "Delta7", "LNO", "LCNO", "SpsP", "IbSpsP", "P1-9", "PEcG", "EL", "GLNO", "OA-AL2i")


def readout_sets(brain, random_seed: int = 0) -> dict[str, np.ndarray]:
    """Anatomical readout populations plus size-matched random controls."""
    ct = brain.cell_type.astype(str)
    sets = {
        "descending": brain.cells(["descending_neuron"]),
        "mushroom_body_output": np.flatnonzero(np.char.startswith(ct, "MBON")),
        "central_complex": np.flatnonzero(np.logical_or.reduce([np.char.startswith(ct, p) for p in CX_PREFIXES])),
    }
    rng = np.random.default_rng(random_seed)
    # random controls drawn from non-sensory, non-visual-projection neurons of matched size
    pool = np.flatnonzero(~np.isin(brain.superclass, ["visual_projection"]) &
                          (np.char.find(brain.superclass.astype(str), "sensory") < 0))
    for name in list(sets):
        sets[f"random_{len(sets[name])}"] = rng.choice(pool, size=len(sets[name]), replace=False)
    return sets


# ---- readout -----------------------------------------------------------------------------------


def ridge_score(F: np.ndarray, task: Task, lambdas=(1e-2, 1e-1, 1.0, 10.0)) -> dict:
    """F: (len(eval_idx), n_features). Standardise, pick lambda on a validation split of the
    training set, refit, score on the held-out test set."""
    Y = task.targets
    tr, te = task.train, ~task.train
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    Z = (F - mu) / sd
    n_tr = tr.sum()
    val = np.zeros(len(F), bool)
    val[np.flatnonzero(tr)[int(0.75 * n_tr):]] = True
    fit = tr & ~val

    def solve(mask, lam):
        Zm, Ym = Z[mask], Y[mask]
        zm, ym = Zm.mean(0), Ym.mean(0)
        A = (Zm - zm).T @ (Zm - zm) + lam * mask.sum() * np.eye(Z.shape[1], dtype=np.float64)
        Wt = np.linalg.solve(A, (Zm - zm).T @ (Ym - ym))
        return Wt, ym - zm @ Wt

    def score(pred, mask):
        y = Y[mask]
        if task.kind == "classification":
            return float((np.sign(pred[:, 0]) == np.sign(y[:, 0])).mean())
        r2 = 1 - ((pred - y) ** 2).sum(0) / np.maximum(((y - y.mean(0)) ** 2).sum(0), 1e-9)
        return float(np.clip(r2, 0, 1).sum())

    best = max(lambdas, key=lambda lam: score(Z[val] @ solve(fit, lam)[0] + solve(fit, lam)[1], val))
    Wt, b = solve(tr, best)
    pred = Z[te] @ Wt + b
    out = {"score": score(pred, te), "lambda": best, "n_features": int(F.shape[1])}
    if task.kind == "regression":
        y = Y[te]
        r2 = 1 - ((pred - y) ** 2).sum(0) / np.maximum(((y - y.mean(0)) ** 2).sum(0), 1e-9)
        out["r2_by_delay"] = np.clip(r2, 0, 1).round(3).tolist()
    return out


# ---- reservoirs ----------------------------------------------------------------------------------


class RateReservoir:
    """rho = spectral radius to normalise by (conn2res protocol), or 1.0 for 'raw' scaling where
    alpha multiplies the per-neuron-normalised connectome directly (same multiplier for real and
    null networks, so the comparison is not confounded by how the null's spectrum differs)."""

    def __init__(self, W_csr: sparse.csr_matrix, alpha: float, rho: float = 1.0):
        self.indptr, self.indices = W_csr.indptr, W_csr.indices
        self.data = (W_csr.data * np.float32(alpha / rho)).astype(np.float32)
        self.n = W_csr.shape[0]
        self.x = np.zeros(self.n, np.float32)

    def step(self, drive: np.ndarray) -> np.ndarray:
        pre = csr_matvec(self.indptr, self.indices, self.data, self.x) + drive
        self.x = np.tanh(pre)
        return self.x


def run_rate(W_csr, alpha: float, rho: float, task: Task, imap, readouts: dict[str, np.ndarray],
             input_scale: float = 2.0) -> dict:
    res = RateReservoir(W_csr, alpha, rho)
    eval_set = {int(t): i for i, t in enumerate(task.eval_idx)}
    feats = {name: np.zeros((len(task.eval_idx), len(idx)), np.float32) for name, idx in readouts.items()}
    drive = np.zeros(res.n, np.float32)
    t0 = time.perf_counter()
    for t in range(len(task.inputs)):
        drive[:] = 0
        for c, (idx, w) in enumerate(imap):
            drive[idx] = w * np.float32(input_scale * task.inputs[t, c])
        x = res.step(drive)
        if t in eval_set:
            for name, idx in readouts.items():
                feats[name][eval_set[t]] = x[idx]
    wall = time.perf_counter() - t0
    out = {name: ridge_score(F, task) for name, F in feats.items()}
    out["_state"] = {"mean_abs_activity": float(np.abs(res.x).mean()), "saturated_fraction": float((np.abs(res.x) > 0.99).mean()),
                     "wall_seconds": round(wall, 1), "ms_per_step": round(1000 * wall / len(task.inputs), 1)}
    return out


def run_spiking(brain, gain: float, task: Task, imap, readouts: dict[str, np.ndarray], input_scale: float = 0.9,
                tau: float = 0.1, seed: int = 0) -> dict:
    """Same protocol on the leaky integrate-and-fire brain. Inputs in [-1, 1] become injected
    voltage in [0, input_scale]. Features are exponentially filtered spike traces (tau seconds)."""
    from flybrain.reservoir import Trace
    brain.gain = float(gain)
    brain.reset(seed=seed)
    traces = {name: Trace(brain, idx=idx, tau=tau) for name, idx in readouts.items()}
    eval_set = {int(t): i for i, t in enumerate(task.eval_idx)}
    feats = {name: np.zeros((len(task.eval_idx), len(idx)), np.float32) for name, idx in readouts.items()}
    spikes = 0
    t0 = time.perf_counter()
    for t in range(len(task.inputs)):
        # flybrain takes one scalar per population, so the per-neuron weights average out here
        inject = [(idx, float(input_scale * w.mean() * (task.inputs[t, c] + 1) / 2)) for c, (idx, w) in enumerate(imap)]
        fired = brain.step(inject=inject)
        spikes += len(fired)
        for name, tr in traces.items():
            f = tr.observe(fired)
            if t in eval_set:
                feats[name][eval_set[t]] = f
    wall = time.perf_counter() - t0
    out = {name: ridge_score(F, task) for name, F in feats.items()}
    out["_state"] = {"mean_rate_hz": float(spikes / len(task.inputs) / brain.n / brain.dt),
                     "wall_seconds": round(wall, 1), "ms_per_step": round(1000 * wall / len(task.inputs), 1)}
    return out
