"""
Server-side aggregation rules. Closes Gaps 1, 2 and 4 from the proposal.

Deliberately framework-agnostic: a client update is just
`Dict[str, np.ndarray]` (a state dict with tensors converted to numpy). That
keeps the contribution — the weighting rule — testable without a GPU, and
makes it trivially portable between the TF and PyTorch ports of the base repo.

Three rules:
    fedavg   w_i = n_i / N                        (McMahan et al., the baseline)
    fedprox  identical aggregation to FedAvg      (FedProx changes the LOCAL
             objective, not the server rule — see note below)
    cbwa     w_i ∝ n_i^α · (ε + F1_i)^γ           (proposed)

NOTE ON FEDPROX. A detail that is easy to get wrong and that reviewers check:
FedProx does *not* modify the aggregation step. Its proximal term (μ/2)||w-w_t||²
is added to each client's local loss. So "reproducing the paper's FedProx
baseline" means enabling `prox_mu` in the client, while the server still runs
FedAvg. `fedprox_aggregate` is an explicit alias so this cannot be fudged.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

StateDict = Dict[str, np.ndarray]


# --------------------------------------------------------------------------
# Weighting rules
# --------------------------------------------------------------------------

def fedavg_weights(n_samples: Sequence[int]) -> np.ndarray:
    n = np.asarray(n_samples, dtype=np.float64)
    if n.sum() <= 0:
        raise ValueError("total sample count must be positive")
    return n / n.sum()


def cbwa_weights(
    n_samples: Sequence[int],
    macro_f1: Sequence[float],
    alpha: float = 0.5,
    gamma: float = 1.0,
    eps: float = 0.05,
    floor: float = 0.01,
) -> np.ndarray:
    """Class-Balance-aware Weighted Aggregation.

    w_i ∝ n_i^alpha · (eps + F1_i)^gamma, renormalised, with a floor.

    alpha  sub-linear damping of sample count. alpha=1 recovers FedAvg's size
           term exactly, alpha=0 gives every client an equal size vote.
           alpha=0.5 is the proposal's default.
    gamma  how sharply to reward cross-class generalisation. gamma=0 disables
           the performance term, reducing CBWA to pure dampened-size weighting
           — which is the ablation that isolates how much the F1 signal is
           actually contributing.
    eps    floor inside the performance term. WITHOUT THIS, a client scoring
           F1=0 gets weight exactly 0 and drops out of the federation
           entirely. Early in training every client scores near zero, so a
           zero-floor rule can silently kill the whole round.
    floor  minimum share of total weight per client, applied after
           normalisation. Guarantees no client is ever fully silenced — a
           hospital that contributed data should always influence the model.

    Returns weights summing to 1.
    """
    n = np.asarray(n_samples, dtype=np.float64)
    f1 = np.asarray(macro_f1, dtype=np.float64)
    if n.shape != f1.shape:
        raise ValueError(f"n_samples {n.shape} and macro_f1 {f1.shape} differ")
    if np.any(n < 0):
        raise ValueError("negative sample counts")
    if np.any((f1 < -1e-9) | (f1 > 1 + 1e-9)):
        raise ValueError("macro_f1 values must lie in [0, 1]")

    size_term = np.power(np.maximum(n, 0.0), alpha)
    perf_term = np.power(np.maximum(f1, 0.0) + eps, gamma)
    w = size_term * perf_term

    if w.sum() <= 0:
        return np.full(len(n), 1.0 / len(n))
    w = w / w.sum()
    return _apply_floor(w, floor)


def _apply_floor(w: np.ndarray, floor: float) -> np.ndarray:
    """Lift every weight to at least `floor`, taking the excess proportionally
    from those above it. Preserves sum=1."""
    k = len(w)
    if floor <= 0 or floor * k >= 1.0:
        return w
    below = w < floor
    if not below.any():
        return w
    deficit = (floor - w[below]).sum()
    above = ~below
    headroom = (w[above] - floor).sum()
    if headroom <= 0:
        return np.full(k, 1.0 / k)
    w = w.copy()
    w[above] -= (w[above] - floor) / headroom * deficit
    w[below] = floor
    return w / w.sum()


def smooth_weights(prev: Optional[np.ndarray], curr: np.ndarray,
                   momentum: float = 0.7) -> np.ndarray:
    """EMA across rounds: w_t = m·w_{t-1} + (1-m)·w_t.

    The macro-F1 term is estimated on a small reference set, so it is noisy —
    with 40 volumes per class, one or two flipped predictions visibly moves a
    client's score. Un-smoothed, that noise makes aggregation weights thrash
    round to round and can destabilise convergence. Set momentum=0 to disable
    and see the raw behaviour.
    """
    if prev is None:
        return curr
    w = momentum * prev + (1.0 - momentum) * curr
    return w / w.sum()


# --------------------------------------------------------------------------
# Applying weights to state dicts
# --------------------------------------------------------------------------

def aggregate(updates: List[StateDict], weights: Sequence[float]) -> StateDict:
    """Weighted average of client state dicts."""
    if not updates:
        raise ValueError("no client updates to aggregate")
    w = np.asarray(weights, dtype=np.float64)
    if len(w) != len(updates):
        raise ValueError(f"{len(updates)} updates but {len(w)} weights")
    if not np.isclose(w.sum(), 1.0):
        w = w / w.sum()

    keys = set(updates[0])
    for i, u in enumerate(updates[1:], 1):
        if set(u) != keys:
            missing, extra = keys - set(u), set(u) - keys
            raise KeyError(f"client {i} state dict mismatch; "
                           f"missing={sorted(missing)[:3]} extra={sorted(extra)[:3]}")

    out: StateDict = {}
    for k in updates[0]:
        ref = updates[0][k]
        # Integer buffers (e.g. BatchNorm num_batches_tracked) must not be
        # averaged into floats — carry the first client's value instead.
        if not np.issubdtype(ref.dtype, np.floating):
            out[k] = ref.copy()
            continue
        acc = np.zeros_like(ref, dtype=np.float64)
        for wi, u in zip(w, updates):
            acc += wi * u[k].astype(np.float64)
        out[k] = acc.astype(ref.dtype)
    return out


def fedavg_aggregate(updates: List[StateDict], n_samples: Sequence[int]) -> StateDict:
    return aggregate(updates, fedavg_weights(n_samples))


# FedProx aggregates exactly like FedAvg; the difference is the local objective.
fedprox_aggregate = fedavg_aggregate


def cbwa_aggregate(updates: List[StateDict], n_samples: Sequence[int],
                   macro_f1: Sequence[float], alpha: float = 0.5,
                   gamma: float = 1.0, eps: float = 0.05, floor: float = 0.01,
                   prev_weights: Optional[np.ndarray] = None,
                   momentum: float = 0.7):
    """Returns (aggregated_state_dict, weights_used)."""
    w = cbwa_weights(n_samples, macro_f1, alpha, gamma, eps, floor)
    w = smooth_weights(prev_weights, w, momentum)
    return aggregate(updates, w), w


# --------------------------------------------------------------------------
# Round scheduling
# --------------------------------------------------------------------------

def select_rule(round_idx: int, warmup_rounds: int = 50,
                rule_after_warmup: str = "cbwa") -> str:
    """Which rule applies at this round.

    CBWA MUST NOT run during warm-up. The base paper's warm-up trains only
    ST and SE for the first 50 rounds; during that phase no client has
    meaningful macro-F1 on a balanced reference set, so the performance term
    is pure noise and would corrupt exactly the phase the paper found most
    important for convergence. Plain FedAvg over the two warm-up sites, then
    switch.
    """
    return "fedavg" if round_idx < warmup_rounds else rule_after_warmup


def describe_weights(sites: Sequence[str], n_samples: Sequence[int],
                     macro_f1: Optional[Sequence[float]] = None,
                     alpha: float = 0.5, **kw) -> str:
    """Human-readable FedAvg vs CBWA comparison — the table for the write-up."""
    fa = fedavg_weights(n_samples)
    lines = [f"{'site':6s} {'n':>7s} {'FedAvg':>9s} {'CBWA':>9s} {'ratio':>7s} {'F1':>6s}"]
    if macro_f1 is None:
        cb, f1 = cbwa_weights(n_samples, np.ones(len(n_samples)), alpha, gamma=0.0, **kw), None
    else:
        cb, f1 = cbwa_weights(n_samples, macro_f1, alpha, **kw), np.asarray(macro_f1)
    for i, s in enumerate(sites):
        f = f"{f1[i]:6.3f}" if f1 is not None else "     -"
        lines.append(f"{s:6s} {n_samples[i]:7d} {fa[i]:9.4f} {cb[i]:9.4f} "
                     f"{cb[i]/max(fa[i],1e-12):7.2f} {f}")
    return "\n".join(lines)
