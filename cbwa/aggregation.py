"""Class-Balance-Aware Weighted Aggregation (CBWA): the server-side rule.

Equation numbers refer to the paper (Agarwal, Patil & Saman, *Class-Balance-Aware
Weighted Aggregation for Federated Pediatric Brain Tumor MRI Classification
and Segmentation*):

* Eq. (1)  theta^{t+1} = sum_k w_k theta_k                 -> :func:`weighted_average`
* Eq. (4)  q_k = n_k^alpha (F_k + eps)^gamma / sum_j (...)  -> :func:`cbwa_scores`
* floor    every hospital keeps at least w_min              -> :func:`apply_floor`
* Eq. (5)  w_k^(t) ∝ m w_k^(t-1) + (1 - m) q_k              -> :func:`momentum_update`

:class:`CBWA` chains the three steps (Algorithm 1, steps 6-8) and keeps the
momentum state between rounds.

Only NumPy is needed. Model parameters are passed as dictionaries of arrays,
for example a PyTorch ``state_dict`` converted with
``{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}``.
"""

from __future__ import annotations

from typing import Dict, Hashable, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "fedavg_weights",
    "cbwa_scores",
    "apply_floor",
    "momentum_update",
    "effective_number",
    "weighted_average",
    "CBWA",
]

# Settings used in the paper (Table 2 and Table 5).
ALPHA = 0.5    # size-dampening exponent
GAMMA = 1.0    # performance exponent
EPS = 0.05     # smoothing constant added to F_k
W_MIN = 0.01   # weight floor per hospital
M = 0.7        # momentum on the weights across rounds


def _as_1d(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain only finite numbers")
    return a


def fedavg_weights(n: Sequence[float]) -> np.ndarray:
    """FedAvg weights: each hospital in proportion to its number of patients."""
    n = _as_1d(n, "n")
    if np.any(n <= 0):
        raise ValueError("every hospital needs at least one training patient")
    return n / n.sum()


def cbwa_scores(
    n: Sequence[float],
    f1: Sequence[float],
    alpha: float = ALPHA,
    gamma: float = GAMMA,
    eps: float = EPS,
) -> np.ndarray:
    """Eq. (4): normalized CBWA weights before the floor and momentum.

    ``q_k = n_k**alpha * (F_k + eps)**gamma / sum_j(...)``

    ``n``  : training patients per hospital (n_k > 0).
    ``f1`` : validation macro-F1 of each hospital's *updated* model, computed
             by the hospital on its own validation patients (0 <= F_k <= 1).
    """
    n = _as_1d(n, "n")
    f1 = _as_1d(f1, "f1")
    if n.shape != f1.shape:
        raise ValueError("n and f1 must have the same length")
    if np.any(n <= 0):
        raise ValueError("every hospital needs at least one training patient")
    if np.any((f1 < 0) | (f1 > 1)):
        raise ValueError("macro-F1 values must lie in [0, 1]")
    if eps <= 0 and gamma != 0:
        raise ValueError("eps must be > 0, otherwise a hospital with F_k = 0 gets weight 0")
    raw = n ** alpha * (f1 + eps) ** gamma
    return raw / raw.sum()


def apply_floor(q: Sequence[float], w_min: float = W_MIN) -> np.ndarray:
    """Weight floor: raise every weight below ``w_min`` to ``w_min``.

    The total deficit is taken from the other hospitals in proportion to their
    excess above the floor, so the result still sums to one. One pass is
    enough (Proposition 2), provided ``K * w_min < 1``.
    """
    q = _as_1d(q, "q")
    if np.any(q < 0):
        raise ValueError("weights must be non-negative")
    q = q / q.sum()
    if w_min <= 0:
        return q.copy()
    k = q.size
    if k * w_min >= 1:
        raise ValueError(f"the floor needs K * w_min < 1 (got K={k}, w_min={w_min})")
    below = q < w_min
    if not below.any():
        return q.copy()
    deficit = float(np.sum(w_min - q[below]))
    excess = q[~below] - w_min                      # >= 0 by construction
    out = q.copy()
    out[below] = w_min
    out[~below] = q[~below] - deficit * excess / excess.sum()
    return out


def momentum_update(w_prev: Sequence[float], q: Sequence[float], m: float = M) -> np.ndarray:
    """Eq. (5): ``w^(t) ∝ m * w^(t-1) + (1 - m) * q``, renormalized."""
    w_prev = _as_1d(w_prev, "w_prev")
    q = _as_1d(q, "q")
    if w_prev.shape != q.shape:
        raise ValueError("w_prev and q must have the same length")
    if not 0 <= m < 1:
        raise ValueError("momentum m must lie in [0, 1)")
    w = m * w_prev + (1 - m) * q
    return w / w.sum()


def effective_number(w: Sequence[float]) -> float:
    """Effective number of participating hospitals, ``K_eff = 1 / sum_k w_k^2``.

    Equals K for uniform weights and 1 when a single hospital has all the weight.
    """
    w = _as_1d(w, "w")
    w = w / w.sum()
    return float(1.0 / np.sum(w ** 2))


def weighted_average(
    params: Sequence[Mapping[str, np.ndarray]],
    weights: Sequence[float],
) -> Dict[str, np.ndarray]:
    """Eq. (1): weighted average of the hospitals' parameters.

    ``params`` is one ``{name: array}`` dictionary per hospital, all with the
    same names and shapes. Averaging is done in 64-bit floating point (as the
    paper's server step does) and each result is cast back to the dtype of the
    first hospital's array. Integer arrays (for example BatchNorm's
    ``num_batches_tracked``) are rounded to the nearest integer.
    """
    weights = _as_1d(weights, "weights")
    if len(params) != weights.size:
        raise ValueError("need exactly one weight per hospital")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-9):
        raise ValueError("weights must be non-negative and sum to one")
    names = list(params[0].keys())
    for p in params[1:]:
        if list(p.keys()) != names:
            raise ValueError("all hospitals must send the same parameter names, in the same order")
    out: Dict[str, np.ndarray] = {}
    for name in names:
        ref = np.asarray(params[0][name])
        acc = np.zeros(ref.shape, dtype=np.float64)
        for w, p in zip(weights, params):
            arr = np.asarray(p[name])
            if arr.shape != ref.shape:
                raise ValueError(f"shape mismatch for parameter '{name}'")
            acc += w * arr.astype(np.float64)
        if np.issubdtype(ref.dtype, np.integer):
            acc = np.rint(acc)
        out[name] = acc.astype(ref.dtype)
    return out


class CBWA:
    """Stateful CBWA server rule: Eq. (4) -> floor -> Eq. (5) momentum.

    One call to :meth:`weights` per communication round (Algorithm 1, steps
    6-8). The momentum state is keyed by hospital ID. It starts from the first
    round's floored weights, and it restarts from fresh weights whenever the
    set of participating hospitals changes or :meth:`reset` is called (for
    example after resuming from a checkpoint that did not store the state).

    Setting ``alpha=1, gamma=0, w_min=0, m=0`` recovers FedAvg exactly.
    """

    def __init__(
        self,
        alpha: float = ALPHA,
        gamma: float = GAMMA,
        eps: float = EPS,
        w_min: float = W_MIN,
        m: float = M,
    ) -> None:
        if not 0 <= m < 1:
            raise ValueError("momentum m must lie in [0, 1)")
        self.alpha, self.gamma, self.eps, self.w_min, self.m = alpha, gamma, eps, w_min, m
        self._prev: Optional[Dict[Hashable, float]] = None

    def reset(self) -> None:
        """Forget the momentum state; the next round starts from fresh weights."""
        self._prev = None

    @property
    def state(self) -> Optional[Dict[Hashable, float]]:
        """Weights of the previous round by hospital ID (``None`` before the first round)."""
        return None if self._prev is None else dict(self._prev)

    def weights(
        self,
        hospital_ids: Sequence[Hashable],
        n: Sequence[float],
        f1: Sequence[float],
    ) -> np.ndarray:
        """Aggregation weights for this round, in the order of ``hospital_ids``."""
        ids = list(hospital_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("hospital IDs must be unique")
        if len(ids) != len(n):
            raise ValueError("need one n and one F1 per hospital")
        q = cbwa_scores(n, f1, self.alpha, self.gamma, self.eps)
        q = apply_floor(q, self.w_min)
        if self._prev is None or set(self._prev) != set(ids) or self.m == 0:
            w = q
        else:
            w_prev = np.array([self._prev[i] for i in ids], dtype=np.float64)
            w = momentum_update(w_prev, q, self.m)
        self._prev = {i: float(x) for i, x in zip(ids, w)}
        return w

    def aggregate(
        self,
        hospital_ids: Sequence[Hashable],
        n: Sequence[float],
        f1: Sequence[float],
        params: Sequence[Mapping[str, np.ndarray]],
    ):
        """One server step: CBWA weights, then the weighted parameter average.

        Returns ``(new_global_params, weights)``.
        """
        w = self.weights(hospital_ids, n, f1)
        return weighted_average(params, w), w
