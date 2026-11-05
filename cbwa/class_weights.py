"""Effective-number class weights for the local loss (Eq. (3); Cui et al., CVPR 2019).

Each hospital computes these from its *own* training labels, so no
information leaves the hospital.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["class_balanced_weights"]

BETA = 0.99  # effective-number parameter used in the paper


def class_balanced_weights(counts: Sequence[float], beta: float = BETA) -> np.ndarray:
    """Eq. (3): ``a_c ∝ (1 - beta) / (1 - beta**N_c)``.

    ``counts`` holds N_c, the number of training patients of each class at one
    hospital. Weights are normalized to a mean of one over the classes that are
    present (N_c > 0). Classes the hospital does not hold get weight 0; they
    never occur in its labels, so their weight has no effect on the loss.

    The result can be passed as the ``weight`` of a weighted cross-entropy
    loss (for example ``torch.nn.CrossEntropyLoss(weight=...)``).
    """
    c = np.asarray(counts, dtype=np.float64)
    if c.ndim != 1 or c.size == 0:
        raise ValueError("counts must be a non-empty 1-D sequence")
    if np.any(c < 0) or not np.all(np.isfinite(c)):
        raise ValueError("counts must be finite and non-negative")
    if not 0 <= beta < 1:
        raise ValueError("beta must lie in [0, 1)")
    present = c > 0
    if not present.any():
        raise ValueError("at least one class must be present")
    a = np.zeros_like(c)
    a[present] = (1.0 - beta) / (1.0 - beta ** c[present])
    a[present] *= present.sum() / a[present].sum()
    return a
