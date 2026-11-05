"""CBWA: Class-Balance-Aware Weighted Aggregation for federated learning.

Independent reference implementation of the aggregation rule described in
Agarwal, Patil & Saman, "Class-Balance-Aware Weighted Aggregation for Federated
Pediatric Brain Tumor MRI Classification and Segmentation".
"""

from .aggregation import (
    ALPHA,
    EPS,
    GAMMA,
    M,
    W_MIN,
    CBWA,
    apply_floor,
    cbwa_scores,
    effective_number,
    fedavg_weights,
    momentum_update,
    weighted_average,
)
from .class_weights import BETA, class_balanced_weights

__version__ = "1.0.0"

__all__ = [
    "CBWA",
    "cbwa_scores",
    "apply_floor",
    "momentum_update",
    "fedavg_weights",
    "effective_number",
    "weighted_average",
    "class_balanced_weights",
    "ALPHA",
    "GAMMA",
    "EPS",
    "W_MIN",
    "M",
    "BETA",
    "__version__",
]
