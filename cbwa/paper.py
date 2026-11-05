"""Numbers from the paper, and checks that recompute them with this package.

Every value below is copied from the paper as printed (rounded). Each check
recomputes the value from the CBWA rule and compares it at the paper's
rounding. ``python scripts/reproduce_paper.py`` prints the full table.

Numbering follows the Elsevier (LaTeX) version of the paper: Section 3.4 is
the case-wise analysis, Section 3.5 the properties, Section 5.6 the weights
of the real federation, Fig. 8 their plot. The Word version numbers the same
items III-D, III-E, V-F and uses Roman numerals for tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, sqrt
from typing import Dict, List, Tuple

import numpy as np

from .aggregation import (
    CBWA,
    apply_floor,
    cbwa_scores,
    effective_number,
    fedavg_weights,
)
from .class_weights import class_balanced_weights

# ---------------------------------------------------------------------------
# Data copied from the paper
# ---------------------------------------------------------------------------

# Four-hospital example of Section 3.4, Table 3 and Fig. 2 (training + validation patients).
EXAMPLE_N = [246, 40, 10, 2]
EXAMPLE_F_CASE5 = [0.40, 0.60, 0.80, 0.90]
EXAMPLE_F_CASE6 = [0.40, 0.60, 0.80, 0.00]
# Case 7 / Fig. 2b: H1-H3 keep the Case 5 scores; H4's validation F1 in rounds 1-10.
CASE7_F4 = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]

# Real federation. Training patients per hospital (Table 4) and the final CBWA
# model's validation macro-F1 per hospital (Table 9, column "F1 CBWA").
HOSPITALS = ["ST", "SE", "CG", "UT", "IN", "CP", "TO", "GO",
             "PH", "DY", "NY", "DU", "BO", "CH", "TM", "KC"]
N_TRAIN = [197, 144, 90, 77, 70, 58, 54, 47, 32, 16, 15, 14, 11, 7, 7, 1]
F1_CBWA_FINAL = [0.594, 0.752, 0.345, 0.497, 0.651, 0.384, 0.426, 0.375,
                 0.748, 0.542, 1.000, 0.857, 0.000, 0.222, 0.333, 1.000]

# Example of Eq. (3) in Section 3.3: "a hospital with 72 MB and 31 EP patients".
# These are ST's patient counts in Table 4 (with 59 DIPG and 84 PA); the ratio of
# the EP and MB weights depends only on the 31 and 72. Order: EP, DIPG, MB, PA.
ST_CLASS_COUNTS = [31, 59, 72, 84]


# ---------------------------------------------------------------------------
# Recomputation
# ---------------------------------------------------------------------------

def table3() -> Dict[str, np.ndarray]:
    """Weights of the four-hospital example under Cases 1-6 (Table 3)."""
    n = EXAMPLE_N
    q6 = cbwa_scores(n, EXAMPLE_F_CASE6)
    return {
        "Cases 1-3: size only": fedavg_weights(n),
        "Case 4: dampened size, equal scores": cbwa_scores(n, [0.5] * 4),
        "Case 5: dampened size x (F + eps)": cbwa_scores(n, EXAMPLE_F_CASE5),
        "Case 6 before the floor": q6,
        "Case 6: Case 5 with F4 = 0 and floor": apply_floor(q6),
    }


def case7_series() -> Tuple[np.ndarray, np.ndarray]:
    """H4's weight in rounds 1-10 of Case 7 (Fig. 2b): without and with momentum."""
    server = CBWA()
    no_mom, with_mom = [], []
    for f4 in CASE7_F4:
        f = EXAMPLE_F_CASE5[:3] + [f4]
        no_mom.append(apply_floor(cbwa_scores(EXAMPLE_N, f))[3])
        with_mom.append(server.weights(["H1", "H2", "H3", "H4"], EXAMPLE_N, f)[3])
    return np.array(no_mom), np.array(with_mom)


def federation_weights() -> Tuple[np.ndarray, np.ndarray]:
    """FedAvg and CBWA weights of the 16 training hospitals (Section 5.6, Fig. 8).

    As in the paper, the CBWA weights are computed once from the final model's
    validation macro-F1 (Eq. (4) with the floor, no momentum).
    """
    fedavg = fedavg_weights(N_TRAIN)
    cbwa = apply_floor(cbwa_scores(N_TRAIN, F1_CBWA_FINAL))
    return fedavg, cbwa


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

@dataclass
class Check:
    where: str      # where the value appears in the paper
    paper: str      # the value as printed in the paper
    computed: str   # the recomputed value, printed the same way
    ok: bool


def _fmt(where: str, paper: str, value: float, decimals: int) -> Check:
    shown = f"{value:.{decimals}f}"
    return Check(where, paper, shown, shown == paper)


def checks() -> List[Check]:
    """All checks against the paper, in reading order."""
    out: List[Check] = []

    # Table 3 (Section 3.4)
    t3 = table3()
    printed = {
        "Cases 1-3: size only": ["0.826", "0.134", "0.034", "0.007"],
        "Case 4: dampened size, equal scores": ["0.590", "0.238", "0.119", "0.053"],
        "Case 5: dampened size x (F + eps)": ["0.464", "0.270", "0.177", "0.088"],
        "Case 6: Case 5 with F4 = 0 and floor": ["0.504", "0.294", "0.192", "0.010"],
    }
    for case, values in printed.items():
        for h, (paper, w) in enumerate(zip(values, t3[case])):
            out.append(_fmt(f"Table 3, {case}, H{h + 1}", paper, w, 3))
    out.append(_fmt("Sec. 3.4, Case 6: H4 weight before the floor", "0.005",
                    t3["Case 6 before the floor"][3], 3))

    # Case 7 / Fig. 2b
    no_mom, with_mom = case7_series()
    out.append(_fmt("Sec. 3.4, Case 7: lowest weight without momentum", "0.010", no_mom.min(), 3))
    out.append(_fmt("Sec. 3.4, Case 7: highest weight without momentum", "0.097", no_mom.max(), 3))
    out.append(_fmt("Sec. 3.4, Case 7: lowest weight with m = 0.7 (rounds 2-10)", "0.05", with_mom[1:].min(), 2))
    out.append(_fmt("Sec. 3.4, Case 7: highest weight with m = 0.7 (rounds 2-10)", "0.08", with_mom[1:].max(), 2))

    # FedAvg as a special case
    n_arr = np.array(N_TRAIN, dtype=float)
    special = CBWA(alpha=1.0, gamma=0.0, w_min=0.0, m=0.0)
    w_special = special.weights(HOSPITALS, n_arr, F1_CBWA_FINAL)
    diff = float(np.max(np.abs(w_special - fedavg_weights(n_arr))))
    out.append(Check("Sec. 3.4: alpha = 1, gamma = 0, w_min = 0, m = 0 recovers FedAvg exactly",
                     "exact", f"max diff {diff:.1e}", diff < 1e-15))

    # Propositions (Section 3.5)
    eps, m, w_min, k = 0.05, 0.7, 0.01, 16
    factor = (1 + eps) / eps
    out.append(Check("Sec. 3.5, Prop. 1: largest change from the score, (1 + eps)/eps", "21",
                     f"{factor:g}", isclose(factor, 21.0)))
    out.append(_fmt("Sec. 3.5, Prop. 1: size term compresses 197 : 1 to about", "14", sqrt(197 / 1), 0))
    out.append(_fmt("Sec. 3.5, Prop. 2: floor reserves at most K * w_min (%)", "16", 100 * k * w_min, 0))
    out.append(_fmt("Sec. 3.5, Prop. 3: a one-round jump moves the weight by (1 - m) (%)", "30", 100 * (1 - m), 0))

    # Eq. (3) example (Section 3.3)
    a = class_balanced_weights(ST_CLASS_COUNTS)
    out.append(_fmt("Sec. 3.3, Eq. (3): EP patient weighted about twice an MB patient (ratio)", "2",
                    a[0] / a[2], 0))

    # Real federation (Section 5.6, Fig. 8)
    fedavg, cbwa = federation_weights()
    idx = {h: i for i, h in enumerate(HOSPITALS)}
    for h, before, after in [("NY", "0.018", "0.067"), ("DU", "0.017", "0.056"),
                             ("ST", "0.235", "0.149"), ("CG", "0.107", "0.062")]:
        out.append(_fmt(f"Sec. 5.6: {h} weight under FedAvg", before, fedavg[idx[h]], 3))
        out.append(_fmt(f"Sec. 5.6: {h} weight under CBWA", after, cbwa[idx[h]], 3))
    out.append(_fmt("Sec. 5.6: BO sits at the floor", "0.010", cbwa[idx["BO"]], 3))
    out.append(_fmt("Sec. 6: KC reaches only", "0.017", cbwa[idx["KC"]], 3))
    two = [idx["ST"], idx["SE"]]
    out.append(_fmt("Sec. 5.6: share of the two largest hospitals, FedAvg (%)", "40.6", 100 * fedavg[two].sum(), 1))
    out.append(_fmt("Sec. 5.6: share of the two largest hospitals, CBWA (%)", "30.8", 100 * cbwa[two].sum(), 1))
    small = np.array(N_TRAIN) <= 32
    out.append(_fmt("Sec. 5.6: number of hospitals with 32 or fewer training patients", "8", small.sum(), 0))
    out.append(_fmt("Sec. 5.6: their share, FedAvg (%)", "12.3", 100 * fedavg[small].sum(), 1))
    out.append(_fmt("Sec. 5.6: their share, CBWA (%)", "29.3", 100 * cbwa[small].sum(), 1))
    out.append(_fmt("Sec. 5.6: effective number of hospitals K_eff, FedAvg", "7.9", effective_number(fedavg), 1))
    out.append(_fmt("Sec. 5.6: effective number of hospitals K_eff, CBWA", "11.0", effective_number(cbwa), 1))

    return out


__all__ = [
    "EXAMPLE_N", "EXAMPLE_F_CASE5", "EXAMPLE_F_CASE6", "CASE7_F4",
    "HOSPITALS", "N_TRAIN", "F1_CBWA_FINAL", "ST_CLASS_COUNTS",
    "table3", "case7_series", "federation_weights", "Check", "checks",
]
