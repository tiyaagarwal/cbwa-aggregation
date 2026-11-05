"""
Local objective. Closes Gap 3: nothing in the base paper corrects a client's
training objective for its OWN internal class skew before the update is sent.

Three pieces:
    ClassBalancedCE   effective-number reweighting (Cui et al., 2019)
    MaskedDiceLoss    Dice that skips normal controls
    JointLoss         CE + Dice, the base paper's combination
    prox_penalty      FedProx's (mu/2)||w - w_global||^2, for the baseline

The weight computation is pure numpy so it can be unit-tested without torch;
only the loss modules need PyTorch.
"""

from typing import Dict, Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:                                    # allows import w/o torch
    _HAS_TORCH = False
    nn = type("nn", (), {"Module": object})            # type: ignore


# --------------------------------------------------------------------------
# Effective-number class weights (numpy, testable)
# --------------------------------------------------------------------------

def effective_number_weights(class_counts: Sequence[int], beta: float = 0.999,
                             normalise: bool = True) -> np.ndarray:
    """w_c ∝ (1 - beta) / (1 - beta^{n_c}), per Cui et al. 2019.

    beta is the only knob and it matters more than it looks:
      beta = 0     -> uniform (no reweighting)
      beta -> 1    -> approaches inverse-frequency weighting
      beta = 0.999 -> sensible when the rarest class has O(100s) of samples

    CALIBRATE BETA TO THE SITE, NOT THE FEDERATION. Effective number saturates
    at 1/(1-beta); with beta=0.999 that is 1000. A site holding 4 EP cases and
    200 MB gets an EP weight ~50x the MB weight, which will dominate its local
    gradient and can make that client's update actively harmful. At the
    per-site counts in this dataset (EP is the rarest class, 210 patients
    across 19 sites), beta=0.99 is the safer starting point. Log the resulting
    weights per site and sanity-check the extremes before training.

    Classes with zero local samples get weight 0 — a client cannot learn a
    class it has never seen, and a nonzero weight there is meaningless.
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    if np.any(counts < 0):
        raise ValueError("negative class counts")
    w = np.zeros_like(counts)
    present = counts > 0
    if not present.any():
        raise ValueError("client has no samples in any class")
    if beta <= 0:
        w[present] = 1.0
    else:
        eff = (1.0 - np.power(beta, counts[present])) / (1.0 - beta)
        w[present] = 1.0 / eff
    if normalise:
        # Mean 1 over PRESENT classes keeps the loss on the same scale as
        # unweighted CE, so the paper's lr=1e-4 transfers without retuning.
        w = w / w[present].mean()
    return w


def weight_report(site_counts: Dict[str, Sequence[int]], classes: Sequence[str],
                  beta: float = 0.99) -> str:
    """Print per-site class weights so extreme values are caught before training."""
    head = f"{'site':6s} " + " ".join(f"{c:>14s}" for c in classes) + "   max/min"
    lines = [head]
    for site, counts in site_counts.items():
        w = effective_number_weights(counts, beta)
        nz = w[w > 0]
        ratio = f"{nz.max()/nz.min():8.1f}x" if nz.size else "       -"
        cells = " ".join(f"{c:5d}/{wi:7.2f}" for c, wi in zip(counts, w))
        flag = "  <-- CHECK" if nz.size and nz.max() / nz.min() > 25 else ""
        lines.append(f"{site:6s} {cells}  {ratio}{flag}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Torch losses
# --------------------------------------------------------------------------

if _HAS_TORCH:

    class ClassBalancedCE(nn.Module):
        """Cross-entropy with per-client effective-number class weights."""

        def __init__(self, class_counts: Sequence[int], beta: float = 0.99,
                     label_smoothing: float = 0.0):
            super().__init__()
            w = effective_number_weights(class_counts, beta)
            self.register_buffer("class_weights", torch.tensor(w, dtype=torch.float32))
            self.label_smoothing = label_smoothing

        def forward(self, logits: "torch.Tensor", target: "torch.Tensor"):
            return F.cross_entropy(logits, target, weight=self.class_weights,
                                   label_smoothing=self.label_smoothing)

    class MaskedDiceLoss(nn.Module):
        """Soft Dice over the 64x256x256 volume, skipping tumour-free cases.

        Normal controls have an all-zero target. Dice against all-zero is
        degenerate — the model minimises it by predicting nothing anywhere,
        which drags the segmentation head toward empty output for real tumours
        too. `has_tumour` gates those samples out of the Dice term entirely
        while they still contribute to the classification term.
        """

        def __init__(self, smooth: float = 1.0):
            super().__init__()
            self.smooth = smooth

        def forward(self, logits, target, has_tumour=None):
            probs = torch.sigmoid(logits)
            b = probs.shape[0]
            p = probs.reshape(b, -1)
            t = target.reshape(b, -1).float()
            inter = (p * t).sum(1)
            dice = (2 * inter + self.smooth) / (p.sum(1) + t.sum(1) + self.smooth)
            loss = 1.0 - dice
            if has_tumour is None:
                return loss.mean()
            m = has_tumour.reshape(-1).float()
            if m.sum() < 1:
                # No tumour cases in this batch. Return a real zero attached to
                # the graph; a bare 0.0 would break backward().
                return (loss * 0.0).sum()
            return (loss * m).sum() / m.sum()

    class JointLoss(nn.Module):
        """CE + Dice, as the base paper uses, with the class-balanced CE."""

        def __init__(self, class_counts: Sequence[int], beta: float = 0.99,
                     dice_weight: float = 1.0, ce_weight: float = 1.0):
            super().__init__()
            self.ce = ClassBalancedCE(class_counts, beta)
            self.dice = MaskedDiceLoss()
            self.dice_weight, self.ce_weight = dice_weight, ce_weight

        def forward(self, cls_logits, seg_logits, labels, masks, has_tumour=None):
            ce = self.ce(cls_logits, labels)
            dc = self.dice(seg_logits, masks, has_tumour)
            return {"loss": self.ce_weight * ce + self.dice_weight * dc,
                    "ce": ce.detach(), "dice": dc.detach()}

    def prox_penalty(model, global_params, mu: float):
        """FedProx: (mu/2) * ||w - w_global||^2, added to the local loss.

        This is the ONLY thing FedProx changes. The server still runs plain
        FedAvg. mu=0 disables it and recovers the paper's reported best setting.
        """
        if mu <= 0:
            return torch.zeros((), device=next(model.parameters()).device)
        acc = 0.0
        for p, g in zip(model.parameters(), global_params):
            acc = acc + ((p - g.detach()) ** 2).sum()
        return 0.5 * mu * acc
