"""
Stage 2 - Build the federation artefacts CBWA actually needs.

Preprocessing is not finished when the images are normalised. The proposed
method needs three things that are *preprocessing outputs*, not training-time
outputs:

    1. Per-client class counts        -> the effective-number local loss weights
    2. Dampened sample-count weights  -> the n_i term of the CBWA rule
    3. A class-balanced reference set -> the macro-F1 term of the CBWA rule

and one thing that protects the whole result:

    4. Patient-level train/val splits -> no slice from a patient in both sides
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .schema import CLASSES, EXTERNAL_HOLDOUT_SITES

# --------------------------------------------------------------------------
# 4. Leak-free splits
# --------------------------------------------------------------------------

def patient_level_split(
    patient_ids: List[str],
    labels: List[int],
    val_frac: float = 0.2,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split frame indices so that no patient appears on both sides.

    Splitting frames at random is the single most common way to get a medical
    imaging result that looks excellent and means nothing: adjacent slices of
    one tumour are near-duplicates, so a random split leaks the answer.
    Stratified by the patient's class so rare classes survive the split.
    """
    pids = np.asarray(patient_ids)
    labs = np.asarray(labels)
    rng = np.random.default_rng(seed)

    uniq = np.unique(pids)
    # One label per patient (mode); frames of a patient share a diagnosis.
    pat_label = {p: np.bincount(labs[pids == p]).argmax() for p in uniq}

    val_patients = set()
    for c in range(len(CLASSES)):
        members = np.array([p for p in uniq if pat_label[p] == c])
        if members.size == 0:
            continue
        rng.shuffle(members)
        n_val = max(1, int(round(val_frac * members.size))) if members.size > 1 else 0
        val_patients.update(members[:n_val].tolist())

    is_val = np.isin(pids, list(val_patients))
    return np.where(~is_val)[0], np.where(is_val)[0]


# --------------------------------------------------------------------------
# 1. Effective-number class weights (Cui et al., 2019) - the local loss
# --------------------------------------------------------------------------

def effective_number_weights(
    class_counts: np.ndarray,
    beta: float = 0.999,
    normalise: bool = True,
) -> np.ndarray:
    """w_c proportional to (1 - beta) / (1 - beta^{n_c}).

    beta is the knob. beta=0 gives uniform weights (no reweighting); beta->1
    approaches inverse-frequency weighting. 0.999 is the paper's default and is
    reasonable when the rarest class has O(100s) of samples. If a site holds
    only a handful of a class, inverse-frequency weighting will hand that
    class an enormous weight and destabilise local training -- prefer a lower
    beta (0.99) there and record the choice.

    Classes with zero local samples get weight 0: a client cannot learn a class
    it has never seen, and a nonzero weight on an absent class is meaningless.
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    w = np.zeros_like(counts)
    present = counts > 0
    if beta <= 0:
        w[present] = 1.0
    else:
        eff = (1.0 - np.power(beta, counts[present])) / (1.0 - beta)
        w[present] = 1.0 / eff
    if normalise and w.sum() > 0:
        # Scale so weights average 1 over PRESENT classes; keeps the loss on a
        # comparable scale to unweighted CE, so learning rates transfer.
        w = w / w[present].mean()
    return w


# --------------------------------------------------------------------------
# 2. Dampened sample-count term of CBWA
# --------------------------------------------------------------------------

def dampened_size_weights(n_per_client: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """n_i^alpha / sum_j n_j^alpha.

    alpha=1 recovers FedAvg exactly. alpha=0 gives every client an equal vote.
    alpha=0.5 (sqrt) is the proposal's "sub-linear" term: ST and SE keep more
    influence than a tiny site, but cannot out-vote the several small sites
    that together hold most of a rare class.

    Report the FedAvg weights alongside these in the write-up -- the size of
    the gap is the mechanism, and a reviewer will want to see it.
    """
    n = np.asarray(n_per_client, dtype=np.float64)
    s = np.power(n, alpha)
    return s / s.sum()


# --------------------------------------------------------------------------
# 3. Class-balanced server reference set - READ THE CAVEAT
# --------------------------------------------------------------------------

def build_reference_set(
    site_dirs: Dict[str, Path],
    per_class: int = 40,
    source_sites: Optional[List[str]] = None,
    exclude_sites: Optional[List[str]] = None,
    seed: int = 0,
) -> Dict:
    """Assemble a small, class-balanced set for the server to score clients on.

    !! PRIVACY CAVEAT -- do not skip this !!
    CBWA scores each client's updated weights on a set held AT THE SERVER. In a
    real federation the server has no patient data, so a reference set carved
    out of client data is not something a hospital network would permit. The
    proposal must say which of these it assumes, and defend it:

      (a) Carved from the warm-up sites (ST, SE) under the existing DUA.
          Simplest, matches the paper's warm-up already treating those sites
          specially. Weakest privacy story -- the server now holds real scans.
      (b) A small public pediatric set (e.g. BraTS-PEDs) as a proxy.
          Best privacy story, but distribution shift may make the macro-F1
          signal noisy. Worth reporting as an ablation.
      (c) Client-reported metrics only -- each client scores its own update on
          its own held-out split and reports a scalar macro-F1. No data moves.
          Cheapest to defend; vulnerable to a client misreporting.

    !! LEAKAGE GUARD !!
    The reference set must never draw from TU/TK/AU. Those are the external
    hold-outs the whole claim rests on. If CBWA's aggregation weights are tuned
    against them, the headline result is tuned on test data and is worthless.
    That is enforced below, not left to discipline.
    """
    rng = np.random.default_rng(seed)
    exclude = set(exclude_sites or []) | set(EXTERNAL_HOLDOUT_SITES)
    sources = [s for s in (source_sites or site_dirs) if s not in exclude]
    if not sources:
        raise ValueError("no eligible source sites after excluding hold-outs")

    pool: Dict[int, List[Tuple[str, int]]] = {c: [] for c in range(len(CLASSES))}
    used_patients: Dict[str, set] = {}
    for site in sources:
        meta = json.loads((Path(site_dirs[site]) / "meta.json").read_text())
        for i, lab in enumerate(meta["labels"]):
            pool[lab].append((site, i))
        used_patients[site] = set(meta["patient_ids"])

    chosen, shortfalls = [], {}
    for c, items in pool.items():
        if not items:
            shortfalls[CLASSES[c]] = 0
            continue
        idx = rng.permutation(len(items))[:per_class]
        chosen.extend([items[i] + (c,) for i in idx])
        if len(idx) < per_class:
            shortfalls[CLASSES[c]] = len(idx)

    if shortfalls:
        print(f"[warn] reference set under-filled for {shortfalls}. "
              f"CBWA's macro-F1 term will be noisy on those classes; consider "
              f"lowering per_class so the set stays balanced.")

    return {
        "entries": [{"site": s, "index": int(i), "label": int(c)} for s, i, c in chosen],
        "per_class_target": per_class,
        "source_sites": sources,
        "excluded_sites": sorted(exclude),
        "shortfalls": shortfalls,
        "patients_consumed": {s: sorted(p) for s, p in used_patients.items()},
        "note": "Patients listed in patients_consumed must be removed from the "
                "corresponding client's training split, or the server is scoring "
                "clients on data they trained on.",
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def build_federation(
    processed_dir: Path,
    val_frac: float = 0.2,
    beta: float = 0.999,
    alpha: float = 0.5,
    ref_per_class: int = 40,
    seed: int = 0,
) -> Dict:
    processed_dir = Path(processed_dir)
    manifest = json.loads((processed_dir / "manifest.json").read_text())
    sites = [e["site"] for e in manifest["sites"]]
    site_dirs = {s: processed_dir / s for s in sites}

    clients, n_per_client = {}, []
    for e in manifest["sites"]:
        site = e["site"]
        meta = json.loads((site_dirs[site] / "meta.json").read_text())
        tr, va = patient_level_split(meta["patient_ids"], meta["labels"], val_frac, seed)
        counts = np.bincount(np.asarray(meta["labels"])[tr], minlength=len(CLASSES))
        clients[site] = {
            "train_idx": tr.tolist(),
            "val_idx": va.tolist(),
            "train_class_counts": counts.tolist(),
            "class_weights": effective_number_weights(counts, beta).round(6).tolist(),
            "n_train": int(tr.size),
            "n_patients": e["n_patients"],
            "is_external_holdout": site in EXTERNAL_HOLDOUT_SITES,
        }
        if site not in EXTERNAL_HOLDOUT_SITES:
            n_per_client.append(int(tr.size))

    in_net = [s for s in sites if s not in EXTERNAL_HOLDOUT_SITES]
    fedavg_w = np.asarray(n_per_client, float) / max(sum(n_per_client), 1)
    cbwa_size_w = dampened_size_weights(np.asarray(n_per_client), alpha)
    for s, fa, cb in zip(in_net, fedavg_w, cbwa_size_w):
        clients[s]["fedavg_weight"] = float(fa)
        clients[s]["cbwa_size_weight"] = float(cb)

    ref = build_reference_set(site_dirs, ref_per_class, source_sites=in_net, seed=seed)

    fed = {"clients": clients, "in_network_sites": in_net,
           "external_holdout_sites": [s for s in sites if s in EXTERNAL_HOLDOUT_SITES],
           "reference_set": ref,
           "params": {"val_frac": val_frac, "beta": beta, "alpha": alpha, "seed": seed}}
    (processed_dir / "federation.json").write_text(json.dumps(fed, indent=2))

    _report(fed, in_net, fedavg_w, cbwa_size_w)
    return fed


def _report(fed, in_net, fedavg_w, cbwa_w):
    print("\n" + "=" * 72)
    print("FEDERATION SUMMARY")
    print("=" * 72)
    print(f"{'site':6s} {'n_train':>8s} {'FedAvg w':>10s} {'CBWA size w':>12s} "
          f"{'ratio':>7s}  missing classes")
    for s, fa, cb in zip(in_net, fedavg_w, cbwa_w):
        miss = [CLASSES[i] for i, k in
                enumerate(fed["clients"][s]["train_class_counts"]) if k == 0]
        print(f"{s:6s} {fed['clients'][s]['n_train']:8d} {fa:10.4f} {cb:12.4f} "
              f"{cb/max(fa,1e-9):7.2f}  {','.join(miss) or '-'}")

    ext = fed["external_holdout_sites"]
    print(f"\nexternal hold-outs present: {ext or 'NONE'}")
    if not ext:
        print("!! TU/TK/AU are absent from the processed release. The proposal's")
        print("!! external evaluation (Sections 5.5, 9, 10.5) cannot run as written.")
        print("!! Fallback: hold out 3 in-network sites chosen to mirror TK/AU's")
        print("!! class imbalance, and say plainly in the write-up that the")
        print("!! external sites are simulated, not the paper's actual ones.")
