"""
Run this in Colab AFTER run_analysis_only.py finished.

It does two things to the /content/processed files you already have:

1. Fixes up each site's meta.json so it has what the splitter needs:
   integer class labels (0-4, matching NormalControl/EP/DIPG/MB/PA) instead
   of just the class name strings, and a has_tumour flag.

2. Builds a proper train/validation split for each site -- by PATIENT, so no
   scan from one patient ends up on both sides -- and computes the real
   class-balance weights for the local loss.

TK and AU are excluded from the split entirely and marked as external
evaluation sites, per the paper's own designation -- their filenames say
"_train" but the paper is explicit that these were never trained on.
"""
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("/content/processed")
CLASSES = ["NormalControl", "EP", "DIPG", "MB", "PA"]
EXTERNAL_HOLDOUT = {"TK", "AU", "TU"}   # TU not downloaded; TK/AU present but held out

manifest = json.loads((OUT_DIR / "manifest.json").read_text())
sites = [e["site"] for e in manifest["sites"]]

federation = {"clients": {}, "in_network_sites": [], "external_holdout_sites": []}

for site in sites:
    d = OUT_DIR / site
    meta = json.loads((d / "meta.json").read_text())
    meta["labels"] = [CLASSES.index(n) for n in meta["label_name"]]
    meta["has_tumour"] = [True] * len(meta["labels"])   # no NormalControl in this release
    (d / "meta.json").write_text(json.dumps(meta))

    if site in EXTERNAL_HOLDOUT:
        federation["external_holdout_sites"].append(site)
        federation["clients"][site] = {"role": "external_eval_only",
                                       "n_patients": len(meta["labels"])}
        print(f"[external] {site:4s} {len(meta['labels']):4d} patients -- "
              f"held out entirely, not split")
        continue

    pids, labs = meta["patient_ids"], meta["labels"]
    rng = np.random.default_rng(0)
    uniq = list(dict.fromkeys(pids))
    pat_label = {p: labs[pids.index(p)] for p in uniq}
    val_patients = set()
    for c in range(len(CLASSES)):
        members = [p for p in uniq if pat_label[p] == c]
        if not members:
            continue
        rng.shuffle(members)
        n_val = max(1, round(0.2 * len(members))) if len(members) > 1 else 0
        val_patients.update(members[:n_val])
    is_val = np.array([p in val_patients for p in pids])
    train_idx = np.where(~is_val)[0].tolist()
    val_idx = np.where(is_val)[0].tolist()

    counts = np.bincount([labs[i] for i in train_idx], minlength=len(CLASSES))
    present = counts > 0
    eff = np.zeros(len(CLASSES))
    beta = 0.99
    eff[present] = (1 - beta ** counts[present]) / (1 - beta)
    w = np.zeros(len(CLASSES))
    w[present] = 1.0 / eff[present]
    if w[present].mean() > 0:
        w = w / w[present].mean()

    federation["clients"][site] = {
        "role": "in_network",
        "train_idx": train_idx, "val_idx": val_idx,
        "n_train": len(train_idx), "n_val": len(val_idx),
        "class_weights": w.round(4).tolist(),
    }
    federation["in_network_sites"].append(site)
    print(f"[in-net]   {site:4s} train={len(train_idx):4d} val={len(val_idx):3d}  "
          f"weights={np.round(w,2).tolist()}")

(OUT_DIR / "federation.json").write_text(json.dumps(federation, indent=2))
print(f"\nWrote {OUT_DIR/'federation.json'}")
print(f"In-network training sites: {len(federation['in_network_sites'])}")
print(f"External evaluation sites: {federation['external_holdout_sites']}")
print("\nEverything needed for training is now in /content/processed/.")
print("Next: copy processed/ to Drive so it survives a session restart, e.g.")
print("  !cp -r /content/processed /content/drive/MyDrive/fedcb_processed")
