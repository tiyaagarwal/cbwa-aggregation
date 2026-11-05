"""
Stage 1 - Preprocessing, matching the base paper's stated procedure.

From the Nature Methods section, the authors' preprocessing is exactly two
steps, deliberately minimal:

    (1) normalise each 3D image to a 0-255 intensity range
    (2) extract 64 congruent axial slices of 256 x 256

No skull-stripping. No atlas registration. They chose this so the model works
end-to-end on raw clinical scans -- including skull, scalp, and head sizes from
infants to teenagers. Adding "better" preprocessing here would depart from the
baseline and make your numbers incomparable to their Tables 3 and 4. Match
them first; ablate later if you want.

Two things worth knowing about that choice:

  - Normalisation is per-volume, so it is computable by each hospital in
    isolation. That is not incidental: a normaliser fitted across the
    federation would move information across a site boundary that in a real
    deployment cannot cross, breaking the FL premise. Keep it per-volume.

  - 64 slices was chosen to absorb the huge range of slice thicknesses
    (1-5 mm) and sequence types across the 19 sites, so a fixed depth works
    for everyone.

Output layout:
    out_dir/
        ST/volumes.npy   uint8 (N, 64, 256, 256)   memmapped
        ST/masks.npy     uint8 (N, 64, 256, 256)
        ST/meta.json     labels, patient ids, has_tumour, tumour fraction
        manifest.json    per-site counts, class histograms, provenance
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import zoom

from .io import list_site_files, load_site_file, site_from_filename
from .schema import (CLASSES, NORMAL_IDX, TARGET_DEPTH, TARGET_HW, Volume,
                     adapt_site)


@dataclass
class PreprocessConfig:
    target_depth: int = TARGET_DEPTH
    target_hw: Tuple[int, int] = TARGET_HW
    # "Congruent" 64 slices. 'center' takes the middle 64 of the stack, which
    # for an axial head study is where the posterior fossa sits. 'resample'
    # rescales the whole stack to 64 along z instead -- safer when stacks are
    # much shorter than 64, since centre-cropping would then pad heavily.
    depth_strategy: str = "center"          # "center" | "resample"
    # If a tumour mask would be cropped away by the depth selection, prefer a
    # window centred on the tumour instead. Losing the tumour entirely turns a
    # tumour case into a silent false negative.
    center_on_tumour: bool = True
    # The release ships uint8 volumes -- the paper's 0-255 normalisation has
    # ALREADY been applied upstream. Re-running it would re-stretch each volume
    # to full range, a different operation, silently diverging from the
    # published baseline. Left False; set True only for raw non-uint8 input.
    renormalise: bool = False


def _as_uint8(vol: np.ndarray) -> np.ndarray:
    """Pass through data that is already 0-255, erroring loudly if it is not."""
    if vol.dtype == np.uint8:
        return vol
    lo, hi = float(np.min(vol)), float(np.max(vol))
    if lo < -0.5 or hi > 255.5:
        raise ValueError(
            f"expected pre-normalised 0-255 data, got range [{lo:.1f}, {hi:.1f}]. "
            "Set PreprocessConfig(renormalise=True) if this really is raw input."
        )
    return np.clip(np.round(vol), 0, 255).astype(np.uint8)


def normalise_volume(vol: np.ndarray) -> np.ndarray:
    """Scale the whole 3D image to 0-255, as the paper specifies.

    Per-volume, not per-slice: a slice-wise rescale would flatten the
    intensity relationship between slices within one study.
    """
    v = vol.astype(np.float32)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-6:
        return np.zeros(v.shape, dtype=np.uint8)
    return np.clip(np.round((v - lo) / (hi - lo) * 255.0), 0, 255).astype(np.uint8)


def select_depth(vol: np.ndarray, mask: np.ndarray, cfg: PreprocessConfig):
    """Reduce/expand the z axis to exactly target_depth congruent slices."""
    d = vol.shape[0]
    target = cfg.target_depth

    if cfg.depth_strategy == "resample" or d < target:
        f = target / d
        vol = zoom(vol.astype(np.float32), (f, 1, 1), order=1)
        mask = zoom(mask.astype(np.float32), (f, 1, 1), order=0)
        vol, mask = _fix_depth(vol, target), _fix_depth(mask, target)
        return vol, (mask > 0.5).astype(np.uint8)

    if d == target:
        return vol, mask

    start = (d - target) // 2
    if cfg.center_on_tumour and mask.any():
        z = np.where(mask.any(axis=(1, 2)))[0]
        centre = int((z.min() + z.max()) // 2)
        start = int(np.clip(centre - target // 2, 0, d - target))
    return vol[start:start + target], mask[start:start + target]


def _fix_depth(arr: np.ndarray, target: int) -> np.ndarray:
    if arr.shape[0] > target:
        return arr[:target]
    if arr.shape[0] < target:
        pad = target - arr.shape[0]
        return np.pad(arr, ((0, pad), (0, 0), (0, 0)), mode="edge")
    return arr


def resample_hw(arr: np.ndarray, hw: Tuple[int, int], is_mask: bool) -> np.ndarray:
    """Resize in-plane to 256x256. Nearest for masks so labels stay {0,1}."""
    if arr.shape[1:] == tuple(hw):
        return arr
    f = (1.0, hw[0] / arr.shape[1], hw[1] / arr.shape[2])
    out = zoom(arr.astype(np.float32), f, order=0 if is_mask else 1)
    out = out[:, :hw[0], :hw[1]]
    ph, pw = hw[0] - out.shape[1], hw[1] - out.shape[2]
    if ph > 0 or pw > 0:
        out = np.pad(out, ((0, 0), (0, max(ph, 0)), (0, max(pw, 0))), mode="constant")
    return (out > 0.5).astype(np.uint8) if is_mask else out


def preprocess_site(pkl_path: Path, site: str, out_dir: Path,
                    cfg: Optional[PreprocessConfig] = None) -> Dict:
    cfg = cfg or PreprocessConfig()
    out_dir = Path(out_dir) / site
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_site_file(pkl_path)
    vols: List[Volume] = adapt_site(raw, site)

    processed, lost_tumour = [], 0
    for v in vols:
        v.validate()
        vol, msk = select_depth(v.volume, v.mask, cfg)
        had_tumour = bool(v.mask.any())
        vol = resample_hw(vol, cfg.target_hw, False)
        msk = resample_hw(msk, cfg.target_hw, True)
        if had_tumour and not msk.any():
            lost_tumour += 1          # depth window missed the tumour entirely
            continue
        v.volume = normalise_volume(vol) if cfg.renormalise else _as_uint8(vol)
        v.mask = msk
        processed.append(v)

    if not processed:
        raise RuntimeError(f"site {site}: no usable volumes")

    n, d, (h, w) = len(processed), cfg.target_depth, cfg.target_hw
    volumes = np.lib.format.open_memmap(
        out_dir / "volumes.npy", mode="w+", dtype=np.uint8, shape=(n, d, h, w))
    masks = np.lib.format.open_memmap(
        out_dir / "masks.npy", mode="w+", dtype=np.uint8, shape=(n, d, h, w))
    for i, v in enumerate(processed):
        volumes[i], masks[i] = v.volume, v.mask
    volumes.flush(); masks.flush()

    meta = {
        "site": site,
        "labels": [v.label for v in processed],
        "patient_ids": [v.patient_id for v in processed],
        "has_tumour": [bool(v.has_tumour) for v in processed],
        "tumour_fraction": [round(v.tumour_voxel_fraction, 6) for v in processed],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta))

    counts = np.bincount(meta["labels"], minlength=len(CLASSES)).tolist()
    tf = [f for f, t in zip(meta["tumour_fraction"], meta["has_tumour"]) if t]
    return {
        "site": site,
        "n_patients": n,
        "class_counts": dict(zip(CLASSES, counts)),
        "missing_classes": [c for c, k in zip(CLASSES, counts) if k == 0],
        "lost_tumour_in_crop": lost_tumour,
        "median_tumour_fraction": round(float(np.median(tf)), 6) if tf else 0.0,
        "gb_on_disk": round(2 * n * d * h * w / 1e9, 3),
        "source_file": pkl_path.name,
    }


def preprocess_all(raw_dir: Path, out_dir: Path,
                   cfg: Optional[PreprocessConfig] = None,
                   site_from_name=site_from_filename) -> Dict:
    cfg = cfg or PreprocessConfig()
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_site_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"no site files (.npy/.npz/.pkl) under {raw_dir}")

    entries, failures = [], []
    for f in files:
        site = site_from_name(f)
        try:
            e = preprocess_site(f, site, out_dir, cfg)
            entries.append(e)
            warn = f"  LOST-TUMOUR={e['lost_tumour_in_crop']}" if e["lost_tumour_in_crop"] else ""
            print(f"[ok]   {site:4s} {e['n_patients']:5d} patients  "
                  f"{e['gb_on_disk']:6.2f} GB  missing={e['missing_classes']}{warn}")
        except Exception as ex:
            failures.append({"file": f.name, "site": site, "error": repr(ex)})
            print(f"[FAIL] {site:4s} {ex}")

    manifest = {"config": asdict(cfg), "classes": CLASSES,
                "sites": entries, "failures": failures, "totals": _totals(entries)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _report(manifest)
    return manifest


def _totals(entries: List[Dict]) -> Dict:
    tot = {c: 0 for c in CLASSES}
    for e in entries:
        for c, k in e["class_counts"].items():
            tot[c] += k
    return {"class_counts": tot,
            "n_patients": sum(e["n_patients"] for e in entries),
            "gb_on_disk": round(sum(e["gb_on_disk"] for e in entries), 2)}


def _report(m: Dict) -> None:
    print("\n" + "=" * 72)
    print("PREPROCESSING SUMMARY")
    print("=" * 72)
    tot = m["totals"]["class_counts"]
    grand = sum(tot.values()) or 1
    print(f"{'class':16s} {'patients':>9s} {'share':>8s} {'sites w/ 0':>12s}")
    for c in CLASSES:
        zero = sum(1 for e in m["sites"] if e["class_counts"][c] == 0)
        print(f"{c:16s} {tot[c]:9d} {100*tot[c]/grand:7.1f}% {zero:12d}")
    print(f"\ntotal {m['totals']['n_patients']} patients across "
          f"{len(m['sites'])} sites, {m['totals']['gb_on_disk']} GB on disk")

    if tot["NormalControl"] == 0:
        print("\n[expected] NormalControl is absent. The paper's normal cohort")
        print("(N=1667) was NOT part of the public release -- it must be")
        print("requested from the authors. Until then this is a 4-class problem.")

    tumour = {c: k for c, k in tot.items() if c != "NormalControl"}
    if any(tumour.values()):
        rarest = min(tumour, key=tumour.get)
        absent = sum(1 for e in m["sites"] if e["class_counts"][rarest] == 0)
        print(f"\nRarest tumour class: {rarest} ({tumour[rarest]} patients), "
              f"absent at {absent}/{len(m['sites'])} sites.")
        print("This is the imbalance CBWA targets. Compare against the paper's")
        print("cohort totals (MB 596, EP 210, PA 335, DIPG 327) -- EP is the")
        print("rarest there and is the class their Fig. 5c ablation tracks.")

    if m["failures"]:
        print(f"\n!! {len(m['failures'])} site(s) failed -- see manifest.json")
