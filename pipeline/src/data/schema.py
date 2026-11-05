"""
Canonical schema -- VOLUME-based, matching the base paper's actual design.

Corrected against the Nature Methods section (fetched from the publisher),
which supersedes the earlier frame-based guess:

  - The model is a 3D-UNet with a Kinetics-pretrained encoder, consuming
    64 slices per datum. The unit of data is a VOLUME, not a frame.
  - Preprocessing is deliberately minimal: normalise each 3D image to a
    0-255 range, extract 64 congruent axial slices of 256x256. No
    skull-stripping, no atlas registration -- the authors wanted end-to-end
    clinical deployability on raw scans.

Canonical record:
    volume     uint8 (64, 256, 256)   axial T2, normalised 0-255
    mask       uint8 (64, 256, 256)   whole-tumour mask, 0/1
    label      int                    class index, see CLASSES
    patient_id str                    unique per patient
    site       str                    two-letter site code

One record == one patient, so train/val splits are leak-free by construction.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

CLASSES = ["NormalControl", "EP", "DIPG", "MB", "PA"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NORMAL_IDX = 0

# Paper cohort totals (Results, "Study cohort"): 1468 tumour subjects.
# Normals (N=1667, from ST) were distributed across sites for training only
# and NEVER appeared in validation or hold-out sets.
PAPER_CLASS_TOTALS = {"MB": 596, "EP": 210, "PA": 335, "DIPG": 327}
NORMAL_COHORT_SIZE = 1667

IN_NETWORK_SITES = ["ST", "SE", "PH", "UT", "CH", "DY", "IN", "CG", "NY",
                    "CP", "DU", "BO", "TO", "GO", "KC", "TM"]
EXTERNAL_HOLDOUT_SITES = ["TU", "TK", "AU"]
WARMUP_SITES = ["ST", "SE"]

TARGET_DEPTH = 64
TARGET_HW = (256, 256)

_LABEL_ALIASES = {
    "normal": "NormalControl", "control": "NormalControl", "nc": "NormalControl",
    "normalcontrol": "NormalControl", "none": "NormalControl", "healthy": "NormalControl",
    "ependymoma": "EP", "ep": "EP",
    "dipg": "DIPG", "diffuseintrinsicpontineglioma": "DIPG", "dmg": "DIPG",
    "medulloblastoma": "MB", "mb": "MB", "medullo": "MB",
    "pilocytic": "PA", "pa": "PA", "pilocyticastrocytoma": "PA", "astrocytoma": "PA",
}


def normalise_label(raw: Any) -> int:
    if isinstance(raw, (int, np.integer)):
        idx = int(raw)
        if not 0 <= idx < len(CLASSES):
            raise ValueError(f"label index {idx} outside 0..{len(CLASSES)-1}")
        return idx
    if isinstance(raw, np.ndarray):
        if raw.ndim == 1 and raw.size == len(CLASSES):
            return int(np.argmax(raw))
        if raw.size == 1:
            return normalise_label(raw.item())
    if isinstance(raw, str):
        key = raw.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
        if key in _LABEL_ALIASES:
            return CLASS_TO_IDX[_LABEL_ALIASES[key]]
        raise ValueError(f"unrecognised label {raw!r}; add it to _LABEL_ALIASES")
    raise TypeError(f"cannot interpret label of type {type(raw)}")


@dataclass
class Volume:
    volume: np.ndarray
    mask: np.ndarray
    label: int
    patient_id: str
    site: str

    def validate(self) -> None:
        if self.volume.ndim != 3:
            raise ValueError(f"volume must be 3D, got {self.volume.shape}")
        if self.mask.shape != self.volume.shape:
            raise ValueError(f"mask {self.mask.shape} != volume {self.volume.shape}")
        if not 0 <= self.label < len(CLASSES):
            raise ValueError(f"bad label {self.label}")
        if self.label == NORMAL_IDX and self.mask.any():
            raise ValueError(f"{self.patient_id}: normal control has non-empty mask")

    @property
    def has_tumour(self) -> bool:
        """Normal controls have no tumour. The Dice term must be skipped for
        them -- Dice against an all-zero target is degenerate."""
        return self.label != NORMAL_IDX and bool(self.mask.any())

    @property
    def tumour_voxel_fraction(self) -> float:
        """The paper found tumour volume varies significantly between sites
        (Fig. 6) -- worth logging per site as a heterogeneity check."""
        return float(self.mask.mean())


# --------------------------------------------------------------------------
# ADAPTER -- the one function to rewrite once Stage 0 reports the real schema
# --------------------------------------------------------------------------

def adapt_site(raw: Any, site: str) -> List[Volume]:
    """Convert one unpickled site object into a list of per-patient Volumes.

    Handles both plausible releases:
      - already-3D volumes per patient
      - 2D 'static frames' that must be REASSEMBLED into volumes by patient id

    The frame case is why patient ids matter so much. Without them the frames
    cannot be reassembled and the paper's volumetric Dice is unreproducible.
    """
    if isinstance(raw, dict) and _looks_like_arrays(raw):
        return _from_parallel_arrays(raw, site)
    if isinstance(raw, dict):
        out = []
        for pid, payload in raw.items():
            out.extend(_from_patient_payload(payload, site, f"{site}_{_as_str(pid)}"))
        return out
    if isinstance(raw, (list, tuple)):
        out = []
        for i, item in enumerate(raw):
            out.extend(_from_patient_payload(item, site, f"{site}_{i:05d}"))
        return out
    raise TypeError(f"unhandled top-level type {type(raw)} for site {site}")


def _key(d: Dict, *cands: str) -> Optional[str]:
    lower = {k.lower(): k for k in d if isinstance(k, str)}
    for c in cands:
        if c in lower:
            return lower[c]
    return None


def _looks_like_arrays(d: Dict) -> bool:
    k = _key(d, "images", "image", "volumes", "x", "data", "frames")
    return k is not None and isinstance(d[k], np.ndarray)


def _from_parallel_arrays(d: Dict, site: str) -> List[Volume]:
    ik = _key(d, "images", "image", "volumes", "x", "data", "frames")
    mk = _key(d, "masks", "mask", "y", "seg", "segmentation")
    lk = _key(d, "labels", "label", "y_cls", "class", "classes", "tumor_type")
    pk = _key(d, "patient_ids", "patient_id", "patients", "subject_ids", "pid", "ids")
    zk = _key(d, "slice_idx", "slice", "z", "slice_index")

    arr = d[ik]
    labels = d[lk] if lk else None
    if labels is None:
        raise KeyError(f"site {site}: no label array. Keys: {list(d)}")
    if pk is None:
        raise KeyError(
            f"site {site}: NO PATIENT-ID ARRAY. Keys: {list(d)}. STOP.\n"
            "Without patient ids you cannot (a) reassemble frames into the "
            "64-slice volumes the paper's model consumes, or (b) split without "
            "leaking a patient across train and val. Email the authors first."
        )
    pids = d[pk]
    masks = d[mk] if mk else np.zeros_like(arr, dtype=np.uint8)

    # Case A: already per-patient volumes, shape (N, D, H, W).
    if arr.ndim == 4:
        return [Volume(np.asarray(arr[i]), (np.asarray(masks[i]) > 0).astype(np.uint8),
                       normalise_label(labels[i]), f"{site}_{_as_str(pids[i])}", site)
                for i in range(len(arr))]

    # Case B: flat 2D frames, shape (N, H, W). Reassemble by patient.
    if arr.ndim == 3:
        return _reassemble_frames(arr, masks, labels, pids,
                                  d[zk] if zk else None, site)

    raise ValueError(f"site {site}: unexpected image array ndim {arr.ndim}")


def _reassemble_frames(imgs, masks, labels, pids, zidx, site) -> List[Volume]:
    pids = np.asarray([_as_str(p) for p in pids])
    order = np.arange(len(imgs)) if zidx is None else np.asarray(zidx)
    out = []
    for pid in dict.fromkeys(pids.tolist()):        # preserve file order
        sel = np.where(pids == pid)[0]
        sel = sel[np.argsort(order[sel])]           # stack in slice order
        out.append(Volume(
            volume=np.stack([np.asarray(imgs[i]) for i in sel]),
            mask=np.stack([(np.asarray(masks[i]) > 0).astype(np.uint8) for i in sel]),
            label=normalise_label(labels[sel[0]]),
            patient_id=f"{site}_{pid}",
            site=site,
        ))
    return out


def _from_patient_payload(item: Any, site: str, pid: str) -> List[Volume]:
    if not isinstance(item, dict):
        raise TypeError(f"expected dict record, got {type(item)}")
    ik = _key(item, "image", "volume", "img", "x", "data")
    mk = _key(item, "mask", "seg", "segmentation", "y_seg")
    lk = _key(item, "label", "class", "tumor_type", "y", "diagnosis")
    pk = _key(item, "patient_id", "subject_id", "pid", "case_id", "study_id")

    vol = np.asarray(item[ik])
    if vol.ndim == 2:                                # single frame -> depth 1
        vol = vol[None]
    msk = (np.asarray(item[mk]) > 0).astype(np.uint8) if mk else np.zeros_like(vol, np.uint8)
    if msk.ndim == 2:
        msk = msk[None]
    return [Volume(vol, msk, normalise_label(item[lk]),
                   f"{site}_{_as_str(item[pk])}" if pk else pid, site)]


def _as_str(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.generic):
        return str(v.item())
    return str(v)
