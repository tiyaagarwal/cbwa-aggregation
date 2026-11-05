"""
File IO for the real FLPedBrain-processed release.

Corrected against the actual repository listing. The release is NOT pickles:

    AU_data_uint8_train.npy    201 MB
    BO_data_uint8_train.npy    117 MB
    CG_data_uint8_train.npy    940 MB
    ...

Four consequences for the earlier code, all handled here:

  1. Files are .npy, not .pkl. Every glob("*.pkl") found nothing.
  2. The site code is the FIRST token, not the last. The old
     `stem.split("_")[-1]` returned "TRAIN" for all 19 files, which would have
     collapsed the entire federation into one client named TRAIN.
  3. .npy files holding dicts or object arrays need allow_pickle=True.
     Without it np.load raises on anything but a plain numeric array.
  4. "uint8" in the filename means intensities are ALREADY 0-255. The paper's
     normalisation step has been applied upstream. Re-normalising per volume
     would re-stretch each volume to full range, which is NOT the same
     operation and would silently diverge from the published baseline.

Size arithmetic says the files hold images and masks together: 9.46 GB at
2 x 4.19 MB per patient is ~1128 patients, matching the paper's "approximately
1200 patients" for the released training split. Confirm with inspect_npy().
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

SITE_FILE_GLOBS = ("*.npy", "*.npz", "*.pkl", "*.pickle")


def site_from_filename(path: Path) -> str:
    """'AU_data_uint8_train.npy' -> 'AU'.

    The old implementation took the LAST underscore token and returned 'TRAIN'
    for every file in the release. Worth an assertion in your own tests.
    """
    stem = Path(path).stem
    token = stem.split("_")[0].upper()
    if not (1 <= len(token) <= 3) or not token.isalpha():
        raise ValueError(
            f"cannot parse a site code from {Path(path).name!r} (got {token!r}). "
            "Pass an explicit site_from_name= to preprocess_all()."
        )
    return token


def split_from_filename(path: Path) -> str:
    """'AU_data_uint8_train.npy' -> 'train'. Returns '' if no split marker."""
    stem = Path(path).stem.lower()
    for s in ("train", "val", "valid", "test", "holdout"):
        if f"_{s}" in stem:
            return s
    return ""


def load_site_file(path: Path) -> Any:
    """Load one site file regardless of container format."""
    path = Path(path)
    if path.suffix == ".npz":
        return dict(np.load(path, allow_pickle=True))
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        # A dict saved via np.save round-trips as a 0-d object array.
        if arr.dtype == object and arr.ndim == 0:
            return arr.item()
        return arr
    import pickle
    with open(path, "rb") as fh:
        return pickle.load(fh)


def list_site_files(raw_dir: Path) -> List[Path]:
    raw_dir = Path(raw_dir)
    files: List[Path] = []
    for g in SITE_FILE_GLOBS:
        files.extend(raw_dir.glob(g))
        files.extend(raw_dir.glob(f"**/{g}"))
    # Skip model checkpoints and anything under demos/.
    files = [f for f in files
             if "checkpoint" not in str(f).lower() and "demo" not in str(f).lower()]
    return sorted(set(files))


def peek_npy_header(path: Path) -> Dict[str, Any]:
    """Read shape and dtype from the .npy header WITHOUT loading the array.

    This is the cheap way to answer 'what is in here' for a 1.5 GB file. Use it
    before committing to a full download or a full load.
    """
    with open(path, "rb") as fh:
        version = np.lib.format.read_magic(fh)
        if version[0] == 1:
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
    n_bytes = Path(path).stat().st_size
    return {"shape": list(shape), "dtype": str(dtype), "fortran_order": fortran,
            "file_mb": round(n_bytes / 1e6, 1),
            "is_object_array": dtype == np.dtype("O")}


def inspect_npy(path: Path) -> Dict[str, Any]:
    """Full structural report for one .npy site file."""
    path = Path(path)
    out: Dict[str, Any] = {
        "filename": path.name,
        "site": site_from_filename(path),
        "split": split_from_filename(path),
    }
    try:
        out["header"] = peek_npy_header(path)
    except Exception as e:
        out["header_error"] = repr(e)

    obj = load_site_file(path)
    out["loaded_type"] = type(obj).__name__

    if isinstance(obj, dict):
        out["keys"] = list(obj)
        out["fields"] = {k: _describe(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        out["array"] = _describe(obj)
        if obj.dtype != object:
            out["interpretation"] = _guess_layout(obj.shape)
    return out


def _describe(v: Any) -> Dict[str, Any]:
    if isinstance(v, np.ndarray):
        d = {"shape": list(v.shape), "dtype": str(v.dtype)}
        if np.issubdtype(v.dtype, np.number) and v.size:
            s = v.reshape(-1)
            if s.size > 1_000_000:
                s = s[:: max(1, s.size // 1_000_000)]
            d.update({"min": float(s.min()), "max": float(s.max()),
                      "mean": round(float(s.astype(np.float64).mean()), 3)})
            u = np.unique(s)
            if u.size <= 12:
                d["unique"] = [float(x) for x in u]
            else:
                d["n_unique_approx"] = int(u.size)
        return d
    return {"type": type(v).__name__, "repr": repr(v)[:120]}


def _guess_layout(shape: Tuple[int, ...]) -> str:
    """Interpret a bare array shape against the paper's 64x256x256 geometry."""
    s = list(shape)
    if len(s) == 5 and s[1] == 2:
        return (f"{s[0]} patients x [image, mask] x {s[2]}x{s[3]}x{s[4]} "
                "-- channel 0 is the volume, channel 1 the mask")
    if len(s) == 5 and s[-1] == 2:
        return (f"{s[0]} patients x {s[1]}x{s[2]}x{s[3]} x [image, mask] "
                "-- last axis is the channel")
    if len(s) == 4 and s[1] == 64:
        return f"{s[0]} patients x 64x{s[2]}x{s[3]} volumes -- masks must be elsewhere"
    if len(s) == 4 and s[-1] == 64:
        return f"{s[0]} patients x {s[1]}x{s[2]} x 64 -- depth is LAST, needs transpose"
    if len(s) == 3:
        return f"{s[0]} flat 2D frames of {s[1]}x{s[2]} -- reassemble by patient id"
    return "unrecognised layout -- inspect manually"
