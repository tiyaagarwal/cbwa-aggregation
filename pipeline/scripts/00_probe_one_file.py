"""
Stage 0 (fast path) — resolve the schema from ONE small file.

KC_data_uint8_train.npy is 16.8 MB against the repository's 9.46 GB. Downloading
just that file answers every schema question for ~0.2% of the transfer, before
you commit to the full pull. If the layout turns out to be something the adapter
does not handle, you have lost seconds rather than an hour.

It also fetches README.md (234 bytes) and lists demos/, which very likely
contains the authors' own loading code — the definitive answer to how these
files are meant to be read.

Usage (in Colab, after `huggingface-cli login`):
    python scripts/00_probe_one_file.py
    python scripts/00_probe_one_file.py --file SE_data_uint8_train.npy
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = "edhlee/FLPedBrain-processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="KC_data_uint8_train.npy",
                    help="smallest site file; override to probe another")
    ap.add_argument("--out", type=Path, default=Path("probe.json"))
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files

    print(f"Listing {REPO} ...")
    try:
        files = list_repo_files(REPO, repo_type="dataset")
    except Exception as e:
        sys.exit(f"Could not list repo ({e}). Run `huggingface-cli login` and "
                 f"confirm your access request was approved.")

    npys = sorted(f for f in files if f.endswith(".npy") and "/" not in f)
    sites = sorted({f.split("_")[0] for f in npys})
    print(f"\n{len(files)} files, {len(npys)} top-level .npy, {len(sites)} sites:")
    print("  " + " ".join(sites))

    # The question that decides your evaluation protocol.
    external = {"TU", "TK", "AU"}
    present = external & set(sites)
    print(f"\nExternal hold-out sites present: {sorted(present) or 'NONE'}")
    if present == external:
        print("  All three are here. The paper's Table 4 protocol is reproducible.")
    elif present:
        print(f"  Only {sorted(present)}. Missing {sorted(external - present)} —")
        print("  still email the authors for the rest.")

    other = [f for f in files if not f.endswith(".npy")]
    print(f"\nNon-.npy files ({len(other)}):")
    for f in other:
        print(f"  {f}")

    report = {"repo": REPO, "n_files": len(files), "sites": sites,
              "external_holdouts_present": sorted(present), "other_files": other}

    # README is tiny and may document the schema outright.
    for small in ("README.md",):
        if small in files:
            try:
                p = hf_hub_download(REPO, small, repo_type="dataset")
                txt = Path(p).read_text()[:2000]
                print(f"\n--- {small} ---\n{txt}\n---")
                report["readme"] = txt
            except Exception as e:
                print(f"could not fetch {small}: {e}")

    demos = [f for f in files if f.startswith("demos/")]
    if demos:
        print(f"\ndemos/ contains {len(demos)} files — likely the authors' own")
        print("loader. Fetch and read these; they settle the schema definitively:")
        for f in demos[:10]:
            print(f"  {f}")
        report["demos"] = demos

    print(f"\nDownloading {args.file} ...")
    local = hf_hub_download(REPO, args.file, repo_type="dataset")
    print(f"  -> {local} ({Path(local).stat().st_size/1e6:.1f} MB)")

    from fedcb_data.io import inspect_npy
    info = inspect_npy(Path(local))
    report["probe"] = info
    print("\n" + "=" * 70)
    print(json.dumps(info, indent=1, default=str))
    print("=" * 70)

    args.out.write_text(json.dumps(report, indent=1, default=str))
    print(f"\nwrote {args.out} — send me this file.")


if __name__ == "__main__":
    main()
