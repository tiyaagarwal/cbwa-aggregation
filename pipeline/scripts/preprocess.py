"""
Continuation: run this AFTER the download is already done (files in /content/raw).
Skips the download step entirely.
"""
import json
from pathlib import Path
import numpy as np

RAW_DIR = Path("/content/raw")
OUT_DIR = Path("/content/processed")
PAPER_TUMOUR_TOTALS = {"EP": 210, "DIPG": 327, "MB": 596, "PA": 335}

files = sorted(RAW_DIR.glob("*_data_uint8_train.npy"))
print(f"Found {len(files)} site files already downloaded.\n")

print("=" * 72)
print("PASS 1 -- read every file once, build label histogram, check the")
print("          label_classes_plane hypothesis")
print("=" * 72)

site_data = {}          # site -> loaded dict, kept in memory (fits: ~9.5GB, 87GB disk/RAM permitting)
label_counts = {}       # raw code -> total patient count across all sites
plane_hypothesis_checks = []

for f in files:
    site = f.stem.split("_")[0].upper()
    d = np.load(f, allow_pickle=True).item()
    site_data[site] = d

    for code in d["label_classes"]:
        label_counts[int(code)] = label_counts.get(int(code), 0) + 1

    # Check: does label_classes_plane match "mask present in this slice"?
    # xs/ys are (N,256,256,64) -> mask-per-slice is ys.sum(axis=(1,2)) > 0
    mask_per_slice = (d["ys_uint8"].sum(axis=(1, 2)) > 0).astype(int)
    match = (mask_per_slice == d["label_classes_plane"]).mean()
    plane_hypothesis_checks.append((site, round(float(match), 4)))

print(f"\n{'site':6s} {'n_patients':>10s}   raw label codes present")
for site, d in site_data.items():
    codes = sorted(set(int(c) for c in d["label_classes"]))
    print(f"{site:6s} {len(d['label_classes']):10d}   {codes}")

print(f"\nGlobal raw label code -> total patient count:")
for code, n in sorted(label_counts.items()):
    print(f"  code {code}: {n} patients")

print(f"\nlabel_classes_plane == (mask present per slice) match rate per site:")
for site, rate in plane_hypothesis_checks:
    flag = "" if rate > 0.95 else "  <-- LOW, hypothesis may be wrong for this site"
    print(f"  {site:6s} {rate:6.2%}{flag}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PROPOSED LABEL MAPPING -- verify this against the printed counts above")
print("=" * 72)
codes_sorted = sorted(label_counts, key=lambda c: -label_counts[c])
classes_sorted = sorted(PAPER_TUMOUR_TOTALS, key=lambda c: -PAPER_TUMOUR_TOTALS[c])
proposed = {}
print(f"{'raw code':>10s} {'observed n':>12s}   {'closest class':>14s} {'paper n':>9s}")
for code, cls in zip(codes_sorted, classes_sorted):
    proposed[code] = cls
    print(f"{code:10d} {label_counts[code]:12d}   {cls:>14s} {PAPER_TUMOUR_TOTALS[cls]:9d}")
print("\nThis maps by RANK (biggest observed count -> biggest paper count), which")
print("only works if the release's site subset preserves that ranking. If two")
print("counts above are close, don't trust the mapping without double-checking")
print("against the paper's Table 2 per-site breakdown.")
print(f"\nproposed mapping: {proposed}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PASS 2 -- write canonical per-site files")
print("=" * 72)
OUT_DIR.mkdir(parents=True, exist_ok=True)
manifest = {"sites": [], "label_mapping_raw_to_name": proposed}

for site, d in site_data.items():
    n = d["xs_uint8"].shape[0]
    # (N, 256, 256, 64) -> (N, 64, 256, 256): move depth to axis 1
    vol = np.transpose(d["xs_uint8"], (0, 3, 1, 2))
    msk = np.transpose(d["ys_uint8"], (0, 3, 1, 2))
    labels_raw = d["label_classes"].astype(int)
    labels_named = [proposed.get(int(c), f"UNKNOWN_{c}") for c in labels_raw]

    site_dir = OUT_DIR / site
    site_dir.mkdir(exist_ok=True)
    np.save(site_dir / "volumes.npy", vol)     # already uint8, already 0-255
    np.save(site_dir / "masks.npy", msk)       # already uint8, 0/1
    meta = {
        "site": site, "n_patients": int(n),
        "patient_ids": [f"{site}_{i:03d}" for i in range(n)],
        "label_raw": labels_raw.tolist(),
        "label_name": labels_named,
    }
    (site_dir / "meta.json").write_text(json.dumps(meta))

    counts = {c: labels_named.count(c) for c in set(labels_named)}
    manifest["sites"].append({"site": site, "n_patients": int(n), "class_counts": counts})
    print(f"[ok] {site:6s} {n:4d} patients  {counts}")

(OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

print("\n" + "=" * 72)
print(f"DONE. Processed data in {OUT_DIR}, {len(site_data)} sites.")
print("Volumes are (n, 64, 256, 256) uint8, already in the paper's 0-255 range.")
print("=" * 72)
print("\nSEND ME THE PRINTED OUTPUT ABOVE (especially the label histogram and")
print("the plane-hypothesis match rates) so I can confirm the mapping is right")
print("before you train on it.")
