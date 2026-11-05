# Paired significance tests: CBWA vs FedAvg on the same patients (no retraining)
import sys, json
from pathlib import Path
import numpy as np, torch
from scipy.stats import wilcoxon, binomtest
ROOT = Path("/kaggle/working"); PROCESSED = ROOT / "processed"; OUT = ROOT / "fedcb_results"
sys.path.insert(0, str(ROOT / "FLPedBrain/FLPedBrain-PyTorch/src"))
from model_pretrained import PedBrainNetPretrained
DEV = torch.device("cuda:0")
fed = json.loads((PROCESSED / "federation.json").read_text())
IN_NET = [s for s in fed["in_network_sites"] if s not in ("AU", "TK")]
SETS = [(s, fed["clients"][s]["val_idx"]) for s in IN_NET] + [("AU", None), ("TK", None)]

def load_model(path):
    m = PedBrainNetPretrained(num_classes=5, pretrained=True, weights_path=str(ROOT / "rgb_imagenet.pt"))
    sd = torch.load(path, map_location="cpu")
    m.load_state_dict(sd.get("model_state_dict", sd))
    return m.to(DEV).eval()

@torch.no_grad()
def run(model, site, idx):
    d = PROCESSED / site
    vols = np.load(d / "volumes.npy", mmap_mode="r"); masks = np.load(d / "masks.npy", mmap_mode="r")
    labels = json.loads((d / "meta.json").read_text())["labels"]
    pred, true, dsc = [], [], []
    for i in (range(len(labels)) if idx is None else idx):
        x = torch.from_numpy(np.asarray(vols[i], np.float32) / 255.0)[None, None].to(DEV)
        m = torch.from_numpy(np.asarray(masks[i], np.float32)).to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            seg, _, _, cls_l = model(x)
        p = (seg[0, 0] > 0.5).float()
        pred.append(int(cls_l.argmax(1))); true.append(int(labels[i]))
        dsc.append(float((2 * (p * m).sum() + 1) / (p.sum() + m.sum() + 1)) if labels[i] > 0 else np.nan)
    return np.array(pred), np.array(true), np.array(dsc)

def macro_f1(p, t):   # same definition as training: mean F1 over classes present in t
    return float(np.mean([2 * np.sum((p == c) & (t == c)) / (np.sum(p == c) + np.sum(t == c)) for c in np.unique(t)]))

R = {}
for name in ("CBWA_pretrained200", "FedAvg_pretrained200"):
    model = load_model(OUT / f"{name}_weights.pth")
    R[name] = {s: run(model, s, idx) for s, idx in SETS}
    del model; torch.cuda.empty_cache()
A, B = R["CBWA_pretrained200"], R["FedAvg_pretrained200"]

def f1_of(res, sites): return np.mean([macro_f1(res[s][0], res[s][1]) for s in sites])
def dice_of(res, sites): return np.mean([np.nanmean(res[s][2]) for s in sites])
rng = np.random.default_rng(0)
def boot_ci(stat, sites, n=2000):    # resample patients within each hospital
    diffs = []
    for _ in range(n):
        ra, rb = {}, {}
        for s in sites:
            j = rng.integers(0, len(A[s][1]), len(A[s][1]))
            ra[s] = tuple(a[j] for a in A[s]); rb[s] = tuple(b[j] for b in B[s])
        diffs.append(stat(ra, sites) - stat(rb, sites))
    return np.percentile(diffs, [2.5, 97.5])

print(f"{'Metric':16s} {'CBWA':>7s} {'FedAvg':>7s} {'Diff':>7s}   95% CI of diff")
summary = {}
for label, stat, sites in (("In-net F1", f1_of, IN_NET), ("In-net Dice", dice_of, IN_NET),
                           ("AU F1", f1_of, ["AU"]), ("AU Dice", dice_of, ["AU"]),
                           ("TK F1", f1_of, ["TK"]), ("TK Dice", dice_of, ["TK"])):
    a, b = stat(A, sites), stat(B, sites); lo, hi = boot_ci(stat, sites)
    summary[label] = dict(cbwa=a, fedavg=b, ci=[lo, hi])
    print(f"{label:16s} {a:7.4f} {b:7.4f} {a - b:+7.4f}   [{lo:+.4f}, {hi:+.4f}]")

def wil(x, y):
    try: return wilcoxon(x, y).pvalue
    except ValueError: return float("nan")
fa = [macro_f1(*A[s][:2]) for s in IN_NET]; fb = [macro_f1(*B[s][:2]) for s in IN_NET]
da = [np.nanmean(A[s][2]) for s in IN_NET]; db = [np.nanmean(B[s][2]) for s in IN_NET]
print(f"\nPer hospital (16): CBWA F1 higher at {sum(x > y for x, y in zip(fa, fb))}, lower at "
      f"{sum(x < y for x, y in zip(fa, fb))}; Wilcoxon p={wil(fa, fb):.4f}")
print(f"Per hospital (16): CBWA Dice higher at {sum(x > y for x, y in zip(da, db))}, lower at "
      f"{sum(x < y for x, y in zip(da, db))}; Wilcoxon p={wil(da, db):.4f}")
for label, sites in (("in-network", IN_NET), ("external AU+TK", ["AU", "TK"])):
    pa = np.concatenate([A[s][0] for s in sites]); pb = np.concatenate([B[s][0] for s in sites])
    t = np.concatenate([A[s][1] for s in sites])
    xa = np.concatenate([A[s][2] for s in sites]); xb = np.concatenate([B[s][2] for s in sites])
    ok = ~np.isnan(xa) & ~np.isnan(xb)
    b_, c_ = int(np.sum((pa == t) & (pb != t))), int(np.sum((pa != t) & (pb == t)))
    mc = binomtest(b_, b_ + c_, 0.5).pvalue if b_ + c_ else float("nan")
    print(f"\nPatients {label} (n={len(t)}): type correct CBWA {np.sum(pa == t)}, FedAvg {np.sum(pb == t)}; "
          f"only-CBWA-right {b_}, only-FedAvg-right {c_}; McNemar p={mc:.4f}")
    print(f"   Dice: CBWA higher for {np.sum(xa[ok] > xb[ok])}/{ok.sum()}, mean diff "
          f"{np.mean(xa[ok] - xb[ok]):+.4f}; Wilcoxon p={wil(xa[ok], xb[ok]):.2e}")
    summary[label] = dict(n=int(len(t)), only_cbwa=b_, only_fedavg=c_, mcnemar_p=mc,
                          dice_wilcoxon_p=wil(xa[ok], xb[ok]))
per_patient = {n: {s: [v.tolist() for v in R[n][s]] for s, _ in SETS} for n in R}
(OUT / "significance_cbwa_vs_fedavg.json").write_text(json.dumps(
    {"summary": summary, "per_patient": per_patient}, indent=1, default=float))
print("\nSaved fedcb_results/significance_cbwa_vs_fedavg.json")
