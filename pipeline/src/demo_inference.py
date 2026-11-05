"""FedCB-PedBrain -- live demo for Review 3.

Runs in a few minutes on a Kaggle GPU (works on CPU too, just slower). It loads the
two trained global models (CBWA and the FedAvg baseline) and shows, for patients
from the external hold-out hospitals (never seen in training) and from in-network
validation sets:
  * the predicted tumour type and its probability vs. the ground truth
  * the predicted tumour segmentation on the axial slice with the largest tumour
  * per-patient Dice
and then how CBWA re-weights the 16 hospitals compared with FedAvg.

Prerequisites (already present in the training notebook's /kaggle/working):
  processed/, FLPedBrain/ (with model_pretrained.py), rgb_imagenet.pt,
  fedcb_results/CBWA_pretrained200_weights.pth and FedAvg_pretrained200_weights.pth
"""
import os, sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt

ROOT      = Path(os.environ.get("FEDCB_ROOT", "/kaggle/working"))
PROCESSED = ROOT / "processed"
OUT_DIR   = ROOT / "fedcb_results"
WEIGHTS_PATH = ROOT / "rgb_imagenet.pt"
sys.path.insert(0, str(ROOT / "FLPedBrain/FLPedBrain-PyTorch/src"))
from model_pretrained import PedBrainNetPretrained

CLASSES   = ["NormalControl", "EP", "DIPG", "MB", "PA"]
N_CLASSES = 5
EXTERNAL  = {"AU", "TK"}
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODELS = {"CBWA (ours)": "CBWA_pretrained200", "FedAvg baseline": "FedAvg_pretrained200"}
RUN_WEIGHT_DEMO = True          # evaluates 16 validation sets (~1 min on GPU)

fed = json.loads((PROCESSED / "federation.json").read_text())
IN_NET = [s for s in fed["in_network_sites"] if s not in EXTERNAL]
print(f"Device: {DEVICE}")


def load_site(site):
    d = PROCESSED / site
    vols = np.load(d / "volumes.npy", mmap_mode="r")
    masks = np.load(d / "masks.npy", mmap_mode="r")
    labels = json.loads((d / "meta.json").read_text())["labels"]
    return vols, masks, labels

def load_model(exp_name):
    m = PedBrainNetPretrained(num_classes=N_CLASSES, pretrained=True,
                              weights_path=str(WEIGHTS_PATH))
    w = OUT_DIR / f"{exp_name}_weights.pth"
    m.load_state_dict(torch.load(w, map_location="cpu"))
    return m.to(DEVICE).eval()

@torch.no_grad()
def predict(model, vol):
    x = torch.from_numpy(vol.astype(np.float32) / 255.0)[None, None].to(DEVICE)
    with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
        seg, _, _, cls_l = model(x)
    prob = F.softmax(cls_l.float(), dim=1)[0].cpu().numpy()
    return prob, (seg[0, 0].float() > 0.5).cpu().numpy()

def dice(pred, mask):
    return (2 * (pred * mask).sum() + 1) / (pred.sum() + mask.sum() + 1)


# ── 1. Training results recorded by the experiment script ───────────────────
print("\n=== Final results recorded during training (last evaluated round) ===")
print(f"{'Model':18s} {'In-net F1':>9s} {'In-net Dice':>11s} {'AU F1':>7s} {'TK F1':>7s}")
for label, exp in MODELS.items():
    f = OUT_DIR / f"{exp}_results.json"
    if f.exists():
        last = [r for r in json.loads(f.read_text()) if "in_net_f1" in r][-1]
        print(f"{label:18s} {last['in_net_f1']:9.4f} {last['in_net_dice']:11.4f} "
              f"{last.get('AU_f1', 0):7.4f} {last.get('TK_f1', 0):7.4f}")

# ── 2. Pick demo patients ────────────────────────────────────────────────────
# one per tumour class from TK (external, 4 classes), one from AU (external),
# and one from each of two in-network validation sets
picks = []
tk_vols, tk_masks, tk_labels = load_site("TK")
seen = set()
for i, l in enumerate(tk_labels):
    if l not in seen:
        seen.add(l); picks.append(("TK", i, "external"))
picks.append(("AU", 0, "external"))
for s in [s for s in IN_NET if fed["clients"][s]["val_idx"]][:2]:
    picks.append((s, fed["clients"][s]["val_idx"][0], "in-network val"))

# ── 3. Run both models on every picked patient ───────────────────────────────
models = {label: load_model(exp) for label, exp in MODELS.items()}
cache = {}
rows = []
for site, idx, kind in picks:
    if site not in cache: cache[site] = load_site(site)
    vols, masks, labels = cache[site]
    vol, mask, true = np.asarray(vols[idx]), np.asarray(masks[idx]).astype(np.float32), labels[idx]
    out = {label: predict(m, vol) for label, m in models.items()}
    rows.append((site, idx, kind, vol, mask, true, out))

print("\n=== Per-patient predictions ===")
for site, idx, kind, vol, mask, true, out in rows:
    print(f"\n{site} patient {idx} ({kind}) -- true type: {CLASSES[true]}")
    for label, (prob, pred) in out.items():
        k = int(prob.argmax())
        ok = "correct" if k == true else "WRONG"
        d = dice(pred, mask) if mask.sum() > 0 else float("nan")
        print(f"   {label:16s} -> {CLASSES[k]:5s} (p={prob[k]:.2f}, {ok})   Dice={d:.3f}")

# ── 4. Figure: MRI | ground truth | CBWA | FedAvg ────────────────────────────
fig, axes = plt.subplots(len(rows), 2 + len(models), figsize=(4 * (2 + len(models)), 3.6 * len(rows)))
axes = np.atleast_2d(axes)
for r, (site, idx, kind, vol, mask, true, out) in enumerate(rows):
    z = int(mask.reshape(mask.shape[0], -1).sum(1).argmax()) if mask.sum() > 0 else vol.shape[0] // 2
    img = vol[z]
    panels = [("MRI", None), ("Ground truth", mask[z])] + \
             [(label, pred[z]) for label, (prob, pred) in out.items()]
    for c, (name, m) in enumerate(panels):
        ax = axes[r, c]
        ax.imshow(img, cmap="gray"); ax.axis("off")
        if m is not None:
            ax.imshow(np.ma.masked_where(m == 0, m), cmap="autumn", alpha=0.5)
        if c == 0:
            ax.set_title(f"{site} #{idx} ({kind})\ntrue: {CLASSES[true]}", fontsize=10)
        elif c == 1:
            ax.set_title("Ground-truth tumour", fontsize=10)
        else:
            prob, pred = out[name]
            k = int(prob.argmax())
            d = dice(pred, mask) if mask.sum() > 0 else float("nan")
            ax.set_title(f"{name}: {CLASSES[k]} (p={prob[k]:.2f})\nDice {d:.3f}",
                         fontsize=10, color="green" if k == true else "red")
plt.tight_layout()
fig.savefig(OUT_DIR / "demo_predictions.png", dpi=130)
plt.show()

# ── 4b. Paper Fig. 9, drawn straight from the 256 x 256 arrays ──────────────
# Journal rules: no cropping or upscaling of an old image, no AI tools, no local
# contrast edits. Each panel is the original slice shown with nearest-neighbour
# pixels; the expert mask is a yellow contour, predictions a red overlay.
# Labels are written out (tick/cross + word), not shown by colour alone.
ROT = -1   # np.rot90 turns: the raw slices have the eyes on the LEFT; -1 (clockwise)
           # brings them to the TOP. If they end up at the bottom, set ROT = 1.
show = lambda a: np.rot90(a, ROT)

def panel(ax, img, pred=None, expert=None):
    ax.imshow(show(img), cmap="gray", interpolation="nearest")
    if pred is not None:                                   # red overlay = predicted tumor
        rgba = np.zeros(pred.shape + (4,)); rgba[pred > 0.5] = (1, 0, 0, 0.45)
        ax.imshow(show(rgba), interpolation="nearest")
    if expert is not None:                                 # expert mask as a yellow line
        ax.contour(show(expert.astype(float)), levels=[0.5], colors="yellow", linewidths=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

paper_rows = [row for row in rows if row[0] == "TK"][:2]
fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.3))
headers = ["MRI (axial slice)", "Expert tumor outline", "CBWA prediction", "FedAvg prediction"]
print("\n=== Paper Fig. 9 ===")
for r, (site, idx, kind, vol, mask, true, out) in enumerate(paper_rows):
    z = int(mask.reshape(mask.shape[0], -1).sum(1).argmax())
    print(f"Patient {r + 1} = {site} patient {idx}, true {CLASSES[true]}: "
          f"axial slice {z + 1} of {vol.shape[0]}, slice size {vol[z].shape}")
    preds = list(out.values())                             # [(prob, pred) CBWA, (prob, pred) FedAvg]
    panel(axes[r, 0], vol[z])
    panel(axes[r, 1], vol[z], expert=mask[z])
    panel(axes[r, 2], vol[z], pred=preds[0][1][z])
    panel(axes[r, 3], vol[z], pred=preds[1][1][z])
    axes[r, 0].set_ylabel(f"Patient {r + 1}\ntrue type: {CLASSES[true]}\naxial slice {z + 1}", fontsize=8.5)
    for c, (prob, pred) in zip((2, 3), preds):
        k = int(prob.argmax())
        mark = "\u2713 correct" if k == true else "\u2717 wrong"
        label = f"{CLASSES[k]} {mark} (p = {prob[k]:.2f})\nDice {dice(pred, mask):.3f}"
        print("  ", headers[c], "->", label.replace("\n", " / "))
        axes[r, c].set_xlabel(label, fontsize=8, color="#146c2e" if k == true else "#a3221c")
for c, h in enumerate(headers):
    axes[0, c].set_title(h, fontsize=8.5)
plt.tight_layout()
fig.savefig(OUT_DIR / "Figure_9.pdf", dpi=600)
fig.savefig(OUT_DIR / "Figure_9.png", dpi=600)
print("Saved", OUT_DIR / "Figure_9.pdf")
plt.show()

# ── 5. How CBWA re-weights hospitals (one aggregation step, final model) ─────
def cbwa_weights(ns, f1s, alpha=0.5, gamma=1.0, eps=0.05, floor=0.01):
    """Same rule as the training script's cbwa(), without momentum."""
    n, f1 = np.array(ns, float), np.array(f1s, float)
    w = np.power(np.maximum(n, 0), alpha) * np.power(np.maximum(f1, 0) + eps, gamma)
    w = w / w.sum()
    if floor > 0 and floor * len(w) < 1:
        below = w < floor
        if below.any():
            deficit = (floor - w[below]).sum(); above = ~below
            head = (w[above] - floor).sum()
            if head > 0:
                w[above] -= (w[above] - floor) / head * deficit
                w[below] = floor
                w /= w.sum()
    return w

if RUN_WEIGHT_DEMO:
    from sklearn.metrics import f1_score
    m = models["CBWA (ours)"]
    ns, f1s = [], []
    for s in IN_NET:
        vols, masks, labels = load_site(s)
        va = fed["clients"][s]["val_idx"]
        y = [labels[i] for i in va]
        p = [int(predict(m, np.asarray(vols[i]))[0].argmax()) for i in va]
        ns.append(len(fed["clients"][s]["train_idx"]))
        f1s.append(f1_score(y, p, labels=sorted(set(y)), average="macro", zero_division=0) if y else 0.0)
    w_avg = np.array(ns, float) / sum(ns)
    w_cb = cbwa_weights(ns, f1s)
    order = np.argsort(ns)
    print("\n=== Aggregation weight per hospital: FedAvg (size only) vs CBWA (size^0.5 x val-F1) ===")
    print(f"{'Site':5s} {'train n':>7s} {'val F1':>7s} {'FedAvg w':>9s} {'CBWA w':>8s}")
    for i in order:
        print(f"{IN_NET[i]:5s} {ns[i]:7d} {f1s[i]:7.3f} {w_avg[i]:9.3f} {w_cb[i]:8.3f}")
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - 0.2, w_avg[order], 0.4, label="FedAvg (sample count)")
    ax.bar(x + 0.2, w_cb[order], 0.4, label="CBWA (dampened size x validation F1)")
    ax.set_xticks(x); ax.set_xticklabels([IN_NET[i] for i in order])
    ax.set_ylabel("aggregation weight"); ax.set_xlabel("hospital (smallest -> largest)")
    ax.set_title("CBWA gives small hospitals a larger voice than FedAvg")
    ax.legend(); plt.tight_layout()
    fig.savefig(OUT_DIR / "demo_cbwa_weights.png", dpi=130)
    plt.show()

print(f"\nFigures saved to {OUT_DIR}")
