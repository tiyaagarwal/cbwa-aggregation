"""
FedCB-PedBrain: Fast dual-GPU training for Kaggle T4 x2.

Speedups vs previous version (zero impact on comparison validity):
  1. Sites split across both GPUs and trained in parallel -- same total
     computation, ~2x wall-clock speedup.
  2. CPU augmentation removed -- applied equally to all 3 methods before,
     so removing it doesn't bias FedAvg vs FedProx vs CBWA. Just changes
     absolute numbers vs. the paper (already a disclosed deviation).
  3. 20 warmup + 30 full rounds (50 total) instead of 200 -- matches the
     scale of the proposal's own synthetic proof-of-concept (40 rounds).

Estimated total time: ~8 hours for all 3 experiments, fits in one
Kaggle session. Resumes automatically from Drive-equivalent checkpoints
if interrupted.
"""

import sys, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROCESSED = Path("/kaggle/working/processed")
REPO_SRC  = Path("/kaggle/working/FLPedBrain/FLPedBrain-PyTorch/src")
OUT_DIR   = Path("/kaggle/working/fedcb_results"); OUT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO_SRC))

CLASSES   = ["NormalControl", "EP", "DIPG", "MB", "PA"]
N_CLASSES = 5
WARMUP_SITES = ["ST", "SE"]
EXTERNAL     = {"AU", "TK"}

N_GPU = torch.cuda.device_count()
DEVICES = [torch.device(f"cuda:{i}") for i in range(max(N_GPU, 1))]
if N_GPU == 0:
    DEVICES = [torch.device("cpu")]
print(f"GPUs available: {N_GPU} -> using {[str(d) for d in DEVICES]}")

fed    = json.loads((PROCESSED / "federation.json").read_text())
IN_NET = [s for s in fed["in_network_sites"] if s not in EXTERNAL]
EXT    = [s for s in fed["clients"] if fed["clients"][s].get("role") == "external_eval_only"]
print(f"In-network sites ({len(IN_NET)}): {IN_NET}")
print(f"External eval sites: {EXT}")

# ═══════════════════════════════════════════════════════════════════════════
# MODEL (unchanged from the memory-fixed version)
# ═══════════════════════════════════════════════════════════════════════════

class Conv3dBN(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, k, stride=stride, padding=k//2, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)

class InceptionModule3D(nn.Module):
    def __init__(self, in_ch, o1, r3, o3, r5, o5, op):
        super().__init__()
        self.b1 = Conv3dBN(in_ch, o1, 1)
        self.b2 = nn.Sequential(Conv3dBN(in_ch, r3, 1), Conv3dBN(r3, o3, 3))
        self.b3 = nn.Sequential(Conv3dBN(in_ch, r5, 1), Conv3dBN(r5, o5, 3))
        self.b4 = nn.Sequential(nn.MaxPool3d(3,1,1), Conv3dBN(in_ch, op, 1))
    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], 1)

class Encoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv3dBN(1, 64, 7, stride=2)
        self.pool1 = nn.MaxPool3d((1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.conv2a = Conv3dBN(64, 64, 1); self.conv2b = Conv3dBN(64, 192, 3)
        self.pool2  = nn.MaxPool3d((1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.inc3b  = InceptionModule3D(192, 64, 96,128,16, 32, 32)
        self.inc3c  = InceptionModule3D(256,128,128,192,32, 96, 64)
        self.pool3  = nn.MaxPool3d(3, stride=2, padding=1)
        self.inc4b  = InceptionModule3D(480,192, 96,208,16, 48, 64)
        self.inc4c  = InceptionModule3D(512,160,112,224,24, 64, 64)
        self.inc4d  = InceptionModule3D(512,128,128,256,24, 64, 64)
        self.inc4e  = InceptionModule3D(512,112,144,288,32, 64, 64)
        self.inc4f  = InceptionModule3D(528,256,160,320,32,128,128)
        self.pool4  = nn.MaxPool3d(2, stride=2)
        self.inc5b  = InceptionModule3D(832,256,160,320,32,128,128)
        self.inc5c  = InceptionModule3D(832,384,192,384,48,128,128)

    def forward(self, x):
        from torch.utils.checkpoint import checkpoint
        ck = lambda f, inp: (checkpoint(f, inp, use_reentrant=False)
                             if self.training else f(inp))
        skips = []
        x = ck(self.conv1, x); skips.append(x)
        x = self.pool1(x)
        x = ck(self.conv2a, x); x = ck(self.conv2b, x); skips.append(x)
        x = self.pool2(x)
        x = ck(self.inc3b, x); x = ck(self.inc3c, x); skips.append(x)
        x = self.pool3(x)
        for m in (self.inc4b, self.inc4c, self.inc4d, self.inc4e, self.inc4f):
            x = ck(m, x)
        skips.append(x)
        x = self.pool4(x)
        x = ck(self.inc5b, x); x = ck(self.inc5c, x)
        return x, skips

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = Conv3dBN(in_ch, out_ch)
    def forward(self, x, target_size):
        x = F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)
        return self.conv(x)

class Decoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = UpBlock(1024, 416); self.c1 = Conv3dBN(416+832, 416)
        self.up2 = UpBlock(416,  240); self.c2 = Conv3dBN(240+480, 240)
        self.up3 = UpBlock(240,   96); self.c3 = Conv3dBN(96+192,   96)
        self.up4 = UpBlock(96,    32); self.c4 = Conv3dBN(32+64,    32)
        self.up5 = UpBlock(32,    16)
        self.final = nn.Conv3d(16, 1, 1)

    def forward(self, x, skips):
        s4,s3,s2,s1 = skips[3],skips[2],skips[1],skips[0]
        x = self.c1(torch.cat([self.up1(x, s4.shape[2:]), s4], 1))
        x = self.c2(torch.cat([self.up2(x, s3.shape[2:]), s3], 1))
        x = self.c3(torch.cat([self.up3(x, s2.shape[2:]), s2], 1))
        x = self.c4(torch.cat([self.up4(x, s1.shape[2:]), s1], 1))
        x = self.up5(x, (64,256,256))
        return self.final(x)

class PedBrainNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.encoder = Encoder3D(); self.decoder = Decoder3D()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(1024,128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes))
    def forward(self, x):
        feat, skips = self.encoder(x)
        seg_logits  = self.decoder(feat, skips)
        cls_logits  = self.classifier(self.pool(feat).flatten(1))
        return torch.sigmoid(seg_logits), F.softmax(cls_logits,1), seg_logits, cls_logits


# ═══════════════════════════════════════════════════════════════════════════
# DATASET  (augmentation OFF -- CPU zoom on 3D volumes is the bottleneck;
#           removed equally for all 3 methods, so comparison stays fair)
# ═══════════════════════════════════════════════════════════════════════════

class ProcessedSiteDataset(Dataset):
    def __init__(self, site, indices=None):
        d = PROCESSED / site
        self.vols  = np.load(d/"volumes.npy", mmap_mode="r")
        self.masks = np.load(d/"masks.npy",   mmap_mode="r")
        meta = json.loads((d/"meta.json").read_text())
        self.labels = np.array(meta["labels"], dtype=np.int64)
        self.indices = np.arange(len(self.labels)) if indices is None else np.array(indices)
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx  = self.indices[i]
        vol  = self.vols[idx].astype(np.float32) / 255.0
        mask = self.masks[idx].astype(np.float32)
        lbl  = int(self.labels[idx])
        return (torch.from_numpy(vol[None].copy()),
                torch.from_numpy(mask[None].copy()),
                torch.tensor(lbl, dtype=torch.long),
                torch.tensor(lbl > 0, dtype=torch.bool))

def make_loader(site, indices=None, bs=1, shuffle=True):
    ds = ProcessedSiteDataset(site, indices)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      num_workers=2, pin_memory=True, drop_last=False)


# ═══════════════════════════════════════════════════════════════════════════
# LOSSES
# ═══════════════════════════════════════════════════════════════════════════

def dice_loss(sig, tgt, has_t, smooth=1.0):
    b = sig.shape[0]
    p = sig.reshape(b,-1); t = tgt.reshape(b,-1).float()
    loss = 1 - (2*(p*t).sum(1)+smooth)/(p.sum(1)+t.sum(1)+smooth)
    m = has_t.float().to(sig.device)
    return (loss*m).sum() / (m.sum()+1e-8)

def prox_pen(model, gparams, mu, device):
    if mu <= 0: return torch.zeros(1, device=device)
    return 0.5*mu*sum(((p-g.detach())**2).sum()
                       for p,g in zip(model.parameters(), gparams))


# ═══════════════════════════════════════════════════════════════════════════
# AGGREGATION  (unchanged -- CPU-side, unaffected by GPU parallelism)
# ═══════════════════════════════════════════════════════════════════════════

def sd2np(sd): return {k: v.cpu().numpy() for k,v in sd.items()}
def np2sd(nd, ref): return {k: torch.tensor(nd[k], dtype=ref[k].dtype) for k in nd}

def wavg(updates, w):
    out = {}
    for k in updates[0]:
        if not np.issubdtype(updates[0][k].dtype, np.floating):
            out[k] = updates[0][k].copy(); continue
        acc = np.zeros_like(updates[0][k], dtype=np.float64)
        for wi,u in zip(w, updates): acc += wi*u[k].astype(np.float64)
        out[k] = acc.astype(updates[0][k].dtype)
    return out

def fedavg(updates, ns):
    n = np.array(ns, dtype=np.float64); return wavg(updates, n/n.sum())

def cbwa(updates, ns, f1s, alpha=0.5, gamma=1.0, eps=0.05, floor=0.01, prev=None, mom=0.7):
    n  = np.array(ns,  dtype=np.float64)
    f1 = np.array(f1s, dtype=np.float64)
    w  = np.power(np.maximum(n,0), alpha) * np.power(np.maximum(f1,0)+eps, gamma)
    w  = w/w.sum() if w.sum()>0 else np.full(len(n),1/len(n))
    k = len(w)
    if floor > 0 and floor*k < 1:
        below = w < floor
        if below.any():
            deficit = (floor - w[below]).sum(); above = ~below
            head = (w[above]-floor).sum()
            if head > 0:
                w = w.copy()
                w[above] -= (w[above]-floor)/head*deficit
                w[below]  = floor
                w /= w.sum()
    if prev is not None:
        w = mom*prev + (1-mom)*w; w /= w.sum()
    return wavg(updates, w), w


# ═══════════════════════════════════════════════════════════════════════════
# DEVICE-AWARE TRAIN / EVAL
# ═══════════════════════════════════════════════════════════════════════════

def local_train(model, device, gsd, site, tidx, cls_w_np, lr=1e-4, epochs=1,
                bs=1, dice_w=0.5, mu=0.0):
    model.load_state_dict(np2sd(gsd, model.state_dict()))
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    cls_w = torch.tensor(cls_w_np, dtype=torch.float32).to(device)
    gparams = [p.clone().detach() for p in model.parameters()] if mu>0 else None
    loader  = make_loader(site, tidx, bs=bs)
    for _ in range(epochs):
        for vol,msk,lbl,has_t in loader:
            vol,msk,lbl = vol.to(device), msk.to(device), lbl.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type=="cuda"):
                seg,_,seg_l,cls_l = model(vol)
                ce = F.cross_entropy(cls_l, lbl, weight=cls_w)
                dc = dice_loss(torch.sigmoid(seg_l.float()), msk, has_t)
                loss = ce + dice_w*dc
                if mu > 0: loss = loss + prox_pen(model, gparams, mu, device)
            if scaler:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
    n = len(tidx) if tidx is not None else \
        len(json.loads((PROCESSED/site/"meta.json").read_text())["labels"])
    return sd2np(model.state_dict()), n

@torch.no_grad()
def eval_site(model, device, site, indices=None, bs=1):
    model.eval()
    loader = make_loader(site, indices, bs=bs, shuffle=False)
    preds, lbls, dices = [], [], []
    for vol,msk,lbl,has_t in loader:
        vol,msk = vol.to(device), msk.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type=="cuda"):
            seg,_,_,cls_l = model(vol)
        preds.append(cls_l.argmax(1).cpu()); lbls.append(lbl)
        for b in range(vol.shape[0]):
            if has_t[b]:
                p = (seg[b]>0.5).float()
                dices.append(((2*(p*msk[b]).sum()+1)/(p.sum()+msk[b].sum()+1)).item())
    preds = torch.cat(preds); lbls = torch.cat(lbls)
    acc = (preds==lbls).float().mean().item()
    f1s = []
    for c in range(N_CLASSES):
        tp=((preds==c)&(lbls==c)).sum().float()
        fp=((preds==c)&(lbls!=c)).sum().float()
        fn=((preds!=c)&(lbls==c)).sum().float()
        d=2*tp+fp+fn; f1s.append((2*tp/d).item() if d>0 else 0.0)
    present = [i for i in range(N_CLASSES) if (lbls==i).any()]
    return {"acc": acc,
            "macro_f1": float(np.mean([f1s[i] for i in present])) if present else 0.0,
            "per_class_f1": f1s,
            "dice": float(np.mean(dices)) if dices else 0.0}


# ═══════════════════════════════════════════════════════════════════════════
# PARALLEL ROUND EXECUTION -- sites chunked across GPUs, each GPU owns its
# chunk sequentially. No two threads ever touch the same device's model.
# ═══════════════════════════════════════════════════════════════════════════

def run_round_parallel(models, sites, gsd, site_tr, site_cw, site_va,
                       lr, epochs, bs, dice_w, mu, use_cbwa, warmup):
    chunks = [sites[i::len(models)] for i in range(len(models))]

    def worker(chunk_idx):
        model, device = models[chunk_idx], DEVICES[chunk_idx]
        results = []
        for site in chunks[chunk_idx]:
            upd, n = local_train(model, device, gsd, site, site_tr[site],
                                 site_cw[site], lr=lr, epochs=epochs,
                                 bs=bs, dice_w=dice_w, mu=mu)
            f1 = None
            if use_cbwa and not warmup:
                model.load_state_dict(np2sd(upd, model.state_dict()))
                f1 = eval_site(model, device, site, site_va[site], bs)["macro_f1"]
            results.append((site, upd, n, f1))
        return results

    with ThreadPoolExecutor(max_workers=len(models)) as ex:
        futures = [ex.submit(worker, i) for i in range(len(models))]
        all_results = [r for fut in futures for r in fut.result()]
    return all_results


def run_experiment(name, n_rounds=50, warmup_rounds=20, local_epochs=1,
                   bs=1, lr=1e-4, dice_w=0.5, mu=0.0, use_cbwa=False,
                   cbwa_alpha=0.5, cbwa_gamma=1.0, eval_every=5, bs_eval=1):

    print(f"\n{'='*72}\nEXPERIMENT: {name}\n{'='*72}")

    # Skip entirely if this experiment already finished in a prior run --
    # without this check, a session restart after completion looks
    # identical to "never started" (the checkpoint is deleted on success),
    # so a naive resume-check would wastefully redo the whole experiment.
    done_path = OUT_DIR / f"{name}_results.json"
    if done_path.exists():
        hist = json.loads(done_path.read_text())
        print(f"ALREADY COMPLETE (found {done_path.name}) -- skipping, "
              f"{len([r for r in hist if 'in_net_f1' in r])} eval points saved")
        model = PedBrainNet(N_CLASSES).to(DEVICES[0])
        w = OUT_DIR / f"{name}_weights.pth"
        if w.exists():
            model.load_state_dict(torch.load(w, map_location=DEVICES[0]))
        return hist, model

    models = [PedBrainNet(N_CLASSES).to(d) for d in DEVICES]
    print(f"Parameters: {sum(p.numel() for p in models[0].parameters()):,}  "
          f"x{len(models)} device copies")

    ckpt_path = OUT_DIR / f"{name}_checkpoint.pth"
    start_round = 0
    history = []
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=DEVICES[0])
        for m in models: m.load_state_dict(ck["model_state_dict"])
        start_round = ck["round"] + 1
        history = json.loads((OUT_DIR / f"{name}_history.json").read_text())
        print(f"RESUMED from round {start_round}")

    site_cw = {s: np.array(fed["clients"][s]["class_weights"], np.float32) for s in IN_NET}
    site_tr = {s: fed["clients"][s]["train_idx"] for s in IN_NET}
    site_va = {s: fed["clients"][s]["val_idx"]   for s in IN_NET}

    gsd = sd2np(models[0].state_dict())
    prev_w = None

    for rnd in range(start_round, n_rounds):
        t0 = time.time()
        warmup = rnd < warmup_rounds
        active = WARMUP_SITES if warmup else IN_NET

        results = run_round_parallel(models, active, gsd, site_tr, site_cw,
                                     site_va, lr, local_epochs, bs, dice_w, mu,
                                     use_cbwa, warmup)
        updates = [r[1] for r in results]; ns = [r[2] for r in results]
        f1s     = [r[3] for r in results if r[3] is not None]

        if use_cbwa and not warmup and f1s:
            gsd, prev_w = cbwa(updates, ns, f1s, cbwa_alpha, cbwa_gamma, prev=prev_w)
        else:
            gsd = fedavg(updates, ns)

        for m in models: m.load_state_dict(np2sd(gsd, m.state_dict()))
        row = {"round": rnd+1, "warmup": warmup}

        if (rnd+1) % eval_every == 0 or rnd == n_rounds-1:
            print(f"\nRound {rnd+1}/{n_rounds} ({'warmup' if warmup else 'full'}) "
                  f"[{time.time()-t0:.0f}s]")
            vm = [eval_site(models[0], DEVICES[0], s, site_va[s], bs_eval) for s in IN_NET]
            row.update({"in_net_f1": float(np.mean([m["macro_f1"] for m in vm])),
                        "in_net_acc": float(np.mean([m["acc"] for m in vm])),
                        "in_net_dice": float(np.mean([m["dice"] for m in vm]))})
            print(f"  In-network — F1:{row['in_net_f1']:.4f} "
                  f"Acc:{row['in_net_acc']:.4f} Dice:{row['in_net_dice']:.4f}")
            for es in EXT:
                em = eval_site(models[0], DEVICES[0], es, None, bs_eval)
                meta_labs = json.loads((PROCESSED/es/"meta.json").read_text())["labels"]
                pf = em["per_class_f1"]
                present = [pf[c] for c in range(N_CLASSES) if any(l==c for l in meta_labs)]
                row[f"{es}_f1"] = em["macro_f1"]; row[f"{es}_dice"] = em["dice"]
                row[f"{es}_worst"] = min(present) if present else 0.0
                print(f"  {es} (ext) — F1:{em['macro_f1']:.4f} "
                      f"Dice:{em['dice']:.4f} Worst:{row[f'{es}_worst']:.4f}")
            torch.save({"round": rnd, "model_state_dict": models[0].state_dict()}, ckpt_path)
            (OUT_DIR / f"{name}_history.json").write_text(json.dumps(history+[row], indent=2))
            print(f"  [checkpointed: round {rnd+1}]")
        history.append(row)

    out = OUT_DIR/f"{name}_results.json"
    out.write_text(json.dumps(history, indent=2))
    torch.save(models[0].state_dict(), OUT_DIR/f"{name}_weights.pth")
    if ckpt_path.exists(): ckpt_path.unlink()
    print(f"Saved -> {out}")
    return history, models[0]


# ═══════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════

HP = dict(n_rounds=50, warmup_rounds=20, local_epochs=1,
          bs=1, lr=1e-4, dice_w=0.5, eval_every=5, bs_eval=1)

print("\n" + "="*72)
print(f"RUNNING ALL 3 EXPERIMENTS -- {N_GPU}-GPU parallel, no augmentation")
print(f"Estimated total: ~8 hours (fits one Kaggle session)")
print("="*72)

results = {}
hist1, _ = run_experiment("FedAvg",  mu=0.0,  use_cbwa=False, **HP)
results["FedAvg"] = hist1
hist2, _ = run_experiment("FedProx", mu=0.01, use_cbwa=False, **HP)
results["FedProx"] = hist2
hist3, _ = run_experiment("CBWA",    mu=0.0,  use_cbwa=True,
                          cbwa_alpha=0.5, cbwa_gamma=1.0, **HP)
results["CBWA"] = hist3

print("\n" + "="*72)
print("FINAL RESULTS")
print("="*72)
print(f"{'Method':10s} {'In-net F1':>10s} {'In-net Dice':>12s} "
      f"{'AU F1':>8s} {'AU Worst':>9s} {'TK F1':>8s} {'TK Worst':>9s}")
for nm, hist in results.items():
    last = [r for r in hist if "in_net_f1" in r][-1]
    print(f"{nm:10s} {last.get('in_net_f1',0):10.4f} {last.get('in_net_dice',0):12.4f} "
          f"{last.get('AU_f1',0):8.4f} {last.get('AU_worst',0):9.4f} "
          f"{last.get('TK_f1',0):8.4f} {last.get('TK_worst',0):9.4f}")

(OUT_DIR/"all_results.json").write_text(json.dumps(results, indent=2))
print(f"\nAll results -> {OUT_DIR}")
