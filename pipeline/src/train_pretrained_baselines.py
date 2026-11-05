import os, sys, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# FEDCB_ROOT / FEDCB_SMOKE / FEDCB_NUM_WORKERS exist only for a local CPU smoke
# test; on Kaggle leave them unset and everything uses /kaggle/working.
ROOT      = Path(os.environ.get("FEDCB_ROOT", "/kaggle/working"))
SMOKE     = bool(os.environ.get("FEDCB_SMOKE"))
NUM_WORKERS = int(os.environ.get("FEDCB_NUM_WORKERS", 2))

PROCESSED = ROOT / "processed"
REPO_SRC  = ROOT / "FLPedBrain/FLPedBrain-PyTorch/src"
OUT_DIR   = ROOT / "fedcb_results"; OUT_DIR.mkdir(exist_ok=True)
WEIGHTS_PATH = ROOT / "rgb_imagenet.pt"
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

# Download pretrained weights if not already present (makes this script
# self-contained / resumable in a fresh session)
if not WEIGHTS_PATH.exists():
    print("Downloading pretrained I3D weights...")
    import urllib.request
    urllib.request.urlretrieve(
        "https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt",
        str(WEIGHTS_PATH))
    print(f"Downloaded: {WEIGHTS_PATH.stat().st_size/1e6:.1f} MB")

from model_pretrained import PedBrainNetPretrained

fed    = json.loads((PROCESSED / "federation.json").read_text())
IN_NET = [s for s in fed["in_network_sites"] if s not in EXTERNAL]
EXT    = [s for s in fed["clients"] if fed["clients"][s].get("role") == "external_eval_only"]
print(f"In-network sites ({len(IN_NET)}): {IN_NET}")
print(f"External eval sites: {EXT}")


# ═══════════════════════════════════════════════════════════════════════════
# DATASET (unchanged -- proven pipeline)
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

def make_loader(site, indices=None, bs=2, shuffle=True):
    ds = ProcessedSiteDataset(site, indices)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

def make_pooled_loader(sites, site_tr, bs=2):
    """Centralized data sharing (Sheller 2020 CDS): all sites' train splits pooled."""
    ds = ConcatDataset([ProcessedSiteDataset(s, site_tr[s]) for s in sites])
    return DataLoader(ds, batch_size=bs, shuffle=True,
                      num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

def site_label_prior(site, tidx):
    labels = np.array(json.loads((PROCESSED/site/"meta.json").read_text())["labels"])
    counts = np.bincount(labels[np.array(tidx)], minlength=N_CLASSES).astype(np.float64)
    return counts / max(counts.sum(), 1)


# ═══════════════════════════════════════════════════════════════════════════
# LOSSES
# ═══════════════════════════════════════════════════════════════════════════

def dice_loss(sig, tgt, has_t, smooth=1.0):
    b = sig.shape[0]
    p = sig.reshape(b,-1); t = tgt.reshape(b,-1).float()
    loss = 1 - (2*(p*t).sum(1)+smooth)/(p.sum(1)+t.sum(1)+smooth)
    m = has_t.float().to(sig.device)
    return (loss*m).sum() / (m.sum()+1e-8)

# FedSat prediction-sensitive loss (Chowdhury & Halder 2024, Eq. 3-4).
# eps1=1, eps2=2 are the paper's own settings.
PSL_EPS1, PSL_EPS2 = 1.0, 2.0

def psl_cost(M, lbl, pred):
    """Per-sample cost: 1 if correct, else M[l, l_hat] min-max scaled into
    [eps1, eps2]. M is the confusion-count matrix accumulated so far."""
    lo, hi = M.min(), M.max()
    scaled = (M[lbl, pred] - lo) / (hi - lo) if hi > lo else torch.zeros(len(lbl))
    cost = PSL_EPS1 + scaled * (PSL_EPS2 - PSL_EPS1)
    return torch.where(lbl == pred, torch.ones_like(cost), cost)

# FedIIC (Wu et al. 2023, arXiv:2206.13803). k1=k2=2, t=0.5, q=0.25 are the
# paper's settings; tau is not stated in the paper text, 0.07 is the SupCon
# default it builds on.
FEDIIC_K1, FEDIIC_K2, FEDIIC_T, FEDIIC_Q, FEDIIC_TAU = 2.0, 2.0, 0.5, 0.25, 0.07

def fediic_intra_loss(z, y, prior, tau=FEDIIC_TAU, t=FEDIIC_T):
    """Eq. 3: supervised contrastive loss with class-prior dynamic temperature
    tau' = (p_i p_j)^t * tau. Computed in fp32 via logsumexp (tau' can be tiny)."""
    p = prior[y].clamp(min=1e-3)
    temp = (p[:, None] * p[None, :]) ** t * tau
    logits = (z @ z.T) / temp
    self_mask = torch.eye(len(y), dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & ~self_mask
    npos = pos.sum(1)
    per_anchor = -log_prob.masked_fill(~pos, 0.0).sum(1) / npos.clamp(min=1)
    valid = npos > 0
    return per_anchor[valid].mean() if valid.any() else z.sum() * 0.0

def augment_view(vol):
    """Second view for FedIIC's multi-view batch: random in-plane crop+resize,
    left-right flip, intensity jitter. vol: B x 1 x D x H x W in [0,1]."""
    B, _, D, H, W = vol.shape
    c = max(int(H * 0.875), 1)
    dx = int(torch.randint(0, H - c + 1, (1,))); dy = int(torch.randint(0, W - c + 1, (1,)))
    v = F.interpolate(vol[..., dx:dx+c, dy:dy+c], size=(D, H, W),
                      mode="trilinear", align_corners=False)
    if torch.rand(1).item() < 0.5:
        v = v.flip(-1)
    scale = torch.empty(B, 1, 1, 1, 1, device=vol.device).uniform_(0.9, 1.1)
    shift = torch.empty(B, 1, 1, 1, 1, device=vol.device).uniform_(-0.05, 0.05)
    return (v * scale + shift).clamp(0, 1)


class FedIICWrapper(nn.Module):
    """Adds FedIIC's 2-layer projection head h(.) on top of the classifier's
    input features g(x). Features are captured with a hook on the final
    Linear(*, N_CLASSES), whose weights also seed the global prototypes."""
    def __init__(self, base, proj_dim=128):
        super().__init__()
        self.base = base
        lin = [m for m in base.modules()
               if isinstance(m, nn.Linear) and m.out_features == N_CLASSES][-1]
        d = lin.in_features
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True),
                                  nn.Linear(d, proj_dim))
        # plain attributes, not registered submodules (no duplicate state_dict keys)
        object.__setattr__(self, "_cls_linear", lin)
        object.__setattr__(self, "_feat", None)
        lin.register_forward_hook(
            lambda m, inp, out: object.__setattr__(self, "_feat", inp[0]))
    def forward(self, x):
        return self.base(x)

def fediic_prototypes(model, steps=100, lr=0.1):
    """Server side (Sec. 2.3): prototypes = h_g(classifier weights), then
    'orthogonalized' by gradient descent on sum_i max_{j!=i} cos(v_i, v_j)
    (Eq. 5), then l2-normalised. Step count / lr are not given in the paper."""
    with torch.no_grad():
        V0 = model.head(model._cls_linear.weight.detach().float())
    V = V0.clone().requires_grad_(True)
    opt = torch.optim.SGD([V], lr=lr)
    eye = torch.eye(V.shape[0], device=V.device)
    for _ in range(steps):
        Vn = F.normalize(V, dim=1)
        S = Vn @ Vn.T - 3.0 * eye
        loss = S.max(dim=1).values.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return F.normalize(V.detach(), dim=1).cpu()


# ═══════════════════════════════════════════════════════════════════════════
# AGGREGATION
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


def fedsat_weights(stats, sigma=1.5, alpha=1.0, beta=1.0):
    """Prioritized-class weighted aggregation from FedSat (Chowdhury & Halder
    2024, arXiv:2407.03862), Eqs. 11-21:
      * worker set = the client itself (the paper allows this, Sec. III-C),
        i.e. each client's counts come from its own validation split;
      * FNR/FPR use the standard definitions FNR_i=(T_i-TP_i)/T_i and
        FPR_i=(Z_i-TP_i)/(sum_r T_r - T_i); the paper's Eq. 12/14 print
        different denominators that look like typos;
      * weights are normalised to sum to 1 (equivalent to Eq. 22 with eta_g=1);
      * sigma is untuned (paper: 'varies from task to task').
    stats: list of dicts with per-class int lists T (targets), Z (predicted),
    TP (true positives). Returns (normalised weights, priority class)."""
    K = len(stats)
    E = np.zeros((K, N_CLASSES))
    acc = np.zeros((K, N_CLASSES))
    T = np.array([s["T"] for s in stats], dtype=np.float64)
    Z = np.array([s["Z"] for s in stats], dtype=np.float64)
    TP = np.array([s["TP"] for s in stats], dtype=np.float64)
    for k in range(K):
        tot = T[k].sum()
        fnr = np.where(T[k] > 0, (T[k]-TP[k]) / np.maximum(T[k], 1), 0.0)
        neg = tot - T[k]
        fpr = np.where(neg > 0, (Z[k]-TP[k]) / np.maximum(neg, 1), 0.0)
        acc[k] = np.where(T[k] > 0, TP[k] / np.maximum(T[k], 1), 0.0)
        E[k] = (alpha * fnr / (fnr.max() if fnr.max() > 0 else 1.0)
                + beta * fpr / (fpr.max() if fpr.max() > 0 else 1.0))
    pc = int(np.argmax(E.sum(0)))                      # Eq. 19
    score = E.sum(1)
    keep = score <= sigma * score.mean()               # Eq. 20
    if not keep.any(): keep[:] = True
    w = np.zeros(K)
    for k in np.where(keep)[0]:                        # Eq. 21
        misclassified = max(1.0, T[k].sum() - TP[k].sum())
        w[k] = (acc[k].sum() * TP[k].sum() / max(score[k], 1e-8)) \
               * max(1.0, TP[k, pc]) / misclassified
    if w.sum() <= 0: w[keep] = 1.0
    return w / w.sum(), pc


# ═══════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL -- fp16 autocast + GradScaler, no gradient checkpointing
# (memory headroom confirmed: bs=2 peaks at 6.55GB on a 15GB GPU)
# ═══════════════════════════════════════════════════════════════════════════

def local_train(model, device, gsd, site, tidx, cls_w_np, lr=1e-4, epochs=1, bs=2,
                mu=0.0, loss="cb", extra=None, loader=None, opt=None):
    """loss: 'cb'     class-balanced CE (site class_weights; Cui et al. 2019)
             'ce'     plain cross-entropy
             'psl'    FedSat prediction-sensitive loss
             'fediic' FedIIC: DALA + intra- and inter-client contrastive
    mu > 0 adds the FedProx proximal term (Li et al. 2020).
    Returns (state dict as numpy, n samples, info dict or None)."""
    model.load_state_dict(np2sd(gsd, model.state_dict()))
    model.train()
    if opt is None:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    # fp16, not bf16 -- this is a T4 (Turing) GPU, which lacks the tensor-core
    # bf16 acceleration that Ampere+ GPUs have. bf16 measured ~2x slower here.
    # fp16 needs a GradScaler to avoid gradient underflow; bf16 did not.
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    cls_w = torch.tensor(cls_w_np, dtype=torch.float32).to(device) if loss == "cb" else None
    gparams = [p.detach().clone() for p in model.parameters()] if mu > 0 else None
    if loader is None:
        loader = make_loader(site, tidx, bs=bs)

    M = torch.zeros(N_CLASSES, N_CLASSES)              # PSL confusion counts (this round)
    info = None
    if loss == "fediic":
        V = extra["protos"].to(device)
        prior = torch.tensor(extra["prior"], dtype=torch.float32, device=device)
        lce = torch.tensor(extra["lce"], dtype=torch.float32, device=device)
        delta = torch.log(lce.clamp(min=1e-6) ** FEDIIC_Q / prior.clamp(min=1e-3))
        loss_sum = torch.zeros(N_CLASSES, device=device)
        cnt = torch.zeros(N_CLASSES, device=device)

    for _ in range(epochs):
        for vol,msk,lbl,has_t in loader:
            vol,msk,lbl = vol.to(device), msk.to(device), lbl.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type=="cuda"):
                seg,_,seg_l,cls_l = model(vol)
                dc = dice_loss(torch.sigmoid(seg_l.float()), msk, has_t)
                if loss == "cb":
                    cls_loss = F.cross_entropy(cls_l, lbl, weight=cls_w)
                elif loss == "ce":
                    cls_loss = F.cross_entropy(cls_l, lbl)
                elif loss == "psl":
                    per = F.cross_entropy(cls_l.float(), lbl, reduction="none")
                    l_cpu, p_cpu = lbl.cpu(), cls_l.argmax(1).detach().cpu()
                    cost = psl_cost(M, l_cpu, p_cpu).to(device)
                    cls_loss = (per * cost).mean()
                    M.index_put_((l_cpu, p_cpu), torch.ones(len(l_cpu)), accumulate=True)
                elif loss == "fediic":
                    f1 = model._feat
                    logits = cls_l.float()
                    cls_loss = F.cross_entropy(logits - delta, lbl)          # DALA, Eq. 6
                    model(augment_view(vol))
                    f2 = model._feat
                    proj = model.head(torch.cat([f1, f2]))
                else:
                    raise ValueError(loss)
            if loss == "fediic":
                with torch.autocast(device.type, enabled=False):
                    z = F.normalize(proj.float(), dim=1)
                    y2 = torch.cat([lbl, lbl])
                    l_intra = fediic_intra_loss(z, y2, prior)
                    l_inter = F.cross_entropy(z @ V.T / FEDIIC_TAU, y2)      # Eq. 4
                    cls_loss = cls_loss + FEDIIC_K1 * l_intra + FEDIIC_K2 * l_inter
                    with torch.no_grad():
                        per = F.cross_entropy(logits.detach(), lbl, reduction="none")
                        loss_sum.index_add_(0, lbl, per)
                        cnt.index_add_(0, lbl, torch.ones_like(per))
            total = cls_loss + 0.5*dc
            if mu > 0:
                prox = sum(((p - g) ** 2).sum()
                           for p, g in zip(model.parameters(), gparams))
                total = total + 0.5 * mu * prox
            if scaler:
                scaler.scale(total).backward(); scaler.step(opt); scaler.update()
            else:
                total.backward(); opt.step()

    if loss == "fediic":
        info = {"loss_sum": loss_sum.cpu().numpy(), "cnt": cnt.cpu().numpy()}
    n = len(loader.dataset)
    return sd2np(model.state_dict()), n, info

@torch.no_grad()
def eval_site(model, device, site, indices=None, bs=2):
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
            "dice": float(np.mean(dices)) if dices else 0.0,
            "T":  [int((lbls==c).sum()) for c in range(N_CLASSES)],
            "Z":  [int((preds==c).sum()) for c in range(N_CLASSES)],
            "TP": [int(((preds==c)&(lbls==c)).sum()) for c in range(N_CLASSES)]}


def run_round_parallel(models, sites, gsd, site_tr, site_cw, site_va, site_prior,
                       lr, epochs, bs, need_eval, mu, loss, extra_common):
    chunks = [sites[i::len(models)] for i in range(len(models))]
    def worker(chunk_idx):
        model, device = models[chunk_idx], DEVICES[chunk_idx]
        results = []
        for site in chunks[chunk_idx]:
            extra = {**extra_common, "prior": site_prior[site]} if extra_common else None
            upd, n, info = local_train(model, device, gsd, site, site_tr[site],
                                       site_cw[site], lr=lr, epochs=epochs, bs=bs,
                                       mu=mu, loss=loss, extra=extra)
            ev = None
            if need_eval:
                model.load_state_dict(np2sd(upd, model.state_dict()))
                ev = eval_site(model, device, site, site_va[site], bs)
            results.append((site, upd, n, ev, info))
        return results
    with ThreadPoolExecutor(max_workers=len(models)) as ex:
        futures = [ex.submit(worker, i) for i in range(len(models))]
        return [r for fut in futures for r in fut.result()]


def make_pretrained_model(device, fediic=False):
    m = PedBrainNetPretrained(num_classes=N_CLASSES, pretrained=True,
                              weights_path=str(WEIGHTS_PATH))
    if fediic:
        m = FedIICWrapper(m)
    return m.to(device)


def run_experiment(name, n_rounds=200, warmup_rounds=50, local_epochs=1,
                   bs=2, lr=1e-4, mode="fl", agg="fedavg", mu=0.0, loss="cb",
                   cbwa_alpha=0.5, cbwa_gamma=1.0, fedsat_sigma=1.5,
                   iil_epochs=10, eval_every=20, bs_eval=2):
    """mode: 'fl'   federated (agg = 'fedavg' | 'cbwa' | 'fedsat')
             'cds'  centralized data sharing: one model on pooled data, 1 epoch per round
             'ciil' cyclic institutional incremental learning: each round the model
                    visits every active site in turn for 1 epoch
             'iil'  institutional incremental learning: a single pass, smallest
                    site first, iil_epochs per site (one 'round' per site)
    Defaults reproduce the original FedAvg_pretrained200 / CBWA_pretrained200
    runs exactly."""
    assert mode in ("fl", "cds", "ciil", "iil") and agg in ("fedavg", "cbwa", "fedsat")
    fediic = loss == "fediic"

    print(f"\n{'='*72}\nEXPERIMENT: {name}  (mode={mode}, agg={agg}, loss={loss}, mu={mu})\n{'='*72}")

    done_path = OUT_DIR / f"{name}_results.json"
    if done_path.exists():
        hist = json.loads(done_path.read_text())
        print(f"ALREADY COMPLETE -- skipping, {len(hist)} eval points saved")
        return hist

    site_cw = {s: np.array(fed["clients"][s]["class_weights"], np.float32) for s in IN_NET}
    site_tr = {s: fed["clients"][s]["train_idx"] for s in IN_NET}
    site_va = {s: fed["clients"][s]["val_idx"]   for s in IN_NET}
    site_prior = {s: site_label_prior(s, site_tr[s]) for s in IN_NET}
    iil_order = sorted(IN_NET, key=lambda s: len(site_tr[s]))   # smallest first (Sheller)
    if mode == "iil":
        n_rounds, warmup_rounds = len(iil_order), 0

    devices = DEVICES if mode == "fl" else DEVICES[:1]          # non-FL modes are sequential
    models = [make_pretrained_model(d, fediic) for d in devices]
    n_params = sum(p.numel() for p in models[0].parameters())
    print(f"Parameters: {n_params:,} (real pretrained I3D encoder)")

    ckpt_path = OUT_DIR / f"{name}_checkpoint.pth"
    start_round = 0
    history = []
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=devices[0])
        for m in models: m.load_state_dict(ck["model_state_dict"])
        start_round = ck["round"] + 1
        history = json.loads((OUT_DIR / f"{name}_history.json").read_text())
        print(f"RESUMED from round {start_round}")

    gsd = sd2np(models[0].state_dict())
    prev_w = None
    lce = np.ones(N_CLASSES)                 # FedIIC per-class average CE (server state)
    ones_w = np.ones(N_CLASSES, np.float32)
    cds_opt = torch.optim.Adam(models[0].parameters(), lr=lr) if mode == "cds" else None

    for rnd in range(start_round, n_rounds):
        t0 = time.time()
        warmup = rnd < warmup_rounds
        active = WARMUP_SITES if warmup else IN_NET
        row = {"round": rnd+1, "warmup": warmup}

        if mode == "fl":
            need_eval = agg in ("cbwa", "fedsat") and not warmup
            extra_common = None
            if fediic:
                extra_common = {"protos": fediic_prototypes(models[0]), "lce": lce}
            results = run_round_parallel(models, active, gsd, site_tr, site_cw, site_va,
                                         site_prior, lr, local_epochs, bs, need_eval,
                                         mu, loss, extra_common)
            updates = [r[1] for r in results]; ns = [r[2] for r in results]
            evs = [r[3] for r in results]
            if agg == "cbwa" and not warmup:
                gsd, prev_w = cbwa(updates, ns, [e["macro_f1"] for e in evs],
                                   cbwa_alpha, cbwa_gamma, prev=prev_w)
            elif agg == "fedsat" and not warmup:
                w, pc = fedsat_weights(evs, sigma=fedsat_sigma)
                gsd = wavg(updates, w)
                row.update({"priority_class": pc, "n_kept": int((w > 0).sum())})
            else:
                gsd = fedavg(updates, ns)
            if fediic:
                tot = sum(r[4]["loss_sum"] for r in results)
                cnt = sum(r[4]["cnt"] for r in results)
                lce = np.where(cnt > 0, tot / np.maximum(cnt, 1), lce)
        elif mode == "cds":
            loader = make_pooled_loader(active, site_tr, bs)
            gsd, _, _ = local_train(models[0], devices[0], gsd, None, None, ones_w,
                                    lr=lr, epochs=local_epochs, bs=bs, mu=mu, loss=loss,
                                    loader=loader, opt=cds_opt)
        elif mode == "ciil":
            for site in active:
                gsd, _, _ = local_train(models[0], devices[0], gsd, site, site_tr[site],
                                        site_cw[site], lr=lr, epochs=local_epochs,
                                        bs=bs, mu=mu, loss=loss)
        else:  # iil
            site = iil_order[rnd]
            gsd, _, _ = local_train(models[0], devices[0], gsd, site, site_tr[site],
                                    site_cw[site], lr=lr, epochs=iil_epochs,
                                    bs=bs, mu=mu, loss=loss)
            row["site"] = site

        for m in models: m.load_state_dict(np2sd(gsd, m.state_dict()))
        print(f"  round {rnd+1}/{n_rounds} done [{time.time()-t0:.0f}s]", flush=True)

        if (rnd+1) % eval_every == 0 or rnd == n_rounds-1:
            print(f"\nRound {rnd+1}/{n_rounds} ({'warmup' if warmup else 'full'}) "
                  f"[{time.time()-t0:.0f}s]")
            vm = [eval_site(models[0], devices[0], s, site_va[s], bs_eval) for s in IN_NET]
            row.update({"in_net_f1": float(np.mean([m["macro_f1"] for m in vm])),
                        "in_net_acc": float(np.mean([m["acc"] for m in vm])),
                        "in_net_dice": float(np.mean([m["dice"] for m in vm]))})
            print(f"  In-network — F1:{row['in_net_f1']:.4f} "
                  f"Acc:{row['in_net_acc']:.4f} Dice:{row['in_net_dice']:.4f}")
            for es in EXT:
                em = eval_site(models[0], devices[0], es, None, bs_eval)
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
    return history


# ═══════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════

HP = dict(n_rounds=200, warmup_rounds=50, local_epochs=1,
          bs=2, lr=1e-4, eval_every=20, bs_eval=2)
if SMOKE:
    HP.update(n_rounds=4, warmup_rounds=2, eval_every=2)

# (experiment name, source paper, config). Order = run priority. Each new
# experiment needs roughly one full GPU session (CDS/CIIL ~2, see notes);
# comment out rows to run them one session at a time. Finished experiments
# are skipped automatically.
EXPERIMENTS = [
    ("CBWA_pretrained200",        "Ours (CBWA)",
        dict(agg="cbwa", loss="cb")),                                      # done
    ("FedAvg_pretrained200",      "Cui et al. 2019 (FedAvg + class-balanced loss)",
        dict(agg="fedavg", loss="cb")),                                    # done
    ("FedAvgPlain_pretrained200", "McMahan et al. 2017 (FedAvg)",
        dict(agg="fedavg", loss="ce")),
    ("FedProx_pretrained200",     "Li et al. 2020 (FedProx)",
        dict(agg="fedavg", loss="ce", mu=0.01)),
    ("FedSat_pretrained200",      "Chowdhury & Halder 2024 (FedSat)",
        dict(agg="fedsat", loss="psl")),
    ("FedIIC_pretrained200",      "Wu et al. 2023 (FedIIC)",
        dict(agg="fedavg", loss="fediic")),
    ("CDS_pretrained200",         "Sheller et al. 2020 (centralized data sharing)",
        dict(mode="cds", loss="ce")),
    ("CIIL_pretrained200",        "Sheller et al. 2020 (cyclic IIL)",
        dict(mode="ciil", loss="ce")),
    ("IIL_pretrained",            "Sheller et al. 2020 (IIL, smallest first)",
        dict(mode="iil", loss="ce", iil_epochs=10, eval_every=4)),
]
if SMOKE:
    EXPERIMENTS[-1] = ("IIL_pretrained", EXPERIMENTS[-1][1],
                       dict(mode="iil", loss="ce", iil_epochs=1, eval_every=2))

# Papers in the literature review that have no trainable method to run.
NOT_RUNNABLE = [
    ("Zhang et al. 2023 (survey)",        "survey -- no method or experiments"),
    ("Kazerooni et al. 2024 (BraTS-PEDs)", "challenge/dataset paper -- no FL method; different dataset"),
    ("Rieke et al. 2020 (npj Dig Med)",    "perspective paper -- no method or experiments"),
]

ALREADY_RAN = {"FedAvg_pretrained200", "CBWA_pretrained200"}
# FEDCB_ONLY="FedProx_pretrained200" (comma-separated) runs just those experiments
ONLY = [n for n in os.environ.get("FEDCB_ONLY", "").split(",") if n]

print("\n" + "="*72)
print("REAL PRETRAINED MODEL -- 200 ROUNDS -- literature baselines vs CBWA")
print("I3D encoder, ImageNet+Kinetics pretrained, 37.9M params, fp16, bs=2")
print("="*72)

results, sources = {}, {}
for name, source, cfg in EXPERIMENTS:
    if ONLY and name not in ONLY:
        continue
    if name in ALREADY_RAN and not SMOKE and not (OUT_DIR / f"{name}_results.json").exists():
        # /kaggle/working was wiped -- do NOT silently retrain a ~9h run
        print(f"WARNING: {name}_results.json missing from {OUT_DIR}; skipping "
              f"(restore it from your downloaded copy to include it in the table)")
        continue
    results[name] = run_experiment(name, **{**HP, **cfg})
    sources[name] = source

print("\n" + "="*72)
print("FINAL RESULTS (last evaluated round)")
print("="*72)
hdr = (f"{'Source paper':48s} {'Experiment':26s} {'In-net F1':>9s} {'In-net Dice':>11s} "
       f"{'AU F1':>7s} {'AU Worst':>8s} {'TK F1':>7s} {'TK Worst':>8s}")
print(hdr)
rows = []
for nm, hist in results.items():
    last = [r for r in hist if "in_net_f1" in r][-1]
    vals = [last.get(k, 0) for k in ("in_net_f1", "in_net_dice", "AU_f1", "AU_worst",
                                     "TK_f1", "TK_worst")]
    rows.append({"source": sources[nm], "experiment": nm,
                 **dict(zip(("in_net_f1", "in_net_dice", "AU_f1", "AU_worst",
                             "TK_f1", "TK_worst"), vals))})
    print(f"{sources[nm]:48s} {nm:26s} {vals[0]:9.4f} {vals[1]:11.4f} "
          f"{vals[2]:7.4f} {vals[3]:8.4f} {vals[4]:7.4f} {vals[5]:8.4f}")
for src, why in NOT_RUNNABLE:
    print(f"{src:48s} {'n/a':26s} {why}")

(OUT_DIR/"all_results_pretrained200_baselines.json").write_text(
    json.dumps({"final": rows, "not_runnable": NOT_RUNNABLE, "history": results}, indent=2))
print(f"\nAll results -> {OUT_DIR}")
