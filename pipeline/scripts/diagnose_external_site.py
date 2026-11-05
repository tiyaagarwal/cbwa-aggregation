"""
Diagnostic: what is each model actually PREDICTING for AU's patients?

AU has 24 patients, all DIPG (class 2). F1=0.0000 for FedProx/CBWA just
means "zero correct DIPG predictions" -- it doesn't say what the model
predicted INSTEAD. This script finds out.

No training, no GPU quota cost beyond a few seconds of inference.
Paste into a new Kaggle cell and run.
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROCESSED = Path("/kaggle/working/processed")
OUT_DIR   = Path("/kaggle/working/fedcb_results")
CLASSES   = ["NormalControl", "EP", "DIPG", "MB", "PA"]
DEVICE    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── model definition -- MUST match train_fedcb_fast.py exactly for weights to load ──
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
        skips = []
        x = self.conv1(x); skips.append(x); x = self.pool1(x)
        x = self.conv2a(x); x = self.conv2b(x); skips.append(x); x = self.pool2(x)
        x = self.inc3b(x); x = self.inc3c(x); skips.append(x); x = self.pool3(x)
        for m in (self.inc4b,self.inc4c,self.inc4d,self.inc4e,self.inc4f): x = m(x)
        skips.append(x); x = self.pool4(x)
        x = self.inc5b(x); x = self.inc5c(x)
        return x, skips

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__(); self.conv = Conv3dBN(in_ch, out_ch)
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
        self.up5 = UpBlock(32,    16); self.final = nn.Conv3d(16, 1, 1)
    def forward(self, x, skips):
        s4,s3,s2,s1 = skips[3],skips[2],skips[1],skips[0]
        x = self.c1(torch.cat([self.up1(x,s4.shape[2:]), s4], 1))
        x = self.c2(torch.cat([self.up2(x,s3.shape[2:]), s3], 1))
        x = self.c3(torch.cat([self.up3(x,s2.shape[2:]), s2], 1))
        x = self.c4(torch.cat([self.up4(x,s1.shape[2:]), s1], 1))
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
        cls_logits  = self.classifier(self.pool(feat).flatten(1))
        return F.softmax(cls_logits, 1), cls_logits

class AUDataset(Dataset):
    def __init__(self):
        d = PROCESSED / "AU"
        self.vols = np.load(d/"volumes.npy", mmap_mode="r")
        self.labels = np.array(json.loads((d/"meta.json").read_text())["labels"])
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        vol = self.vols[i].astype(np.float32)/255.0
        return torch.from_numpy(vol[None].copy()), int(self.labels[i])

# ── run diagnostic for each method ──────────────────────────────────────────
ds = AUDataset()
loader = DataLoader(ds, batch_size=1, shuffle=False)
print(f"AU: {len(ds)} patients, true labels: "
      f"{[CLASSES[l] for l in np.unique(ds.labels)]} "
      f"(all should be DIPG)\n")

for method in ["FedAvg", "FedProx", "CBWA"]:
    wpath = OUT_DIR / f"{method}_weights.pth"
    if not wpath.exists():
        print(f"{method}: weights not found, skipping"); continue

    model = PedBrainNet(5).to(DEVICE)
    sd = torch.load(wpath, map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()

    preds, confidences, all_probs = [], [], []
    with torch.no_grad():
        for vol, lbl in loader:
            vol = vol.to(DEVICE)
            probs, _ = model(vol)
            pred = probs.argmax(1).item()
            conf = probs.max(1).values.item()
            preds.append(pred); confidences.append(conf)
            all_probs.append(probs.cpu().numpy()[0])

    pred_names = [CLASSES[p] for p in preds]
    counts = {c: pred_names.count(c) for c in set(pred_names)}
    all_probs = np.array(all_probs)  # (24, 5)
    mean_dipg_prob = all_probs[:, CLASSES.index("DIPG")].mean()
    mean_probs_per_class = {c: round(float(all_probs[:, i].mean()), 3)
                            for i, c in enumerate(CLASSES)}

    print(f"{method}:")
    print(f"  Argmax prediction distribution: {counts}")
    print(f"  Mean confidence in winning class: {np.mean(confidences):.3f}")
    print(f"  Correctly predicted DIPG (argmax): {pred_names.count('DIPG')}/{len(ds)}")
    print(f"  Mean probability assigned to DIPG (true class): {mean_dipg_prob:.4f}")
    print(f"  Mean probability per class: {mean_probs_per_class}")
    print(f"  -> {'Model has near-zero DIPG signal even before argmax -- real collapse' if mean_dipg_prob < 0.05 else 'Model has some DIPG signal but another class wins -- close call, not full collapse'}")
    print()
