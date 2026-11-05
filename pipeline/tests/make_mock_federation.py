# Builds a tiny fake /kaggle/working tree: same file layout + model interface as the real one.
import json, sys, numpy as np
from pathlib import Path
root = Path(sys.argv[1]); rng = np.random.default_rng(0)
proc = root/"processed"; proc.mkdir(parents=True, exist_ok=True)
in_net = ["ST", "SE", "BO", "KC"]; ext = ["AU", "TK"]
clients = {}
for s in in_net + ext:
    d = proc/s; d.mkdir(exist_ok=True)
    n = 6
    vols = rng.integers(0, 255, (n, 8, 32, 32), dtype=np.uint8)
    masks = (rng.random((n, 8, 32, 32)) > 0.8).astype(np.uint8)
    labels = [int(x) for x in rng.integers(1, 5, n)]
    if s == "AU": labels = [2]*n
    np.save(d/"volumes.npy", vols); np.save(d/"masks.npy", masks)
    (d/"meta.json").write_text(json.dumps({"labels": labels}))
    c = {"class_weights": [0.0, 1.0, 1.5, 0.8, 1.2], "train_idx": [0,1,2,3], "val_idx": [4,5]}
    if s in ext: c["role"] = "external_eval_only"
    clients[s] = c
(proc/"federation.json").write_text(json.dumps({"in_network_sites": in_net + ext, "clients": clients}))
src = root/"FLPedBrain/FLPedBrain-PyTorch/src"; src.mkdir(parents=True, exist_ok=True)
(src/"model_pretrained.py").write_text('''
import torch, torch.nn as nn, torch.nn.functional as F
class PedBrainNetPretrained(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, weights_path=None):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv3d(1, 4, 3, padding=1), nn.BatchNorm3d(4), nn.ReLU())
        self.seg = nn.Conv3d(4, 1, 1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, num_classes))
    def forward(self, x):
        f = self.enc(x); seg_l = self.seg(f)
        cls_l = self.classifier(self.pool(f).flatten(1))
        return torch.sigmoid(seg_l), F.softmax(cls_l, 1), seg_l, cls_l
''')
(root/"rgb_imagenet.pt").write_bytes(b"mock")
print("mock ready at", root)
