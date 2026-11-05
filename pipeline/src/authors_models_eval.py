# Evaluate the base study's released models (Lee et al. 2024) on the same external patients as CBWA.
# Logs in with the HF_TOKEN Kaggle secret (a token with access to the gated FLPedBrain repository; never printed).
import sys, json
from pathlib import Path
import numpy as np, torch
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, list_repo_files, hf_hub_download
from scipy.stats import wilcoxon
login(token=UserSecretsClient().get_secret("HF_TOKEN"), add_to_git_credential=False)
ROOT = Path("/kaggle/working"); PROCESSED = ROOT / "processed"; OUT = ROOT / "fedcb_results"
sys.path.insert(0, str(ROOT / "FLPedBrain/FLPedBrain-PyTorch/src"))
from model_pretrained import PedBrainNetPretrained
DEV = torch.device("cuda:0")
REPO = "edhlee/FLPedBrain-processed"

files = list_repo_files(REPO, repo_type="dataset")
wanted = {n: f for f in files for n in ("fl_end_weights.pth", "cds_end_weights.pth") if f.endswith(n)}
print("found:", wanted)

def load(path):
    m = PedBrainNetPretrained(num_classes=5, pretrained=True, weights_path=str(ROOT / "rgb_imagenet.pt"))
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("model_state_dict", sd.get("state_dict", sd))
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"{Path(path).name}: {len(sd)} tensors, missing {len(missing)}, unexpected {len(unexpected)}")
    if missing or unexpected:
        print("   e.g. missing", missing[:3], "unexpected", unexpected[:3])
    return m.to(DEV).eval(), not missing and not unexpected

@torch.no_grad()
def run(model, site):
    d = PROCESSED / site
    vols = np.load(d / "volumes.npy", mmap_mode="r"); masks = np.load(d / "masks.npy", mmap_mode="r")
    labels = json.loads((d / "meta.json").read_text())["labels"]
    pred, dsc = [], []
    for i in range(len(labels)):
        x = torch.from_numpy(np.asarray(vols[i], np.float32) / 255.0)[None, None].to(DEV)
        m = torch.from_numpy(np.asarray(masks[i], np.float32)).to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            seg, _, _, cls_l = model(x)
        p = (seg[0, 0] > 0.5).float()
        pred.append(int(cls_l.argmax(1))); dsc.append(float((2 * (p * m).sum() + 1) / (p.sum() + m.sum() + 1)))
    return np.array(pred), np.array(labels), np.array(dsc)

def macro_f1(p, t):
    return float(np.mean([2 * np.sum((p == c) & (t == c)) / (np.sum(p == c) + np.sum(t == c)) for c in np.unique(t)]))

ours = json.loads((OUT / "significance_cbwa_vs_fedavg.json").read_text())["per_patient"]["CBWA_pretrained200"]
rows = {}
for name, f in wanted.items():
    model, exact = load(hf_hub_download(REPO, f, repo_type="dataset"))
    if not exact:
        print(f"   {name} does not match this model definition exactly - results below are NOT reliable")
    for site in ("AU", "TK"):
        p, t, d = run(model, site)
        cb_d = np.array(ours[site][2])
        rows[(name, site)] = dict(f1=macro_f1(p, t), dice=float(d.mean()), exact=exact)
        print(f"{name:20s} {site}: F1 {macro_f1(p, t):.4f}  Dice {d.mean():.4f}  | CBWA Dice {cb_d.mean():.4f}, "
              f"CBWA higher for {np.sum(cb_d > d)}/{len(d)}, Wilcoxon p={wilcoxon(cb_d, d).pvalue:.3g}")
    del model; torch.cuda.empty_cache()
(OUT / "authors_models_eval.json").write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in rows.items()}, indent=1))
print("Saved fedcb_results/authors_models_eval.json")
