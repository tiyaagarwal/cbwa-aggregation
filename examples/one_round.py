"""Algorithm 1 in a toy federation: five CBWA rounds with NumPy "models".

    python examples/one_round.py

Each hospital trains locally (here: the global parameters plus a small random
step), scores its updated model on its own validation patients (here: a given
macro-F1), and sends (parameters, n_k, F_k). The server weights the updates
with CBWA and averages them. Replace the two marked lines with real local
training and evaluation.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbwa import CBWA, effective_number, fedavg_weights  # noqa: E402

rng = np.random.default_rng(0)
hospitals = ["H1", "H2", "H3", "H4"]
n = [246, 40, 10, 2]                                   # patients per hospital
global_params = {"layer.weight": np.zeros((2, 3)), "layer.bias": np.zeros(2)}

# Validation macro-F1 that each hospital reports in rounds 1-5 (H4 is noisy).
f1_per_round = [
    [0.40, 0.60, 0.80, 1.00],
    [0.42, 0.61, 0.79, 0.00],
    [0.45, 0.62, 0.81, 1.00],
    [0.47, 0.64, 0.80, 0.00],
    [0.50, 0.65, 0.82, 1.00],
]

server = CBWA()   # alpha = 0.5, gamma = 1, eps = 0.05, w_min = 0.01, m = 0.7
print("FedAvg weights:", np.round(fedavg_weights(n), 3), f"K_eff = {effective_number(fedavg_weights(n)):.2f}\n")

for t, f1 in enumerate(f1_per_round, start=1):
    updates = []
    for k in range(len(hospitals)):
        # --- local training on hospital k's own data (toy stand-in) ---
        updates.append({name: p + 0.01 * rng.normal(size=p.shape) for name, p in global_params.items()})
        # --- F_k = macro-F1 of the updated model on hospital k's validation patients: f1[k] ---
    global_params, w = server.aggregate(hospitals, n, f1, updates)
    print(f"round {t}: F1 = {f1}  ->  CBWA weights {np.round(w, 3)}  K_eff = {effective_number(w):.2f}")
