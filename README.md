# CBWA: Class-Balance-Aware Weighted Aggregation

Reference implementation of the aggregation rule from:

> T. Agarwal, C. Patil and S. Saman, *Class-Balance-Aware Weighted Aggregation for Federated Pediatric Brain Tumor MRI Classification and Segmentation*, Results in Engineering (manuscript in preparation).

**What this repository is.** An independent, tested implementation of the CBWA server rule, written from the paper's description. It reproduces every number the paper derives from the rule: **44 of 44 values match** ([results/paper_numbers.md](results/paper_numbers.md)).

**What it is not.** It is not the training pipeline used for the paper's experiments (I3D model, FLPedBrain data loading, Kaggle scripts). That pipeline is available from the corresponding author on reasonable request.

## The rule

A federated server usually weights each hospital by its size (FedAvg). CBWA also uses how good each hospital's update is. After local training, hospital *k* scores its updated model on its **own** validation patients and sends one number, the macro-F1 $F_k$. The server then computes:

$$\hat w_k = n_k^{\alpha}\,(F_k+\varepsilon)^{\gamma}, \qquad q_k = \hat w_k \Big/ \sum_j \hat w_j \qquad \text{(Eq. 4)}$$

1. **Floor.** Any $q_k$ below $w_{\min}$ is raised to $w_{\min}$. The deficit is taken from the other hospitals in proportion to their excess above the floor.
2. **Momentum.** $w_k^{(t)} \propto m\,w_k^{(t-1)} + (1-m)\,q_k$, renormalized (Eq. 5).
3. **Average.** $\theta^{t+1} = \sum_k w_k^{(t)}\,\theta_k$ (Eq. 1).

Each hospital trains with an effective-number class-balanced loss, $a_c \propto (1-\beta)/(1-\beta^{N_c})$ (Eq. 3). No data, features or predictions leave a hospital; the only extra traffic is one scalar per hospital per round.

| Setting | Symbol | Paper value |
|---|---|---|
| Size-dampening exponent | α | 0.5 |
| Performance exponent | γ | 1 |
| Smoothing constant | ε | 0.05 |
| Weight floor | w_min | 0.01 |
| Momentum | m | 0.7 |
| Effective-number parameter | β | 0.99 |

With α = 1, γ = 0, w_min = 0 and m = 0, CBWA is exactly FedAvg.

## Install

Python 3.9 or newer. The only dependency is NumPy.

```bash
git clone https://github.com/tiyaagarwal/cbwa-aggregation
cd cbwa-aggregation
pip install -e ".[test]"
```

## Quick start

```python
import numpy as np
from cbwa import CBWA

server = CBWA()                                   # paper settings
ids = ["ST", "SE", "CG", "KC"]
n   = [197, 144, 90, 1]                           # training patients per hospital
f1  = [0.59, 0.75, 0.35, 1.00]                    # validation macro-F1 sent by each hospital
client_params = [{"fc.weight": np.full((2, 2), float(k))} for k in range(4)]   # one dict per hospital

new_params, w = server.aggregate(ids, n, f1, client_params)   # CBWA weights + Eq. (1)
print(w.round(3))                                             # [0.383 0.41  0.162 0.045]
```

Call `aggregate` once per round, or `weights` if you average the parameters yourself. Each call advances the momentum state, so don't call both in the same round. `client_params` is one `{name: numpy array}` dictionary per hospital. With PyTorch:

```python
import torch

client_params = [{k: v.detach().cpu().numpy() for k, v in sd.items()} for sd in client_state_dicts]
new_params, w = server.aggregate(ids, n, f1, client_params)
global_model.load_state_dict({k: torch.from_numpy(v) for k, v in new_params.items()})
```

The class weights of the local loss:

```python
from cbwa import class_balanced_weights

weights = class_balanced_weights([31, 59, 72, 84])     # EP, DIPG, MB, PA counts at one hospital
loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
```

`examples/one_round.py` runs Algorithm 1 for five rounds in a toy federation.

## Reproduce the paper's numbers

```bash
python scripts/reproduce_paper.py
```

The script recomputes the following and compares each value with the paper at the paper's rounding:

- **Table 3:** Cases 1–6 of the four-hospital example.
- **Fig. 2b:** the Case 7 momentum series.
- **Section 3.5:** the constants of Propositions 1–3.
- **Section 5.6 and Fig. 8:** the FedAvg and CBWA weights of the 16 hospitals (from Tables 4 and 9), the shares (40.6% → 30.8%; 12.3% → 29.3%) and K_eff (7.9 → 11.0).

It exits with an error if any value differs. The full output is in [results/paper_numbers.md](results/paper_numbers.md).

## Tests

```bash
python -m pytest
```

There are 77 tests:

- The CBWA checks of the paper's Table 6 (weights form a distribution, the floor, the exact momentum update, the parameter average).
- Randomized checks of Propositions 1–3, 2,000 trials each.
- Edge cases: integer buffers, mismatched models, invalid input.
- One test per reproduced paper value.

## Implementation notes

- **n_k** is the number of training patients at hospital k. **F_k** is the macro-F1 over the tumor types present at that hospital, computed by the hospital on its own validation patients.
- **One floor pass is enough.** When K·w_min < 1, no hospital ends below the floor and the weights still sum to one (Proposition 2).
- **Momentum** starts from the first round's floored weights. It restarts from fresh weights when the set of participating hospitals changes (for example after a warm-up phase) or after `reset()`, for instance when a run resumes from a checkpoint that did not save the state.
- **Averaging** runs in 64-bit floating point. Each result is cast back to the parameter's dtype, and integer buffers are rounded.

## Repository layout

```
cbwa/aggregation.py    Eq. (1), (4), floor, Eq. (5), CBWA server class, K_eff
cbwa/class_weights.py  Eq. (3)
cbwa/paper.py          the paper's numbers and the checks that recompute them
scripts/reproduce_paper.py
examples/one_round.py
tests/
results/paper_numbers.md
```

## Citation

Please cite the paper. To cite this software, use [CITATION.cff](CITATION.cff) (GitHub shows a "Cite this repository" button).

## License

MIT; see [LICENSE](LICENSE).

## Use of generative AI

This code was written with the help of an AI assistant (Claude, Anthropic). Every value it reports is checked against the paper by the test suite.
