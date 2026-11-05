# Unit tests for the CBWA weighting rule, extracted from the training script by AST.
import ast, sys, numpy as np
src = open(sys.argv[1]).read()
ns = {"np": np}
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name in ("wavg", "fedavg", "cbwa"):
        exec(compile(ast.Module([node], []), "x", "exec"), ns)
cbwa, wavg = ns["cbwa"], ns["wavg"]
ups = [{"w": np.full(3, float(i))} for i in range(4)]
ns_ = [246, 40, 10, 2]                       # ST-like big site ... KC-like tiny site
# 1 weights are a distribution
_, w = cbwa(ups, ns_, [0.5, 0.5, 0.5, 0.5]); assert abs(w.sum() - 1) < 1e-12 and (w > 0).all()
# 2 dampening: with equal F1, CBWA gives the biggest site less weight than FedAvg
fedavg_w = np.array(ns_) / sum(ns_); assert w[0] < fedavg_w[0] and w[3] > fedavg_w[3]
# 3 performance term: raising one site's F1 raises its weight
_, w2 = cbwa(ups, ns_, [0.5, 0.5, 0.9, 0.5]); assert w2[2] > w[2]
# 4 floor: a site with F1 = 0 still gets at least the 0.01 floor
_, w3 = cbwa(ups, ns_, [0.9, 0.9, 0.9, 0.0]); assert w3[3] >= 0.01 - 1e-12
# 5 momentum: weights = 0.7 * previous + 0.3 * fresh (both already sum to 1)
prev = np.array([0.7, 0.1, 0.1, 0.1])
_, w4 = cbwa(ups, ns_, [0.5, 0.5, 0.5, 0.5], prev=prev)
assert np.allclose(w4, 0.7 * prev + 0.3 * w)
# 6 aggregation is the weighted average of client parameters
agg, w5 = cbwa(ups, ns_, [0.5, 0.5, 0.5, 0.5]); assert np.allclose(agg["w"], sum(wi * i for i, wi in enumerate(w5)))
print("CBWA_TESTS_OK  weights:", np.round(w, 3), "vs FedAvg:", np.round(fedavg_w, 3))
