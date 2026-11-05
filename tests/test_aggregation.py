"""Unit tests for the CBWA rule.

The first block mirrors the CBWA checks listed in the paper's Table 6
("Verification of the pipeline"); the rest covers edge cases and input checks.
"""

import numpy as np
import pytest

from cbwa import (
    CBWA,
    apply_floor,
    cbwa_scores,
    class_balanced_weights,
    effective_number,
    fedavg_weights,
    momentum_update,
    weighted_average,
)

RNG = np.random.default_rng(2026)
N4 = [246, 40, 10, 2]


def random_federation(k, rng=RNG):
    n = rng.integers(1, 300, size=k).astype(float)
    f1 = rng.uniform(0, 1, size=k)
    return n, f1


# --- Table 6 checks --------------------------------------------------------------

def test_weights_form_a_distribution():
    for k in range(2, 30):
        n, f1 = random_federation(k)
        for w in (cbwa_scores(n, f1), apply_floor(cbwa_scores(n, f1)), CBWA().weights(range(k), n, f1)):
            assert np.all(w >= 0)
            assert np.isclose(w.sum(), 1.0)


def test_equal_f1_largest_below_smallest_above_fedavg():
    q = cbwa_scores(N4, [0.7] * 4)
    fa = fedavg_weights(N4)
    assert q[0] < fa[0]      # largest hospital loses weight
    assert q[-1] > fa[-1]    # smallest hospital gains weight


def test_higher_f1_gives_higher_weight():
    n, f1 = random_federation(10)
    base = cbwa_scores(n, f1)
    for k in range(10):
        better = f1.copy()
        better[k] = min(1.0, f1[k] + 0.1)
        if better[k] > f1[k]:
            assert cbwa_scores(n, better)[k] > base[k]


def test_zero_f1_still_receives_the_floor():
    q = cbwa_scores(N4, [0.4, 0.6, 0.8, 0.0])
    assert q[3] < 0.01                     # 0.005 before the floor
    w = apply_floor(q, 0.01)
    assert np.isclose(w[3], 0.01)
    assert np.isclose(w.sum(), 1.0)


def test_momentum_is_exactly_07_previous_plus_03_q():
    w_prev = np.array([0.5, 0.3, 0.15, 0.05])
    q = np.array([0.4, 0.3, 0.2, 0.1])
    assert np.allclose(momentum_update(w_prev, q, 0.7), 0.7 * w_prev + 0.3 * q)


def test_aggregate_equals_weighted_parameter_average():
    rng = np.random.default_rng(1)
    params = [{"conv.weight": rng.normal(size=(3, 2)).astype(np.float32),
               "fc.bias": rng.normal(size=4).astype(np.float32)} for _ in range(4)]
    w = np.array([0.4, 0.3, 0.2, 0.1])
    avg = weighted_average(params, w)
    for name in params[0]:
        expected = sum(wk * p[name].astype(np.float64) for wk, p in zip(w, params))
        assert avg[name].dtype == np.float32
        assert np.allclose(avg[name], expected, atol=1e-6)


# --- Behaviour of the stateful server ----------------------------------------------

def test_first_round_uses_floored_scores_then_momentum():
    server = CBWA()
    ids = ["A", "B", "C", "D"]
    w1 = server.weights(ids, N4, [0.4, 0.6, 0.8, 1.0])
    assert np.allclose(w1, apply_floor(cbwa_scores(N4, [0.4, 0.6, 0.8, 1.0])))
    q2 = apply_floor(cbwa_scores(N4, [0.4, 0.6, 0.8, 0.0]))
    w2 = server.weights(ids, N4, [0.4, 0.6, 0.8, 0.0])
    assert np.allclose(w2, 0.7 * w1 + 0.3 * q2)


def test_momentum_follows_hospital_ids_not_positions():
    server = CBWA()
    server.weights(["A", "B", "C"], [10, 20, 30], [0.5, 0.5, 0.5])
    prev = server.state
    w = server.weights(["C", "A", "B"], [30, 10, 20], [0.5, 0.5, 0.5])
    q = apply_floor(cbwa_scores([30, 10, 20], [0.5, 0.5, 0.5]))
    expected = 0.7 * np.array([prev["C"], prev["A"], prev["B"]]) + 0.3 * q
    assert np.allclose(w, expected / expected.sum())


def test_state_restarts_when_participants_change_or_on_reset():
    server = CBWA()
    server.weights(["ST", "SE"], [197, 144], [0.6, 0.7])          # warm-up hospitals
    n = [197, 144, 90]
    f1 = [0.6, 0.7, 0.4]
    fresh = apply_floor(cbwa_scores(n, f1))
    assert np.allclose(server.weights(["ST", "SE", "CG"], n, f1), fresh)
    server.reset()
    assert server.state is None
    assert np.allclose(server.weights(["ST", "SE", "CG"], n, f1), fresh)


def test_fedavg_is_a_special_case():
    n, f1 = random_federation(16)
    server = CBWA(alpha=1.0, gamma=0.0, w_min=0.0, m=0.0)
    for _ in range(3):
        assert np.allclose(server.weights(range(16), n, f1), fedavg_weights(n), rtol=0, atol=1e-15)
        f1 = RNG.uniform(0, 1, size=16)


def test_aggregate_returns_params_and_weights():
    params = [{"w": np.full(2, float(i))} for i in range(4)]
    new, w = CBWA().aggregate(["A", "B", "C", "D"], N4, [0.4, 0.6, 0.8, 0.9], params)
    assert np.allclose(new["w"], np.full(2, np.dot(w, [0, 1, 2, 3])))


# --- Floor, momentum and averaging edge cases ---------------------------------------

def test_floor_is_a_no_op_when_nobody_is_below_it():
    q = np.array([0.4, 0.3, 0.2, 0.1])
    assert np.allclose(apply_floor(q, 0.01), q)
    assert np.allclose(apply_floor(q, 0.0), q)


def test_floor_requires_k_times_wmin_below_one():
    with pytest.raises(ValueError):
        apply_floor([0.5, 0.3, 0.1, 0.1], 0.25)


def test_hospital_exactly_at_the_floor_keeps_its_weight():
    q = np.array([0.6, 0.3, 0.095, 0.005])
    q = q / q.sum()
    w = apply_floor(q, 0.095)
    assert np.isclose(w[2], 0.095) and np.isclose(w[3], 0.095)


def test_weighted_average_rounds_integer_buffers():
    params = [{"num_batches_tracked": np.array(10, dtype=np.int64)},
              {"num_batches_tracked": np.array(13, dtype=np.int64)}]
    out = weighted_average(params, [0.5, 0.5])
    assert out["num_batches_tracked"].dtype == np.int64
    assert int(out["num_batches_tracked"]) == 12      # 11.5 rounds to the even integer


def test_weighted_average_rejects_mismatched_models():
    with pytest.raises(ValueError):
        weighted_average([{"a": np.zeros(2)}, {"b": np.zeros(2)}], [0.5, 0.5])
    with pytest.raises(ValueError):
        weighted_average([{"a": np.zeros(2)}, {"a": np.zeros(3)}], [0.5, 0.5])
    with pytest.raises(ValueError):
        weighted_average([{"a": np.zeros(2)}, {"a": np.zeros(2)}], [0.7, 0.7])


def test_effective_number():
    assert np.isclose(effective_number(np.full(16, 1 / 16)), 16)
    assert np.isclose(effective_number([1.0, 0.0, 0.0]), 1)


@pytest.mark.parametrize("n, f1", [
    ([10, 0], [0.5, 0.5]),          # a hospital without patients
    ([10, 20], [0.5, 1.2]),         # F1 above 1
    ([10, 20], [0.5, -0.1]),        # F1 below 0
    ([10, 20, 30], [0.5, 0.5]),     # lengths differ
    ([10, np.nan], [0.5, 0.5]),     # not a number
])
def test_invalid_inputs_are_rejected(n, f1):
    with pytest.raises(ValueError):
        cbwa_scores(n, f1)


def test_duplicate_hospital_ids_are_rejected():
    with pytest.raises(ValueError):
        CBWA().weights(["A", "A"], [10, 20], [0.5, 0.5])


# --- Class-balanced weights (Eq. (3)) -----------------------------------------------

def test_class_weights_mean_one_over_present_classes():
    a = class_balanced_weights([31, 59, 72, 84, 0])
    assert a[4] == 0
    assert np.isclose(a[:4].mean(), 1.0)


def test_rarer_class_gets_larger_weight():
    a = class_balanced_weights([31, 72])
    assert a[0] > a[1]
    assert 1.8 < a[0] / a[1] < 2.0          # "about twice" (Section 3.3)


def test_beta_zero_gives_uniform_weights():
    assert np.allclose(class_balanced_weights([5, 50, 500], beta=0.0), 1.0)


def test_class_weights_reject_bad_input():
    with pytest.raises(ValueError):
        class_balanced_weights([0, 0])
    with pytest.raises(ValueError):
        class_balanced_weights([3, 4], beta=1.0)
