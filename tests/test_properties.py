"""Randomized checks of Propositions 1-3 (Section 3.5 of the paper)."""

import numpy as np

from cbwa import apply_floor, cbwa_scores, momentum_update

RNG = np.random.default_rng(7)
TRIALS = 2000


def test_proposition_1_bounded_reweighting():
    """q_i / q_j = (n_i/n_j)^alpha ((F_i+eps)/(F_j+eps))^gamma, and the score factor is bounded."""
    for _ in range(TRIALS):
        k = RNG.integers(2, 20)
        n = RNG.integers(1, 300, size=k).astype(float)
        f1 = RNG.uniform(0, 1, size=k)
        alpha = RNG.uniform(0, 1)
        gamma = RNG.uniform(0, 2)
        eps = RNG.uniform(0.01, 0.2)
        q = cbwa_scores(n, f1, alpha, gamma, eps)
        i, j = RNG.choice(k, size=2, replace=False)
        size = (n[i] / n[j]) ** alpha
        score = ((f1[i] + eps) / (f1[j] + eps)) ** gamma
        assert np.isclose(q[i] / q[j], size * score, rtol=1e-12)
        lo, hi = (eps / (1 + eps)) ** gamma, ((1 + eps) / eps) ** gamma
        assert lo - 1e-12 <= score <= hi + 1e-12


def test_proposition_1_bound_is_reached():
    eps = 0.05
    q = cbwa_scores([10, 10], [1.0, 0.0], eps=eps)
    assert np.isclose(q[0] / q[1], (1 + eps) / eps)        # factor 21


def test_proposition_2_guaranteed_participation():
    for _ in range(TRIALS):
        k = RNG.integers(2, 40)
        w_min = RNG.uniform(0, 0.99 / k)
        q = RNG.dirichlet(np.full(k, RNG.uniform(0.05, 2.0)))
        w = apply_floor(q, w_min)
        assert np.all(w >= w_min - 1e-12)
        assert np.isclose(w.sum(), 1.0)
        # hospitals above the floor are never pushed below it, and never gain weight
        above = q >= w_min
        assert np.all(w[above] <= q[above] + 1e-12)
        # momentum is a convex combination of two floored distributions: the floor is kept
        w_prev = apply_floor(RNG.dirichlet(np.ones(k)), w_min)
        m = RNG.uniform(0, 0.99)
        assert np.all(momentum_update(w_prev, w, m) >= w_min - 1e-12)


def test_proposition_3_damped_response():
    for _ in range(TRIALS):
        k = RNG.integers(2, 20)
        m = RNG.uniform(0, 0.99)
        w_prev = RNG.dirichlet(np.ones(k))
        q1, q2 = RNG.dirichlet(np.ones(k)), RNG.dirichlet(np.ones(k))
        # both inputs are distributions, so the renormalization in Eq. (5) changes nothing
        change = momentum_update(w_prev, q2, m) - momentum_update(w_prev, q1, m)
        assert np.allclose(change, (1 - m) * (q2 - q1), atol=1e-12)


def test_proposition_3_effect_of_one_noisy_round_decays_as_m_to_the_s():
    m, k, rounds = 0.7, 5, 12
    q = RNG.dirichlet(np.ones(k))
    noisy = RNG.dirichlet(np.ones(k))
    w_a = w_b = q.copy()
    for s in range(rounds):
        w_a = momentum_update(w_a, noisy if s == 0 else q, m)
        w_b = momentum_update(w_b, q, m)
        # after the noisy round, the gap shrinks by a factor m every round
        assert np.allclose(w_a - w_b, (1 - m) * m ** s * (noisy - q), atol=1e-12)
