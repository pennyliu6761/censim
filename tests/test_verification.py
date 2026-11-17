"""Verification tests (Section 4.4): the simulator and learners behave as the model says."""
import numpy as np
from scipy import stats
from censim.learners_stat import km_quantile, make_kmx, par, saa, tob
from censim.newsvendor import draw_demand, true_quantile
from censim.simulate import run_synthetic_cell
from censim.theory import par_fixed_point


def test_infinite_order_reveals_demand():
    rng = np.random.default_rng(0)
    D = draw_demand("normal", 100, 0.4, 1000, rng)
    q = np.full_like(D, 1e12)
    assert np.allclose(np.minimum(D, q), D)


def test_learners_recover_optimum_without_censoring():
    """With uncensored data every learner must estimate q* (checks the estimators, not the loop)."""
    rng = np.random.default_rng(1)
    D = draw_demand("normal", 100, 0.4, (5000, 1), rng)
    C = np.zeros_like(D)
    qs = true_quantile("normal", 100, 0.4, 0.9)
    for fn in (par, saa, tob):
        assert abs(fn(D, C, 0.9)[0] / qs - 1.0) < 0.02


def test_kmx_recovers_from_low_start():
    out, _ = run_synthetic_cell("normal", 0.4, 0.9, 0.7, {"KMX": make_kmx(0.05)}, reps=30, horizon=182, seed=1)
    assert abs(out["KMX"]["ratio"].mean() - 1.0) < 0.05


def test_prop1_spiral_down_tau_half():
    rng = np.random.default_rng(2)
    d = draw_demand("normal", 100, 0.4, 200_000, rng)
    q = true_quantile("normal", 100, 0.4, 0.5)
    for _ in range(20):
        y = np.minimum(d, q)
        qn = y.mean() + stats.norm.ppf(0.5) * y.std()
        assert qn < q
        q = qn


def test_prop2_fixed_point_below_optimum():
    assert par_fixed_point("normal", 100, 0.4, 0.9, n=100_000) < 1.0


def test_prop3_saa_never_exceeds_past_maximum():
    rng = np.random.default_rng(3)
    D = draw_demand("normal", 100, 0.4, (56, 30), rng)
    q = rng.uniform(60, 140, (56, 30))
    Y = np.minimum(D, q)
    assert np.all(saa(Y, (D >= q).astype(float), 0.9) <= q.max(0) + 1e-9)


def test_km_not_identified_below_optimum_and_kmx_explores():
    rng = np.random.default_rng(4)
    D = draw_demand("normal", 100, 0.4, 56, rng)
    q = np.full(56, 0.7 * true_quantile("normal", 100, 0.4, 0.9))
    y, c = np.minimum(D, q), (D >= q).astype(float)
    v, ok = km_quantile(y, c, 0.9)
    assert not ok and np.isclose(v, q[0])
    assert make_kmx(0.05)(y[:, None], c[:, None], 0.9)[0] > q[0]


def test_tobit_recovers_normal_parameters():
    rng = np.random.default_rng(5)
    D = rng.normal(100, 30, (4000, 1))
    q = 110.0
    Y, C = np.minimum(D, q), (D >= q).astype(float)
    est = tob(Y, C, 0.5)
    assert abs(est[0] - 100) < 3
