"""Analytical properties (Section 3.4).

Proposition 2: the PAR fixed point solves q = E[min(D, q)] + z_tau * SD[min(D, q)] (Eq. 4).
We evaluate the large-window map by Monte Carlo and iterate it from q*.
"""
import numpy as np
from scipy import stats
from .newsvendor import draw_demand, true_quantile


def par_map(q, d_sample, tau):
    y = np.minimum(d_sample, q)
    return y.mean() + stats.norm.ppf(tau) * y.std()


def par_fixed_point(dist, mu, cv, tau, n=400_000, seed=7, max_iter=2000, tol=1e-7):
    """Iterates the PAR map from q*. Returns q_dagger / q*.
    For tau <= 0.5 the map is strictly decreasing to 0 (Proposition 1), so the result tends to 0."""
    d = draw_demand(dist, mu, cv, n, np.random.default_rng(seed))
    qs = true_quantile(dist, mu, cv, tau)
    q = qs
    for _ in range(max_iter):
        qn = par_map(q, d, tau)
        if abs(qn - q) < tol * mu:
            break
        q = qn
    return q / qs
