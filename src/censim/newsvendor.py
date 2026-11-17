"""Newsvendor primitives: cost, optimal quantile, synthetic demand (Eq. 1 in the paper)."""
import numpy as np
from scipy import stats


def underage_cost(tau: float) -> float:
    """Underage cost b for overage cost h = 1, so that tau = b / (b + h)."""
    return tau / (1.0 - tau)


def cost(q, d, tau):
    """Newsvendor cost C(q, D) = h (q - D)^+ + b (D - q)^+ with h = 1."""
    b = underage_cost(tau)
    return np.maximum(q - d, 0.0) + b * np.maximum(d - q, 0.0)


def draw_demand(dist: str, mu: float, cv: float, size, rng):
    """Synthetic demand with mean mu and coefficient of variation cv."""
    sd = cv * mu
    if dist == "normal":
        return np.clip(rng.normal(mu, sd, size), 0.0, None)
    if dist == "lognormal":
        s2 = np.log(1.0 + cv ** 2)
        return rng.lognormal(np.log(mu) - s2 / 2.0, np.sqrt(s2), size)
    raise ValueError(dist)


def true_quantile(dist: str, mu: float, cv: float, tau: float) -> float:
    """Optimal order q* = F^{-1}(tau)."""
    sd = cv * mu
    if dist == "normal":
        return max(stats.norm.ppf(tau, mu, sd), 0.0)
    s2 = np.log(1.0 + cv ** 2)
    return float(np.exp(np.log(mu) - s2 / 2.0 + np.sqrt(s2) * stats.norm.ppf(tau)))
