"""Closed-loop simulator (Algorithm 1).

At each refit epoch the learner is retrained on its OWN last W observations (Y, C);
the order is held for R days; the simulator realises demand, returns only min(D, q) and the
stockout flag, and books the true cost. All learners in a cell share the same demand paths
(common random numbers).
"""
import numpy as np
from .newsvendor import cost, draw_demand, true_quantile


def run_synthetic_cell(dist, cv, tau, start, learners, reps=100, horizon=364, window=56,
                       refit=7, last=112, mu=100.0, seed=0):
    """Returns {learner: dict(ratio=(reps,), cost_ratio=(reps,), traj=(weeks,))}."""
    rng = np.random.default_rng(seed)
    T = window + horizon
    D = draw_demand(dist, mu, cv, (T, reps), rng)                     # common random numbers
    qs = true_quantile(dist, mu, cv, tau)
    L = slice(T - last, T)
    opt_cost = cost(qs, D[L], tau).mean(0)
    out = {}
    for name, fn in learners.items():
        q = np.full((T, reps), start * qs)
        Y, C = np.minimum(D, q), (D >= q).astype(float)
        for t0 in range(window, T, refit):
            qn = np.maximum(fn(Y[t0 - window:t0], C[t0 - window:t0], tau), 0.0)
            sl = slice(t0, min(t0 + refit, T))
            q[sl] = qn
            Y[sl] = np.minimum(D[sl], qn)
            C[sl] = (D[sl] >= qn).astype(float)
        wk = q[window:][::7] / qs                                        # weeks x reps
        hit = wk >= 0.95
        rec = np.where(hit.any(0), hit.argmax(0), np.nan)                # first week at >= 0.95 q*
        out[name] = dict(ratio=q[L].mean(0) / qs,
                         cost_ratio=cost(q[L], D[L], tau).mean(0) / opt_cost,
                         traj=(q[window:].mean(1) / qs)[::7], recover_week=rec)
    return out, qs


def run_real_closed_loop(panel, fit_predict, tau, refit, seed, warm=28, full_information=False):
    """One learner on the real-data panel.

    panel: dict with D (T x K latent demand), Yreal, Creal (real censored history), covariates.
    fit_predict(Y, C, t_end, tau, seed, exact_labels) -> callable(t, Y, C) giving orders for day t.
    full_information=True: the learner always observes true demand (same-type benchmark).
    Returns Q (T x K) with NaN before the loop starts.
    """
    D = panel["D"]
    T, K = D.shape
    Y = np.zeros_like(D)
    C = np.zeros_like(D)
    Q = np.full_like(D, np.nan)
    if full_information:
        Y[:warm] = D[:warm]
    else:
        Y[:warm] = panel["Yreal"][:warm]
        C[:warm] = panel["Creal"][:warm]
    predict = None
    for t in range(warm, T):
        if (t - warm) % refit == 0:
            predict = fit_predict(Y, C, t, tau, seed, exact_labels=full_information)
        q = predict(t, Y, C)
        Q[t] = q
        if full_information:
            Y[t] = D[t]
            C[t] = 0.0
        else:
            Y[t] = np.minimum(D[t], q)
            C[t] = (D[t] >= q).astype(float)
    return Q
