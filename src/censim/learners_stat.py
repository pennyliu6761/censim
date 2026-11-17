"""Statistical learners for the synthetic experiment (E1).

Each learner maps a window of own sales Y (W x reps) and stockout flags C (W x reps)
to the next order quantity for every replication (column).

  PAR  normal fit to sales, censoring ignored          (Eq. 3, left)
  SAA  sample tau-quantile of sales, censoring ignored (Eq. 3, right)
  TOB  censored-normal (Tobit) MLE
  KM   plug-in Kaplan-Meier tau-quantile; largest observation if not identified
  KMX  as KM, but orders (1 + eps) x largest observation when not identified (exploration)
  SAAX as SAA, but orders (1 + eps) x the SAA order when the stockout share of the last k days
       exceeds 1 - tau, the share expected at the optimal order (the rule used for QNTX on real data)
"""
import numpy as np
from scipy import optimize, stats


def par(Y, C, tau):
    return np.maximum(Y.mean(0) + stats.norm.ppf(tau) * Y.std(0), 0.0)


def saa(Y, C, tau):
    return np.quantile(Y, tau, axis=0)


def tobit_fit(Y, C):
    """Right-censored normal MLE for every column; returns (mean, sd)."""
    n = Y.shape[1]
    m0 = Y.mean(0)
    s0 = np.maximum(Y.std(0), 1e-3 * np.maximum(np.abs(m0), 1.0))

    def f(p):
        m, ls = p[:n], p[n:]
        s = np.exp(ls)
        z = (Y - m) / s
        lsf = stats.norm.logsf(z)
        ll = np.where(C == 1, lsf, -0.5 * z ** 2 - 0.5 * np.log(2 * np.pi) - ls)
        mills = np.exp(stats.norm.logpdf(z) - lsf)
        dm = np.where(C == 1, mills / s, z / s)
        dls = np.where(C == 1, mills * z, z ** 2 - 1)
        return -ll.sum(), -np.concatenate([dm.sum(0), dls.sum(0)])

    x = optimize.minimize(f, np.concatenate([m0 * 1.1, np.log(s0)]), jac=True, method="L-BFGS-B",
                          bounds=[(None, None)] * n + [(-8, 8)] * n).x
    return x[:n], np.exp(x[n:])


def tob(Y, C, tau):
    m, s = tobit_fit(Y, C)
    return np.maximum(m + s * stats.norm.ppf(tau), 0.0)


def km_quantile(y, c, tau):
    """Plug-in Kaplan-Meier tau-quantile. Returns (value, identified)."""
    o = np.argsort(y, kind="stable")
    y, c = y[o], c[o]
    n, surv, i = len(y), 1.0, 0
    while i < n:
        j = i
        while j < n and y[j] == y[i]:
            j += 1
        d = int((c[i:j] == 0).sum())
        if d > 0:
            surv *= 1.0 - d / (n - i)
            if 1.0 - surv >= tau:
                return float(y[i]), True
        i = j
    return float(y.max()), False


def km(Y, C, tau):
    return np.array([km_quantile(Y[:, r], C[:, r], tau)[0] for r in range(Y.shape[1])])


def make_kmx(eps: float):
    def kmx(Y, C, tau):
        out = np.empty(Y.shape[1])
        for r in range(Y.shape[1]):
            v, ok = km_quantile(Y[:, r], C[:, r], tau)
            out[r] = v if ok else v * (1.0 + eps)
        return out
    return kmx


def make_saax(eps: float, k: int = 7):
    def saax(Y, C, tau):
        q = saa(Y, C, tau)
        return np.where(C[-k:].mean(0) > 1.0 - tau, q * (1.0 + eps), q)
    return saax


def registry(eps: float = 0.05):
    return {"PAR": par, "SAA": saa, "TOB": tob, "KM": km, "KMX": make_kmx(eps)}
