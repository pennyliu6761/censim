"""Statistics: store-level cost decomposition, bootstrap CIs, Wilcoxon + Holm, survival analysis."""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def bootstrap_ci(x, n=2000, seed=0, stat=np.mean):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = np.array([stat(rng.choice(x, len(x), replace=True)) for _ in range(n)])
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def rolling_ratio(Q, Qref, w=7):
    """7-day rolling ratio of summed orders (T x K) -> (T-w+1 x K)."""
    cs = lambda a: np.vstack([np.zeros((1, a.shape[1])), np.cumsum(a, 0)])
    return (cs(Q)[w:] - cs(Q)[:-w]) / np.maximum(cs(Qref)[w:] - cs(Qref)[:-w], 1e-9)


def first_event(mask, w=7):
    """Index of first True per column (+ w to express in days since loop start); no event -> censored."""
    ev = mask.any(0)
    dur = np.where(ev, mask.argmax(0) + w, mask.shape[0] + w - 1)
    return dur, ev.astype(int)


def kaplan_meier_table(durations, events, groups):
    rows = []
    for g in pd.unique(groups):
        m = groups == g
        sf = SurvfuncRight(durations[m], events[m])
        for t, s in zip(sf.surv_times, sf.surv_prob):
            rows.append(dict(group=g, time=float(t), surv=float(s)))
    return pd.DataFrame(rows)


def logrank(durations, events, groups):
    stat, p = survdiff(durations, events, groups)
    return float(stat), float(p)


def cox(df, duration, event, covariates):
    """Cox PH model on standardized covariates (HR per 1 SD), Efron ties.
    Newton's method can overshoot on its first step when events are rare; if it returns NaN we refit
    with BFGS from zero, which maximises the same partial likelihood.
    Also returns a proportional-hazards check per covariate: the Spearman correlation between the
    Schoenfeld residual and event time (Grambsch-Therneau idea); |rho| near 0 supports PH."""
    d = df[[duration, event] + covariates].dropna()
    X = ((d[covariates] - d[covariates].mean()) / d[covariates].std()).to_numpy(float)
    t, e = d[duration].to_numpy(float), d[event].to_numpy(float)
    mod = PHReg(t, X, status=e, ties="efron")
    r, method = None, None
    # Newton first; if it fails (NaN or error, which depends on the platform's numerics), the same
    # partial likelihood is maximized with other optimizers. The table reports which one was used.
    for meth, kw in [("newton", {}), ("bfgs", dict(maxiter=1000, disp=False)),
                     ("lbfgs", dict(maxiter=1000, disp=False)), ("nm", dict(maxiter=20000, disp=False))]:
        try:
            with np.errstate(all="ignore"):
                cand = mod.fit(method=meth, **kw)
            if np.all(np.isfinite(cand.params)) and np.all(np.isfinite(cand.bse)):
                r, method = cand, meth
                break
        except Exception:
            continue
    if r is None:
        raise RuntimeError("partial likelihood could not be maximized with newton, bfgs, lbfgs or nm")
    ci = np.exp(r.conf_int())
    sr = r.schoenfeld_residuals
    ok = e == 1
    rho, prho = zip(*[stats.spearmanr(sr[ok, j], t[ok]) for j in range(X.shape[1])])
    return pd.DataFrame({"HR": np.exp(r.params), "lo95": ci[:, 0], "hi95": ci[:, 1], "p": r.pvalues,
                         "ph_rho": rho, "ph_p": prho, "fit": method},
                        index=covariates), int(e.sum()), len(d)


def wilcoxon_greater(a, b):
    """One-sided Wilcoxon signed-rank test that a > b (paired by store)."""
    diff = np.asarray(a) - np.asarray(b)
    if np.allclose(diff, 0):
        return 1.0
    return float(stats.wilcoxon(diff, alternative="greater").pvalue)
