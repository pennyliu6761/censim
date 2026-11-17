"""Step 7: two additional synthetic experiments.
  E1-S  sensitivity to the window W and the refit interval R (one factor at a time around W=56, R=7)
  E1-X  exploration trade-off: KMX and SAAX for several eps, with KM and SAA (eps = 0) as reference
"""
import itertools
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from common import load_config, log
from censim.learners_stat import registry, make_kmx, make_saax, km, saa
from censim.simulate import run_synthetic_cell


def summarise(out, keys):
    rows = []
    for m, v in out.items():
        rec = v["recover_week"]
        rows.append(dict(**keys, learner=m, ratio_mean=v["ratio"].mean(), cost_ratio=v["cost_ratio"].mean(),
                         cost_ratio_p10=np.quantile(v["cost_ratio"], 0.1), cost_ratio_p90=np.quantile(v["cost_ratio"], 0.9),
                         share_recovered=np.mean(~np.isnan(rec)),
                         recover_week_median=np.nanmedian(rec) if np.any(~np.isnan(rec)) else np.nan))
    return rows


def sens_cell(c, x, seed):
    dist, cv, tau, start, W, R = c
    learners = {k: v for k, v in registry(x["kmx_eps"]).items() if k in x["learners"]}
    out, _ = run_synthetic_cell(dist, cv, tau, start, learners, reps=x["reps"], horizon=x["horizon"],
                                window=W, refit=R, last=x["last"], seed=seed)
    return summarise(out, dict(dist=dist, cv=cv, tau=tau, start=start, window=W, refit=R))


def eps_cell(c, x, seed):
    dist, cv, tau, start = c
    learners = {"KM": km, "SAA": saa}
    for e in x["eps"]:
        learners[f"KMX:{e}"] = make_kmx(e)
        learners[f"SAAX:{e}"] = make_saax(e)
    out, _ = run_synthetic_cell(dist, cv, tau, start, learners, reps=x["reps"], horizon=x["horizon"],
                                window=x["window"], refit=x["refit"], last=x["last"], seed=seed)
    return summarise(out, dict(dist=dist, cv=cv, tau=tau, start=start))


def main():
    cfg = load_config()
    x = cfg["e1x"]
    out = cfg["_out"] / "e1"
    s = x["sensitivity"]
    grid = [(W, s["base_refit"]) for W in s["windows"]] + [(s["base_window"], R) for R in s["refits"]]
    grid = list(dict.fromkeys(grid))
    cells = [(d, cv, t, st, W, R) for d, cv, t, st in itertools.product(s["dists"], s["cvs"], s["taus"], s["starts"])
             for W, R in grid]
    # same demand paths for every (W, R) of a setting: seed depends on the setting only
    base = {k: int(v) for k, v in zip(itertools.product(s["dists"], s["cvs"], s["taus"], s["starts"]),
                                      np.random.default_rng(x["seed"]).integers(0, 2**31 - 1, 10_000))}
    log(f"E1-S: {len(cells)} runs of {len(s['learners'])} learners")
    xs = dict(s, reps=x["reps"], horizon=x["horizon"], last=x["last"], kmx_eps=x["kmx_eps"])
    res = Parallel(n_jobs=x["n_jobs"])(delayed(sens_cell)(c, xs, base[c[:4]]) for c in cells)
    pd.DataFrame([r for rr in res for r in rr]).to_csv(out / "e1_sensitivity.csv", index=False)
    e = x["exploration"]
    cells = list(itertools.product(e["dists"], e["cvs"], e["taus"], e["starts"]))
    seeds = np.random.default_rng(x["seed"] + 1).integers(0, 2**31 - 1, len(cells))
    log(f"E1-X: {len(cells)} settings, eps = {e['eps']}")
    xe = dict(e, reps=x["reps"], horizon=x["horizon"], last=x["last"], window=s["base_window"], refit=s["base_refit"])
    res = Parallel(n_jobs=x["n_jobs"])(delayed(eps_cell)(c, xe, int(sd)) for c, sd in zip(cells, seeds))
    pd.DataFrame([r for rr in res for r in rr]).to_csv(out / "e1_exploration.csv", index=False)
    log("E1 extra done")


if __name__ == "__main__":
    main()
