"""Step 1: synthetic experiment E1 (RQ1, RQ2) and the Proposition 2 fixed points."""
import itertools
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from common import load_config, log, dump_json
from censim.learners_stat import registry
from censim.simulate import run_synthetic_cell
from censim.theory import par_fixed_point


def cell(c, e, seed):
    dist, cv, tau, start = c
    learners = {k: v for k, v in registry(e["kmx_eps"]).items() if k in e["learners"]}
    out, _ = run_synthetic_cell(dist, cv, tau, start, learners, reps=e["reps"], horizon=e["horizon"],
                                window=e["window"], refit=e["refit"], last=e["last"], seed=seed)
    rows, traj = [], {}
    for m, v in out.items():
        rows.append(dict(dist=dist, cv=cv, tau=tau, start=start, learner=m,
                         ratio_mean=v["ratio"].mean(), ratio_p10=np.quantile(v["ratio"], 0.1),
                         ratio_p90=np.quantile(v["ratio"], 0.9), cost_ratio=v["cost_ratio"].mean(),
                         cost_ratio_p10=np.quantile(v["cost_ratio"], 0.1),
                         cost_ratio_p90=np.quantile(v["cost_ratio"], 0.9)))
        traj[f"{dist}|{cv}|{tau}|{start}|{m}"] = v["traj"].round(4).tolist()
    return rows, traj


def main():
    cfg = load_config()
    e = cfg["e1"]
    out = cfg["_out"] / "e1"
    cells = list(itertools.product(e["dists"], e["cvs"], e["taus"], e["starts"]))
    seeds = np.random.default_rng(e["seed"]).integers(0, 2**31 - 1, len(cells))
    log(f"E1: {len(cells)} cells x {len(e['learners'])} learners x {e['reps']} replications")
    res = Parallel(n_jobs=e["n_jobs"])(delayed(cell)(c, e, int(s)) for c, s in zip(cells, seeds))
    rows = [r for rr, _ in res for r in rr]
    traj = {k: v for _, tt in res for k, v in tt.items()}
    pd.DataFrame(rows).to_csv(out / "e1_summary.csv", index=False)
    dump_json(traj, out / "e1_trajectories.json")
    th = []
    for dist, cv, tau in itertools.product(e["dists"], e["cvs"], e["taus"]):
        th.append(dict(dist=dist, cv=cv, tau=tau, theory_fixed_point=par_fixed_point(dist, 100.0, cv, tau)))
    pd.DataFrame(th).to_csv(out / "e1_theory.csv", index=False)
    log("E1 done")


if __name__ == "__main__":
    main()
