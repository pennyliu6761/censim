"""Step 8: exploration on the real data. The quantile learner (and, for comparison, the squared-error
learner) raises its order by (1 + eps) for a series whose recent stockout share exceeds 1 - tau.
Only the censored path is run: on the full-information path no stockouts are recorded, so the
benchmark is identical to FI_QNT / FI_MSE from step 2 (same seed), which is reused."""
import itertools
import numpy as np
from joblib import Parallel, delayed
from common import load_config, log
from censim.ground_truth import load_panel
from censim.learners_ml import make_fit_predict
from censim.simulate import run_real_closed_loop


def job(cfg, v, tau, refit, seed):
    e, x = cfg["e2"], cfg["e2x"]
    path = cfg["_out"] / "e2" / f"e2x_{v}_tau{tau}_R{refit}_s{seed}.npz"
    if path.exists():
        return str(path), "cached"
    panel = load_panel(cfg["_out"] / "data" / f"panel_{v}.npz")
    res = {}
    for m in x["learners"]:
        fp = make_fit_predict(m, panel, dict(e["xgb"]))
        res[f"Q_{m}"] = run_real_closed_loop(panel, fp, tau, refit, seed, warm=e["warm"])[e["warm"]:]
    np.savez_compressed(path, **res)
    return str(path), "done"


def main():
    cfg = load_config()
    x = cfg["e2x"]
    jobs = list(itertools.product(x["ground_truth"], x["taus"], [x["refit"]], x["seeds"]))
    log(f"E2-X: {len(jobs)} jobs, learners {x['learners']}")
    res = Parallel(n_jobs=cfg["e2"]["n_jobs"], verbose=10)(delayed(job)(cfg, *j) for j in jobs)
    for p, s in res:
        log(f"{s}: {p}")
    log("E2 exploration done")


if __name__ == "__main__":
    main()
