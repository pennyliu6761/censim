"""Step 2: real-data closed loop E2 (RQ3). Each job = (ground truth, tau, refit, seed) and runs every
learner twice: under its own censored feedback and as its same-type full-information benchmark."""
import itertools
import numpy as np
from joblib import Parallel, delayed
from common import load_config, log
from censim.ground_truth import load_panel
from censim.learners_ml import make_fit_predict
from censim.simulate import run_real_closed_loop


def job(cfg, v, tau, refit, seed):
    e = cfg["e2"]
    path = cfg["_out"] / "e2" / f"e2_{v}_tau{tau}_R{refit}_s{seed}.npz"
    if path.exists():
        return str(path), "cached"
    panel = load_panel(cfg["_out"] / "data" / f"panel_{v}.npz")
    params = dict(e["xgb"])
    res = {}
    for m in e["learners"]:
        fp = make_fit_predict(m, panel, params)
        res[f"Q_{m}"] = run_real_closed_loop(panel, fp, tau, refit, seed, warm=e["warm"])[e["warm"]:]
        res[f"Q_FI_{m}"] = run_real_closed_loop(panel, fp, tau, refit, seed, warm=e["warm"],
                                                full_information=True)[e["warm"]:]
    np.savez_compressed(path, D=panel["D"][e["warm"]:], store=panel["store"], **res)
    return str(path), "done"


def main():
    cfg = load_config()
    e = cfg["e2"]
    jobs = [(v, t, r, s) for v, t, r, s in itertools.product(cfg["data"]["ground_truth"], e["taus"], e["refit"], e["seeds"])]
    rb = e["robustness"]
    jobs += [(v, t, r, s) for v, t, r, s in itertools.product(rb["ground_truth"], rb["taus"], rb["refit"], rb["seeds"])]
    jobs = list(dict.fromkeys(jobs))
    log(f"E2: {len(jobs)} jobs, {len(e['learners'])} learners x 2 runs each")
    res = Parallel(n_jobs=e["n_jobs"], verbose=10)(delayed(job)(cfg, *j) for j in jobs)
    for p, s in res:
        log(f"{s}: {p}")
    log("E2 done")


if __name__ == "__main__":
    main()
