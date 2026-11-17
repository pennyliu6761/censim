"""Step 9: hyperparameter sensitivity of the real-data experiment. One hyperparameter of the gradient
boosting learners is changed at a time; every learner is run on the censored path and as its
same-type full-information benchmark, exactly as in step 2. The base setting is step 2 with the same seed."""
import itertools
import numpy as np
from joblib import Parallel, delayed
from common import load_config, log
from censim.ground_truth import load_panel
from censim.learners_ml import make_fit_predict
from censim.simulate import run_real_closed_loop


def job(cfg, name, override, v, tau, refit, seed):
    e = cfg["e2"]
    path = cfg["_out"] / "e2" / f"e2h_{name}_{v}_tau{tau}_R{refit}_s{seed}.npz"
    if path.exists():
        return str(path), "cached"
    panel = load_panel(cfg["_out"] / "data" / f"panel_{v}.npz")
    params = dict(e["xgb"], **override)
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
    h = cfg["e2h"]
    jobs = [(n, o, v, t, h["refit"], s) for (n, o), v, t, s in
            itertools.product(h["variants"].items(), h["ground_truth"], h["taus"], h["seeds"])]
    log(f"E2-H: {len(jobs)} jobs ({len(h['variants'])} variants), {len(cfg['e2']['learners'])} learners x 2 runs each")
    res = Parallel(n_jobs=cfg["e2"]["n_jobs"], verbose=10)(delayed(job)(cfg, *j) for j in jobs)
    for p, s in res:
        log(f"{s}: {p}")
    log("E2 hyperparameter sensitivity done")


if __name__ == "__main__":
    main()
