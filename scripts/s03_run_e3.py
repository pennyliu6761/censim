"""Step 3: survival analysis E3 (RQ4) on the main E2 configuration.
Failure: 7-day order ratio to the same-type benchmark falls below fail_threshold.
Escape (SKUs that start low): the ratio reaches escape_threshold."""
import warnings
import numpy as np
import pandas as pd
from common import load_config, log
from censim import analysis as an
from censim.ground_truth import load_panel

COVS = ["log_mean_demand", "cv_demand", "promo_share", "initial_censoring"]


def covariates(panel, warm):
    D = panel["D"][:warm]
    return dict(log_mean_demand=np.log(D.mean(0) + 0.1),
                cv_demand=D.std(0) / np.maximum(D.mean(0), 1e-9),
                promo_share=panel["activity_flag"].mean(0),
                initial_censoring=panel["Creal"][:warm].mean(0))


def main():
    cfg = load_config()
    e, e3 = cfg["e2"], cfg["e3"]
    v, R = e3["ground_truth"], e3["refit"]
    panel = load_panel(cfg["_out"] / "data" / f"panel_{v}.npz")
    cov = covariates(panel, e["warm"])
    rows = []
    thresholds = sorted(set([e3["fail_threshold"]] + e3["threshold_sensitivity"]))
    for tau in e["taus"]:
        for seed in e["seeds"]:
            z = np.load(cfg["_out"] / "e2" / f"e2_{v}_tau{tau}_R{R}_s{seed}.npz")
            for m in e["learners"]:
                ratio = an.rolling_ratio(z[f"Q_{m}"], z[f"Q_FI_{m}"])
                low = ratio[0] < e3["escape_threshold"]
                esc_d, esc_e = an.first_event(ratio >= e3["escape_threshold"])
                for th in thresholds:
                    f_d, f_e = an.first_event(ratio < th)
                    for k in range(ratio.shape[1]):
                        rows.append(dict(tau=tau, seed=seed, learner=m, threshold=th, sku=k,
                                         fail_dur=f_d[k], fail=f_e[k], start_low=bool(low[k]),
                                         esc_dur=esc_d[k], esc=esc_e[k], **{c: cov[c][k] for c in COVS}))
    S = pd.DataFrame(rows)
    out = cfg["_out"] / "e3"
    S.to_csv(out / "e3_events.csv.gz", index=False)
    main_th = e3["fail_threshold"]
    km_rows, lr_rows, cox_rows = [], [], []
    for tau in e["taus"]:
        d = S[(S["tau"] == tau) & (S["threshold"] == main_th) & S["start_low"]]
        if d.empty:
            continue
        kt = an.kaplan_meier_table(d["esc_dur"].to_numpy(float), d["esc"].to_numpy(int), d["learner"].to_numpy())
        kt["tau"] = tau
        km_rows.append(kt)
        stat, p = an.logrank(d["esc_dur"].to_numpy(float), d["esc"].to_numpy(int), d["learner"].to_numpy())
        # seeds pooled repeat each SKU; the seed-0 test uses each SKU once and is the one to report
        d0 = d[d["seed"] == e["seeds"][0]]
        stat0, p0 = an.logrank(d0["esc_dur"].to_numpy(float), d0["esc"].to_numpy(int), d0["learner"].to_numpy())
        lr_rows.append(dict(tau=tau, chi2_pooled=stat, p_pooled=p, chi2_seed0=stat0, p_seed0=p0,
                            df=d0["learner"].nunique() - 1, n_seed0=len(d0)))
        # Cox models use ONE seed so that each SKU enters once (independent observations);
        # the other seeds are used only for the descriptive shares and Kaplan-Meier curves.
        s0 = e["seeds"][0]
        for m in e["learners"]:
            fail_sub = S[(S["tau"] == tau) & (S["threshold"] == main_th) & (S["learner"] == m) & (S["seed"] == s0)]
            esc_sub = d[(d["learner"] == m) & (d["seed"] == s0)]
            for outcome, dur, ev, sub in [("failure", "fail_dur", "fail", fail_sub), ("escape", "esc_dur", "esc", esc_sub)]:
                n_ev = int(sub[ev].sum())
                if n_ev < 50 or n_ev == len(sub):
                    log(f"Cox skipped for tau={tau} {m} {outcome}: {n_ev} events of {len(sub)}")
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    try:
                        tab, n_ev, n = an.cox(sub, dur, ev, COVS)
                    except Exception as ex:  # report, do not hide
                        log(f"Cox failed for tau={tau} {m} {outcome}: {ex}")
                        continue
                if tab["fit"].iloc[0] != "newton":
                    log(f"Cox for tau={tau} {m} {outcome}: Newton failed, fitted with {tab['fit'].iloc[0]}")
                if tab["HR"].isna().any():
                    log(f"Cox did not converge for tau={tau} {m} {outcome}; not reported")
                    continue
                tab = tab.reset_index().rename(columns={"index": "covariate"})
                tab["tau"], tab["learner"], tab["outcome"], tab["events"], tab["n"] = tau, m, outcome, n_ev, n
                cox_rows.append(tab)
    pd.concat(km_rows).to_csv(out / "e3_km_escape.csv", index=False)
    pd.DataFrame(lr_rows).to_csv(out / "e3_logrank.csv", index=False)
    if cox_rows:
        pd.concat(cox_rows).to_csv(out / "e3_cox.csv", index=False)
    log("E3 done")


if __name__ == "__main__":
    main()
