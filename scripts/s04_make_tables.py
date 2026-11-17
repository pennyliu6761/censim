"""Step 4: tables (E2 cost decomposition with tests, Proposition 2 check, E1 cost and order ratios,
failure thresholds, Cox models) and the inputs of the real-data trajectory and robustness figures.
Output files and the paper tables they belong to are listed in README.md."""
import itertools
import numpy as np
import pandas as pd
from common import load_config, log, dump_json
from censim import analysis as an
from censim.newsvendor import cost


def store_sums(x, store):
    _, inv = np.unique(store, return_inverse=True)
    return np.bincount(inv, weights=x)


def e1_tables(cfg, out):
    s = pd.read_csv(cfg["_out"] / "e1" / "e1_summary.csv")
    th = pd.read_csv(cfg["_out"] / "e1" / "e1_theory.csv")
    s.pivot_table(index=["dist", "cv", "tau", "start"], columns="learner", values="cost_ratio").round(3).to_csv(out / "tableA2_e1_cost_ratio.csv")
    s.pivot_table(index=["dist", "cv", "tau", "start"], columns="learner", values="ratio_mean").round(3).to_csv(out / "tableA3_e1_order_ratio.csv")
    par = s[(s["learner"] == "PAR") & (s["start"] == 1.0)][["dist", "cv", "tau", "ratio_mean"]] if 1.0 in cfg["e1"]["starts"] else s[s["learner"] == "PAR"].groupby(["dist", "cv", "tau"], as_index=False)["ratio_mean"].mean()
    a = par.merge(th, on=["dist", "cv", "tau"])
    a["abs_gap"] = (a["ratio_mean"] - a["theory_fixed_point"]).abs()
    a["note"] = np.where(a["tau"] <= 0.5, "Prop. 1: no positive fixed point; simulation still declining at horizon end",
                         "Prop. 2: compare ratio_mean with theory_fixed_point")
    a.loc[a["tau"] <= 0.5, ["theory_fixed_point", "abs_gap"]] = np.nan  # Prop. 1: the map has no positive fixed point
    a.round(4).to_csv(out / "tableA1_prop2_theory_vs_simulation.csv", index=False)
    log(f"Prop. 2 max |theory - simulation| for tau > 0.5: {a[a['tau'] > 0.5]['abs_gap'].max():.3f}")


def e2_tables(cfg, out):
    e = cfg["e2"]
    rows, robust, weekly = [], [], {}
    configs = [(v, t, r) for v, t, r in itertools.product(cfg["data"]["ground_truth"], e["taus"], e["refit"])]
    rb = e["robustness"]
    configs += [(v, t, r) for v, t, r in itertools.product(rb["ground_truth"], rb["taus"], rb["refit"])]
    configs = list(dict.fromkeys(configs))
    for v, tau, R in configs:
        seeds = e["seeds"] if R in e["refit"] else rb["seeds"]
        per = {}  # learner -> list over seeds of store-level arrays
        for seed in seeds:
            p = cfg["_out"] / "e2" / f"e2_{v}_tau{tau}_R{R}_s{seed}.npz"
            if not p.exists():
                continue
            z = np.load(p)
            D, store = z["D"], z["store"]
            for m in e["learners"]:
                for kind in [m, f"FI_{m}"]:
                    Q = z[f"Q_{kind}"]
                    per.setdefault(kind, {"cost": [], "fill": [], "waste": []})
                    per[kind]["cost"].append(store_sums(cost(Q, D, tau).sum(0), store))
                    per[kind]["fill"].append(np.minimum(Q, D).sum() / D.sum())
                    per[kind]["waste"].append(np.maximum(Q - D, 0).sum() / Q.sum())
                if v == "A" and R == e["refit"][0]:
                    n = (Q.shape[0] // 7) * 7
                    wk = z[f"Q_{m}"][:n].reshape(-1, 7, Q.shape[1]).sum((1, 2)) / z[f"Q_FI_{m}"][:n].reshape(-1, 7, Q.shape[1]).sum((1, 2))
                    weekly.setdefault(f"{tau}|{m}", []).append(wk.round(4).tolist())
        if not per:
            continue
        C = {k: np.mean(vv["cost"], axis=0) for k, vv in per.items()}           # average over seeds, per store
        best_fi = min([f"FI_{m}" for m in e["learners"]], key=lambda k: C[k].sum())
        base = C[best_fi]
        for m in e["learners"]:
            cens = (C[m] - C[f"FI_{m}"]) / base
            gap = (C[f"FI_{m}"] - base) / base
            tot = (C[m] - base) / base
            rel = C[m] / C[f"FI_{m}"]
            lo, hi = an.bootstrap_ci(rel)
            rows.append(dict(ground_truth=v, tau=tau, refit=R, learner=m, best_benchmark=best_fi,
                             censoring_loss_pct=100 * cens.mean(), censoring_loss_lo=float(np.round(100 * an.bootstrap_ci(cens)[0], 1)),
                             censoring_loss_hi=float(np.round(100 * an.bootstrap_ci(cens)[1], 1)),
                             model_gap_pct=100 * gap.mean(), total_excess_pct=100 * tot.mean(),
                             cost_vs_same_type=rel.mean(), cost_vs_same_type_lo=lo, cost_vs_same_type_hi=hi,
                             fill_rate=np.mean(per[m]["fill"]), fill_rate_FI=np.mean(per[f"FI_{m}"]["fill"]),
                             waste_rate=np.mean(per[m]["waste"]), waste_rate_FI=np.mean(per[f"FI_{m}"]["waste"]),
                             p_censoring_loss=an.wilcoxon_greater(C[m], C[f"FI_{m}"]), n_stores=len(base)))
            robust.append(dict(setting=f"GT {v}, τ={tau}, R={R}", learner=m, mean=rel.mean(), lo=lo, hi=hi))
    t5 = pd.DataFrame(rows)
    t5["p_holm"] = an.holm(t5["p_censoring_loss"].to_numpy())
    t5.round(4).to_csv(out / "table4_e2_cost_decomposition.csv", index=False)
    pd.DataFrame(robust).to_csv(out / "robustness_input.csv", index=False)
    dump_json(weekly, out / "fig07_weekly_input.json")


def e3_tables(cfg, out):
    p = cfg["_out"] / "e3" / "e3_cox.csv"
    if p.exists():
        c = pd.read_csv(p)
        c.round(3).to_csv(out / "tableA8_cox.csv", index=False)
    S = pd.read_csv(cfg["_out"] / "e3" / "e3_events.csv.gz")
    summ = S.groupby(["tau", "threshold", "learner"]).agg(fail_share=("fail", "mean"), n=("fail", "size")).reset_index()
    low = S[S["start_low"]].groupby(["tau", "threshold", "learner"]).agg(n_low_start=("esc", "size"), escape_share=("esc", "mean")).reset_index()
    summ.merge(low, on=["tau", "threshold", "learner"], how="left").round(3).to_csv(out / "tableA7_failure_thresholds.csv", index=False)


def main():
    cfg = load_config()
    out = cfg["_out"] / "tables"
    e1_tables(cfg, out)
    e2_tables(cfg, out)
    e3_tables(cfg, out)
    log("tables written to " + str(out))


if __name__ == "__main__":
    main()
