"""Step 10: the remaining paper figures and tables. Reads only outputs of steps 0-9.

Figures: E1 cost heatmap, exploration trade-off, cost decomposition, SKU-level order ratios, Cox hazard
ratios, exploration on the real data, robustness, example series. Tables: W and R, exploration step,
store and SKU distributions, robustness, exploration on the real data.
Output files and the paper figures and tables they belong to are listed in README.md."""
import numpy as np
import pandas as pd
from common import load_config, log
from censim import analysis as an
from censim import plotting_ext as px
from censim.newsvendor import cost


def store_sums(v, store):
    return pd.Series(v).groupby(store).sum().to_numpy()


def files(cfg, prefix, v, tau, R, seeds):
    out = []
    for seed in seeds:
        p = cfg["_out"] / "e2" / f"{prefix}_{v}_tau{tau}_R{R}_s{seed}.npz"
        if p.exists():
            out.append((seed, p))
    return out


def rel_to_same_type(z_list, learners, tau):
    """Per-store cost averaged over the given npz files -> {learner: per-store C_m / C_FI_m}."""
    per = {}
    for z in z_list:
        D, store = z["D"], z["store"]
        for m in learners:
            for k in [m, f"FI_{m}"]:
                per.setdefault(k, []).append(store_sums(cost(z[f"Q_{k}"], D, tau).sum(0), store))
    return {m: np.mean(per[m], 0) / np.mean(per[f"FI_{m}"], 0) for m in learners}


def main():
    cfg = load_config()
    o, f, t = cfg["_out"], cfg["_out"] / "figures", cfg["_out"] / "tables"
    e1, e2 = cfg["e1"], cfg["e2"]
    R = e2["refit"][0]

    # E1 cost heatmap (low start); full numbers in tableA2_e1_cost_ratio.csv
    s = pd.read_csv(o / "e1" / "e1_summary.csv")
    px.fig_cost_heatmap(s, e1["learners"], f / "fig05_e1_cost_heatmap.png", starts=(min(e1["starts"]),))

    # exploration trade-off (tableA5); window and refit interval (tableA4)
    p = o / "e1" / "e1_exploration.csv"
    if p.exists():
        ex = pd.read_csv(p)
        px.fig_exploration_single(ex, f / "fig06_exploration_tradeoff.png")
        ex.groupby(["learner", "start", "tau"]).agg(
            cost_ratio_mean=("cost_ratio", "mean"), cost_ratio_min=("cost_ratio", "min"), cost_ratio_max=("cost_ratio", "max"),
            order_ratio_mean=("ratio_mean", "mean"), share_recovered=("share_recovered", "mean"),
            recover_week_median=("recover_week_median", "median")).round(3).reset_index().to_csv(t / "tableA5_e1_exploration.csv", index=False)
    p = o / "e1" / "e1_sensitivity.csv"
    if p.exists():
        sens = pd.read_csv(p)
        a4 = sens.pivot_table(index=["dist", "tau", "start", "window", "refit"], columns="learner",
                              values=["ratio_mean", "cost_ratio"]).round(3)
        a4.columns = [f"{v}_{m}" for v, m in a4.columns]
        a4.to_csv(t / "tableA4_e1_window_refit.csv")

    # cost decomposition (numbers with CIs and tests in table4_e2_cost_decomposition.csv)
    t4 = pd.read_csv(t / "table4_e2_cost_decomposition.csv")
    px.fig_cost_decomposition(t4, f / "fig08_cost_decomposition.png", gt="A", refit=R)

    # SKU-level order ratios (tau = 0.9), store and SKU distributions (tableA6), example series
    a6, sku09, examples = [], {}, []
    for tau in e2["taus"]:
        fl = files(cfg, "e2", "A", tau, R, e2["seeds"])
        if not fl:
            continue
        zs = [np.load(p) for _, p in fl]
        rel = rel_to_same_type(zs, e2["learners"], tau)
        z = zs[0]
        for m in e2["learners"]:
            q, qf = z[f"Q_{m}"][-14:].sum(0), z[f"Q_FI_{m}"][-14:].sum(0)
            v = q[qf > 0] / qf[qf > 0]
            if tau == max(e2["taus"]):
                sku09[m] = v
            r = rel[m]
            a6.append(dict(tau=tau, learner=m, store_p5=np.quantile(r, 0.05), store_median=np.median(r),
                           store_p95=np.quantile(r, 0.95), store_share_above_1=np.mean(r > 1),
                           sku_n=len(v), sku_share_below_090=np.mean(v < 0.9),
                           sku_share_090_110=np.mean((v >= 0.9) & (v <= 1.1)), sku_share_above_110=np.mean(v > 1.1),
                           sku_median=np.median(v)))
        D = z["D"]; meanD = D.mean(0); cand = meanD >= np.median(meanD)
        for m, rule, name in [("MSE", lambda r: r < 0.6, "spiral-down"), ("QNT", lambda r: (r > 0.6) & (r < 0.9), "lock-in")]:
            if not ((m == "MSE" and tau == 0.5) or (m == "QNT" and tau in (0.5, 0.9))):
                continue
            q, qf = z[f"Q_{m}"], z[f"Q_FI_{m}"]
            r = q[-14:].sum(0) / np.maximum(qf[-14:].sum(0), 1e-9)
            idx = np.where(cand & rule(r))[0]
            if len(idx):
                k = idx[np.argsort(meanD[idx])[len(idx) // 2]]       # median demand within the group
                examples.append(dict(title=f"{m}, τ = {tau}: {name}", D=D[:, k],
                                     series=[(f"{m}, censored", px.COLOR[m], "-", q[:, k]),
                                             (f"{m}, full information", px.COLOR[m], ":", qf[:, k])]))
    if a6:
        pd.DataFrame(a6).round(3).to_csv(t / "tableA6_e2_store_sku_distribution.csv", index=False)
    if sku09:
        px.fig_sku_ecdf_single(sku09, f / "fig09_sku_order_ratio.png", max(e2["taus"]))
    if examples:
        px.fig_example_skus(examples, f / "figA1_example_series.png")

    # Cox hazard ratios at the highest critical ratio (all models in tableA8_cox.csv)
    cox = pd.read_csv(t / "tableA8_cox.csv")
    px.fig_cox_forest(cox[cox["tau"] == max(e2["taus"])], f / "fig11_cox_hazard_ratios.png")

    # exploration on the real data (table5_e2_exploration.csv)
    x = cfg.get("e2x")
    rows = []
    if x:
        for tau in x["taus"]:
            fl = dict(files(cfg, "e2", x["ground_truth"][0], tau, x["refit"], x["seeds"]))
            fx = dict(files(cfg, "e2x", x["ground_truth"][0], tau, x["refit"], x["seeds"]))
            seeds = [sd for sd in fl if sd in fx]
            if not seeds:
                continue
            per = {}
            for sd in seeds:
                z, zx = np.load(fl[sd]), np.load(fx[sd])
                D, store = z["D"], z["store"]
                Q = {"QNT": z["Q_QNT"], "MSE": z["Q_MSE"], **{m: zx[f"Q_{m}"] for m in x["learners"]}}
                for b in ["QNT", "MSE"]:
                    per.setdefault(f"FI_{b}", []).append(store_sums(cost(z[f"Q_FI_{b}"], D, tau).sum(0), store))
                for m, q in Q.items():
                    per.setdefault(m, []).append(store_sums(cost(q, D, tau).sum(0), store))
                    per.setdefault(f"fill_{m}", []).append(np.minimum(q, D).sum() / D.sum())
                    per.setdefault(f"waste_{m}", []).append(np.maximum(q - D, 0).sum() / q.sum())
                    b = m.split(":")[0].rstrip("X")
                    per.setdefault(f"ratio_{m}", []).append(q[-14:].sum() / z[f"Q_FI_{b}"][-14:].sum())
            C = {k: np.mean(v, 0) for k, v in per.items() if not k.startswith(("fill_", "waste_", "ratio_"))}
            for m in ["QNT", *[l for l in x["learners"] if l.startswith("QNTX")], "MSE", *[l for l in x["learners"] if l.startswith("MSEX")]]:
                if m not in C:
                    continue
                b = m.split(":")[0].rstrip("X")
                r = C[m] / C[f"FI_{b}"]
                lo, hi = an.bootstrap_ci(r)
                rows.append(dict(tau=tau, learner=m, mean=r.mean(), lo=lo, hi=hi,
                                 order_ratio_last14=np.mean(per[f"ratio_{m}"]),
                                 fill_rate=np.mean(per[f"fill_{m}"]), waste_rate=np.mean(per[f"waste_{m}"]),
                                 cost_change_vs_no_exploration_pct=100 * (C[m].sum() / C[b].sum() - 1),
                                 p_lower_than_no_exploration=an.wilcoxon_greater(C[b], C[m]) if m != b else np.nan,
                                 n_stores=len(r), n_seeds=len(seeds)))
    if rows:
        df = pd.DataFrame(rows)
        k = df["p_lower_than_no_exploration"].notna()
        df.loc[k, "p_holm"] = an.holm(df.loc[k, "p_lower_than_no_exploration"].to_numpy())
        df.round(4).to_csv(t / "table5_e2_exploration.csv", index=False)
        px.fig_real_exploration_single(df, f / "fig12_real_exploration.png")

    # robustness: ground truth B, daily refit, hyperparameters (tableA9_robustness.csv)
    h = cfg.get("e2h", {})
    by_tau, a9 = {}, []
    for tau in sorted(set(e2["taus"]) & set(h.get("taus", e2["taus"]))):
        def add(label, rr, source):
            for m in e2["learners"]:
                r = rr[m]; lo, hi = an.bootstrap_ci(r)
                by_tau.setdefault(tau, []).append((label, m, r.mean(), lo, hi))
                a9.append(dict(tau=tau, setting=label, learner=m, mean=r.mean(), lo=lo, hi=hi, source=source))
        for v, lab in [("A", "Base (version A, R = 7)"), ("B", "Ground truth version B")]:
            fl = files(cfg, "e2", v, tau, R, e2["seeds"])
            if fl:
                add(lab, rel_to_same_type([np.load(p) for _, p in fl], e2["learners"], tau), f"{len(fl)} seeds")
        rb = e2["robustness"]
        for Rr in rb["refit"]:
            fl = files(cfg, "e2", rb["ground_truth"][0], tau, Rr, rb["seeds"])
            if fl and tau in rb["taus"]:
                add(f"Daily refit (R = {Rr})", rel_to_same_type([np.load(p) for _, p in fl], e2["learners"], tau), "seed 0")
        names = {"trees": "Trees", "depth": "Max depth", "eta": "Learning rate"}
        for name in h.get("variants", {}):
            fl = files(cfg, f"e2h_{name}", h["ground_truth"][0], tau, h["refit"], h["seeds"])
            if fl:
                key, val = name.split("_", 1)
                add(f"{names.get(key, key)} {val}", rel_to_same_type([np.load(p) for _, p in fl], e2["learners"], tau), "seed 0")
    if by_tau:
        pd.DataFrame(a9).round(4).to_csv(t / "tableA9_robustness.csv", index=False)
        px.fig_robustness_two(by_tau, f / "fig13_robustness.png")
    log("paper figures and tables written")


if __name__ == "__main__":
    main()
