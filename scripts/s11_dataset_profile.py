"""Step 11: descriptive profile of the 200-store sample (Fig. 4 of the paper).

Uses the same store sample and latent-demand construction (ground truth A) as the experiments.
Outputs: figures/fig04_dataset_profile.png/.pdf, tables/dataset_profile_series.csv,
tables/dataset_profile_summary.json
"""
import numpy as np
import pandas as pd

from common import load_config, log, dump_json
from censim import ground_truth as gt
from censim.plotting import figure, save, COLOR, INK, MUTED, GRID
import matplotlib.pyplot as plt


def main():
    cfg = load_config()
    out = cfg["_out"]
    path = cfg["_root"] / cfg["data"]["train_parquet"]
    warm = cfg["e2"]["warm"]
    stores = gt.sample_stores(path, cfg["data"]["n_stores"], cfg["data"]["store_seed"])
    df, days = gt.load_stores(path, stores)
    log(f"{len(stores)} stores, {df['key'].nunique():,} series, {len(days)} days")
    panel = gt.build_panel(df, "A")
    D, Y, C, promo = panel["D"], panel["Yreal"], panel["Creal"], panel["activity_flag"]

    mean_d = D.mean(0)
    sd_d = D.std(0, ddof=1)
    cv_d = np.where(mean_d > 0, sd_d / np.maximum(mean_d, 1e-12), np.nan)
    cens = C.mean(0)
    cens_warm = C[:warm].mean(0)
    ser = pd.DataFrame(dict(store=panel["store"], product=panel["product"], mean_latent_demand=mean_d,
                            cv_latent_demand=cv_d, censored_share=cens, censored_share_warmup=cens_warm,
                            promotion_share=promo.mean(0), sales_to_demand=Y.sum(0) / np.maximum(D.sum(0), 1e-12)))
    ser.to_csv(out / "tables" / "dataset_profile_series.csv", index=False)

    q = lambda x, p: float(np.nanpercentile(x, p))
    summ = dict(n_stores=int(len(stores)), n_series=int(D.shape[1]), n_days=int(D.shape[0]), warmup_days=int(warm),
                censored_share_all_days=float(C.mean()), censored_share_warmup=float(C[:warm].mean()),
                censored_share_loop_days=float(C[warm:].mean()),
                sales_over_latent_demand=float(Y.sum() / D.sum()),
                mean_latent_demand_median=q(mean_d, 50), mean_latent_demand_p10=q(mean_d, 10), mean_latent_demand_p90=q(mean_d, 90),
                cv_median=q(cv_d, 50), cv_p10=q(cv_d, 10), cv_p90=q(cv_d, 90),
                series_censored_share_median=q(cens, 50), series_censored_share_p10=q(cens, 10), series_censored_share_p90=q(cens, 90),
                share_series_censored_over_half_days=float((cens > 0.5).mean()),
                promotion_share_median=q(promo.mean(0), 50),
                daily_censored_share_min=float(C.mean(1).min()), daily_censored_share_max=float(C.mean(1).max()))
    dump_json(summ, out / "tables" / "dataset_profile_summary.json")
    for k, v in summ.items():
        log(f"{k}: {v}")

    fig, axs = figure(4.8, 2, 2)
    a, b, c, d = axs.ravel()
    a.hist(np.log10(mean_d[mean_d > 0]), bins=40, color=COLOR["PAR"])
    a.set_xlabel("Mean daily latent demand (log$_{10}$ units)")
    a.set_title("(a) Demand level per series")
    b.hist(np.clip(cv_d, 0, 3), bins=40, color=COLOR["PAR"])
    b.set_xlabel("CV of latent demand")
    b.set_title("(b) Demand variability per series")
    c.hist(cens, bins=np.linspace(0, 1, 26), color=COLOR["SAA"])
    c.axvline(np.median(cens), color=INK, lw=1.2, ls="--")
    c.text(0.97, 0.93, f"median {np.median(cens):.2f}", transform=c.transAxes, ha="right", va="top", color=INK)
    c.set_xlabel("Share of days with a stockout")
    c.set_title("(c) Censoring per series")
    t = np.arange(1, D.shape[0] + 1)
    d.axvspan(0.5, warm + 0.5, color="#ecebe6")
    d.plot(t, C.mean(1), color=COLOR["SAA"], lw=1.8)
    d.text(warm / 2, 0.02, "warm-up", ha="center", color=MUTED)
    d.text(warm + (D.shape[0] - warm) / 2, 0.02, "simulation period", ha="center", color=MUTED)
    d.set_ylim(0, 1); d.set_xlim(0.5, D.shape[0] + 0.5)
    d.set_xlabel("Day of the 90-day sample")
    d.set_ylabel("Share of series censored")
    d.set_title("(d) Censoring over time")
    for ax in (a, b, c):
        ax.set_ylabel("Number of series")
    for ax in axs.ravel():
        ax.grid(axis="y", color=GRID, lw=0.8)
    p = out / "figures" / "fig04_dataset_profile.png"
    save(fig, p)
    log(f"saved {p}")


if __name__ == "__main__":
    main()
