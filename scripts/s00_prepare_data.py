"""Step 0: sample stores, build latent-demand panels (ground truth A and B), hourly profile for Fig. 3."""
from common import load_config, log, dump_json
from censim import ground_truth as gt


def main():
    cfg = load_config()
    out = cfg["_out"] / "data"
    path = cfg["_root"] / cfg["data"]["train_parquet"]
    stores = gt.sample_stores(path, cfg["data"]["n_stores"], cfg["data"]["store_seed"])
    log(f"sampled {len(stores)} stores")
    df, days = gt.load_stores(path, stores)
    log(f"loaded {len(df):,} rows, {df['key'].nunique():,} store-SKU series, {len(days)} days ({days[0]} to {days[-1]})")
    meta = dict(stores=[int(s) for s in stores], days=[str(days[0]), str(days[-1])], n_days=len(days),
                n_series=int(df["key"].nunique()), real_censoring_rate=float((df["stock_hour6_22_cnt"] > 0).mean()))
    for v in cfg["data"]["ground_truth"]:
        panel = gt.build_panel(df, v)
        gt.save_panel(panel, out / f"panel_{v}.npz")
        meta[f"uplift_{v}"] = panel["uplift"]
        log(f"ground truth {v}: latent demand exceeds observed sales by {100 * panel['uplift']:.1f}%")
    prof, run_share = gt.hourly_profile(path, cfg["data"]["hourly_profile_row_groups"])
    prof.to_csv(out / "hourly_profile.csv", index=False)
    meta["stockout_days_single_run_to_close"] = run_share
    dump_json(meta, out / "meta.json")
    log(f"done; {100 * run_share:.1f}% of stockout days are a single run to closing time")


if __name__ == "__main__":
    main()
