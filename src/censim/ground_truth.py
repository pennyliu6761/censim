"""FreshRetailNet-50K loading and latent-demand (ground-truth) construction.

Latent demand = observed daily sales + imputed demand in stockout hours, where the imputed
demand of hour h is the series' mean sales rate in hour h over in-stock hours (additive imputation).
  Version A: impute every stockout hour.
  Version B: do not impute stockout hours before the day's first in-stock hour
             (treated as not-yet-replenished, i.e. no lost demand).
hours_stock_status == 1 means out of stock (checked: its 06-21 sum equals stock_hour6_22_cnt).
"""
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

SCALAR = ["store_id", "product_id", "dt", "sale_amount", "stock_hour6_22_cnt", "discount",
          "holiday_flag", "activity_flag", "precpt"]
COVARIATES = ["discount", "holiday_flag", "activity_flag", "precpt", "dow"]


def sample_stores(parquet_path, n_stores, seed):
    stores = pq.read_table(parquet_path, columns=["store_id"]).column(0).to_numpy()
    uniq = np.unique(stores)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(uniq, size=min(n_stores, len(uniq)), replace=False))


def load_stores(parquet_path, stores):
    tab = ds.dataset(parquet_path).to_table(filter=ds.field("store_id").isin([int(s) for s in stores]))
    df = tab.to_pandas().sort_values(["store_id", "product_id", "dt"]).reset_index(drop=True)
    days = sorted(df["dt"].unique())
    df["t"] = df["dt"].map({d: i for i, d in enumerate(days)})
    df["dow"] = pd.to_datetime(df["dt"]).dt.dayofweek
    df["key"] = df["store_id"].astype(str) + "_" + df["product_id"].astype(str)
    return df, days


def latent_demand(df, version):
    H = np.stack(df["hours_sale"].values)
    S = np.stack(df["hours_stock_status"].values)
    instock = S == 0
    out = np.empty(len(df))
    sales = df["sale_amount"].to_numpy()
    for _, idx in df.groupby("key").indices.items():
        h, s = H[idx], instock[idx]
        n_in = s.sum(0)
        rate = np.where(n_in > 0, (h * s).sum(0) / np.maximum(n_in, 1), 0.0)
        missing = ~s
        if version == "B":
            first = np.where(s.any(1), s.argmax(1), 24)
            missing = missing & (np.arange(24)[None, :] >= first[:, None])
        out[idx] = sales[idx] + (missing * rate[None, :]).sum(1)
    return out


def build_panel(df, version):
    df = df.copy()
    df["D"] = latent_demand(df, version)
    df["C_real"] = (df["stock_hour6_22_cnt"] > 0).astype(float)
    keys = df["key"].unique()

    def piv(col):
        return df.pivot(index="t", columns="key", values=col)[keys].to_numpy(dtype=float)

    panel = dict(D=piv("D"), Yreal=piv("sale_amount"), Creal=piv("C_real"),
                 store=np.array([int(k.split("_")[0]) for k in keys]),
                 product=np.array([int(k.split("_")[1]) for k in keys]))
    for c in COVARIATES:
        panel[c] = piv(c)
    panel["uplift"] = float(panel["D"].sum() / panel["Yreal"].sum() - 1.0)
    return panel


def save_panel(panel, path):
    np.savez_compressed(path, **panel)


def load_panel(path):
    z = np.load(path)
    return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def hourly_profile(parquet_path, n_row_groups=100, seed=0):
    """Share of SKU-days out of stock and share of sales, by hour (Fig. 3), plus the share of
    stockout days whose stockout is a single run to closing time."""
    f = pq.ParquetFile(parquet_path)
    rng = np.random.default_rng(seed)
    groups = rng.choice(f.num_row_groups, size=min(n_row_groups, f.num_row_groups), replace=False)
    parts = [f.read_row_group(int(g), columns=["hours_stock_status", "hours_sale"]).to_pandas() for g in groups]
    d = pd.concat(parts)
    S = np.stack(d["hours_stock_status"].values)
    H = np.stack(d["hours_sale"].values)
    so = S[:, 6:22]
    any_so = so.sum(1) > 0
    runs = [(r.argmax() >= 0) and np.all(r[np.argmax(r):] == 1) for r in so[any_so]]
    return pd.DataFrame({"hour": np.arange(24), "stockout_share": S.mean(0),
                         "sales_share": H.sum(0) / H.sum()}), float(np.mean(runs))
