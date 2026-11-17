"""Section 5 figures. Same style and colors as plotting.py: every figure is TEXT_W inches wide and all
text is FS = 10 pt at that size; color follows the learner, never its rank; one y-axis per panel."""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator
from .plotting import COLOR, LABEL, INK, MUTED, GRID, REF, TEXT_W, figure, save, legend_top

SEQ = LinearSegmentedColormap.from_list("seq", ["#f3f7fd", "#9cc3ee", "#2a78d6", "#123e7a"])
TAU_MARK = {0.5: "o", 0.75: "s", 0.9: "^"}
EPS_SHADE = {0.02: "#f6c09c", 0.05: "#ef8f55", 0.1: "#c9531f", 0.2: "#8a3412"}
BASE_GREY = "#6b6a65"


def _unique_handles(axs):
    hs, ls_ = [], []
    for ax in np.atleast_1d(axs).ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in ls_:
                hs.append(h); ls_.append(l)
    return hs, ls_


def fig_cost_heatmap(s, learners, path, starts=(0.7,)):
    """E1: cost ratio for every demand setting (rows) and learner (columns)."""
    starts = [st for st in starts if st in set(s["start"])]
    rows = s[["dist", "cv", "tau"]].drop_duplicates().sort_values(["dist", "cv", "tau"], ascending=[False, True, True])
    fig, axs = figure(0.27 * len(rows) + 1.0, 1, len(starts), sharey=True, squeeze=False)
    axs = axs[0]
    vmax = max(4.0, float(s["cost_ratio"].max()))
    norm = LogNorm(vmin=1.0, vmax=vmax)
    for ax, st in zip(axs, starts):
        M = np.array([[s[(s.dist == r.dist) & (s.cv == r.cv) & (s.tau == r.tau) & (s.start == st) & (s.learner == m)]["cost_ratio"].mean()
                       for m in learners] for r in rows.itertuples()])
        im = ax.imshow(np.clip(M, 1.0, None), cmap=SEQ, norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if norm(max(v, 1.0)) > 0.55 else INK)
        ax.set_xticks(range(len(learners))); ax.set_xticklabels(learners)
        ax.xaxis.tick_top()
        if len(starts) > 1:
            ax.set_title(f"Starting order {st} q*")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axs[0].set_yticks(range(len(rows)))
    axs[0].set_yticklabels([f"{'Normal' if r.dist == 'normal' else 'Lognormal'}, CV {r.cv}, τ = {r.tau}"
                            for r in rows.itertuples()])
    cb = fig.colorbar(im, ax=axs, fraction=0.05, pad=0.02, aspect=30)
    ticks = [v for v in (1, 1.25, 1.5, 2, 3, 4, 6) if v <= vmax]
    cb.set_ticks(ticks); cb.set_ticklabels([f"{v:g}" for v in ticks]); cb.minorticks_off()
    cb.set_label("Cost ratio (log scale)")
    save(fig, path)


def fig_exploration_single(ex, path, ytop=1.15):
    """Mean cost ratio over all settings against eps.
    Color = learner family (KMX purple, SAAX orange); line style and marker = starting order
    (0.7 q*: solid line, filled circles; q*: dashed line, open squares). Two separate legends.
    The y-axis is cut at ytop; the eps = 0 points from the low start lie above it and are marked."""
    fig, ax = figure(3.6)
    fams = [("KMX", "KM", COLOR["KMX"]), ("SAAX", "SAA", COLOR["SAA"])]
    starts = [(0.7, "-", "o", True), (1.0, "--", "s", False)]
    clipped, xs = [], set()
    for fam, base, col in fams:
        for st, ls, mk, filled in starts:
            d = ex[ex["start"] == st]
            if d.empty:
                continue
            labs = [base] + sorted({l for l in d["learner"] if l.startswith(fam + ":")}, key=lambda z: float(z.split(":")[1]))
            x = np.array([0.0 if l == base else float(l.split(":")[1]) for l in labs])
            y = np.array([d[d["learner"] == l]["cost_ratio"].mean() for l in labs])
            if np.all(np.isnan(y)):
                continue
            xs.update(x.tolist())
            over = y > ytop
            ax.plot(x[~over], y[~over], color=col, ls=ls, lw=1.8, marker=mk, ms=6,
                    mfc=col if filled else "white", mec=col, mew=1.5)
            if over.any():
                ax.plot(x[over], np.full(over.sum(), ytop), ls="none", marker="^", ms=8, color=col, clip_on=False)
                clipped += [f"{base} {v:.2f}" for v in y[over]]
    if clipped:
        ax.annotate("▲ off scale from 0.7 q*: " + ", ".join(clipped), (0.0, ytop), xytext=(14, -2),
                    textcoords="offset points", color=MUTED, va="top")
    ax.axhline(1, color=REF, lw=1)
    ax.set_ylim(0.99, ytop)
    ax.set_xticks(sorted(xs)); ax.set_xticklabels([f"{v:g}" for v in sorted(xs)])
    ax.set_xlabel("Exploration step ε (ε = 0: KM and SAA without exploration)")
    n_set = len(ex[["dist", "cv", "tau"]].drop_duplicates())
    ax.set_ylabel(f"Cost / cost of ordering q*\n(mean of {n_set} demand settings)")
    ax.grid(color=GRID, lw=0.8)
    h1 = [Line2D([], [], color=c, lw=2.5) for _, _, c in fams]
    l1 = ["KMX (KM + exploration)", "SAAX (SAA + exploration)"]
    h2 = [Line2D([], [], color=MUTED, ls=ls, lw=1.8, marker=mk, ms=6, mfc=MUTED if f else "white", mec=MUTED, mew=1.5)
          for _, ls, mk, f in starts]
    l2 = ["0.7 q*", "q* (1.0 q*)"]
    fig.legend(h1, l1, title="Learner", loc="outside upper left", ncol=1, alignment="left")
    fig.legend(h2, l2, title="Starting order", loc="outside upper right", ncol=1, alignment="left",
               handlelength=3)
    save(fig, path)


def fig_cost_decomposition(t5, path, gt="A", refit=7):
    d = t5[(t5["ground_truth"] == gt) & (t5["refit"] == refit)]
    taus = sorted(d["tau"].unique()); learners = ["MSE", "QNT", "AFT"]
    fig, ax = figure(0.3 * len(taus) * len(learners) + 1.3)
    y, ticks, labels = 0, [], []
    for tau in taus:
        for m in learners:
            r = d[(d["tau"] == tau) & (d["learner"] == m)].iloc[0]
            c, g = r["censoring_loss_pct"], r["model_gap_pct"]
            ax.barh(y, c, color=COLOR[m], height=0.7, edgecolor="white", lw=1)
            ax.barh(y, g, left=max(c, 0), color=COLOR[m], alpha=0.35, hatch="////", height=0.7, edgecolor="white", lw=1)
            ax.text(max(c, 0) + g + 1.5, y, f"{c + g:.0f}%", va="center", color=INK)
            ticks.append(y); labels.append(f"τ = {tau}, {m}"); y += 1
        y += 0.6
    ax.axvline(0, color=REF, lw=1)
    ax.set_yticks(ticks); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlabel("Excess cost over the best full-information learner (%)")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.08)
    ax.grid(axis="x", color=GRID, lw=0.8)
    h = [Patch(color=BASE_GREY, label="Censoring loss"),
         Patch(facecolor=BASE_GREY, alpha=0.35, hatch="////", label="Model gap")]
    ax.legend(handles=h, loc="lower right")
    save(fig, path)


def fig_sku_ecdf_single(v, path, tau):
    """v[learner] -> SKU-level order ratio over the last 14 days, one critical ratio."""
    fig, ax = figure(3.4)
    for m in ["MSE", "QNT", "AFT"]:
        if m in v:
            x = np.sort(np.clip(v[m], 0, 2))
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=COLOR[m], lw=1.8, label=LABEL[m])
    ax.axvline(1, color=REF, lw=1); ax.axvline(0.9, color=REF, lw=1, ls=":")
    ax.set_xlim(0, 2); ax.set_ylim(0, 1)
    ax.set_xlabel(f"SKU order / same-type full-information order (last 14 days, τ = {tau})")
    ax.set_ylabel("Share of SKUs (cumulative)")
    ax.grid(color=GRID, lw=0.8); ax.legend(loc="lower right")
    save(fig, path)


def fig_example_skus(examples, path):
    """examples: list of dict(title, D, series=[(label, color, linestyle, values)])."""
    fig, axs = figure(3.3, 1, len(examples), squeeze=False)
    axs = axs[0]
    for ax, ex in zip(axs, examples):
        t = np.arange(1, len(ex["D"]) + 1)
        ax.bar(t, ex["D"], color="#dcdbd4", width=0.9, label="Latent demand")
        for lab, col, ls, v in ex["series"]:
            ax.plot(t, v, color=col, ls=ls, lw=1.6, label=lab)
        ax.set_title(ex["title"].replace(": ", ":\n", 1))
        ax.set_xlabel("Day"); ax.grid(axis="y", color=GRID, lw=0.8)
    axs[0].set_ylabel("Units per day")
    hs, ls_ = _unique_handles(axs)
    legend_top(fig, hs, ls_, 3)
    save(fig, path)


def fig_cox_forest(cox, path):
    covs = ["initial_censoring", "cv_demand", "promo_share", "log_mean_demand"]
    names = {"initial_censoring": "Initial\ncensoring", "cv_demand": "Demand CV",
             "promo_share": "Promotion\nshare", "log_mean_demand": "Log mean\ndemand"}
    outcomes = [o for o in ["failure", "escape"] if o in set(cox["outcome"])]
    fig, axs = figure(3.6, 1, len(outcomes), sharey=True, squeeze=False)
    axs = axs[0]
    for ax, oc in zip(axs, outcomes):
        d = cox[cox["outcome"] == oc]
        models = d[["learner", "tau"]].drop_duplicates().sort_values(["learner", "tau"]).values.tolist()
        k = len(models)
        for i, cv in enumerate(covs):
            for j, (m, tau) in enumerate(models):
                r = d[(d["covariate"] == cv) & (d["learner"] == m) & (d["tau"] == tau)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = i + (j - (k - 1) / 2) * 0.8 / max(k, 1)
                ax.errorbar(r["HR"], y, xerr=[[r["HR"] - r["lo95"]], [r["hi95"] - r["HR"]]], fmt=TAU_MARK.get(tau, "o"),
                            color=COLOR[m], ms=5, capsize=0, lw=1.2, label=LABEL[m] if i == 0 else None)
        ax.axvline(1, color=REF, lw=1)
        ax.set_xscale("log")
        lo_, hi_ = ax.get_xlim()
        tk = [v for v in (0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5) if lo_ <= v <= hi_]
        ax.xaxis.set_major_locator(FixedLocator(tk)); ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([f"{v:g}" for v in tk])
        ax.set_title("(a) Time to failure" if oc == "failure" else "(b) Time to recovery")
        ax.set_xlabel("Hazard ratio per SD\n(log scale, 95% CI)")
        ax.grid(axis="x", color=GRID, lw=0.8)
    axs[0].set_yticks(range(len(covs))); axs[0].set_yticklabels([names[c] for c in covs])
    axs[0].invert_yaxis()                 # y is shared, so invert once
    hs, ls_ = _unique_handles(axs)
    legend_top(fig, hs, ls_, len(ls_))
    save(fig, path)


def fig_real_exploration_single(df, path):
    """df: tau, learner, mean, lo, hi. One panel; rows grouped by tau."""
    def col(m):
        if m.startswith("QNTX:"):
            return EPS_SHADE.get(float(m.split(":")[1]), COLOR["QNT"])
        if m.startswith("MSEX:"):
            return COLOR["MSE"]
        return BASE_GREY

    def lab(m):
        return f"{m.split(':')[0]}, ε = {m.split(':')[1]}" if ":" in m else f"{m}, no exploration"
    taus = sorted(df["tau"].unique())
    rows, labels, seps = [], [], []
    for tau in taus:
        d = df[df["tau"] == tau]
        ms = ["QNT"] + sorted([m for m in d["learner"] if m.startswith("QNTX:")], key=lambda z: float(z.split(":")[1])) \
            + ["MSE"] + sorted([m for m in d["learner"] if m.startswith("MSEX:")])
        for m in ms:
            if m in set(d["learner"]):
                r = d[d["learner"] == m].iloc[0]
                rows.append((col(m), r["mean"], r["lo"], r["hi"])); labels.append(f"τ = {tau}, {lab(m)}")
        seps.append(len(rows) - 0.5)
    fig, ax = figure(0.24 * len(rows) + 1.0)
    for k, (c, m, lo, hi) in enumerate(rows):
        ax.errorbar(m, k, xerr=[[m - lo], [hi - m]], fmt="o", color=c, ms=5, capsize=2, lw=1.2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels)
    ax.invert_yaxis(); ax.axvline(1, color=REF, lw=1); ax.grid(axis="x", color=GRID, lw=0.8)
    for s_ in seps[:-1]:
        ax.axhline(s_, color=GRID, lw=1.2)
    ax.set_xlabel("Cost / same-model full-information cost\n(95% CI over stores)")
    save(fig, path)


def fig_robustness_two(rows_by_tau, path):
    """rows_by_tau[tau] -> list of (setting_label, learner, mean, lo, hi). Two panels compared directly."""
    taus = sorted(rows_by_tau)
    seen = list(dict.fromkeys([r[0] for t in taus for r in rows_by_tau[t]]))
    first = [x for x in seen if x.startswith(("Base", "Ground truth", "Daily refit"))]
    settings = sorted(first, key=lambda z: ("Base" not in z, "Ground" not in z)) + [x for x in seen if x not in first]
    learners = ["MSE", "QNT", "AFT"]
    fig, axs = figure(0.32 * len(settings) + 1.4, 1, len(taus), sharey=True, squeeze=False)
    for ax, tau in zip(axs[0], taus):
        for r in rows_by_tau[tau]:
            i = settings.index(r[0]); j = learners.index(r[1])
            y = i + (j - 1) * 0.22
            ax.errorbar(r[2], y, xerr=[[r[2] - r[3]], [r[4] - r[2]]], fmt="o", color=COLOR[r[1]], ms=5, capsize=0, lw=1.2,
                        label=LABEL[r[1]] if i == 0 else None)
        ax.axvline(1, color=REF, lw=1); ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_title(f"τ = {tau}")
        ax.set_xlabel("Cost / same-type\nfull-information cost")
    axs[0, 0].set_yticks(range(len(settings))); axs[0, 0].set_yticklabels(settings)
    axs[0, 0].invert_yaxis()
    hs, ls_ = _unique_handles(axs)
    legend_top(fig, hs, ls_, 3)
    save(fig, path)
