"""Figures for the paper. One consistent style; color follows the learner, never its rank.

Every figure is drawn at its printed size (TEXT_W inches wide, the text width of an A4 page with
1-inch margins) and saved without cropping, so that all text appears at FS = 10 pt in the manuscript
when the figure is inserted at width TEXT_W."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TEXT_W = 6.2          # printed width of every figure (inches); insert at this width in Word
FS = 10               # font size of all text in all figures (pt)

COLOR = {"PAR": "#2a78d6", "MSE": "#2a78d6", "SAA": "#eb6834", "QNT": "#eb6834",
         "TOB": "#1baf7a", "AFT": "#1baf7a", "KM": "#c98a00", "KMX": "#7a4fc4"}
LABEL = {"PAR": "PAR (parametric)", "SAA": "SAA (sample quantile)", "TOB": "TOB (Tobit)",
         "KM": "KM (Kaplan–Meier)", "KMX": "KMX (KM + exploration)",
         "MSE": "MSE (squared error)", "QNT": "QNT (quantile loss)", "AFT": "AFT (censoring-aware)"}
INK, MUTED, GRID, REF = "#1a1a19", "#6b6a65", "#e6e5e0", "#b9b8b0"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
        "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS, "legend.title_fontsize": FS,
        "figure.titlesize": FS, "axes.titlelocation": "left", "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8985",
        "axes.labelcolor": "#3d3d3a", "xtick.color": MUTED, "ytick.color": MUTED,
        "legend.frameon": False, "savefig.dpi": 300, "mathtext.fontset": "dejavusans"})


def figure(height, nrows=1, ncols=1, **kw):
    """New figure at the printed width, with constrained layout so that nothing is clipped."""
    style()
    return plt.subplots(nrows, ncols, figsize=(TEXT_W, height), layout="constrained", **kw)


def save(fig, path):
    """Save at exactly the printed size (no bbox cropping), as PNG and PDF."""
    fig.savefig(path, dpi=300)
    fig.savefig(str(path).replace(".png", ".pdf"))
    plt.close(fig)


def legend_top(fig, handles, labels, ncol, **kw):
    fig.legend(handles, labels, loc="outside upper center", ncol=ncol, **kw)


def fig_hourly_profile(prof, path):
    """Two stacked panels sharing the hour axis (no dual y-axis)."""
    fig, (a1, a2) = figure(4.0, 2, 1, sharex=True)
    a1.bar(prof["hour"], prof["sales_share"], color=COLOR["PAR"], width=0.8)
    a1.set_ylabel("Share of daily sales")
    a1.set_title("(a) Sales by hour")
    a2.plot(prof["hour"], prof["stockout_share"], color=COLOR["SAA"], lw=2, marker="o", ms=3)
    a2.set_ylabel("Share out of stock")
    a2.set_title("(b) SKU-days out of stock by hour")
    a2.set_xlabel("Hour of day"); a2.set_xticks(range(0, 24, 2))
    for a in (a1, a2):
        a.grid(axis="y", color=GRID, lw=0.8)
    save(fig, path)


def fig_synthetic_trajectories(traj, learners, taus, path, start=0.7, dist="normal", cv=0.4):
    fig, axs = figure(3.4, 1, len(taus), sharey=True, squeeze=False)
    for ax, tau in zip(axs[0], taus):
        for m in learners:
            v = np.asarray(traj[f"{dist}|{cv}|{tau}|{start}|{m}"])
            ax.plot(np.arange(len(v)), v, color=COLOR[m], lw=1.8, ls="--" if m == "KM" else "-", label=LABEL[m])
        ax.axhline(1, color=REF, lw=1)
        ax.set_title(f"τ = {tau}")
        ax.set_xlabel("Week"); ax.grid(axis="y", color=GRID, lw=0.8)
    axs[0, 0].set_ylabel("Order / optimal order")
    h, l = axs[0, 0].get_legend_handles_labels()
    legend_top(fig, h, l, 3, columnspacing=1.0, handlelength=1.5)   # five long labels: tighter spacing keeps them inside 6.2 in
    save(fig, path)


def fig_real_trajectories(weekly, taus, path):
    """weekly[(tau, learner)] -> array (seeds x weeks) of order ratio to the same-type benchmark."""
    fig, axs = figure(3.2, 1, len(taus), sharey=True, squeeze=False)
    for ax, tau in zip(axs[0], taus):
        for m in ["MSE", "QNT", "AFT"]:
            a = np.asarray(weekly[(tau, m)])
            w = np.arange(1, a.shape[1] + 1)
            ax.plot(w, a.mean(0), color=COLOR[m], lw=1.8, marker="o", ms=4, label=LABEL[m])
            if a.shape[0] > 1:
                ax.fill_between(w, a.min(0), a.max(0), color=COLOR[m], alpha=0.15, lw=0)
        ax.axhline(1, color=REF, lw=1)
        ax.set_title(f"τ = {tau}")
        ax.set_xlabel("Week of closed loop"); ax.grid(axis="y", color=GRID, lw=0.8)
    axs[0, 0].set_ylabel("Order / same-type\nfull-information order")
    h, l = axs[0, 0].get_legend_handles_labels()
    legend_top(fig, h, l, 3)
    save(fig, path)


def fig_escape_curves(km_table, path, title_groups=None):
    fig, ax = figure(3.3)
    for g, d in km_table.groupby("group"):
        t = np.concatenate([[0], d["time"].to_numpy()])
        s = np.concatenate([[1], d["surv"].to_numpy()])
        ax.step(t, s, where="post", color=COLOR.get(g, MUTED), lw=1.8, label=LABEL.get(g, g))
    ax.set_xlabel("Days since start of closed loop")
    ax.set_ylabel("Share of low-start SKUs\nnot yet recovered")
    ax.set_ylim(0, 1.02); ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(loc="upper right")
    save(fig, path)
