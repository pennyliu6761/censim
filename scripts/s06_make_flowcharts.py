"""Step 6: conceptual figures. Fig. 1 (closed-loop simulator) and Fig. 2 (latent-demand construction).
They contain no results, so they can be drawn before or after the experiments."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from common import load_config, log

INK, MUTED, EDGE = "#1a1a19", "#6b6a65", "#8a8985"
BLUE, GREEN, TINT = "#2a78d6", "#1baf7a", "#f3f2ee"
PRINT_W = 6.2      # printed width of both figures (inches); insert them in the manuscript at this width
UNITS_W = 66       # drawing units across the width (box coordinates below are in these units)
U = PRINT_W / UNITS_W   # inches per drawing unit
FS = 10            # every text in the figures is 10 pt at the printed width


def box(ax, x, y, w, h, lines, edge=EDGE, fill="white", size=FS):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=fill, ec=edge, lw=1.3))
    n, gap = len(lines), 2.4
    y0 = y + h / 2 + gap * (n - 1) / 2
    for i, s in enumerate(lines):
        ax.text(x + w / 2, y0 - i * gap, s, ha="center", va="center", fontsize=size,
                color=INK if i == 0 else "#45443f", fontweight="bold" if i == 0 else "normal")


def arrow(ax, x0, y0, x1, y1, color=EDGE, rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, shrinkA=0, shrinkB=0,
                                connectionstyle=f"arc3,rad={rad}", mutation_scale=13))


def note(ax, x, y, s, ha="center", va="center", color=MUTED, size=FS, **kw):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color, **kw)


def canvas(units_h, ymin=0.0):
    """Figure exactly PRINT_W wide; the axes span UNITS_W drawing units across and units_h units up."""
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(PRINT_W, units_h * U))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, UNITS_W); ax.set_ylim(ymin, ymin + units_h); ax.axis("off")
    return fig, ax


def save(fig, path):
    fig.savefig(path, dpi=300)
    fig.savefig(str(path).replace(".png", ".pdf"))
    plt.close(fig)


def fig1_simulator(path):
    fig, ax = canvas(52, ymin=-5.0)
    h = 11
    rows = [(30, BLUE, "Censored path", ["Warm-up", "real sales and", "stockout flags"],
             ["Record", "sales = min(D, q)", "and stockout flag"]),
            (2, GREEN, "Full information", ["Warm-up", "latent demand"], ["Record", "demand D"])]
    for y, col, lab, hist, rec in rows:
        note(ax, 1.6, y + h / 2, lab, color=col, rotation=90, fontweight="bold")
        box(ax, 3.5, y, 12, h, hist, edge=col)
        box(ax, 18.5, y, 13.5, h, ["Refit learner", "every R days on", "the last W days"], edge=col)
        box(ax, 34.5, y, 12.5, h, ["Set order", "q for each day"], edge=col)
        box(ax, 49.5, y, 16, h, rec, edge=col, fill=TINT)
        arrow(ax, 15.5, y + h / 2, 18.5, y + h / 2)
        arrow(ax, 32, y + h / 2, 34.5, y + h / 2)
        arrow(ax, 47, y + h / 2, 49.5, y + h / 2)
    arrow(ax, 57.5, 41, 25.5, 41, rad=0.22)
    note(ax, 41.5, 45.6, "records feed the next refit")
    arrow(ax, 57.5, 2, 25.5, 2, rad=-0.22)
    note(ax, 41.5, -3.6, "records feed the next refit")
    box(ax, 49.5, 16.5, 16, 10, ["Demand D", "same draws on", "both paths"])
    arrow(ax, 57.5, 26.5, 57.5, 30)
    arrow(ax, 57.5, 16.5, 57.5, 13)
    box(ax, 18.5, 16.5, 28.5, 10, ["Score every day", "cost of q given D, and order", "relative to full information"], fill=TINT)
    arrow(ax, 49.5, 21.5, 47, 21.5)
    save(fig, path)


def fig2_ground_truth(path):
    fig, ax = canvas(44, ymin=-0.5)
    h = 11
    box(ax, 0.5, 26, 14, h, ["Hourly records", "sales and stock", "flag by hour"])
    box(ax, 17.5, 26, 14.5, h, ["In-stock rate", "mean hourly sales", "on in-stock days"])
    arrow(ax, 14.5, 31.5, 17.5, 31.5)
    box(ax, 34.5, 32, 14.5, 11, ["Version A", "fill every", "stockout hour"], edge=BLUE)
    box(ax, 34.5, 20, 14.5, 11, ["Version B", "fill hours after the", "first in-stock hour"], edge=GREEN)
    arrow(ax, 32, 33, 34.5, 37)
    arrow(ax, 32, 30, 34.5, 25.5)
    box(ax, 50.5, 26, 15, h, ["Latent demand", "sales + rate over", "filled hours"], fill=TINT)
    arrow(ax, 49, 37, 50.5, 34)
    arrow(ax, 49, 25.5, 50.5, 29)
    note(ax, 57.3, 16, "above recorded sales:\nA +24.6%, B +17.8%", ha="right")
    box(ax, 0.5, 1.5, 14, h, ["Daily records", "sales and", "stockout flag"])
    box(ax, 17.5, 1.5, 14.5, h, ["Warm-up", "first 28 days,", "as recorded"])
    arrow(ax, 14.5, 7, 17.5, 7)
    box(ax, 34.5, 1.5, 31, h, ["Closed-loop simulation (Fig. 1)", "days 29 to 90, each order scored",
                               "against latent demand"])
    arrow(ax, 32, 7, 34.5, 7)
    arrow(ax, 58.5, 18.5, 58.5, 12.5)
    note(ax, 0.5, 19, "11,476 store–product\nseries, 90 days", ha="left")
    save(fig, path)


def main():
    cfg = load_config()
    f = cfg["_out"] / "figures"
    fig1_simulator(f / "fig01_closed_loop.png")
    fig2_ground_truth(f / "fig02_ground_truth.png")
    log("flowcharts written to " + str(f))


if __name__ == "__main__":
    main()
