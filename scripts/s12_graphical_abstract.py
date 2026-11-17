"""Step 12: graphical abstract (Elsevier format: landscape, 2.5:1, readable at 13 x 5 cm).
It is a schematic; the numbers in the right panel are those reported in the paper (abstract, Tables 5, 6 and 9).
Output: figures/graphical_abstract.png (300 dpi, 3000 x 1200 px) and .pdf."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from common import load_config, log
from censim.plotting import COLOR, INK, MUTED

BLUE, ORANGE, GREEN, PURPLE = COLOR["PAR"], COLOR["SAA"], COLOR["TOB"], COLOR["KMX"]
TINT, EDGE = "#f3f2ee", "#8a8985"
W, H = 100, 40                                   # drawing units; the figure is 10 x 4 inches


def panel(ax, x, w, title):
    ax.add_patch(FancyBboxPatch((x, 6.5), w, 31.5, boxstyle="round,pad=0,rounding_size=1.5", fc="white", ec=EDGE, lw=1.4))
    ax.text(x + 1.5, 35.6, title, fontsize=15, fontweight="bold", color=INK, va="center")


def box(ax, x, y, w, h, text, ec=EDGE, fc="white", fs=12.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.25)


def arrow(ax, a, b, rad=0.0, color=EDGE):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.8, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


def main():
    cfg = load_config()
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    # 1. the closed loop
    panel(ax, 1, 25, "Closed loop")
    box(ax, 3.5, 23, 20, 8, "Learner sets\norder q", ec=BLUE)
    box(ax, 3.5, 9, 20, 8, "Only sales min(D, q)\nare recorded", ec=ORANGE, fc=TINT, fs=12)
    arrow(ax, (10, 23), (10, 17))
    arrow(ax, (17, 17), (17, 23), color=ORANGE)
    ax.text(18, 20, "retrain", fontsize=12, color=ORANGE, va="center")

    # 2. two failure modes (schematic, no data)
    panel(ax, 28, 32, "Two failure modes")
    ax.plot([30.5, 41], [31.5, 31.5], color=MUTED, lw=1.4, ls="--")
    xs = [30.5 + 2.1 * i for i in range(6)]
    ys = [30 - 2.0 * i for i in range(6)]
    for i in range(5):
        ax.plot([xs[i], xs[i + 1], xs[i + 1]], [ys[i], ys[i], ys[i + 1]], color=BLUE, lw=2.4)
    ax.text(43, 26, "Spiral-down\nparametric,\nτ ≤ 0.5", fontsize=12, color=INK, va="center", linespacing=1.2)
    ax.plot([30.5, 41], [16.5, 16.5], color=MUTED, lw=1.4, ls="--")
    ax.plot([30.5, 41], [12, 12], color=ORANGE, lw=2.4)
    ax.text(43, 13.5, "Lock-in\nquantile\nlearners", fontsize=12, color=INK, va="center", linespacing=1.2)
    ax.text(35.7, 8.3, "- - -  optimal order", fontsize=10.5, color=MUTED, ha="center")

    # 3. results (numbers as reported in the paper)
    panel(ax, 62, 37, "Costs and remedies")
    ax.text(64, 30.5, "Synthetic, 27,000 runs", fontsize=12, fontweight="bold", color=INK)
    ax.text(64, 27.7, "lock-in: up to 4.08 × optimal cost", fontsize=12, color=INK)
    ax.text(64, 25.0, "2% exploration: 1.83 → 1.02 ×", fontsize=12, color=PURPLE)
    ax.text(64, 20.0, "Real data, 11,476 series", fontsize=12, fontweight="bold", color=INK)
    ax.text(64, 17.2, "censoring adds 5.6–96.0% cost", fontsize=12, color=INK)
    ax.text(64, 14.5, "backtest winner costs most (τ ≥ 0.75)", fontsize=12, color=INK)
    ax.text(64, 11.8, "stockout-rate rule: up to −21.2%", fontsize=12, color=GREEN)

    ax.text(W / 2, 2.6, "Evaluate ordering models in the loop they will run in, not only by backtest",
            ha="center", va="center", fontsize=13, color=INK, style="italic")
    out = cfg["_out"] / "figures" / "graphical_abstract.png"
    fig.savefig(out, dpi=300)
    fig.savefig(str(out).replace(".png", ".pdf"))
    plt.close(fig)
    log(f"graphical abstract written to {out}")


if __name__ == "__main__":
    main()
