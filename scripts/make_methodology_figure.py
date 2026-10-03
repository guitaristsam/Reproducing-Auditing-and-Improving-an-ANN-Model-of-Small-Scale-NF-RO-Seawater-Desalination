"""Methodology flowchart (Figure 1).

Usage: python make_methodology_figure.py [output_dir]

Writes methodology.pdf and methodology.png, by default next to this script.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1a1a1a"
MUTED = "#4d5560"
RULE = "#b9c0c9"

# light fills so the figure still reads in greyscale
DATA_FC, DATA_EC = "#eef1f5", "#8a96a5"
NB1_FC, NB1_EC = "#e8f0f8", "#5f8fbf"
NB2_FC, NB2_EC = "#eaf3ec", "#62976d"
OUT_FC, OUT_EC = "#faf1e4", "#b98f50"

FS_TITLE = 9.5
FS_BODY = 8.5
FS_NOTE = 8.0
FS_BAND = 9.0


def box(ax, x, y, w, h, title, body, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.014",
                                facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=2))
    ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center",
            fontsize=FS_TITLE, weight="bold", color=INK, zorder=3)
    ax.text(x + w / 2, y + h * 0.32, body, ha="center", va="center",
            fontsize=FS_BODY, color=MUTED, zorder=3, linespacing=1.35)


def arrow(ax, p, q, color=INK, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, linewidth=1.1,
                                 color=color, zorder=4, shrinkA=0, shrinkB=3,
                                 connectionstyle=f"arc3,rad={rad}"))


def band(ax, y, h, label, ec):
    ax.add_patch(FancyBboxPatch((0.01, y), 0.98, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                facecolor=ec, edgecolor=ec, linewidth=0.8, alpha=0.08, zorder=0))
    ax.text(0.03, y + h / 2, label, ha="center", va="center", rotation=90,
            fontsize=FS_BAND, weight="bold", color=ec, zorder=1)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.6, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    L, R, W = 0.07, 0.535, 0.395     # left column x, right column x, column width
    lc, rc = L + W / 2, R + W / 2

    # data
    band(ax, 0.770, 0.220, "Data", DATA_EC)
    box(ax, L, 0.900, R + W - L, 0.075, "73 operating points, 5 inputs, 3 outputs",
        "time, pressure, temperature, feed flow, feed conductivity", "#ffffff", DATA_EC)
    box(ax, L, 0.785, W, 0.080, "Clean the data",
        "drop the s/n row counter\nrow 45: 3.29 °C → 29.29 °C", DATA_FC, DATA_EC)
    box(ax, R, 0.785, W, 0.080, "Scale to [−1, 1]",
        "min-max, as in the\nsource paper", DATA_FC, DATA_EC)
    arrow(ax, (lc, 0.891), (lc, 0.874))
    arrow(ax, (L + W + 0.010, 0.825), (R - 0.010, 0.825))

    # modelling
    band(ax, 0.335, 0.420, "Modelling", "#7d8a99")
    rows = (0.645, 0.515, 0.385)
    h = 0.085
    box(ax, L, rows[0], W, h, "Notebook 1: reproduce",
        "ANN 5-8-3 and MLR, 60/13 split\nLM, 12 restarts", NB1_FC, NB1_EC)
    box(ax, L, rows[1], W, h, "Score in-sample (n = 60)",
        "α, β, R, RMSE, MAE, AARD\nin engineering units", NB1_FC, NB1_EC)
    box(ax, L, rows[2], W, h, "Audit the source tables",
        "recompute printed values,\ncheck internal identities", NB1_FC, NB1_EC)
    box(ax, R, rows[0], W, h, "Notebook 2: re-evaluate",
        "10-fold CV × 5 on all 73 rows\nscaler fitted inside each fold", NB2_FC, NB2_EC)
    box(ax, R, rows[1], W, h, "Nine models",
        "MLR, ridge, two ANNs, SVR, GP,\nRF, XGBoost, RF + XGBoost", NB2_FC, NB2_EC)
    box(ax, R, rows[2], W, h, "Random forest selected",
        "chosen on the same folds;\nselection bias disclosed", NB2_FC, NB2_EC)

    arrow(ax, (R + 0.06, 0.776), (lc + 0.08, rows[0] + h + 0.010), color=NB1_EC, rad=-0.12)
    arrow(ax, (rc, 0.776), (rc, rows[0] + h + 0.010), color=NB2_EC)
    for x, c in ((lc, NB1_EC), (rc, NB2_EC)):
        arrow(ax, (x, rows[0] - 0.009), (x, rows[1] + h + 0.010), color=c)
        arrow(ax, (x, rows[1] - 0.009), (x, rows[2] + h + 0.010), color=c)

    ax.text(0.5, 0.352, "same data and network, different protocol", ha="center", va="center",
            fontsize=FS_NOTE, color=MUTED, style="italic", zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#f5f6f7", edgecolor="none"))

    # findings
    band(ax, 0.030, 0.290, "Findings", OUT_EC)
    box(ax, L, 0.205, W, h, "Seven errors in the source",
        "RMSE < MAE on the MLR rows;\nAARD is a sum (factor 100/60)", OUT_FC, OUT_EC)
    box(ax, L, 0.075, W, h, "Published R is in-sample",
        "0.969 is a training fit,\nnot a target to beat", OUT_FC, OUT_EC)
    box(ax, R, 0.205, W, h, "Out-of-fold gain",
        "mean R 0.579 → 0.816 (+40.9 %)\nAARD −35 %", OUT_FC, OUT_EC)
    box(ax, R, 0.075, W, h, "Intervals and sensitivity",
        "GP coverage 90.7–93.4 %;\ntime and temperature dominate", OUT_FC, OUT_EC)
    for x in (lc, rc):
        arrow(ax, (x, rows[2] - 0.009), (x, 0.205 + h + 0.010), color=OUT_EC)
        arrow(ax, (x, 0.205 - 0.009), (x, 0.075 + h + 0.010), color=OUT_EC)

    ax.text(0.5, 0.047, "Percentages are only computed within one protocol.",
            ha="center", va="center", fontsize=FS_NOTE, color=MUTED, style="italic")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(os.path.join(out_dir, "methodology.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(out_dir, "methodology.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
