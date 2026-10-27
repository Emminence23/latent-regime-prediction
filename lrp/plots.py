"""Figures for the five parts, in one consistent style."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Validated categorical palette, used in fixed order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 6,
})


def _log2_axis(ax, ticks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()


def _label_min(ax, xs, ys, fmt="{:.2f}"):
    i = min(range(len(ys)), key=ys.__getitem__)
    ax.annotate("best " + fmt.format(ys[i]), (xs[i], ys[i]), xytext=(0, -16),
                textcoords="offset points", ha="center", fontsize=8, color=INK)


def plot_part1(iters, maes, knee, path):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(iters, maes, marker="o", color=COLORS[0])
    k = iters.index(knee)
    ax.annotate(f"chosen: {knee} iterations\nMAE {maes[k]:.2f}", (knee, maes[k]),
                xytext=(10, 22), textcoords="offset points", fontsize=8, color=INK,
                arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.set_xscale("log")
    ax.set(xlabel="EM iterations (log scale)", ylabel="Test MAE (ppb)",
           title="Part 1: test error vs EM iterations (4 components)")
    fig.savefig(path)
    plt.close(fig)


def plot_part2(comps, maes, path):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(comps, maes, marker="o", color=COLORS[0])
    _log2_axis(ax, comps)
    _label_min(ax, comps, maes)
    ax.set(xlabel="Number of components", ylabel="Test MAE (ppb)",
           title="Part 2: test error vs number of components (5 iterations)")
    fig.savefig(path)
    plt.close(fig)


def plot_part3(groups, path):
    """One box per component, showing the true O3 of the training points it claims."""
    ks = sorted(groups)
    fig, axes = plt.subplots(1, len(ks), figsize=(4.2 * len(ks), 3.8), sharex=True)
    for ax, k in zip(axes, ks):
        order = sorted(range(k), key=lambda c: np.median(groups[k][c]) if len(groups[k][c]) else 0)
        data = [groups[k][c] for c in order]
        bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True, showfliers=False,
                        medianprops={"color": INK}, whiskerprops={"color": MUTED},
                        capprops={"color": MUTED})
        for patch, c in zip(bp["boxes"], order):
            patch.set(facecolor=COLORS[c], edgecolor="white", alpha=0.85)
        ax.set_yticks(range(1, k + 1))
        ax.set_yticklabels([f"comp {c} (n={len(groups[k][c])})" for c in order], fontsize=7)
        ax.set(title=f"{k} components", xlabel="Reference O3 (ppb)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Part 3: true O3 of the points each component claims", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_part4(comps, all_maes, ablation, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(comps, all_maes, marker=MARKERS[0], color=COLORS[0], lw=2.6, label="all features")
    for i, (feat, maes) in enumerate(ablation.items(), start=1):
        ax.plot(comps, maes, marker=MARKERS[i], color=COLORS[i], lw=1.4, ls="--",
                label=f"without {feat}")
    _log2_axis(ax, comps)
    ax.set(xlabel="Number of components", ylabel="Test MAE (ppb)",
           title="Part 4: dropping one feature at a time (5 iterations)")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(path)
    plt.close(fig)


def plot_part5(trees, mixtures, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = [p["pred_time"] * 1e3 for p in trees]
    y = [p["mae"] for p in trees]
    ax.scatter(x, y, marker=MARKERS[0], color=COLORS[0], s=36, zorder=3,
               edgecolor="white", linewidth=1, label="Decision tree (depth 2 to 30)")
    best = min(trees, key=lambda p: p["mae"])
    ax.annotate(f"best tree, depth {best['max_depth']}: {best['mae']:.2f}",
                (best["pred_time"] * 1e3, best["mae"]), xytext=(8, -4),
                textcoords="offset points", fontsize=8, color=INK)
    x = [p["pred_time"] * 1e3 for p in mixtures]
    y = [p["mae"] for p in mixtures]
    ax.plot(x, y, marker=MARKERS[1], color=COLORS[1], label="Mixture (1 to 32 components)")
    for p, xi, yi in zip(mixtures, x, y):
        ax.annotate(f"k={p['n_components']}", (xi, yi), xytext=(0, 7), ha="center",
                    textcoords="offset points", fontsize=7, color=MUTED)
    ax.set_xscale("log")
    ax.set(xlabel="Prediction time for the whole test set (ms, log scale)",
           ylabel="Test MAE (ppb)", title="Part 5: speed vs accuracy")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)
