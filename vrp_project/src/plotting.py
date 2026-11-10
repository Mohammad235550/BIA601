# Author: noor
"""
Charts for the report (saved as PNG files in results/plots):
- convergence curves (Iteration -> Best Fitness) for every size
- the same curves against the number of fitness evaluations
- bar charts of solution quality and execution time
- boxplots: spread of the final result over all runs (stability)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # only save files, do not open a window
import matplotlib.pyplot as plt

SIZES = ["Small", "Medium", "Large", "XLarge"]
ALGORITHMS = ["GA", "PSO", "ACO"]

# one color, line style and marker per algorithm (readable in black and white too)
COLOR = {"GA": "#6250d6", "PSO": "#eda100", "ACO": "#1baf7a"}
LINE = {"GA": "-", "PSO": "--", "ACO": "-."}
MARKER = {"GA": "o", "PSO": "s", "ACO": "^"}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linestyle": ":",
    "axes.axisbelow": True,
    "legend.frameon": False,
})


def get_optimum(rows, key):
    """The exact CVRP optimum of the Small instance (from the DP)."""
    for r in rows:
        if r["algorithm"] == "DP (exact CVRP optimum)":
            return r[key]
    return None


def plot_convergence(convergence, out_dir="results/plots", stats=None):
    os.makedirs(out_dir, exist_ok=True)
    optimum = get_optimum(stats or [], "mean_fitness")

    # two versions of every chart: x = iteration, and x = number of evaluations
    for x_key, x_label, suffix in [("iteration", "Iteration", ""),
                                   ("evaluations", "Fitness evaluations", "_evals")]:
        for size in SIZES:
            plt.figure(figsize=(7, 5))
            for name in ALGORITHMS:
                rows = [r for r in convergence if r["size"] == size and r["algorithm"] == name]
                x = [r[x_key] for r in rows]
                mean = [r["mean_best_fitness"] for r in rows]
                low = [r["min_best_fitness"] for r in rows]
                high = [r["max_best_fitness"] for r in rows]
                plt.plot(x, mean, label=name, color=COLOR[name], linestyle=LINE[name],
                         linewidth=2, marker=MARKER[name], markersize=6,
                         markevery=max(1, len(x) // 8))
                # shaded band = from the best run to the worst run
                plt.fill_between(x, low, high, color=COLOR[name], alpha=0.15, linewidth=0)
            if size == "Small" and optimum is not None:
                plt.axhline(optimum, color="#333333", linestyle=":", linewidth=1.5,
                            label=f"Exact CVRP optimum ({optimum:.2f})")
            plt.xlabel(x_label)
            plt.ylabel("Best Fitness (total distance)")
            plt.title(f"Convergence — {size} instance\n(line = mean of all runs, band = best..worst run)",
                      fontsize=11)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{out_dir}/convergence_{size}{suffix}.png", dpi=150)
            plt.close()


def bar_chart(stats, value_key, error_key, ylabel, title, out_path):
    width = 0.25
    x = np.arange(len(SIZES))
    plt.figure(figsize=(8, 5))
    for i, name in enumerate(ALGORITHMS):
        rows = [r for size in SIZES for r in stats if r["size"] == size and r["algorithm"] == name]
        values = [r[value_key] for r in rows]
        errors = [r[error_key] for r in rows] if error_key else None
        plt.bar(x + i * width, values, width=width * 0.92, label=name, yerr=errors, capsize=4,
                color=COLOR[name], edgecolor="white")
    plt.xticks(x + width, SIZES)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_quality_comparison(stats, out_dir="results/plots"):
    os.makedirs(out_dir, exist_ok=True)
    bar_chart(stats, "mean_fitness", "std_fitness", "Mean Best Fitness (total distance)",
              "Solution quality by algorithm and problem size", f"{out_dir}/quality_comparison.png")


def plot_time_comparison(stats, out_dir="results/plots"):
    os.makedirs(out_dir, exist_ok=True)
    bar_chart(stats, "mean_time_sec", None, "Mean Execution Time (s)",
              "Execution time by algorithm and problem size", f"{out_dir}/time_comparison.png")


def plot_boxplots(runs, out_dir="results/plots"):
    """One box per algorithm and size: how the final result changes from run to run."""
    os.makedirs(out_dir, exist_ok=True)
    optimum = get_optimum(runs, "best_fitness")
    fig, axes = plt.subplots(1, len(SIZES), figsize=(16, 4.5))
    for ax, size in zip(axes, SIZES):
        data = []
        for name in ALGORITHMS:
            data.append([r["best_fitness"] for r in runs if r["size"] == size and r["algorithm"] == name])
        box = ax.boxplot(data, patch_artist=True, widths=0.6,
                         medianprops=dict(color="black", linewidth=1.5))
        for patch, name in zip(box["boxes"], ALGORITHMS):
            patch.set_facecolor(COLOR[name])
            patch.set_alpha(0.6)
        ax.set_xticks([1, 2, 3], ALGORITHMS)
        ax.set_title(size)
        if size == "Small" and optimum is not None:
            ax.axhline(optimum, color="#333333", linestyle=":", linewidth=1.5)
    axes[0].set_ylabel("Best Fitness (total distance)")
    n_runs = len(data[0])
    fig.suptitle(f"Spread of the final result over {n_runs} runs (dotted line = exact optimum, Small)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/boxplots.png", dpi=150)
    plt.close(fig)
