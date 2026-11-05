"""
Charts for the report:
- convergence curves (Iteration -> Best Fitness) for each size
- the same curves against the number of fitness evaluations
- bar charts of solution quality and execution time
All charts are saved as PNG files in results/plots.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # save to files only, no window
import matplotlib.pyplot as plt

SIZES = ["Small", "Medium", "Large"]
ALGORITHMS = ["GA", "PSO", "ACO"]


def plot_convergence(convergence, out_dir="results/plots", stats=None):
    os.makedirs(out_dir, exist_ok=True)

    # exact optimum (Small only) drawn as a dashed line
    optimum = {}
    for r in stats or []:
        if r["algorithm"] == "DP (exact CVRP optimum)":
            optimum[r["size"]] = r["mean_fitness"]

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
                line, = plt.plot(x, mean, label=name)
                # shaded band = best run .. worst run
                plt.fill_between(x, low, high, color=line.get_color(), alpha=0.2)
            if size in optimum:
                plt.axhline(optimum[size], color="black", linestyle="--", linewidth=1,
                            label=f"Exact CVRP optimum ({optimum[size]:.2f})")
            plt.xlabel(x_label)
            plt.ylabel("Best Fitness (total distance)")
            plt.title(f"Convergence — {size} instance (mean, band = best..worst run)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{out_dir}/convergence_{size}{suffix}.png", dpi=150)
            plt.close()


def bar_chart(stats, value_key, error_key, ylabel, title, out_path):
    table = {(r["size"], r["algorithm"]): r for r in stats}
    width = 0.25
    x = np.arange(len(SIZES))

    plt.figure(figsize=(8, 5))
    for i, name in enumerate(ALGORITHMS):
        values = [table[(s, name)][value_key] for s in SIZES]
        errors = [table[(s, name)][error_key] for s in SIZES] if error_key else None
        plt.bar(x + i * width, values, width=width, label=name, yerr=errors, capsize=4)
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
