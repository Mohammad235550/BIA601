# Author: Shaima Shakhashiro
Runs all experiments:
- 3 problem sizes (Small, Medium, Large)
- GA, PSO, ACO, each repeated n_runs times with a fixed seed (reproducible)
- the two DP baselines once, on Small only
Saves raw_runs.csv, convergence.csv and summary_stats.csv in the output folder.
"""
import csv
import os
import time
import tracemalloc
import numpy as np

from .problem import generate_instance, build_distance_matrix
from .ga import run_ga
from .pso import run_pso
from .aco import run_aco
from .dp_baseline import held_karp, exact_cvrp

# problem sizes (Q = 50 for all)
SIZES = {
    "Small":  dict(n_customers=10, capacity=50.0, seed=1),
    "Medium": dict(n_customers=30, capacity=50.0, seed=2),
    "Large":  dict(n_customers=60, capacity=50.0, seed=3),
}

ALGORITHMS = {"GA": run_ga, "PSO": run_pso, "ACO": run_aco}
SEED_OFFSET = {"GA": 0, "PSO": 1, "ACO": 2}   # different seeds for each algorithm
N_RUNS = 10


def save_csv(path, rows, columns):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def measure_memory(algo, instance, dist, seed):
    """Peak memory (MB) of one run, measured with tracemalloc.
    Done in a separate run so it does not slow down the timed runs."""
    tracemalloc.start()
    algo(instance, dist, seed=seed)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def run_all(n_runs=N_RUNS, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    runs = []          # one row per run
    convergence = []   # one row per iteration (mean over the runs)
    memory = {}

    for size, cfg in SIZES.items():
        instance = generate_instance(name=size, **cfg)
        dist = build_distance_matrix(instance)

        # --- DP baselines (Small only) ---
        if size == "Small":
            t0 = time.perf_counter()
            cost, _ = held_karp(dist)
            runs.append(dict(size=size, algorithm="DP (exact, no capacity)", run=0,
                             best_fitness=cost, time_sec=time.perf_counter() - t0, evaluations=""))

            t0 = time.perf_counter()
            cost, _ = exact_cvrp(dist, instance.demands, instance.capacity)
            runs.append(dict(size=size, algorithm="DP (exact CVRP optimum)", run=0,
                             best_fitness=cost, time_sec=time.perf_counter() - t0, evaluations=""))

        # --- GA, PSO, ACO: n_runs each ---
        for name, algo in ALGORITHMS.items():
            histories = []
            for r in range(n_runs):
                seed = 1000 * r + SEED_OFFSET[name]
                t0 = time.perf_counter()
                result = algo(instance, dist, seed=seed)
                elapsed = time.perf_counter() - t0

                runs.append(dict(size=size, algorithm=name, run=r,
                                 best_fitness=result["best_fitness"], time_sec=elapsed,
                                 evaluations=result["evaluations"]))
                histories.append(result["history"])
                evals_per_iter = result["evaluations"] // len(result["history"])

            # runs stop at different iterations: repeat the last value so all have the same length
            length = max(len(h) for h in histories)
            curves = np.array([h + [h[-1]] * (length - len(h)) for h in histories])
            for it in range(length):
                convergence.append(dict(size=size, algorithm=name, iteration=it,
                                        evaluations=(it + 1) * evals_per_iter,
                                        mean_best_fitness=float(curves[:, it].mean()),
                                        std_best_fitness=float(curves[:, it].std()),
                                        min_best_fitness=float(curves[:, it].min()),
                                        max_best_fitness=float(curves[:, it].max())))

            memory[(size, name)] = measure_memory(algo, instance, dist, SEED_OFFSET[name])

    stats = summarize(runs, memory)

    save_csv(f"{out_dir}/raw_runs.csv", runs,
             ["size", "algorithm", "run", "best_fitness", "time_sec", "evaluations"])
    save_csv(f"{out_dir}/convergence.csv", convergence,
             ["size", "algorithm", "iteration", "evaluations", "mean_best_fitness",
              "std_best_fitness", "min_best_fitness", "max_best_fitness"])
    save_csv(f"{out_dir}/summary_stats.csv", stats,
             ["size", "algorithm", "mean_fitness", "std_fitness", "min_fitness", "max_fitness",
              "mean_time_sec", "mean_evaluations", "peak_memory_mb"])
    return runs, convergence, stats


def summarize(runs, memory):
    """Mean / std / best / worst for every (size, algorithm)."""
    stats = []
    for size in SIZES:
        names = sorted({r["algorithm"] for r in runs if r["size"] == size})
        for name in names:
            rows = [r for r in runs if r["size"] == size and r["algorithm"] == name]
            fit = np.array([r["best_fitness"] for r in rows])
            times = np.array([r["time_sec"] for r in rows])
            evals = [r["evaluations"] for r in rows if r["evaluations"] != ""]
            stats.append(dict(size=size, algorithm=name,
                              mean_fitness=float(fit.mean()),
                              std_fitness=float(fit.std()),
                              min_fitness=float(fit.min()),
                              max_fitness=float(fit.max()),
                              mean_time_sec=float(times.mean()),
                              mean_evaluations=float(np.mean(evals)) if evals else "",
                              peak_memory_mb=memory.get((size, name), "")))
    return stats
