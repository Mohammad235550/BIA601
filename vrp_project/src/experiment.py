# Author: Shaima Shakhashiro
"""
Runs all the experiments:
- 4 problem sizes (Small, Medium, Large, XLarge)
- GA, PSO and ACO, each one run n_runs times with fixed seeds (so it can be repeated)
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

# problem sizes (vehicle capacity Q = 50 for all of them)
SIZES = {
    "Small":  dict(n_customers=10, capacity=50.0, seed=1),
    "Medium": dict(n_customers=30, capacity=50.0, seed=2),
    "Large":  dict(n_customers=60, capacity=50.0, seed=3),
    "XLarge": dict(n_customers=100, capacity=50.0, seed=4),
}

ALGORITHMS = {"GA": run_ga, "PSO": run_pso, "ACO": run_aco}
SEED_OFFSET = {"GA": 0, "PSO": 1, "ACO": 2}   # each algorithm gets different seeds


def all_instances():
    """Generate the 4 problem sizes."""
    for size in SIZES:
        yield size, generate_instance(name=size, **SIZES[size])


def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def measure_memory(algo, instance, dist, seed):
    """Peak memory (MB) of one run. Done in a separate run so the timed runs are not slowed down."""
    tracemalloc.start()
    algo(instance, dist, seed=seed)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak / (1024 * 1024)


def run_all(n_runs=30, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    runs = []          # one row for every single run
    convergence = []   # one row for every iteration (mean over all runs)
    memory = {}

    for size, instance in all_instances():
        dist = build_distance_matrix(instance)

        # ---- DP baselines (Small only) ----
        if size == "Small":
            start = time.perf_counter()
            cost, _ = held_karp(dist)
            runs.append(dict(size=size, algorithm="DP (exact, no capacity)", run=0,
                             best_fitness=cost, time_sec=time.perf_counter() - start, evaluations=""))

            start = time.perf_counter()
            cost, _ = exact_cvrp(dist, instance.demands, instance.capacity)
            runs.append(dict(size=size, algorithm="DP (exact CVRP optimum)", run=0,
                             best_fitness=cost, time_sec=time.perf_counter() - start, evaluations=""))

        # ---- GA, PSO, ACO: n_runs each ----
        for name, algo in ALGORITHMS.items():
            histories = []
            for r in range(n_runs):
                seed = 1000 * r + SEED_OFFSET[name]
                start = time.perf_counter()
                result = algo(instance, dist, seed=seed)
                elapsed = time.perf_counter() - start

                runs.append(dict(size=size, algorithm=name, run=r,
                                 best_fitness=result["best_fitness"], time_sec=elapsed,
                                 evaluations=result["evaluations"]))
                histories.append(result["history"])
            evals_per_iter = result["evaluations"] // len(result["history"])

            # runs stop at different iterations, so we repeat the last value of
            # the short runs to make all curves the same length
            length = max(len(h) for h in histories)
            curves = [h + [h[-1]] * (length - len(h)) for h in histories]
            for it in range(length):
                values = np.array([curve[it] for curve in curves])
                convergence.append(dict(size=size, algorithm=name, iteration=it,
                                        evaluations=(it + 1) * evals_per_iter,
                                        mean_best_fitness=float(values.mean()),
                                        std_best_fitness=float(values.std()),
                                        min_best_fitness=float(values.min()),
                                        max_best_fitness=float(values.max())))

            memory[(size, name)] = measure_memory(algo, instance, dist, SEED_OFFSET[name])

    stats = summarize(runs, memory)
    save_csv(f"{out_dir}/raw_runs.csv", runs)
    save_csv(f"{out_dir}/convergence.csv", convergence)
    save_csv(f"{out_dir}/summary_stats.csv", stats)
    return runs, convergence, stats


def summarize(runs, memory):
    """Mean / std / best / worst of every (size, algorithm)."""
    stats = []
    for size in SIZES:
        names = sorted(set(r["algorithm"] for r in runs if r["size"] == size))
        for name in names:
            rows = [r for r in runs if r["size"] == size and r["algorithm"] == name]
            fitness = np.array([r["best_fitness"] for r in rows])
            times = np.array([r["time_sec"] for r in rows])
            evals = [r["evaluations"] for r in rows if r["evaluations"] != ""]
            stats.append(dict(size=size, algorithm=name,
                              mean_fitness=float(fitness.mean()),
                              std_fitness=float(fitness.std()),
                              min_fitness=float(fitness.min()),
                              max_fitness=float(fitness.max()),
                              mean_time_sec=float(times.mean()),
                              mean_evaluations=float(np.mean(evals)) if evals else "",
                              peak_memory_mb=memory.get((size, name), "")))
    return stats
