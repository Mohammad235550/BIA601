#Author:zenab alsoutary
"""
Correctness tests:

    python verify.py

1) exact_cvrp gives the same answer as brute force (15 small instances).
2) held_karp gives the same answer as brute force TSP (same 15 instances).
3) the PSO swap sequence really turns a route into its target.
4) GA, PSO and ACO return valid solutions (every customer once, no route
   over capacity) and the reported fitness is correct, on every instance.
5) the CVRPLIB benchmark is read correctly: the published optimal solution,
   evaluated with our code, gives exactly the published cost (27591).
"""
import itertools
import numpy as np

from src.problem import generate_instance, build_distance_matrix, route_distance, evaluate, load_cvrplib
from src.dp_baseline import held_karp, exact_cvrp
from src.pso import swap_sequence, run_pso
from src.ga import run_ga
from src.aco import run_aco
from src.experiment import SIZES, BENCHMARKS, all_instances


def brute_force_cvrp(instance, dist):
    """Try every order of the customers and every way to cut it into routes."""
    n = instance.n
    best = float("inf")
    for perm in itertools.permutations(range(n)):
        for cuts in range(2 ** (n - 1)):
            routes, current = [], [perm[0]]
            for k in range(1, n):
                if (cuts >> (k - 1)) & 1:      # cut before customer k
                    routes.append(current)
                    current = []
                current.append(perm[k])
            routes.append(current)
            if all(sum(instance.demands[c] for c in r) <= instance.capacity for r in routes):
                best = min(best, sum(route_distance(r, dist) for r in routes))
    return best


def brute_force_tsp(instance, dist):
    return min(route_distance(list(p), dist) for p in itertools.permutations(range(instance.n)))


def check(ok, message):
    print(("PASS  " if ok else "FAIL  ") + message)
    return ok


def main():
    results = []

    # 1) + 2) DP vs brute force
    wrong_cvrp = wrong_tsp = 0
    for s in range(15):
        inst = generate_instance("test", 6, capacity=15.0, seed=100 + s)
        dist = build_distance_matrix(inst)
        if abs(exact_cvrp(dist, inst.demands, inst.capacity)[0] - brute_force_cvrp(inst, dist)) > 1e-9:
            wrong_cvrp += 1
        if abs(held_karp(dist)[0] - brute_force_tsp(inst, dist)) > 1e-9:
            wrong_tsp += 1
    results.append(check(wrong_cvrp == 0, "exact CVRP DP == brute force on 15 random instances"))
    results.append(check(wrong_tsp == 0, "Held-Karp == brute force TSP on 15 random instances"))

    # 3) PSO swap sequence
    rng = np.random.default_rng(0)
    reached = 0
    for _ in range(1000):
        source = rng.permutation(20).tolist()
        target = rng.permutation(20).tolist()
        route = source.copy()
        for i, j, _ in swap_sequence(source, target, 1.0):
            route[i], route[j] = route[j], route[i]
        if route == target:
            reached += 1
    results.append(check(reached == 1000, f"PSO swap sequence reaches its target ({reached}/1000)"))

    # 4) valid solutions from every algorithm on every instance
    invalid = 0
    for size, inst in all_instances():
        dist = build_distance_matrix(inst)
        for algo, key in [(run_ga, "route"), (run_pso, "solution"), (run_aco, "tour")]:
            result = algo(inst, dist, seed=0)
            perm = result[key]
            total, routes, _ = evaluate(perm, inst, dist)
            every_customer_once = sorted(perm) == list(range(inst.n))
            capacity_ok = all(sum(inst.demands[c] for c in r) <= inst.capacity for r in routes)
            fitness_ok = abs(total - result["best_fitness"]) < 1e-9
            if not (every_customer_once and capacity_ok and fitness_ok):
                invalid += 1
    results.append(check(invalid == 0, "GA / PSO / ACO return valid solutions with correct fitness (all sizes)"))

    # 5) benchmark file + published optimal solution
    for name, cfg in BENCHMARKS.items():
        inst = load_cvrplib(cfg["path"])
        dist = build_distance_matrix(inst)
        routes = []
        with open(cfg["path"].replace(".vrp", ".sol")) as f:
            for line in f:
                if line.startswith("Route"):
                    routes.append([int(c) - 1 for c in line.split(":")[1].split()])
        served = sorted(c for r in routes for c in r)
        cost = sum(route_distance(r, dist) for r in routes)
        ok = (served == list(range(inst.n))
              and all(sum(inst.demands[c] for c in r) <= inst.capacity for r in routes)
              and abs(cost - cfg["best_known"]) < 1e-9)
        results.append(check(ok, f"{name}: published optimal solution costs {cost:.0f} with our code "
                                 f"(expected {cfg['best_known']})"))

    print("\nAll checks passed." if all(results) else "\nSome checks FAILED.")


if __name__ == "__main__":
    main()
