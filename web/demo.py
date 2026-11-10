# Author: Mohammad Nour
"""
Glue code for the web demo (index.html).
It runs inside the browser with Pyodide and only CALLS the project code in
vrp_project/src: same instances, same seeds and same algorithms as main.py.
"""
import json
import sys
import time
import types

try:
    import tracemalloc  # noqa: F401  (used by experiment.py for the memory test)
except ImportError:      # some browser builds of Python do not ship it
    sys.modules["tracemalloc"] = types.ModuleType("tracemalloc")

from src.experiment import SIZES, SEED_OFFSET, ALGORITHMS
from src.problem import generate_instance, build_distance_matrix, evaluate, route_distance
from src.dp_baseline import exact_cvrp, held_karp

_cache = {}


def _get(size):
    if size not in _cache:
        inst = generate_instance(size, **SIZES[size])
        _cache[size] = (inst, build_distance_matrix(inst))
    return _cache[size]


def _routes_info(routes, inst, dist):
    return [dict(customers=[int(c) for c in r],
                 load=float(sum(inst.demands[c] for c in r)),
                 length=float(route_distance(r, dist))) for r in routes]


def instance_json(size):
    inst, _ = _get(size)
    return json.dumps(dict(size=size, depot=[float(v) for v in inst.depot],
                           coords=inst.coords.tolist(), demands=inst.demands.tolist(),
                           capacity=float(inst.capacity), n=int(inst.n)))


def run_json(size, algo, run):
    """One run of GA / PSO / ACO with the same seed as run `run` of the experiments."""
    inst, dist = _get(size)
    seed = 1000 * int(run) + SEED_OFFSET[algo]
    start = time.perf_counter()
    res = ALGORITHMS[algo](inst, dist, seed=seed)
    elapsed = time.perf_counter() - start
    total, routes, vehicles = evaluate(res["solution"], inst, dist)
    return json.dumps(dict(algo=algo, seed=seed, best=float(res["best_fitness"]), check=float(total),
                           history=[float(h) for h in res["history"]],
                           evaluations=int(res["evaluations"]), vehicles=int(vehicles),
                           routes=_routes_info(routes, inst, dist), time=elapsed))


def dp_json(size):
    """Exact CVRP optimum and Held-Karp lower bound (Small only)."""
    inst, dist = _get(size)
    start = time.perf_counter()
    cost, routes = exact_cvrp(dist, inst.demands, inst.capacity)
    elapsed = time.perf_counter() - start
    lower_bound, _ = held_karp(dist)
    return json.dumps(dict(algo="DP", best=float(cost), heldkarp=float(lower_bound),
                           vehicles=len(routes), routes=_routes_info(routes, inst, dist), time=elapsed))
