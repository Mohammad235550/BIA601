# Author: Diana Souliman
"""
Discrete Particle Swarm Optimization (PSO) for CVRP.

A particle is a permutation of the customers, so the velocity is a
sequence of swap operators (Swap Sequence).
In every iteration each particle does three steps:
  1) exploration: 2 random swaps (instead of the inertia term)
  2) move towards its pbest: apply each swap with probability alfa
  3) move towards gbest:     apply each swap with probability beta
Stop: after `iterations` or when gbest does not improve for `patience` iterations.
"""
import numpy as np
from .problem import total_cost


def swap_sequence(source, target, probability):
    """List of swaps (i, j, probability) that turns `source` into `target`."""
    current = list(source)
    position = {c: i for i, c in enumerate(current)}   # where each customer is now
    swaps = []
    for i in range(len(current)):
        if current[i] != target[i]:
            j = position[target[i]]
            swaps.append((i, j, probability))
            position[current[i]], position[current[j]] = j, i
            current[i], current[j] = current[j], current[i]
    return swaps


def apply_swaps(route, swaps, rng):
    for i, j, prob in swaps:
        if rng.random() <= prob:
            route[i], route[j] = route[j], route[i]
    return route


def run_pso(instance, dist, n_particles=40, iterations=300,
            alfa=0.6, beta=0.8, n_random_swaps=2, patience=60, seed=0):
    rng = np.random.default_rng(seed)
    n = instance.n

    # initial swarm: random permutations
    swarm = [rng.permutation(n).tolist() for _ in range(n_particles)]
    pbest = [p.copy() for p in swarm]
    pbest_cost = [total_cost(p, instance, dist) for p in swarm]

    i = int(np.argmin(pbest_cost))
    gbest = pbest[i].copy()
    gbest_cost = pbest_cost[i]

    history = [gbest_cost]
    no_improve = 0
    it = 0

    while it < iterations and no_improve < patience:
        improved = False
        for p in range(n_particles):
            new = swarm[p].copy()

            # 1) exploration: random swaps
            for _ in range(n_random_swaps):
                a, b = rng.integers(0, n, size=2)
                new[a], new[b] = new[b], new[a]

            # 2) towards pbest, 3) towards gbest
            new = apply_swaps(new, swap_sequence(new, pbest[p], alfa), rng)
            new = apply_swaps(new, swap_sequence(new, gbest, beta), rng)

            swarm[p] = new
            cost = total_cost(new, instance, dist)
            if cost < pbest_cost[p]:
                pbest_cost[p] = cost
                pbest[p] = new.copy()
            if cost < gbest_cost:
                gbest_cost = cost
                gbest = new.copy()
                improved = True

        history.append(gbest_cost)
        no_improve = 0 if improved else no_improve + 1
        it += 1

    return {"solution": gbest,
            "best_fitness": gbest_cost,
            "history": history,
            "evaluations": n_particles * len(history)}
