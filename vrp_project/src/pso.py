# Author: Diana Souliman
"""
Discrete Particle Swarm Optimization (PSO) for CVRP.

A particle is an order (permutation) of the customers, so its "velocity" is a
list of swaps (Swap Sequence). In every iteration each particle does 3 steps:
  1) exploration: 2 random swaps (instead of the inertia term)
  2) move towards its pbest: apply each swap with probability alfa
  3) move towards gbest:     apply each swap with probability beta
Stop: after `iterations`, or when gbest does not improve for `patience` iterations.
"""
import numpy as np
from .problem import total_cost


def swap_sequence(source, target):
    """List of swaps (i, j) that turns `source` into `target`."""
    current = list(source)
    swaps = []
    for i in range(len(current)):
        if current[i] != target[i]:
            j = current.index(target[i])          # where is the customer we need?
            swaps.append((i, j))
            current[i], current[j] = current[j], current[i]
    return swaps


def move_towards(order, target, probability, rng):
    """Apply every swap of the sequence with the given probability."""
    for i, j in swap_sequence(order, target):
        if rng.random() <= probability:
            order[i], order[j] = order[j], order[i]
    return order


def run_pso(instance, dist, n_particles=40, iterations=300,
            alfa=0.6, beta=0.8, n_random_swaps=2, patience=60, seed=0):
    rng = np.random.default_rng(seed)
    n = instance.n

    # first swarm: random orders
    swarm = [rng.permutation(n).tolist() for _ in range(n_particles)]
    pbest = [p.copy() for p in swarm]
    pbest_cost = [total_cost(p, instance, dist) for p in swarm]

    i = pbest_cost.index(min(pbest_cost))
    gbest = pbest[i].copy()
    gbest_cost = pbest_cost[i]

    history = [gbest_cost]
    no_improve = 0

    for it in range(iterations):
        if no_improve >= patience:
            break
        improved = False

        for p in range(n_particles):
            new = swarm[p].copy()

            # 1) exploration: random swaps
            for _ in range(n_random_swaps):
                a, b = rng.integers(0, n, size=2)
                new[a], new[b] = new[b], new[a]

            # 2) towards pbest, 3) towards gbest
            new = move_towards(new, pbest[p], alfa, rng)
            new = move_towards(new, gbest, beta, rng)

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
        if improved:
            no_improve = 0
        else:
            no_improve += 1

    return {"solution": gbest,
            "best_fitness": gbest_cost,
            "history": history,
            "evaluations": n_particles * len(history)}
