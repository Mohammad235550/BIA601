# Author: Mohammad Nour
"""
Ant Colony Optimization (ACO) for CVRP.

Each ant builds an order of all customers. The next customer is chosen
with probability proportional to  tau^alpha * eta^beta
  tau = pheromone on the edge, eta = 1 / distance (closer is better).
The order is then cut into routes with the same Greedy Split.
After every iteration: evaporation (tau = (1 - rho) * tau), then every ant
adds Q / L on the edges it used (L = total distance of its solution).
Stop: after `iterations`, or when the best does not improve for `patience` iterations.
"""
import numpy as np
from .problem import total_cost


def build_tour(n, pheromone, heuristic, alpha, beta, rng):
    """One ant visits all customers, starting from a random customer."""
    current = int(rng.integers(0, n))
    tour = [current]
    not_visited = [c for c in range(n) if c != current]

    while len(not_visited) > 0:
        weights = pheromone[current, not_visited] ** alpha * heuristic[current, not_visited] ** beta
        probs = weights / weights.sum()
        nxt = not_visited[rng.choice(len(not_visited), p=probs)]
        tour.append(nxt)
        not_visited.remove(nxt)
        current = nxt
    return tour


def run_aco(instance, dist, n_ants=30, iterations=200, alpha=1.0, beta=3.0,
            evaporation=0.5, q=100.0, patience=60, seed=0):
    rng = np.random.default_rng(seed)
    n = instance.n

    # heuristic eta = 1 / distance between two customers (index 0 of dist is the depot)
    heuristic = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                heuristic[i, j] = 1.0 / dist[i + 1, j + 1]

    pheromone = np.ones((n, n))
    best = None
    best_cost = float("inf")
    no_improve = 0
    history = []

    for it in range(iterations):
        tours = [build_tour(n, pheromone, heuristic, alpha, beta, rng) for _ in range(n_ants)]
        costs = [total_cost(t, instance, dist) for t in tours]

        # did we find a better solution?
        i = costs.index(min(costs))
        if costs[i] < best_cost:
            best_cost = float(costs[i])
            best = tours[i].copy()
            no_improve = 0
        else:
            no_improve += 1
        history.append(best_cost)

        # pheromone: evaporation, then every ant adds Q / L on its edges
        pheromone *= (1 - evaporation)
        for k in range(n_ants):
            deposit = q / costs[k]
            tour = tours[k]
            for a, b in zip(tour[:-1], tour[1:]):
                pheromone[a, b] += deposit
                pheromone[b, a] += deposit

        if no_improve >= patience:
            break

    return {"solution": best,
            "best_fitness": best_cost,
            "history": history,
            "evaluations": n_ants * len(history)}
