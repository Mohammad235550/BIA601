# Author: Mohammad Nour
"""
Ant Colony Optimization (ACO) for CVRP.

Each ant builds a tour over the customers. The next customer is chosen
with probability proportional to  tau^alpha * eta^beta
  tau = pheromone on the edge, eta = 1 / distance (closer is better).
The tour is then cut into routes with the same Greedy Split.
After each iteration: evaporation (tau = (1 - rho) * tau), then every ant
adds Q / L on the edges it used (L = length of its solution).
Stop: after `iterations` or when the best does not improve for `patience` iterations.
"""
import numpy as np
from .problem import total_cost


def build_tour(n, pheromone, heuristic, alpha, beta, rng):
    not_visited = set(range(n))
    current = int(rng.integers(0, n))      # random start customer
    tour = [current]
    not_visited.discard(current)

    for _ in range(n - 1):
        candidates = list(not_visited)
        weights = (pheromone[current, candidates] ** alpha) * (heuristic[current, candidates] ** beta)
        if weights.sum() > 0:
            probs = weights / weights.sum()
        else:
            probs = np.ones(len(candidates)) / len(candidates)
        nxt = candidates[rng.choice(len(candidates), p=probs)]
        tour.append(nxt)
        not_visited.discard(nxt)
        current = nxt
    return tour


def run_aco(instance, dist, n_ants=30, iterations=200, alpha=1.0, beta=3.0,
            evaporation=0.5, q=100.0, patience=60, seed=0):
    rng = np.random.default_rng(seed)
    n = instance.n

    # heuristic eta = 1 / distance between customers (index 0 is the depot, skip it)
    customer_dist = dist[1:, 1:]
    heuristic = np.zeros((n, n))
    heuristic[customer_dist > 0] = 1.0 / customer_dist[customer_dist > 0]

    pheromone = np.ones((n, n))
    best_tour = None
    best_cost = float("inf")
    no_improve = 0
    history = []

    for it in range(iterations):
        tours = [build_tour(n, pheromone, heuristic, alpha, beta, rng) for _ in range(n_ants)]
        costs = np.array([total_cost(t, instance, dist) for t in tours])

        # update the best solution
        i = np.argmin(costs)
        if costs[i] < best_cost:
            best_cost = float(costs[i])
            best_tour = tours[i].copy()
            no_improve = 0
        else:
            no_improve += 1
        history.append(best_cost)

        # pheromone update: evaporation, then deposit Q / L
        pheromone *= (1 - evaporation)
        for tour, cost in zip(tours, costs):
            deposit = q / cost
            for a, b in zip(tour[:-1], tour[1:]):
                pheromone[a, b] += deposit
                pheromone[b, a] += deposit

        if no_improve >= patience:
            break

    return {"tour": best_tour,
            "best_fitness": best_cost,
            "history": history,
            "evaluations": n_ants * len(history)}
