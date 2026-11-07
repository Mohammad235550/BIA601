#Author: Yousef Jamel Abuaishah
"""
Genetic Algorithm (GA) for CVRP.

- Chromosome: a permutation of the customers.
- Selection: tournament (pick k random individuals, keep the best).
- Crossover: Order Crossover (OX), the child is always a valid permutation.
- Mutation: swap mutation (swap two customers).
- Elitism: the best individual goes to the next generation unchanged.
- Stop: after `generations` or when the best does not improve for `patience` generations.
"""
import numpy as np
from .problem import total_cost


def tournament(population, costs, k, rng):
    picked = rng.integers(0, len(population), size=k)
    best = picked[np.argmin(costs[picked])]
    return population[best]


def order_crossover(parent_a, parent_b, rng):
    size = len(parent_a)
    start, end = sorted(rng.integers(0, size, size=2))
    child = [None] * size
    # 1) copy a slice from parent A
    child[start:end] = parent_a[start:end]
    # 2) fill the empty places with the remaining customers in parent B's order
    taken = set(child[start:end])
    remaining = [c for c in parent_b if c not in taken]
    k = 0
    for i in range(size):
        if child[i] is None:
            child[i] = remaining[k]
            k += 1
    return child


def swap_mutation(route, rate, rng):
    route = route.copy()
    for i in range(len(route)):
        if rng.random() < rate:
            j = rng.integers(0, len(route))
            route[i], route[j] = route[j], route[i]
    return route


def run_ga(instance, dist, pop_size=60, generations=300, crossover_rate=0.9,
           mutation_rate=0.02, tournament_k=3, patience=60, seed=0):
    rng = np.random.default_rng(seed)

    # initial population: random permutations
    population = [rng.permutation(instance.n).tolist() for _ in range(pop_size)]

    best_route = None
    best_cost = float("inf")
    no_improve = 0
    history = []          # best cost after each generation (for the convergence curve)

    for gen in range(generations):
        costs = np.array([total_cost(ind, instance, dist) for ind in population])

        # update the best solution
        i = np.argmin(costs)
        if costs[i] < best_cost:
            best_cost = float(costs[i])
            best_route = population[i].copy()
            no_improve = 0
        else:
            no_improve += 1
        history.append(best_cost)

        if no_improve >= patience:
            break

        # build the next generation
        new_population = [best_route.copy()]          # elitism
        while len(new_population) < pop_size:
            parent_a = tournament(population, costs, tournament_k, rng)
            parent_b = tournament(population, costs, tournament_k, rng)
            if rng.random() < crossover_rate:
                child = order_crossover(parent_a, parent_b, rng)
            else:
                child = parent_a.copy()
            child = swap_mutation(child, mutation_rate, rng)
            new_population.append(child)
        population = new_population

    return {"route": best_route,
            "best_fitness": best_cost,
            "history": history,
            "evaluations": pop_size * len(history)}
