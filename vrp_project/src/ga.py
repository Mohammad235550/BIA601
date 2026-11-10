# Author: Yousef Jamel Abuaishah
"""
Genetic Algorithm (GA) for CVRP.

- Chromosome: an order (permutation) of the customers.
- Selection: tournament (pick 3 random individuals, keep the best one).
- Crossover: Order Crossover (OX), the child is always a valid order.
- Mutation: swap mutation (swap two customers).
- Elitism: the best individual goes to the next generation unchanged.
- Stop: after `generations`, or when the best does not improve for `patience` generations.
"""
import numpy as np
from .problem import total_cost


def tournament(population, costs, k, rng):
    """Pick k random individuals and return the best of them."""
    picked = rng.integers(0, len(population), size=k)
    best = picked[0]
    for i in picked:
        if costs[i] < costs[best]:
            best = i
    return population[best]


def order_crossover(parent_a, parent_b, rng):
    """OX: copy a slice from parent A, fill the rest in parent B's order."""
    size = len(parent_a)
    start, end = sorted(rng.integers(0, size, size=2))
    middle = parent_a[start:end]
    rest = [c for c in parent_b if c not in middle]
    return rest[:start] + middle + rest[start:]


def swap_mutation(order, rate, rng):
    """Every position is swapped with a random position with probability `rate`."""
    order = order.copy()
    for i in range(len(order)):
        if rng.random() < rate:
            j = rng.integers(0, len(order))
            order[i], order[j] = order[j], order[i]
    return order


def run_ga(instance, dist, pop_size=60, generations=300, crossover_rate=0.9,
           mutation_rate=0.02, tournament_k=3, patience=60, seed=0):
    rng = np.random.default_rng(seed)

    # first generation: random orders
    population = [rng.permutation(instance.n).tolist() for _ in range(pop_size)]

    best = None
    best_cost = float("inf")
    no_improve = 0
    history = []      # best cost after every generation (for the convergence curve)

    for gen in range(generations):
        costs = [total_cost(ind, instance, dist) for ind in population]

        # did we find a better solution?
        i = costs.index(min(costs))
        if costs[i] < best_cost:
            best_cost = float(costs[i])
            best = population[i].copy()
            no_improve = 0
        else:
            no_improve += 1
        history.append(best_cost)

        if no_improve >= patience:
            break

        # next generation
        new_population = [best.copy()]          # elitism
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

    return {"solution": best,
            "best_fitness": best_cost,
            "history": history,
            "evaluations": pop_size * len(history)}
