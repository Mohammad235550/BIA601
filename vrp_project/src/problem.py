"""
CVRP problem: data, distances and the fitness function.

A solution is a list of customers in visiting order, e.g. [3, 0, 7, ...].
We cut this list into vehicle routes: we add customers to the current
vehicle until the next one would exceed the capacity, then we start a
new vehicle (Greedy Split). So every solution is always feasible.

The same representation is used by GA, PSO and ACO.
"""
import numpy as np


class CVRPInstance:
    def __init__(self, name, depot, coords, demands, capacity):
        self.name = name          # "Small", "Medium" or "Large"
        self.depot = depot        # (x, y) of the depot
        self.coords = coords      # (x, y) of every customer
        self.demands = demands    # demand of every customer
        self.capacity = capacity  # capacity of one vehicle (Q)
        self.n = len(coords)      # number of customers


def generate_instance(name, n_customers, capacity=50.0, seed=0, grid_size=100.0):
    """Random depot and customers in a 100 x 100 square, demands from 1 to 9."""
    rng = np.random.default_rng(seed)
    depot = rng.uniform(0, grid_size, size=2)
    coords = rng.uniform(0, grid_size, size=(n_customers, 2))
    demands = rng.integers(1, 10, size=n_customers).astype(float)
    return CVRPInstance(name, depot, coords, demands, capacity)


def build_distance_matrix(instance):
    """Euclidean distances. Index 0 = depot, index i+1 = customer i."""
    points = np.vstack([instance.depot, instance.coords])
    n = len(points)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.sqrt(((points[i] - points[j]) ** 2).sum())
    return dist


def split_into_routes(perm, instance):
    """Greedy Split: cut the customer order into routes that respect the capacity."""
    routes = []
    current = []
    load = 0.0
    for customer in perm:
        demand = instance.demands[customer]
        if load + demand > instance.capacity and current:
            routes.append(current)   # this vehicle is full, start a new one
            current = []
            load = 0.0
        current.append(customer)
        load += demand
    if current:
        routes.append(current)
    return routes


def route_distance(route, dist):
    """Length of one route: depot -> customers -> depot."""
    if not route:
        return 0.0
    total = dist[0, route[0] + 1]
    for a, b in zip(route[:-1], route[1:]):
        total += dist[a + 1, b + 1]
    total += dist[route[-1] + 1, 0]
    return total


def total_cost(perm, instance, dist):
    """Fitness: total distance of all routes (smaller is better)."""
    routes = split_into_routes(perm, instance)
    return sum(route_distance(r, dist) for r in routes)


def evaluate(perm, instance, dist):
    """Returns (total distance, routes, number of vehicles)."""
    routes = split_into_routes(perm, instance)
    total = sum(route_distance(r, dist) for r in routes)
    return total, routes, len(routes)
