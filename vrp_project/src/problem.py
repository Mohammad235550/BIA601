# Author: Rashed Bahnasi
# Author: zenab alsoutary
"""
CVRP problem: generating the data, the distance matrix and the fitness function.

A solution is a list of customers in visiting order, for example [3, 0, 7, ...].
We cut this list into vehicle routes (Greedy Split): we keep adding customers
to the current vehicle, and when the next customer does not fit we start a
new vehicle. So every solution is always feasible.

GA, PSO and ACO all use this same representation.
"""
import math
import numpy as np


class CVRPInstance:
    def __init__(self, name, depot, coords, demands, capacity):
        self.name = name          # "Small", "Medium", "Large" or "XLarge"
        self.depot = depot        # (x, y) of the depot
        self.coords = coords      # (x, y) of every customer
        self.demands = demands    # demand of every customer
        self.capacity = capacity  # capacity of one vehicle (Q)
        self.n = len(coords)      # number of customers


def generate_instance(name, n_customers, capacity=50.0, seed=0):
    """Depot and customers at random points in a 100 x 100 square, demands from 1 to 9."""
    rng = np.random.default_rng(seed)
    depot = rng.uniform(0, 100, size=2)
    coords = rng.uniform(0, 100, size=(n_customers, 2))
    demands = rng.integers(1, 10, size=n_customers).astype(float)
    return CVRPInstance(name, depot, coords, demands, capacity)


def build_distance_matrix(instance):
    """Euclidean distances. Point 0 is the depot, point i+1 is customer i."""
    points = [instance.depot] + list(instance.coords)
    n = len(points)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            dist[i, j] = math.sqrt(dx * dx + dy * dy)
    return dist


def split_into_routes(order, instance):
    """Greedy Split: cut the customer order into routes that respect the capacity."""
    routes = []
    current = []
    load = 0
    for customer in order:
        demand = instance.demands[customer]
        if load + demand > instance.capacity and len(current) > 0:
            routes.append(current)      # this vehicle is full -> start a new one
            current = []
            load = 0
        current.append(customer)
        load += demand
    if len(current) > 0:
        routes.append(current)
    return routes


def route_distance(route, dist):
    """Length of one route: depot -> customers -> depot."""
    if len(route) == 0:
        return 0.0
    total = dist[0, route[0] + 1]                       # depot -> first customer
    for k in range(len(route) - 1):
        total += dist[route[k] + 1, route[k + 1] + 1]   # customer -> next customer
    total += dist[route[-1] + 1, 0]                     # last customer -> depot
    return total


def total_cost(order, instance, dist):
    """Fitness = total distance of all routes (smaller is better)."""
    routes = split_into_routes(order, instance)
    return sum(route_distance(r, dist) for r in routes)


def evaluate(order, instance, dist):
    """Returns (total distance, routes, number of vehicles)."""
    routes = split_into_routes(order, instance)
    return total_cost(order, instance, dist), routes, len(routes)
