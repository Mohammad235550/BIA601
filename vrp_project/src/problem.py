# Author: Rashed Bahnasi
# Author: zenab alsoutary
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
    def __init__(self, name, depot, coords, demands, capacity, rounded=False):
        self.name = name          # "Small", "Medium", "Large" or a benchmark name
        self.depot = depot        # (x, y) of the depot
        self.coords = coords      # (x, y) of every customer
        self.demands = demands    # demand of every customer
        self.capacity = capacity  # capacity of one vehicle (Q)
        self.n = len(coords)      # number of customers
        self.rounded = rounded    # True for CVRPLIB files: distances rounded to integers


def generate_instance(name, n_customers, capacity=50.0, seed=0, grid_size=100.0):
    """Random depot and customers in a 100 x 100 square, demands from 1 to 9."""
    rng = np.random.default_rng(seed)
    depot = rng.uniform(0, grid_size, size=2)
    coords = rng.uniform(0, grid_size, size=(n_customers, 2))
    demands = rng.integers(1, 10, size=n_customers).astype(float)
    return CVRPInstance(name, depot, coords, demands, capacity)


def load_cvrplib(path):
    """Read a benchmark file from CVRPLIB (.vrp format, EUC_2D).
    Node 1 in the file is the depot, the other nodes are the customers."""
    with open(path) as f:
        lines = [line.strip() for line in f]
    name = capacity = None
    coords, demands = {}, {}
    section = None
    for line in lines:
        if not line or line == "EOF":
            continue
        if line.startswith("NAME"):
            name = line.split(":")[1].strip()
        elif line.startswith("CAPACITY"):
            capacity = float(line.split(":")[1])
        elif line.startswith("NODE_COORD_SECTION"):
            section = "coords"
        elif line.startswith("DEMAND_SECTION"):
            section = "demands"
        elif line.startswith("DEPOT_SECTION"):
            section = "depot"
        elif section == "coords":
            node, x, y = line.split()
            coords[int(node)] = (float(x), float(y))
        elif section == "demands":
            node, q = line.split()
            demands[int(node)] = float(q)
    customers = sorted(k for k in coords if k != 1)
    return CVRPInstance(name,
                        depot=np.array(coords[1]),
                        coords=np.array([coords[k] for k in customers]),
                        demands=np.array([demands[k] for k in customers]),
                        capacity=capacity,
                        rounded=True)


def build_distance_matrix(instance):
    """Euclidean distances. Index 0 = depot, index i+1 = customer i.
    For CVRPLIB files the distance is rounded to the nearest integer
    (the official EUC_2D rule), so our costs are comparable to the published ones."""
    points = np.vstack([instance.depot, instance.coords])
    n = len(points)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.sqrt(((points[i] - points[j]) ** 2).sum())
            if instance.rounded:
                dist[i, j] = np.floor(dist[i, j] + 0.5)
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
