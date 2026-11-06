#Author:Hasan safwan aksimi
"""
Dynamic Programming baselines (Small instance only, n <= 15).

1) held_karp: exact TSP with one vehicle and NO capacity limit.
   Its result is a lower bound: no CVRP solution can be shorter.
   Complexity O(n^2 * 2^n).

2) exact_cvrp: the real optimum WITH the capacity limit, in two steps:
   a) for every subset S of customers: shortest route depot -> S -> depot
   b) split all customers into subsets with demand <= Q at minimum total cost
      (set partitioning).

A subset of customers is stored as a bitmask: bit i = 1 means customer i is in it.
"""
INF = float("inf")


def shortest_paths_from_depot(dist, n):
    """path[mask][last] = shortest path depot -> all customers in mask, ending at `last`."""
    full = 1 << n
    path = [[INF] * n for _ in range(full)]
    parent = [[-1] * n for _ in range(full)]
    for c in range(n):
        path[1 << c][c] = dist[0, c + 1]

    for mask in range(1, full):
        for last in range(n):
            if path[mask][last] == INF:
                continue
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue                      # already visited
                new_mask = mask | (1 << nxt)
                cost = path[mask][last] + dist[last + 1, nxt + 1]
                if cost < path[new_mask][nxt]:
                    path[new_mask][nxt] = cost
                    parent[new_mask][nxt] = last
    return path, parent


def rebuild_order(parent, mask, last):
    """Follow the parents back to get the visiting order."""
    order = []
    while last != -1:
        order.append(last)
        prev = parent[mask][last]
        mask &= ~(1 << last)
        last = prev
    return order[::-1]


def held_karp(dist):
    """Exact TSP (one vehicle, no capacity). Returns (cost, order)."""
    n = dist.shape[0] - 1
    if n > 15:
        raise ValueError("Held-Karp is only used for small instances (n <= 15).")
    path, parent = shortest_paths_from_depot(dist, n)

    everyone = (1 << n) - 1
    best_cost, best_last = INF, -1
    for last in range(n):
        cost = path[everyone][last] + dist[last + 1, 0]   # go back to the depot
        if cost < best_cost:
            best_cost, best_last = cost, last
    return float(best_cost), rebuild_order(parent, everyone, best_last)


def exact_cvrp(dist, demands, capacity):
    """Exact CVRP optimum (with capacity). Returns (cost, routes)."""
    n = dist.shape[0] - 1
    if n > 15:
        raise ValueError("exact_cvrp is only used for small instances (n <= 15).")
    full = 1 << n
    path, parent = shortest_paths_from_depot(dist, n)

    # step a) cost of one route that serves exactly the customers in `mask`
    route_cost = [INF] * full
    route_last = [-1] * full
    load = [0.0] * full
    for mask in range(1, full):
        for last in range(n):
            cost = path[mask][last] + dist[last + 1, 0]
            if cost < route_cost[mask]:
                route_cost[mask], route_last[mask] = cost, last
        load[mask] = sum(demands[c] for c in range(n) if mask & (1 << c))

    # step b) best[mask] = cheapest way to serve all customers in `mask`
    best = [INF] * full
    choice = [0] * full
    best[0] = 0.0
    for mask in range(1, full):
        first = mask & -mask              # lowest customer in mask must be in the first route
        sub = mask
        while sub:                        # try every sub-route of mask
            if (sub & first) and load[sub] <= capacity:
                cost = best[mask ^ sub] + route_cost[sub]
                if cost < best[mask]:
                    best[mask], choice[mask] = cost, sub
            sub = (sub - 1) & mask

    # rebuild the routes
    routes = []
    mask = full - 1
    while mask:
        sub = choice[mask]
        routes.append(rebuild_order(parent, sub, route_last[sub]))
        mask ^= sub
    return float(best[full - 1]), routes
