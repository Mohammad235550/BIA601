# Author: Hasan safwan aksimi
"""
Dynamic Programming baselines (only for the Small instance, n <= 15).

1) held_karp: exact TSP with ONE vehicle and NO capacity limit.
   No CVRP solution can be shorter, so its result is a lower bound.
   Complexity O(n^2 * 2^n).

2) exact_cvrp: the real optimum WITH the capacity limit, in two steps:
   a) for every group of customers: the shortest route depot -> group -> depot
   b) split all customers into groups (each group's demand <= Q)
      with the smallest total distance.

A group of customers is stored as one number (bitmask): bit i = 1 means
customer i is in the group. Example with 4 customers: 0b0101 = customers 0 and 2.
"""
INF = float("inf")


def has(group, c):
    """Is customer c in the group?"""
    return (group >> c) & 1 == 1


def add(group, c):
    """The group with customer c added."""
    return group + 2 ** c


def is_part_of(small, big):
    """Is every customer of `small` also in `big`?"""
    return small | big == big


def shortest_paths(dist, n):
    """path[group][last] = shortest path that starts at the depot, visits all
    customers in `group` and ends at customer `last`."""
    path = [[INF] * n for _ in range(2 ** n)]
    parent = [[-1] * n for _ in range(2 ** n)]
    for c in range(n):
        path[2 ** c][c] = dist[0, c + 1]             # depot -> c

    for group in range(1, 2 ** n):
        for last in range(n):
            if path[group][last] == INF:
                continue
            for nxt in range(n):
                if has(group, nxt):                  # already visited
                    continue
                new_group = add(group, nxt)
                cost = path[group][last] + dist[last + 1, nxt + 1]
                if cost < path[new_group][nxt]:
                    path[new_group][nxt] = cost
                    parent[new_group][nxt] = last
    return path, parent


def rebuild_order(parent, group, last):
    """Follow the parents backwards to get the visiting order."""
    order = []
    while last != -1:
        order.append(last)
        prev = parent[group][last]
        group = group - 2 ** last                    # remove `last` from the group
        last = prev
    return order[::-1]


def held_karp(dist):
    """Exact TSP (one vehicle, no capacity). Returns (cost, order)."""
    n = len(dist) - 1
    if n > 15:
        raise ValueError("Held-Karp is only used for small instances (n <= 15).")
    path, parent = shortest_paths(dist, n)

    everyone = 2 ** n - 1
    best_cost, best_last = INF, -1
    for last in range(n):
        cost = path[everyone][last] + dist[last + 1, 0]   # back to the depot
        if cost < best_cost:
            best_cost, best_last = cost, last
    return float(best_cost), rebuild_order(parent, everyone, best_last)


def exact_cvrp(dist, demands, capacity):
    """Exact CVRP optimum (with capacity). Returns (cost, routes)."""
    n = len(dist) - 1
    if n > 15:
        raise ValueError("exact_cvrp is only used for small instances (n <= 15).")
    path, parent = shortest_paths(dist, n)

    # step a) best single route for every group, and the group's total demand
    route_cost = [INF] * 2 ** n
    route_last = [-1] * 2 ** n
    load = [0.0] * 2 ** n
    for group in range(1, 2 ** n):
        for last in range(n):
            cost = path[group][last] + dist[last + 1, 0]
            if cost < route_cost[group]:
                route_cost[group], route_last[group] = cost, last
        load[group] = sum(demands[c] for c in range(n) if has(group, c))

    # step b) best[group] = cheapest way to serve all customers in `group` with several routes.
    # We try every smaller group `sub` inside `group` as one route, and the rest is
    # already solved. To not count the same split twice, `sub` must contain the
    # first customer of `group`.
    best = [INF] * 2 ** n
    choice = [0] * 2 ** n
    best[0] = 0.0
    for group in range(1, 2 ** n):
        first = 0
        while not has(group, first):
            first += 1
        for sub in range(1, group + 1):
            if not is_part_of(sub, group) or not has(sub, first):
                continue
            if load[sub] > capacity:
                continue
            cost = best[group - sub] + route_cost[sub]
            if cost < best[group]:
                best[group], choice[group] = cost, sub

    # rebuild the routes
    routes = []
    group = 2 ** n - 1
    while group > 0:
        sub = choice[group]
        routes.append(rebuild_order(parent, sub, route_last[sub]))
        group = group - sub
    return float(best[2 ** n - 1]), routes
