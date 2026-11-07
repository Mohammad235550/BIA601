Author:noor
# CVRP Optimization — GA / PSO / ACO / DP Baseline

Compares three metaheuristics on the **Capacitated Vehicle Routing Problem
(CVRP)**: route a fleet of vehicles from a single depot to a set of
customers, respecting a per-vehicle capacity limit, minimizing total travel
distance.

- **Genetic Algorithm (GA)** — order-crossover + swap mutation over a
  permutation of customers.
- **Particle Swarm Optimization (PSO)** — discrete swap-sequence encoding:
  a few random swaps per round for exploration, then a particle's velocity
  is a list of swap operators generated from pbest and gbest, applied
  probabilistically to the current permutation.
- **Ant Colony Optimization (ACO)** — ants build tours using pheromone +
  distance heuristic, pheromone updated by tour quality.
- **Dynamic Programming (Held-Karp)** — exact baseline, Small instance only,
  solves the single-vehicle TSP (capacity ignored) to give a true lower
  bound for comparison.
- **Exact CVRP DP** (`exact_cvrp` in `src/dp_baseline.py`) — Small instance
  only: Held-Karp route cost for every customer subset + set-partitioning DP
  over capacity-feasible routes. Gives the true optimum *with* the capacity
  constraint, so we can check whether the metaheuristics reach it.

All three metaheuristics share the same solution representation (a
permutation of customers, split into capacity-feasible routes — see
`src/problem.py`), so their results are directly comparable.

## Setup

```bash
cd vrp_project
pip install -r requirements.txt
```

Tested on Python 3.12. Only `numpy` and `matplotlib` — no other dependencies.

## Running

```bash
python main.py --runs 10
```

- `--runs N`: how many repetitions per algorithm per problem size (used to
  measure result stability via mean/std). Default 10.
- `--out DIR`: output directory. Default `results/`.

This generates three CVRP instances (Small = 10 customers, Medium = 30,
Large = 60), runs GA/PSO/ACO `N` times each on every instance (plus the
two exact DP baselines once each, on Small only), and writes:

```
results/
  raw_runs.csv        # every individual run: size, algorithm, run #, best_fitness, time_sec, evaluations
  convergence.csv     # iteration / evaluations -> mean, std, min, max best_fitness over all runs
  summary_stats.csv   # mean/std/min/max fitness, mean time, mean evaluations, peak memory (MB)
  plots/
    convergence_Small.png / _Medium.png / _Large.png   # Iteration -> Best Fitness (mean of all runs, band = best..worst run)
    convergence_*_evals.png                             # same, x-axis = fitness evaluations (fair budget view)
    quality_comparison.png                              # mean solution quality per algorithm/size
    time_comparison.png                                 # mean execution time per algorithm/size
```

Peak memory is measured with `tracemalloc` in a separate run (same seed as
run 0), so it does not slow down the timed runs.

Instance generation is seeded (see `src/experiment.py: SIZES`), and every
algorithm run uses a fixed, deterministic seed — running `main.py` again
reproduces identical results.

## Correctness checks

```bash
python verify.py
```

Checks that (1) the exact CVRP DP and Held-Karp match a brute-force search
on 15 random 6-customer instances, (2) the PSO swap sequence turns a route
exactly into its target, and (3) every algorithm returns a valid solution
(each customer visited once, no route over capacity) whose reported fitness
matches the recomputed total distance.

## Project layout

```
src/
  problem.py       # CVRP instance generation, solution decoding, cost evaluation
  ga.py            # Genetic Algorithm
  pso.py           # Particle Swarm Optimization
  aco.py           # Ant Colony Optimization
  dp_baseline.py   # Held-Karp exact TSP baseline + exact CVRP DP (Small only)
  experiment.py    # experiment runner across sizes/algorithms/repetitions
  plotting.py      # convergence + comparison charts
main.py            # CLI entry point
verify.py          # correctness checks (brute force, PSO operators, feasibility)
requirements.txt   # numpy, matplotlib
```

## Tuning

Hyperparameters (population size, iterations, mutation rate, ACO
alpha/beta/evaporation, PSO alfa/beta swap probabilities and n_random_swaps,
stopping patience) are the keyword defaults on `run_ga`, `run_pso`, and `run_aco` in `src/`. Problem
sizes and vehicle capacity are set in `SIZES` inside `src/experiment.py`.
