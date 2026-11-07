# Author: Shaima Shakhashiro
"""
Main file: runs all experiments and draws the charts.

    python main.py --runs 10
"""
import argparse
from src.experiment import run_all
from src.plotting import plot_convergence, plot_quality_comparison, plot_time_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10, help="runs per algorithm and size")
    parser.add_argument("--out", type=str, default="results", help="output folder")
    args = parser.parse_args()

    print(f"Running experiments ({args.runs} runs per algorithm and size)...")
    runs, convergence, stats = run_all(n_runs=args.runs, out_dir=args.out)

    print("Drawing charts...")
    plot_convergence(convergence, out_dir=f"{args.out}/plots", stats=stats)
    plot_quality_comparison(stats, out_dir=f"{args.out}/plots")
    plot_time_comparison(stats, out_dir=f"{args.out}/plots")

    print("\n=== Results ===")
    print(f"{'size':<12}{'algorithm':<26}{'mean':>10}{'std':>9}{'best':>10}{'time(s)':>9}{'evals':>8}")
    for r in stats:
        evals = f"{r['mean_evaluations']:.0f}" if r["mean_evaluations"] != "" else "-"
        print(f"{r['size']:<12}{r['algorithm']:<26}{r['mean_fitness']:>10.2f}{r['std_fitness']:>9.2f}"
              f"{r['min_fitness']:>10.2f}{r['mean_time_sec']:>9.2f}{evals:>8}")
    for r in stats:
        if r["gap_to_optimum_pct"] != "":
            print(f"{r['size']}: {r['algorithm']} mean gap to optimum = {r['gap_to_optimum_pct']:.2f}%")
    print(f"\nResults saved in '{args.out}/'")


if __name__ == "__main__":
    main()
