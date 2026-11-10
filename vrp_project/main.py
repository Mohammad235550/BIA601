# Author: Shaima Shakhashiro
"""
Main file: runs all the experiments and draws the charts.

    python main.py --runs 30
"""
import argparse
from src.experiment import run_all
from src.plotting import plot_convergence, plot_quality_comparison, plot_time_comparison, plot_boxplots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30, help="runs per algorithm and size")
    parser.add_argument("--out", type=str, default="results", help="output folder")
    args = parser.parse_args()

    print(f"Running experiments ({args.runs} runs per algorithm and size)...")
    runs, convergence, stats = run_all(n_runs=args.runs, out_dir=args.out)

    print("Drawing charts...")
    plots = args.out + "/plots"
    plot_convergence(convergence, out_dir=plots, stats=stats)
    plot_quality_comparison(stats, out_dir=plots)
    plot_time_comparison(stats, out_dir=plots)
    plot_boxplots(runs, out_dir=plots)

    print("\n=== Results ===")
    print(f"{'size':<12}{'algorithm':<26}{'mean':>10}{'std':>9}{'best':>10}{'time(s)':>9}{'evals':>8}")
    for r in stats:
        evals = "-" if r["mean_evaluations"] == "" else f"{r['mean_evaluations']:.0f}"
        print(f"{r['size']:<12}{r['algorithm']:<26}{r['mean_fitness']:>10.2f}{r['std_fitness']:>9.2f}"
              f"{r['min_fitness']:>10.2f}{r['mean_time_sec']:>9.2f}{evals:>8}")
    print(f"\nResults saved in '{args.out}/'")


if __name__ == "__main__":
    main()
