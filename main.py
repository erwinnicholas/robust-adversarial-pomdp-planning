import argparse
import time

import run_performance_evaluation
import run_adversary_analysis
import run_ablation_study
import run_scenario_simulation


# -----------------------------
# Experiment registry
# -----------------------------

EXPERIMENTS = {
    "performance": run_performance_evaluation.main,
    "adversary": run_adversary_analysis.main,
    "ablation": run_ablation_study.main,
    "scenario": run_scenario_simulation.main,
}


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="RIBDP experiment orchestrator"
    )

    parser.add_argument(
        "--run",
        type=str,
        default="all",
        choices=["all"] + list(EXPERIMENTS.keys()),
        help="Which experiment to run"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed"
    )

    parser.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable caching / fast execution mode"
    )

    parser.add_argument(
        "--use-torch",
        action="store_true",
        help="Use torch for numeric kernels (log/exp); optional GPU support"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device to use if --use-torch is enabled"
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort immediately if any experiment fails"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


# -----------------------------
# Runner
# -----------------------------

def run_experiment(name, fn, args):
    print(f"\n=== Running: {name} ===")
    start = time.time()

    try:
        fn(
            seed=args.seed,
            fast=args.fast,
            use_torch=args.use_torch,
            device=args.device,
            verbose=args.verbose,
        )
    except TypeError:
        # Backward compatibility: experiment script does not accept flags yet
        fn()
    except Exception as e:
        print(f"[ERROR] {name} failed: {e}")
        if args.fail_fast:
            raise
    else:
        elapsed = time.time() - start
        print(f"[OK] {name} completed in {elapsed:.2f}s")


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()

    print("========================================")
    print("RIBDP Experiment Orchestrator")
    print(f"Seed       : {args.seed}")
    print(f"Run mode   : {args.run}")
    print(f"Fast mode  : {args.fast}")
    print(f"Use torch  : {args.use_torch}")
    print(f"Device     : {args.device}")
    print(f"Fail-fast  : {args.fail_fast}")
    print("========================================")

    start_time = time.time()

    if args.run == "all":
        for name, fn in EXPERIMENTS.items():
            run_experiment(name, fn, args)
    else:
        run_experiment(args.run, EXPERIMENTS[args.run], args)

    total_time = time.time() - start_time

    print("========================================")
    print(f"All requested experiments finished in {total_time:.2f}s")
    print("========================================")


if __name__ == "__main__":
    main()
