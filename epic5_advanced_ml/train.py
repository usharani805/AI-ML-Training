
"""Command-line interface for the Epic 5 ML pipeline."""

import argparse
import sys
from pathlib import Path

from epic5_advanced_ml.project_final_ml_pipeline import (
    MLPipelineRunner,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run the final Epic 5 ML pipeline."
    )

    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        required=True,
        help="Choose classification or regression.",
    )

    parser.add_argument(
        "--model",
        choices=["best"],
        default="best",
        help="Load the previously saved best model.",
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Train candidates and tune the top two models.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory containing or storing model results.",
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
        or f"epic5_advanced_ml/outputs/{args.task}"
    )

    runner = MLPipelineRunner(
        task=args.task,
        output_dir=output_dir,
    )

    try:
        if args.tune:
            leaderboard = runner.run_all_models()
            leaderboard = runner.tune_top_candidates(n=2)

            model_path = runner.save_best(output_dir)

        else:
            runner.load_best(output_dir)

            leaderboard = runner.leaderboard
            model_path = output_dir / "model.joblib"

        print("\nFinal Model Leaderboard:")
        if leaderboard.empty:
            print("No leaderboard data is available.")
        else:
            print(leaderboard.to_string(index=False))

        print("\nBest Model:")
        print(runner.best_model_name)

        print("\nSaved Model:")
        print(model_path)

        if not args.tune:
            print("\nLoaded the saved model without retraining.")

        print("\nResults Directory:")
        print(output_dir)

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())