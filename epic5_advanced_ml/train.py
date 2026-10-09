"""Command-line interface for the Epic 5 ML pipeline."""

import argparse

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
        default="best",
        choices=["best"],
        help="Select the best model from the leaderboard.",
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Tune the top two candidate models.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the saved model and results.",
    )

    args = parser.parse_args()

    runner = MLPipelineRunner(
        task=args.task,
        output_dir=args.output_dir,
    )

    leaderboard = runner.run_all_models()

    print("\nModel Leaderboard:")
    print(leaderboard.to_string(index=False))

    if args.tune:
        runner.tune_top_candidates(n=2)

    output_dir = args.output_dir or str(runner.output_dir)
    metrics = runner.save_best(output_dir)

    print("\nFinal Model Metrics:")
    print(metrics)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()