"""
fill_letter_validation.py

Fill only the validation_results CSVs for a completed letter training run.
Uses saved best checkpoints; no training. Run from project root:
  python code/fill_letter_validation.py
"""
import argparse
import os

import init_training
import training

# Reuse letter_training helpers and defaults
from letter_training import _resolve_path, build_cfg, get_emnist_dataset


def main():
    parser = argparse.ArgumentParser(description="Save validation_results for completed letter run.")
    parser.add_argument("--npz", type=str, default="./data/processed/emnist_letters_upright.npz")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save-path", type=str, default="./Out/letter")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    # Minimal args for build_cfg (only used fields)
    args.learning_rate = 5e-4
    args.epochs = 16
    args.trainstats_per_epoch = 24
    args.validations_per_epoch = 8
    args.early_stop_criterium = 15
    args.regularization_weight = 0.5

    cfg = build_cfg(args)
    dataset = get_emnist_dataset(cfg, _resolve_path(args.npz))
    models = init_training.get_models(cfg)
    training.load_models(models, cfg, prefix="best")
    training_pipeline = init_training.get_training_pipeline(cfg)

    print("Running validation and saving results...")
    output, performance = training.get_validation_results(
        dataset, models, training_pipeline, cfg
    )
    training.save_validation_results(None, performance, cfg)
    print("Done. validation_performance.csv and performance_summary.csv written.")


if __name__ == "__main__":
    main()
