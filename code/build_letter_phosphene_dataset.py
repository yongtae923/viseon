"""
build_letter_phosphene_dataset.py

Build a phosphene-map dataset from the trained letter model and EMNIST letters.
Each original image gets one phosphene map; indices/labels match the source npz.
Saves to data/processed/emnist_letters_phosphenes.npz.
Run from project root:  python code/build_letter_phosphene_dataset.py
"""
import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

import init_training
import training
from letter_training import _resolve_path, build_cfg, get_emnist_dataset


def _run_forward_and_collect(dataloader, forward_fn, models, cfg, device, desc="phosphenes"):
    """Run model over dataloader and collect phosphenes and labels."""
    phosphenes_list = []
    labels_list = []
    for batch in tqdm(dataloader, desc=desc, unit="batch"):
        image, label = batch
        with torch.no_grad():
            out = forward_fn(batch, models, cfg, to_cpu=True)
        phosphenes_list.append(out["phosphenes"])
        labels_list.append(label.cpu())
    phosphenes = torch.cat(phosphenes_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return phosphenes.numpy(), labels.numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Build phosphene map dataset from best letter model and EMNIST."
    )
    parser.add_argument("--npz", type=str, default="./data/processed/emnist_letters_upright.npz")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save-path", type=str, default="./Out/letter")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--out",
        type=str,
        default="./data/processed/emnist_letters_phosphenes.npz",
        help="Output npz path for phosphene dataset.",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Limit number of training samples (default: all).",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Limit number of test samples (default: all).",
    )
    args = parser.parse_args()

    args.learning_rate = 5e-4
    args.epochs = 16
    args.trainstats_per_epoch = 24
    args.validations_per_epoch = 8
    args.early_stop_criterium = 15
    args.regularization_weight = 0.5

    cfg = build_cfg(args)
    dataset_dict = get_emnist_dataset(cfg, _resolve_path(args.npz))
    models = init_training.get_models(cfg)
    training.load_models(models, cfg, prefix="best")

    for m in models.values():
        if isinstance(m, torch.nn.Module):
            m.eval()

    forward_fn = init_training.get_training_pipeline(cfg)["forward"]

    trainloader = torch.utils.data.DataLoader(
        dataset_dict["trainset"],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    valloader = torch.utils.data.DataLoader(
        dataset_dict["valset"],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    if args.max_train is not None:
        trainloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(
                dataset_dict["trainset"], range(min(args.max_train, len(dataset_dict["trainset"])))
            ),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )
    if args.max_test is not None:
        valloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(
                dataset_dict["valset"], range(min(args.max_test, len(dataset_dict["valset"])))
            ),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

    train_phosphenes, train_labels = _run_forward_and_collect(
        trainloader, forward_fn, models, cfg, cfg["device"], desc="train"
    )
    test_phosphenes, test_labels = _run_forward_and_collect(
        valloader, forward_fn, models, cfg, cfg["device"], desc="test"
    )

    out_path = _resolve_path(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        train_phosphenes=train_phosphenes,
        train_labels=train_labels,
        test_phosphenes=test_phosphenes,
        test_labels=test_labels,
    )
    print(f"Saved: {out_path}")
    print(f"  train_phosphenes {train_phosphenes.shape}, train_labels {train_labels.shape}")
    print(f"  test_phosphenes  {test_phosphenes.shape}, test_labels  {test_labels.shape}")
    print("  Labels 0--25 = a--z; indices match original emnist_letters_upright.npz.")


if __name__ == "__main__":
    main()
