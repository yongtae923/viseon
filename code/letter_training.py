"""
letter_training.py

Train a letter reconstruction model on the EMNIST letters dataset.
Run from project root:  python code/letter_training.py
"""
import argparse
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

import init_training
import training
from local_datasets import create_circular_mask

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(PROJECT_ROOT, path_value))


class EMNISTLettersNPZ(Dataset):
    """EMNIST letters dataset loaded from a preprocessed npz file."""

    def __init__(
        self,
        npz_path: str,
        train: bool = True,
        device: str = "cuda:0",
        imsize: Tuple[int, int] = (128, 128),
        circular_mask: bool = True,
    ):
        super().__init__()
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset not found: {npz_path}")

        data = np.load(npz_path)
        images_key = "train_images" if train else "test_images"
        labels_key = "train_labels" if train else "test_labels"

        self.images = torch.from_numpy(data[images_key]).float().unsqueeze(1) / 255.0
        self.labels = torch.from_numpy(data[labels_key]).long()
        self.device = device
        self.resize = T.Resize(imsize)
        self.normalizer = T.Normalize(mean=[0.459], std=[0.227])

        if circular_mask:
            self._mask = create_circular_mask(*imsize).view(1, *imsize)
        else:
            self._mask = None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        x = self.resize(self.images[idx])
        x = self.normalizer(x)
        if self._mask is not None:
            x = x * self._mask
        y = self.labels[idx]
        return x.to(self.device), y.to(self.device)


def build_cfg(args) -> Dict:
    return {
        # general
        "use_deterministic_algorithms": False,
        "batch_size": args.batch_size,
        "device": args.device,
        "gpu": 0,
        "save_path": _resolve_path(args.save_path),
        "save_output": ["phosphenes", "stimulation", "reconstruction"],
        # simulator
        "base_config": _resolve_path("./_config/exp2/simulator_config.yaml"),
        "phosphene_map": _resolve_path("./_config/phosphene_maps/DefaultCoordinateMap_1000_phosphenes.pickle"),
        # model
        "model_architecture": "end-to-end-autoencoder",
        "in_channels": 1,
        "n_electrodes": 1000,
        "output_scaling": 128.0e-6,
        "output_steps": "None",
        "out_channels": 1,
        "encoder_out_activation": "relu",
        "decoder_out_activation": "sigmoid",
        # optimization
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "trainstats_per_epoch": args.trainstats_per_epoch,
        "validations_per_epoch": args.validations_per_epoch,
        "early_stop_criterium": args.early_stop_criterium,
        # pipeline
        "pipeline": "unconstrained-image-autoencoder",
        "regularization_weight": args.regularization_weight,
        "print_epoch_timing": True,
        "log_stimulation_histogram": False,
    }


def get_emnist_dataset(cfg: Dict, npz_path: str):
    trainset = EMNISTLettersNPZ(
        npz_path=npz_path,
        train=True,
        device=cfg["device"],
        imsize=(128, 128),
        circular_mask=True,
    )
    valset = EMNISTLettersNPZ(
        npz_path=npz_path,
        train=False,
        device=cfg["device"],
        imsize=(128, 128),
        circular_mask=True,
    )

    trainloader = DataLoader(trainset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    valloader = DataLoader(valset, batch_size=cfg["batch_size"], shuffle=False, drop_last=True)
    example_batch = next(iter(valloader))
    cfg["circular_mask"] = trainset._mask.to(cfg["device"])

    return {
        "trainset": trainset,
        "valset": valset,
        "trainloader": trainloader,
        "valloader": valloader,
        "example_batch": example_batch,
    }


def main():
    parser = argparse.ArgumentParser(description="Train letter reconstruction on EMNIST npz.")
    parser.add_argument(
        "--npz",
        type=str,
        default="./data/processed/emnist_letters_upright.npz",
        help="Path to preprocessed EMNIST letters npz file.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--save-path", type=str, default="./Out/letter", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--trainstats-per-epoch", type=int, default=24)
    parser.add_argument("--validations-per-epoch", type=int, default=8)
    parser.add_argument("--early-stop-criterium", type=int, default=15)
    parser.add_argument("--regularization-weight", type=float, default=0.5)
    parser.add_argument(
        "-s",
        "--save-output",
        action="store_true",
        help="Save processed validation outputs after training.",
    )
    args = parser.parse_args()

    cfg = build_cfg(args)
    dataset = get_emnist_dataset(cfg, _resolve_path(args.npz))
    models = init_training.get_models(cfg)
    training_pipeline = init_training.get_training_pipeline(cfg)
    logging = init_training.get_logging(cfg)

    training.train(dataset, models, training_pipeline, logging, cfg)
    training.save_models(models, cfg, prefix="final")

    training.load_models(models, cfg, prefix="best")
    training.save_output_history(logging, cfg)
    training.save_training_summary(logging, cfg)
    output, performance = training.get_validation_results(dataset, models, training_pipeline, cfg)
    if not args.save_output:
        output = None
    training.save_validation_results(output, performance, cfg)


if __name__ == "__main__":
    main()
