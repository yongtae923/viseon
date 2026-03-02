"""
letter_download.py

Download and preprocess EMNIST letters dataset into ./data/processed/emnist_letters_upright.npz.
Run from project root:  python code/letter_download.py
"""
import os
import numpy as np
import torch
from torchvision.datasets import EMNIST

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def fix_emnist_orientation(images: torch.Tensor) -> torch.Tensor:
    """Rotate/flip EMNIST letters so they are upright like MNIST."""
    images = torch.transpose(images, 1, 2)
    images = torch.flip(images, dims=[2])
    return images


def main():
    root = os.path.join(PROJECT_ROOT, "data")
    out_dir = os.path.join(root, "processed")
    os.makedirs(out_dir, exist_ok=True)

    # Use already downloaded EMNIST letters data under ./data/EMNIST
    train_ds = EMNIST(root=root, split="letters", train=True, download=False)
    test_ds = EMNIST(root=root, split="letters", train=False, download=False)

    # torchvision EMNIST letters labels are 1..26 -> convert to 0..25
    train_images = fix_emnist_orientation(train_ds.data).numpy().astype(np.uint8)
    test_images = fix_emnist_orientation(test_ds.data).numpy().astype(np.uint8)
    train_labels = (train_ds.targets - 1).numpy().astype(np.uint8)
    test_labels = (test_ds.targets - 1).numpy().astype(np.uint8)

    output_path = os.path.join(out_dir, "emnist_letters_upright.npz")
    np.savez_compressed(
        output_path,
        train_images=train_images,   # (N, 28, 28), uint8
        train_labels=train_labels,   # (N,), 0..25
        test_images=test_images,     # (N, 28, 28), uint8
        test_labels=test_labels,     # (N,), 0..25
    )

    print(f"Saved: {output_path}")
    print(f"Train: images={train_images.shape}, labels={train_labels.shape}")
    print(f"Test : images={test_images.shape}, labels={test_labels.shape}")
    print("Label mapping: 0='a', ..., 25='z'")


if __name__ == "__main__":
    main()