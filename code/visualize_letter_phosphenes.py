"""
visualize_letter_phosphenes.py

Visualize a few examples of original EMNIST letter images
and their corresponding phosphene maps (built by build_letter_phosphene_dataset.py).

Run from project root, e.g.:
  python code/visualize_letter_phosphenes.py
This script is lightweight and avoids importing torch/torchvision.
"""
import argparse
import os
from typing import Sequence, Optional

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(PROJECT_ROOT, path_value))


def _pick_indices(n_total: int, num: int, indices: Optional[Sequence[int]] = None):
    """Return a list of indices to visualize."""
    if indices:
        idx = [i for i in indices if 0 <= i < n_total]
        if not idx:
            raise ValueError("No valid indices to visualize.")
        return idx[:num]
    # default: first num samples
    return list(range(min(num, n_total)))


def visualize_pairs(
    orig_npz: str,
    phos_npz: str,
    split: str = "train",
    num: int = 4,
    indices: Optional[Sequence[int]] = None,
    out_path: Optional[str] = None,
) -> None:
    """Create a single figure with num rows: [original | phosphene] and save it."""
    orig_path = _resolve_path(orig_npz)
    phos_path = _resolve_path(phos_npz)

    print(f"[visualize] orig_npz  -> {orig_path}")
    print(f"[visualize] phos_npz  -> {phos_path}")

    orig = np.load(orig_path)
    phos = np.load(phos_path)

    print(f"[visualize] loaded orig keys: {list(orig.keys())}")
    print(f"[visualize] loaded phos keys: {list(phos.keys())}")

    if split == "train":
        images = orig["train_images"]  # (N, 28, 28)
        labels = orig["train_labels"]
        phos_maps = phos["train_phosphenes"]  # (N, 1, H, W)
        phos_labels = phos["train_labels"]
    elif split in ("test", "val"):
        images = orig["test_images"]
        labels = orig["test_labels"]
        phos_maps = phos["test_phosphenes"]
        phos_labels = phos["test_labels"]
    else:
        raise ValueError("split must be 'train' or 'test'")

    print(f"[visualize] images shape: {images.shape}")
    print(f"[visualize] phos_maps shape: {phos_maps.shape}")

    assert len(images) == len(
        phos_maps
    ), "Original and phosphene datasets must have same number of samples."

    n_total = len(images)
    idxs = _pick_indices(n_total, num, indices)
    print(f"[visualize] n_total={n_total}, using indices={idxs}")

    n_rows = len(idxs)
    fig, axes = plt.subplots(n_rows, 2, figsize=(4 * 2, 3 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)  # unify indexing

    for row, i in enumerate(idxs):
        img = images[i]
        ph = phos_maps[i]
        if ph.ndim == 3:  # (1, H, W)
            ph = ph[0]

        label = int(labels[i])
        ph_label = int(phos_labels[i])

        ax_orig = axes[row, 0]
        ax_phos = axes[row, 1]

        ax_orig.imshow(img, cmap="gray")
        ax_orig.set_title(f"Orig idx {i}, label {label}")
        ax_orig.axis("off")

        ax_phos.imshow(ph, cmap="gray")
        ax_phos.set_title(f"Phosphene idx {i}, label {ph_label}")
        ax_phos.axis("off")

    plt.tight_layout()
    if out_path is None:
        out_path = f"Out/letter/letter_phosphene_examples_{split}.png"
    # ensure directory exists
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize EMNIST letter images and corresponding phosphene maps."
    )
    parser.add_argument(
        "--orig-npz",
        type=str,
        default="./data/processed/emnist_letters_upright.npz",
        help="Path to original EMNIST letters npz.",
    )
    parser.add_argument(
        "--phos-npz",
        type=str,
        default="./data/processed/emnist_letters_phosphenes.npz",
        help="Path to phosphene dataset npz produced by build_letter_phosphene_dataset.py.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which split to visualize.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=4,
        help="Number of examples (rows) to show.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit indices to visualize (overrides default first num).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path (e.g. Out/letter/letter_phosphenes.png).",
    )
    args = parser.parse_args()

    visualize_pairs(
        orig_npz=args.orig_npz,
        phos_npz=args.phos_npz,
        split=args.split,
        num=args.num,
        indices=args.indices,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()

