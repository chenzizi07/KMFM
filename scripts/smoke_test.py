from __future__ import annotations

import numpy as np
import torch

from kmfm.data import HSIPatchDataset, standardize_cube
from kmfm.metrics import classification_metrics
from kmfm.model import LASSFNet
from kmfm.splits import TRAIN, make_random_pixel_split, make_spatial_block_split


def synthetic_scene() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    height, width, bands, classes = 24, 24, 16, 3
    labels = np.zeros((height, width), dtype=np.int64)
    # Repeated small cells make every spatial block contain all three classes.
    for row in range(height):
        for col in range(width):
            labels[row, col] = ((row // 2 + col // 2) % classes) + 1
    signatures = rng.normal(size=(classes, bands)).astype(np.float32)
    cube = signatures[labels - 1] + 0.15 * rng.normal(size=(height, width, bands)).astype(np.float32)
    return cube, labels


def main() -> None:
    cube, labels = synthetic_scene()
    random_split = make_random_pixel_split(labels, train_per_class=5, val_per_class=3, seed=0)
    spatial_split = make_spatial_block_split(
        labels,
        train_per_class=5,
        val_per_class=3,
        min_test_per_class=5,
        block_size=4,
        buffer_pixels=0,
        seed=0,
        trials=32,
    )
    assert not np.any(random_split.train_mask & random_split.test_mask)
    assert not np.any(spatial_split.train_mask & spatial_split.test_mask)
    normalized, _ = standardize_cube(cube, random_split.train_mask)
    dataset = HSIPatchDataset(
        normalized,
        labels,
        random_split.train_mask,
        patch_size=7,
        region_map=random_split.region_map,
        region_value=TRAIN,
        allow_full_context=True,
    )
    patch, target, _, context_mask = dataset[0]
    model = LASSFNet(bands=cube.shape[-1], num_classes=3, hidden_dim=16)
    output = model(patch.unsqueeze(0), context_mask=context_mask.unsqueeze(0))
    loss = torch.nn.functional.cross_entropy(output["logits"], target.unsqueeze(0))
    loss.backward()
    metrics, confusion = classification_metrics(np.array([0, 1, 2]), np.array([0, 2, 2]), 3)
    assert confusion.sum() == 3
    assert abs(metrics["oa"] - 2 / 3) < 1e-12
    assert torch.isfinite(loss)
    print("Smoke test passed")


if __name__ == "__main__":
    main()
