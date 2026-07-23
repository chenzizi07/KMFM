from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation


IGNORE = 0
TRAIN = 1
VAL = 2
TEST = 3
BUFFER = 4


@dataclass(frozen=True)
class SplitBundle:
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    region_map: np.ndarray
    metadata: dict[str, Any]


def _counts_by_class(mask: np.ndarray, labels: np.ndarray, num_classes: int) -> list[int]:
    return [int(np.sum(mask & (labels == class_id))) for class_id in range(1, num_classes + 1)]


def _validate_disjoint(train: np.ndarray, val: np.ndarray, test: np.ndarray) -> None:
    overlap = train.astype(np.uint8) + val.astype(np.uint8) + test.astype(np.uint8)
    if np.any(overlap > 1):
        raise ValueError("train/val/test masks overlap")


def make_random_pixel_split(
    labels: np.ndarray,
    train_per_class: int = 30,
    val_per_class: int = 10,
    seed: int = 0,
) -> SplitBundle:
    labels = np.asarray(labels, dtype=np.int64)
    num_classes = int(labels.max())
    rng = np.random.default_rng(seed)
    train = np.zeros_like(labels, dtype=bool)
    val = np.zeros_like(labels, dtype=bool)
    test = np.zeros_like(labels, dtype=bool)
    for class_id in range(1, num_classes + 1):
        coords = np.argwhere(labels == class_id)
        required = train_per_class + val_per_class + 1
        if len(coords) < required:
            raise ValueError(
                f"Class {class_id} has {len(coords)} pixels; at least {required} are required."
            )
        order = rng.permutation(len(coords))
        train_coords = coords[order[:train_per_class]]
        val_coords = coords[order[train_per_class : train_per_class + val_per_class]]
        test_coords = coords[order[train_per_class + val_per_class :]]
        train[tuple(train_coords.T)] = True
        val[tuple(val_coords.T)] = True
        test[tuple(test_coords.T)] = True

    region_map = np.full(labels.shape, TEST, dtype=np.int8)
    _validate_disjoint(train, val, test)
    metadata = {
        "protocol": "random_pixel",
        "seed": int(seed),
        "train_per_class": int(train_per_class),
        "val_per_class": int(val_per_class),
        "allow_full_context": True,
        "counts": {
            "train": _counts_by_class(train, labels, num_classes),
            "val": _counts_by_class(val, labels, num_classes),
            "test": _counts_by_class(test, labels, num_classes),
        },
    }
    return SplitBundle(train, val, test, region_map, metadata)


def _block_grid(shape: tuple[int, int], block_size: int) -> tuple[np.ndarray, int, int]:
    height, width = shape
    grid_h = int(np.ceil(height / block_size))
    grid_w = int(np.ceil(width / block_size))
    rows = np.arange(height)[:, None] // block_size
    cols = np.arange(width)[None, :] // block_size
    return (rows * grid_w + cols).astype(np.int32), grid_h, grid_w


def _assign_blocks(
    block_counts: np.ndarray,
    grid_h: int,
    grid_w: int,
    ratios: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    num_blocks, num_classes = block_counts.shape
    targets = ratios[:, None] * block_counts.sum(axis=0, keepdims=True)
    current = np.zeros((3, num_classes), dtype=np.float64)
    assignment = np.full(num_blocks, -1, dtype=np.int8)
    labelled_blocks = np.flatnonzero(block_counts.sum(axis=1) > 0)
    jitter = rng.random(len(labelled_blocks))
    order = labelled_blocks[np.lexsort((jitter, -block_counts[labelled_blocks].sum(axis=1)))]

    for block_id in order:
        counts = block_counts[block_id]
        scores = np.zeros(3, dtype=np.float64)
        row, col = divmod(int(block_id), grid_w)
        neighbours = []
        for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = row + d_row, col + d_col
            if 0 <= rr < grid_h and 0 <= cc < grid_w:
                neighbour_assignment = assignment[rr * grid_w + cc]
                if neighbour_assignment >= 0:
                    neighbours.append(int(neighbour_assignment))
        for split_id in range(3):
            deficit = np.maximum(targets[split_id] - current[split_id], 0.0)
            coverage = np.sum((deficit / np.maximum(targets[split_id], 1.0)) * counts)
            projected = current[split_id] + counts
            overflow = np.sum(np.maximum(projected - targets[split_id], 0.0) / np.maximum(targets[split_id], 1.0))
            neighbour_bonus = 0.2 * neighbours.count(split_id)
            scores[split_id] = coverage - 0.15 * overflow + neighbour_bonus + rng.normal(0.0, 1e-5)
        chosen = int(np.argmax(scores))
        assignment[block_id] = chosen
        current[chosen] += counts

    # Assign empty blocks using nearby assignments; fall back to target ratios.
    for block_id in np.flatnonzero(assignment < 0):
        row, col = divmod(int(block_id), grid_w)
        nearby: list[int] = []
        for radius in range(1, max(grid_h, grid_w) + 1):
            for rr in range(max(0, row - radius), min(grid_h, row + radius + 1)):
                for cc in range(max(0, col - radius), min(grid_w, col + radius + 1)):
                    candidate = int(assignment[rr * grid_w + cc])
                    if candidate >= 0:
                        nearby.append(candidate)
            if nearby:
                break
        assignment[block_id] = int(np.bincount(nearby, minlength=3).argmax()) if nearby else int(rng.choice(3, p=ratios))
    return assignment


def _boundary_mask(region_map: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(region_map, dtype=bool)
    vertical = region_map[1:, :] != region_map[:-1, :]
    horizontal = region_map[:, 1:] != region_map[:, :-1]
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    return boundary


def make_spatial_block_split(
    labels: np.ndarray,
    train_per_class: int = 30,
    val_per_class: int = 10,
    min_test_per_class: int = 20,
    block_size: int = 32,
    buffer_pixels: int = 3,
    region_ratios: tuple[float, float, float] = (0.25, 0.15, 0.60),
    seed: int = 0,
    trials: int = 128,
) -> SplitBundle:
    """Create block-disjoint regions and fixed-count train/validation centers.

    The optimizer retries block assignments and chooses the one with the smallest
    per-class shortfall and ratio error. It raises instead of silently accepting a
    split that cannot provide the requested samples.
    """

    labels = np.asarray(labels, dtype=np.int64)
    if block_size < 2:
        raise ValueError("block_size must be >= 2")
    ratios = np.asarray(region_ratios, dtype=np.float64)
    if ratios.shape != (3,) or np.any(ratios <= 0):
        raise ValueError("region_ratios must contain three positive values")
    ratios /= ratios.sum()
    num_classes = int(labels.max())
    block_ids, grid_h, grid_w = _block_grid(labels.shape, block_size)
    num_blocks = grid_h * grid_w
    block_counts = np.zeros((num_blocks, num_classes), dtype=np.int64)
    for class_id in range(1, num_classes + 1):
        ids, counts = np.unique(block_ids[labels == class_id], return_counts=True)
        block_counts[ids, class_id - 1] = counts

    best: tuple[float, np.ndarray, np.ndarray] | None = None
    required = np.array(
        [train_per_class, val_per_class, min_test_per_class], dtype=np.int64
    )[:, None]
    for trial in range(int(trials)):
        rng = np.random.default_rng(seed * 10007 + trial)
        assignment = _assign_blocks(block_counts, grid_h, grid_w, ratios, rng)
        region = assignment[block_ids] + 1
        if buffer_pixels > 0:
            buffer = binary_dilation(_boundary_mask(region), iterations=int(buffer_pixels))
            region = region.astype(np.int8)
            region[buffer] = BUFFER
        counts = np.zeros((3, num_classes), dtype=np.int64)
        for split_id, region_value in enumerate((TRAIN, VAL, TEST)):
            for class_id in range(1, num_classes + 1):
                counts[split_id, class_id - 1] = int(np.sum((region == region_value) & (labels == class_id)))
        shortfall = np.maximum(required - counts, 0)
        missing_penalty = float(shortfall.sum() * 1000)
        totals = np.maximum(block_counts.sum(axis=0), 1)
        ratio_error = float(np.mean(np.abs(counts / totals[None, :] - ratios[:, None])))
        fragmentation = float(np.mean(_boundary_mask(np.where(region == BUFFER, 0, region))))
        score = missing_penalty + ratio_error + 0.05 * fragmentation
        if best is None or score < best[0]:
            best = (score, region.copy(), counts.copy())
        if missing_penalty == 0 and ratio_error < 0.12:
            break

    assert best is not None
    _, region_map, available_counts = best
    if np.any(available_counts < required):
        details = {
            "available_train": available_counts[0].tolist(),
            "available_val": available_counts[1].tolist(),
            "available_test": available_counts[2].tolist(),
            "required_train": int(train_per_class),
            "required_val": int(val_per_class),
            "required_test": int(min_test_per_class),
        }
        raise ValueError(
            "Unable to create a valid spatial-block split. Increase trials, reduce "
            f"block/buffer size, or lower fixed counts. Details: {json.dumps(details)}"
        )

    rng = np.random.default_rng(seed)
    train = np.zeros_like(labels, dtype=bool)
    val = np.zeros_like(labels, dtype=bool)
    test = (region_map == TEST) & (labels > 0)
    for class_id in range(1, num_classes + 1):
        train_coords = np.argwhere((region_map == TRAIN) & (labels == class_id))
        val_coords = np.argwhere((region_map == VAL) & (labels == class_id))
        train_coords = train_coords[rng.permutation(len(train_coords))[:train_per_class]]
        val_coords = val_coords[rng.permutation(len(val_coords))[:val_per_class]]
        train[tuple(train_coords.T)] = True
        val[tuple(val_coords.T)] = True

    _validate_disjoint(train, val, test)
    metadata = {
        "protocol": "spatial_block",
        "seed": int(seed),
        "train_per_class": int(train_per_class),
        "val_per_class": int(val_per_class),
        "min_test_per_class": int(min_test_per_class),
        "block_size": int(block_size),
        "buffer_pixels": int(buffer_pixels),
        "region_ratios": ratios.tolist(),
        "trials": int(trials),
        "allow_full_context": False,
        "counts": {
            "train": _counts_by_class(train, labels, num_classes),
            "val": _counts_by_class(val, labels, num_classes),
            "test": _counts_by_class(test, labels, num_classes),
            "available_by_region": available_counts.tolist(),
        },
    }
    return SplitBundle(train, val, test, region_map.astype(np.int8), metadata)


def save_split(path: str | Path, split: SplitBundle) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(split.metadata, sort_keys=True, ensure_ascii=False)
    np.savez_compressed(
        path,
        train_mask=split.train_mask.astype(np.uint8),
        val_mask=split.val_mask.astype(np.uint8),
        test_mask=split.test_mask.astype(np.uint8),
        region_map=split.region_map.astype(np.int8),
        metadata_json=np.asarray(payload),
    )


def load_split(path: str | Path) -> SplitBundle:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        split = SplitBundle(
            train_mask=data["train_mask"].astype(bool),
            val_mask=data["val_mask"].astype(bool),
            test_mask=data["test_mask"].astype(bool),
            region_map=data["region_map"].astype(np.int8),
            metadata=metadata,
        )
    _validate_disjoint(split.train_mask, split.val_mask, split.test_mask)
    return split


def split_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
