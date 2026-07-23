from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


@dataclass(frozen=True)
class HSIData:
    cube: np.ndarray
    labels: np.ndarray
    num_classes: int
    class_mapping: dict[int, int]
    data_key: str
    gt_key: str


def _read_mat_arrays(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    try:
        raw = loadmat(path)
        return {
            key: np.asarray(value)
            for key, value in raw.items()
            if not key.startswith("__") and isinstance(value, np.ndarray)
        }
    except (NotImplementedError, ValueError):
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                f"{path} appears to be MATLAB v7.3; install h5py to read it."
            ) from exc
        arrays: dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as handle:
            for key, value in handle.items():
                if hasattr(value, "shape"):
                    arrays[key] = np.asarray(value)
        return arrays


def _select_array(
    arrays: dict[str, np.ndarray], key: str | None, ndim: int, kind: str
) -> tuple[str, np.ndarray]:
    if key:
        if key not in arrays:
            raise KeyError(f"{kind} key {key!r} not found. Available keys: {sorted(arrays)}")
        value = np.squeeze(arrays[key])
        if value.ndim != ndim:
            raise ValueError(f"{kind} key {key!r} has shape {value.shape}, expected {ndim}D")
        return key, value

    candidates = [
        (name, np.squeeze(value))
        for name, value in arrays.items()
        if np.squeeze(value).ndim == ndim
    ]
    if not candidates:
        raise ValueError(f"No {ndim}D array found for {kind}. Available: {sorted(arrays)}")
    candidates.sort(key=lambda item: item[1].size, reverse=True)
    return candidates[0]


def _align_cube_to_labels(cube: np.ndarray, labels: np.ndarray) -> np.ndarray:
    for permutation in itertools.permutations(range(3)):
        candidate = np.transpose(cube, permutation)
        if candidate.shape[:2] == labels.shape:
            return candidate
    raise ValueError(
        f"Cannot align HSI cube shape {cube.shape} with label shape {labels.shape}. "
        "Provide files whose first two aligned dimensions are the scene height and width."
    )


def _normalize_labels(labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    labels = np.asarray(np.rint(labels), dtype=np.int64)
    positive = sorted(int(value) for value in np.unique(labels) if value > 0)
    if not positive:
        raise ValueError("Ground truth contains no positive class labels.")
    mapping = {old: new for new, old in enumerate(positive, start=1)}
    normalized = np.zeros_like(labels, dtype=np.int64)
    for old, new in mapping.items():
        normalized[labels == old] = new
    return normalized, mapping


def load_hsi(
    data_path: str | Path,
    gt_path: str | Path,
    data_key: str | None = None,
    gt_key: str | None = None,
) -> HSIData:
    """Load and align a MATLAB HSI cube and 2-D ground-truth label map.

    When keys are omitted, the largest 3-D and 2-D arrays are selected. Positive
    labels are remapped to contiguous values 1..C; zero remains background.
    """

    data_arrays = _read_mat_arrays(data_path)
    gt_arrays = data_arrays if Path(data_path).resolve() == Path(gt_path).resolve() else _read_mat_arrays(gt_path)
    selected_data_key, cube = _select_array(data_arrays, data_key, ndim=3, kind="HSI")
    selected_gt_key, labels = _select_array(gt_arrays, gt_key, ndim=2, kind="ground truth")
    cube = _align_cube_to_labels(np.asarray(cube), np.asarray(labels))
    labels, mapping = _normalize_labels(labels)
    cube = np.asarray(cube, dtype=np.float32)
    if not np.isfinite(cube).all():
        raise ValueError("HSI cube contains NaN or infinite values.")
    return HSIData(
        cube=cube,
        labels=labels,
        num_classes=len(mapping),
        class_mapping=mapping,
        data_key=selected_data_key,
        gt_key=selected_gt_key,
    )


def standardize_cube(
    cube: np.ndarray,
    fit_mask: np.ndarray,
    clip: float | None = 8.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit per-band z-score statistics only on the declared training support."""

    if cube.shape[:2] != fit_mask.shape:
        raise ValueError(f"cube spatial shape {cube.shape[:2]} != fit mask {fit_mask.shape}")
    pixels = cube[np.asarray(fit_mask, dtype=bool)]
    if pixels.size == 0:
        raise ValueError("Preprocessing fit mask contains no pixels.")
    mean = pixels.mean(axis=0, dtype=np.float64)
    std = pixels.std(axis=0, dtype=np.float64)
    std = np.where(std < 1e-6, 1.0, std)
    normalized = (cube.astype(np.float32) - mean.astype(np.float32)) / std.astype(np.float32)
    if clip is not None:
        normalized = np.clip(normalized, -float(clip), float(clip))
    return normalized.astype(np.float32), {
        "name": "train_support_zscore",
        "clip": clip,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "fit_pixel_count": int(pixels.shape[0]),
    }


class HSIPatchDataset(Dataset):
    """Patch dataset with an explicit cross-split context guard.

    `center_mask` chooses supervised center pixels. If `allow_full_context` is
    false, only pixels whose `region_map` equals `region_value` remain visible;
    all other context is zeroed. This makes the spatial-block protocol enforceable
    in code rather than only in prose.
    """

    def __init__(
        self,
        cube: np.ndarray,
        labels: np.ndarray,
        center_mask: np.ndarray,
        patch_size: int,
        region_map: np.ndarray | None = None,
        region_value: int | None = None,
        allow_full_context: bool = False,
    ) -> None:
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")
        if cube.shape[:2] != labels.shape or labels.shape != center_mask.shape:
            raise ValueError("cube, labels and center_mask spatial shapes must match")
        if not allow_full_context:
            if region_map is None or region_value is None:
                raise ValueError("Guarded context requires region_map and region_value")
            if region_map.shape != labels.shape:
                raise ValueError("region_map shape must match labels")

        self.patch_size = int(patch_size)
        self.pad = patch_size // 2
        self.allow_full_context = bool(allow_full_context)
        self.region_value = region_value
        self.coords = np.argwhere(np.asarray(center_mask, dtype=bool) & (labels > 0)).astype(np.int64)
        if len(self.coords) == 0:
            raise ValueError("Dataset split contains no labelled center pixels")

        self.labels = np.asarray(labels, dtype=np.int64)
        self.cube_padded = np.pad(
            np.asarray(cube, dtype=np.float32),
            ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode="reflect",
        )
        if region_map is None:
            region_map = np.zeros_like(labels, dtype=np.int8)
        self.region_padded = np.pad(
            np.asarray(region_map, dtype=np.int8),
            ((self.pad, self.pad), (self.pad, self.pad)),
            mode="constant",
            constant_values=0,
        )

    def __len__(self) -> int:
        return int(len(self.coords))

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row, col = (int(value) for value in self.coords[index])
        row_p, col_p = row + self.pad, col + self.pad
        patch = self.cube_padded[
            row_p - self.pad : row_p + self.pad + 1,
            col_p - self.pad : col_p + self.pad + 1,
        ].copy()
        if not self.allow_full_context:
            visible = self.region_padded[
                row_p - self.pad : row_p + self.pad + 1,
                col_p - self.pad : col_p + self.pad + 1,
            ] == int(self.region_value)
            patch *= visible[..., None].astype(np.float32)
        else:
            visible = np.ones((self.patch_size, self.patch_size), dtype=bool)
        target = int(self.labels[row, col] - 1)
        patch_tensor = torch.from_numpy(np.transpose(patch, (2, 0, 1)))
        return (
            patch_tensor,
            torch.tensor(target, dtype=torch.long),
            torch.tensor([row, col], dtype=torch.long),
            torch.from_numpy(visible.astype(np.float32)),
        )
