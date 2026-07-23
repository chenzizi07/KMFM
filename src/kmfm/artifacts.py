from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def json_dump(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "pid": os.getpid(),
    }
    try:
        snapshot["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        snapshot["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        snapshot["git_commit"] = None
        snapshot["git_dirty"] = None
    return snapshot


def build_manifest(
    run_dir: str | Path,
    input_files: dict[str, str | Path],
    output_files: list[str | Path],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    inputs = {
        name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
        for name, path in input_files.items()
    }
    outputs: dict[str, Any] = {}
    for path in output_files:
        resolved = Path(path)
        if resolved.exists() and resolved.is_file():
            outputs[str(resolved.relative_to(run_dir))] = {
                "sha256": sha256_file(resolved),
                "bytes": resolved.stat().st_size,
            }
    manifest = {"inputs": inputs, "outputs": outputs}
    if extra:
        manifest.update(extra)
    return manifest


def save_confusion_csv(path: str | Path, confusion: np.ndarray) -> None:
    confusion = np.asarray(confusion, dtype=np.int64)
    header = ",".join(["true\\pred"] + [str(index) for index in range(confusion.shape[1])])
    rows = [header]
    for index, row in enumerate(confusion):
        rows.append(",".join([str(index)] + [str(int(value)) for value in row]))
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")
