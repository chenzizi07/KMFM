from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from kmfm.artifacts import sha256_file
from kmfm.data import load_hsi
from kmfm.splits import make_random_pixel_split, make_spatial_block_split, save_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a reproducible HSI split artifact")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--data-key")
    parser.add_argument("--gt-key")
    parser.add_argument("--protocol", choices=("random_pixel", "spatial_block"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-per-class", type=int, default=30)
    parser.add_argument("--val-per-class", type=int, default=10)
    parser.add_argument("--min-test-per-class", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--buffer-pixels", type=int, default=3)
    parser.add_argument(
        "--region-ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.25, 0.15, 0.60),
    )
    parser.add_argument("--trials", type=int, default=128)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    hsi = load_hsi(args.data_path, args.gt_path, args.data_key, args.gt_key)
    if args.protocol == "random_pixel":
        split = make_random_pixel_split(
            hsi.labels, args.train_per_class, args.val_per_class, args.seed
        )
    else:
        split = make_spatial_block_split(
            hsi.labels,
            train_per_class=args.train_per_class,
            val_per_class=args.val_per_class,
            min_test_per_class=args.min_test_per_class,
            block_size=args.block_size,
            buffer_pixels=args.buffer_pixels,
            region_ratios=tuple(args.region_ratios),
            seed=args.seed,
            trials=args.trials,
        )
    metadata = dict(split.metadata)
    metadata.update(
        {
            "data_path": str(Path(args.data_path).resolve()),
            "gt_path": str(Path(args.gt_path).resolve()),
            "data_sha256": sha256_file(args.data_path),
            "gt_sha256": sha256_file(args.gt_path),
            "cube_shape": list(hsi.cube.shape),
            "label_shape": list(hsi.labels.shape),
            "num_classes": hsi.num_classes,
            "class_mapping": hsi.class_mapping,
            "data_key": hsi.data_key,
            "gt_key": hsi.gt_key,
        }
    )
    split = replace(split, metadata=metadata)
    save_split(args.output, split)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Saved split: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
