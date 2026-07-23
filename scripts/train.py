from __future__ import annotations

import argparse
import json
from pathlib import Path

from kmfm.engine import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one immutable dataset/protocol/model/seed run")
    parser.add_argument("--config", required=True, help="JSON experiment configuration")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split-path")
    parser.add_argument("--experiment")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.seed is not None:
        config["seed"] = args.seed
    if args.split_path is not None:
        config["protocol"]["split_path"] = args.split_path
    if args.experiment is not None:
        config["output"]["experiment"] = args.experiment
    run_dir = run_experiment(config)
    print(f"Successful run: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
