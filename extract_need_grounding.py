#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, List, Sequence


def load_samples(path: Path) -> Sequence[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of samples in {path}, found {type(data).__name__}")
    return data


def filter_samples(samples: Iterable[dict[str, Any]], threshold: float) -> List[dict[str, Any]]:
    filtered: List[dict[str, Any]] = []
    for sample in samples:
        value = sample.get("gaze_normalized_l2_error")
        if value is None or (isinstance(value, (int, float)) and value > threshold):
            filtered.append(sample)
    return filtered


def write_output(samples: Sequence[dict[str, Any]], source_path: Path) -> Path:
    output_path = source_path.parent / "need_grounding.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(samples, fh, indent=2)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract samples whose 'gaze_normalized_l2_error' is null or above the given threshold. "
            "Writes results to 'need_grounding.json' in the source file's parent directory."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to the dataset evaluation JSON file (e.g. dataset_evaluation_results/.../person_level_results.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Threshold for 'gaze_normalized_l2_error' (default: 0.1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.source)
    filtered = filter_samples(samples, args.threshold)
    output_path = write_output(filtered, args.source)
    print(f"Wrote {len(filtered)} samples to {output_path}")


if __name__ == "__main__":
    main()
