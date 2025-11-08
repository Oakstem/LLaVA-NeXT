#!/usr/bin/env python3
"""Combine conversation JSON files into a single dataset with simple metrics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine gaze conversation JSON files into a single dataset."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing conversation JSON files (each holding a list of samples).",
    )
    parser.add_argument(
        "--output-root",
        default="training_datasets/qwen_sets",
        help="Root directory where the timestamped dataset directory is created.",
    )
    parser.add_argument(
        "--timestamp",
        help="Optional timestamp override for the output directory (format YYYYMMDD_HHMMSS).",
    )
    parser.add_argument(
        "--output-name",
        default="combined_conversations.json",
        help="Filename for the combined dataset JSON.",
    )
    parser.add_argument(
        "--pattern",
        default="*_conversation.json",
        help="Glob pattern (relative to input_dir) for selecting JSON files to combine.",
    )
    return parser.parse_args()


def load_conversations(files: Iterable[Path]) -> List[dict]:
    combined: List[dict] = []
    for path in files:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            combined.extend(data)
            continue
        if isinstance(data, dict):
            results = data.get("results")
            if isinstance(results, list):
                combined.extend(results)
                continue
        print(f"Skipping {path}: unsupported top-level JSON type {type(data).__name__}")
    return combined


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    json_files = sorted(input_dir.glob(args.pattern))
    if not json_files:
        raise FileNotFoundError(
            f"No files matching pattern '{args.pattern}' found in {input_dir}"
        )

    combined = load_conversations(json_files)
    if not combined:
        raise ValueError(
            "No samples were loaded. Check the input files or adjust --pattern."
        )
    total_samples = len(combined)
    in_samples = sum(1 for sample in combined if str(sample.get("in_or_out", "0")) == "1")

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root).expanduser() / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    combined_path = output_dir / args.output_name
    with combined_path.open("w", encoding="utf-8") as fh:
        json.dump(combined, fh, ensure_ascii=False, indent=2)

    metrics = {
        "total_samples": total_samples,
        "in_or_out_positive": in_samples,
        "source_dir": str(input_dir),
        "file_count": len(json_files),
    }
    metrics_path = output_dir / "set_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    print(f"Combined {total_samples} samples from {len(json_files)} files.")
    print(f"In-out positives: {in_samples}")
    print(f"Saved dataset: {combined_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
