#!/usr/bin/env python
"""Filter dataset localization results by normalized L2 error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter dataset localization evaluation results by normalized L2 error threshold."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the dataset localization results JSON file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the filtered conversation JSON file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Keep samples whose normalized L2 error is strictly below this value (default: 0.1).",
    )
    return parser.parse_args()


def extract_normalized_l2(sample: dict) -> float | None:
    detections = sample.get("gaze_detections") or {}
    if not detections:
        return None

    # dataset results only contain a single detection entry, take the first value defensively
    first_detection = next(iter(detections.values()))
    return first_detection.get("gaze_normalized_l2_error")


def compute_statistics(errors: Iterable[float]) -> Tuple[int, float]:
    errors_list: List[float] = list(errors)
    if not errors_list:
        return 0, float("nan")
    return len(errors_list), mean(errors_list)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    normalized_errors: List[float] = []
    filtered_samples: List[dict] = []
    filtered_errors: List[float] = []
    skipped_without_error = 0

    for sample in samples:
        error = extract_normalized_l2(sample)
        if error is None:
            skipped_without_error += 1
            continue

        normalized_errors.append(error)
        if error < args.threshold:
            filtered_samples.append(sample)
            filtered_errors.append(error)

    total_count, total_mean = compute_statistics(normalized_errors)
    filtered_count, filtered_mean = compute_statistics(filtered_errors)

    print(f"Loaded {total_count} samples with normalized L2 error available.")
    if skipped_without_error:
        print(f"Skipped {skipped_without_error} samples without normalized L2 error.")
    print(
        f"Filtered {filtered_count} samples below threshold {args.threshold:.4f} "
        f"({filtered_count / total_count:.2%} of available samples)." if total_count else "Filtered 0 samples."
    )
    print(f"Average normalized L2 error before filtering: {total_mean:.6f}" if total_count else "No errors to average.")
    if filtered_count:
        print(f"Average normalized L2 error after filtering: {filtered_mean:.6f}")
    else:
        print("No samples met the threshold; not writing output.")

    if filtered_samples:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(filtered_samples, f, indent=2)
        print(f"Wrote filtered samples to {output_path}")
        print(f"Resulting dataset contains {len(filtered_samples)} samples.")


if __name__ == "__main__":
    main()
