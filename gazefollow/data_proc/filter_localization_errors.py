#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]


def load_records(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Expected a list of records in the input JSON")
    return data


def collect_normalized_errors(records: Iterable[dict]) -> Dict[str, float]:
    errors: Dict[str, float] = {}
    for record in records:
        record_id = str(record.get("id"))
        detections = record.get("gaze_detections")
        if not detections:
            continue

        error_values: List[float] = []
        if isinstance(detections, dict):
            iterator = detections.values()
        elif isinstance(detections, list):
            iterator = detections
        else:
            continue

        for detection in iterator:
            if not isinstance(detection, dict):
                continue
            error = detection.get("gaze_normalized_l2_error")
            if isinstance(error, (float, int)):
                error_values.append(float(error))

        if error_values:
            errors[record_id] = min(error_values)
    return errors


def plot_histogram(
    errors: Iterable[float], bins: int, output_path: Path
) -> Optional[Path]:
    if plt is None:
        return None
    plt.figure(figsize=(8, 5))
    plt.hist(list(errors), bins=bins, color="#0072B2", alpha=0.8, edgecolor="black")
    plt.title("Distribution of Normalized L2 Error (in_out=1)")
    plt.xlabel("Normalized L2 Error")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def write_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def summarize_errors(errors: List[float]) -> Tuple[float, float, float, float]:
    if not errors:
        raise ValueError("Cannot summarize an empty list of errors")
    return (
        min(errors),
        max(errors),
        statistics.mean(errors),
        statistics.median(errors),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter localization records by normalized L2 error and generate "
            "a histogram for analysis."
        )
    )
    parser.add_argument("input_path", type=Path, help="Path to the normalized JSON file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Threshold for normalized L2 error to keep in_out=1 samples (default: 0.2)",
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=40,
        help="Number of bins to use for the histogram (default: 40)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Optional path to write the filtered JSON (defaults to <input>_filtered.json).",
    )
    parser.add_argument(
        "--histogram-output",
        type=Path,
        help=(
            "Optional path to save the histogram image "
            "(defaults to <input>_l2_histogram.png)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.input_path
    records = load_records(input_path)

    in_image_records = [record for record in records if record.get("in_out") == 1]
    out_image_records = [record for record in records if record.get("in_out") == 0]

    errors_by_id = collect_normalized_errors(in_image_records)
    error_values = list(errors_by_id.values())
    valid_error_records = [record for record in in_image_records if str(record.get("id")) in errors_by_id]

    histogram_path = (
        args.histogram_output
        if args.histogram_output
        else input_path.with_name(f"{input_path.stem}_l2_histogram.png")
    )
    if error_values:
        saved_path = plot_histogram(error_values, args.histogram_bins, histogram_path)
        if saved_path:
            print(f"Saved histogram to {saved_path}")
        else:
            print(
                "matplotlib is not available; skipped histogram generation. "
                "Install matplotlib to enable plotting."
            )
    else:
        print("No normalized L2 errors found; histogram was not generated.")

    threshold = args.threshold
    high_error_ids = {
        record_id for record_id, error in errors_by_id.items() if error > threshold
    }

    low_error_ids = {record_id for record_id, error in errors_by_id.items() if record_id not in high_error_ids}
    low_error_values = [errors_by_id[record_id] for record_id in low_error_ids]

    filtered_records = [
        record
        for record in records
        if record.get("in_out") == 0 or str(record.get("id")) in low_error_ids
    ]

    output_path = (
        args.output_path
        if args.output_path
        else input_path.with_name(f"{input_path.stem}_filtered.json")
    )
    write_json(output_path, filtered_records)
    print(f"Wrote filtered records to {output_path}")

    total_records = len(records)
    total_in_image = len(in_image_records)
    total_out_image = len(out_image_records)
    total_with_errors = len(error_values)
    total_high_error = len(high_error_ids)
    total_filtered = len(filtered_records)
    total_low_error = len(low_error_ids)

    print("\n=== Filter Summary ===")
    print(f"Total records: {total_records}")
    print(f"in_out=1 records: {total_in_image}")
    print(f"in_out=0 records: {total_out_image}")
    print(f"Records with normalized L2 error: {total_with_errors}")
    print(f"Records above threshold ({threshold}): {total_high_error}")
    print(f"Filtered dataset size: {total_filtered}")

    if error_values:
        min_error, max_error, mean_error, median_error = summarize_errors(error_values)
        print("\n=== Error Statistics (in_out=1) ===")
        print(f"Min normalized L2 error: {min_error:.4f}")
        print(f"Max normalized L2 error: {max_error:.4f}")
        print(f"Mean normalized L2 error: {mean_error:.4f}")
        print(f"Median normalized L2 error: {median_error:.4f}")

    if total_high_error:
        high_error_values = [errors_by_id[record_id] for record_id in high_error_ids]
        min_high, max_high, mean_high, median_high = summarize_errors(high_error_values)
        print("\n=== High Error Statistics ===")
        print(f"Min high error: {min_high:.4f}")
        print(f"Max high error: {max_high:.4f}")
        print(f"Mean high error: {mean_high:.4f}")
        print(f"Median high error: {median_high:.4f}")
    
    if total_low_error:
        low_min, low_max, low_mean, low_median = summarize_errors(low_error_values)
        print("\n=== Low Error Statistics ===")
        print(f"Min low error: {low_min:.4f}")
        print(f"Max low error: {low_max:.4f}")
        print(f"Mean low error: {low_mean:.4f}")
        print(f"Median low error: {low_median:.4f}")
    else:
        print("No error statistics available.")


if __name__ == "__main__":
    main()
