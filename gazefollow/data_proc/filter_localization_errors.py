#!/usr/bin/env python3

"""Filter gaze localization results by normalized L2 error.

This script trims noisy in-image gaze samples from a localization JSON dump by
keeping only detections whose minimum normalized L2 error is below a threshold
and optionally saves a histogram of the error distribution. Use it after
running localization evaluation (e.g., qwen3vl outputs) when you want a cleaned
dataset for further analysis or downstream tasks. If provided with a
conversation JSON, it also emits a filtered conversation file that keeps only
records surviving the localization filter.

Example:
    python gazefollow/data_proc/filter_localization_errors.py \\
        dataset_evaluation_results/.../dataset_localization_results.json \\
        --threshold 0.2 --histogram-bins 40 \\
        --conversation-json training_datasets/.../val.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

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


def normalize_in_out_flag(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return 1 if int(value) >= 1 else 0
    except (TypeError, ValueError):
        return None


def get_in_out_flags(record: dict) -> Tuple[Optional[int], Optional[int]]:
    gt_flag = normalize_in_out_flag(record.get("gt_in_out"))
    pred_value = record.get("predicted_in_out")
    if pred_value is None:
        pred_value = record.get("pred_in_out")
    pred_flag = normalize_in_out_flag(pred_value)
    legacy_flag = normalize_in_out_flag(record.get("in_out"))
    if gt_flag is None:
        gt_flag = legacy_flag
    if pred_flag is None:
        pred_flag = legacy_flag
    return gt_flag, pred_flag


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
    plt.title("Distribution of Normalized L2 Error (gt_in_out=1)")
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
        default=0.15,
        help="Threshold for normalized L2 error to keep gt_in_out=1 samples (default: 0.15)",
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
    parser.add_argument(
        "--keep-mismatched-inout",
        action="store_true",
        help="Keep samples where predicted_in_out differs from gt_in_out (default: drop them).",
    )
    parser.add_argument(
        "--conversation-json",
        type=Path,
        help="Optional conversation JSON to filter using the localization results.",
    )
    parser.add_argument(
        "--conversation-output",
        type=Path,
        help=(
            "Optional path to write the filtered conversation JSON "
            "(defaults to <conversation_json>_filtered.json)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.input_path
    records = load_records(input_path)
    metrics: dict = {
        "input_path": str(input_path),
        "threshold": args.threshold,
        "keep_mismatched_inout": bool(args.keep_mismatched_inout),
    }

    flag_lookup = {
        str(record.get("id")): get_in_out_flags(record) for record in records
    }
    mismatched_ids = {
        record_id
        for record_id, (gt_flag, pred_flag) in flag_lookup.items()
        if gt_flag is not None and pred_flag is not None and gt_flag != pred_flag
    }
    in_image_records = [
        record for record in records if flag_lookup[str(record.get("id"))][0] == 1
    ]
    out_image_records = [
        record for record in records if flag_lookup[str(record.get("id"))][0] == 0
    ]

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
            metrics["histogram_path"] = str(saved_path)
        else:
            print(
                "matplotlib is not available; skipped histogram generation. "
                "Install matplotlib to enable plotting."
            )
            metrics["histogram_path"] = None
    else:
        print("No normalized L2 errors found; histogram was not generated.")
        metrics["histogram_path"] = None

    threshold = args.threshold
    high_error_ids = {
        record_id for record_id, error in errors_by_id.items() if error > threshold
    }

    low_error_ids = {record_id for record_id, error in errors_by_id.items() if record_id not in high_error_ids}
    low_error_values = [errors_by_id[record_id] for record_id in low_error_ids]

    drop_mismatched = not args.keep_mismatched_inout
    filtered_records = []
    dropped_mismatched = 0
    for record in records:
        record_id = str(record.get("id"))
        gt_flag, _ = flag_lookup.get(record_id, (None, None))
        if drop_mismatched and record_id in mismatched_ids:
            dropped_mismatched += 1
            continue
        if gt_flag == 0:
            filtered_records.append(record)
        elif gt_flag == 1 and record_id in low_error_ids:
            filtered_records.append(record)
        elif gt_flag is None and not drop_mismatched:
            filtered_records.append(record)

    output_path = (
        args.output_path
        if args.output_path
        else input_path.with_name(f"{input_path.stem}_filtered.json")
    )
    write_json(output_path, filtered_records)
    print(f"Wrote filtered records to {output_path}")

    kept_ids: Set[str] = {str(record.get("id")) for record in filtered_records}
    total_records = len(records)
    total_in_image = len(in_image_records)
    total_out_image = len(out_image_records)
    total_with_errors = len(error_values)
    total_mismatched = len(mismatched_ids)
    total_high_error = len(high_error_ids)
    total_filtered = len(filtered_records)
    total_low_error = len(low_error_ids)

    metrics.update(
        {
            "output_path": str(output_path),
            "counts": {
                "total_records": total_records,
                "gt_in_out_1_records": total_in_image,
                "gt_in_out_0_records": total_out_image,
                "records_with_normalized_l2_error": total_with_errors,
                "predicted_in_out_mismatched": total_mismatched,
                "dropped_mismatched": dropped_mismatched,
                "records_above_threshold": total_high_error,
                "records_below_or_equal_threshold": total_low_error,
                "filtered_dataset_size": total_filtered,
            },
        }
    )

    print("\n=== Filter Summary ===")
    print(f"Total records: {total_records}")
    print(f"gt_in_out=1 records: {total_in_image}")
    print(f"gt_in_out=0 records: {total_out_image}")
    print(f"Records with normalized L2 error: {total_with_errors}")
    print(f"predicted_in_out != gt_in_out: {total_mismatched} (dropped: {dropped_mismatched})")
    print(f"Records above threshold ({threshold}): {total_high_error}")
    print(f"Filtered dataset size: {total_filtered}")

    if error_values:
        min_error, max_error, mean_error, median_error = summarize_errors(error_values)
        metrics["error_stats_in_out_1"] = {
            "min": min_error,
            "max": max_error,
            "mean": mean_error,
            "median": median_error,
        }
        print("\n=== Error Statistics (in_out=1) ===")
        print(f"Min normalized L2 error: {min_error:.4f}")
        print(f"Max normalized L2 error: {max_error:.4f}")
        print(f"Mean normalized L2 error: {mean_error:.4f}")
        print(f"Median normalized L2 error: {median_error:.4f}")
    else:
        metrics["error_stats_in_out_1"] = None

    if total_high_error:
        high_error_values = [errors_by_id[record_id] for record_id in high_error_ids]
        min_high, max_high, mean_high, median_high = summarize_errors(high_error_values)
        metrics["high_error_stats"] = {
            "min": min_high,
            "max": max_high,
            "mean": mean_high,
            "median": median_high,
        }
        print("\n=== High Error Statistics ===")
        print(f"Min high error: {min_high:.4f}")
        print(f"Max high error: {max_high:.4f}")
        print(f"Mean high error: {mean_high:.4f}")
        print(f"Median high error: {median_high:.4f}")
    else:
        metrics["high_error_stats"] = None

    if total_low_error:
        low_min, low_max, low_mean, low_median = summarize_errors(low_error_values)
        metrics["low_error_stats"] = {
            "min": low_min,
            "max": low_max,
            "mean": low_mean,
            "median": low_median,
        }
        print("\n=== Low Error Statistics ===")
        print(f"Min low error: {low_min:.4f}")
        print(f"Max low error: {low_max:.4f}")
        print(f"Mean low error: {low_mean:.4f}")
        print(f"Median low error: {low_median:.4f}")
    else:
        print("No error statistics available.")
        metrics["low_error_stats"] = None

    if args.conversation_json:
        convo_path: Path = args.conversation_json
        conversation_records = load_records(convo_path)
        conversation_filtered = [
            record for record in conversation_records if str(record.get("id")) in kept_ids
        ]
        conversation_output = (
            args.conversation_output
            if args.conversation_output
            else convo_path.with_name(f"{convo_path.stem}_filtered.json")
        )
        write_json(conversation_output, conversation_filtered)

        convo_total = len(conversation_records)
        convo_filtered_total = len(conversation_filtered)
        missing_ids = len(kept_ids - {str(record.get("id")) for record in conversation_records})

        metrics["conversation"] = {
            "input_path": str(convo_path),
            "output_path": str(conversation_output),
            "total_records": convo_total,
            "filtered_size": convo_filtered_total,
            "dropped_records": convo_total - convo_filtered_total,
            "localization_ids_not_found_in_conversation": missing_ids,
        }

        print(f"\nWrote filtered conversations to {conversation_output}")
        print("\n=== Conversation Filter Summary ===")
        print(f"Total conversation records: {convo_total}")
        print(f"Filtered conversation size: {convo_filtered_total}")
        print(f"Dropped conversation records: {convo_total - convo_filtered_total}")
        if missing_ids:
            print(f"Localization IDs not found in conversation file: {missing_ids}")
    else:
        metrics["conversation"] = None

    metrics_output = output_path.with_name(f"{output_path.stem}_metrics.json")
    write_json(metrics_output, metrics)
    print(f"\nSaved metrics summary to {metrics_output}")


if __name__ == "__main__":
    main()
