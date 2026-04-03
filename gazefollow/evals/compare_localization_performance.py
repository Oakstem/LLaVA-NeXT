#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# Add parent directory to path to enable imports from gazefollow
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gazefollow.qwen3vl_utils import infer_in_out_from_phrase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare localization performance metrics across evaluation result files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("mini_eval_batch"),
        help="Root directory to search for localization result JSON files (default: evaluation_results).",
    )
    parser.add_argument(
        "--baseline-substring",
        type=str,
        default=None,
        help="Optional substring to identify the baseline localization file. If omitted, baseline diff comparisons are skipped.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Number of samples with the highest normalized L2 error difference to report (default: 10).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to write JSON file with top difference details (default: <root>/localization_top_diffs.json).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path to write CSV file with per-dataset total metrics (default: <root>/localization_total_metrics.csv).",
    )
    parser.add_argument(
        "--output-invalid-coordinates-json",
        type=Path,
        default=None,
        help="Path to write JSON file with per-dataset image IDs without valid gaze coordinates (default: <root>/localization_invalid_coordinates_ids_<timestamp>.json).",
    )
    return parser.parse_args()


def find_localization_files(root: Path) -> List[Path]:
    if not root.exists():
        return []

    candidates = [
        path
        # for path in root.rglob("dataset_*localization*.json")
        for path in root.rglob("model_generation*.json")
        if path.is_file()
    ]
    candidates2 = [
        path
        for path in root.rglob("dataset_*localization*.json")
        if path.is_file()
    ]
    return sorted(candidates + candidates2)


def normalize_in_out(raw_value: object) -> Optional[int]:
    if raw_value is None:
        return None

    if isinstance(raw_value, bool):
        return int(raw_value)

    if isinstance(raw_value, (int, float)):
        return int(raw_value)

    if isinstance(raw_value, str):
        value = raw_value.strip().lower()
        if value in {"1", "true", "in", "inside"}:
            return 1
        if value in {"0", "false", "out", "outside"}:
            return 0

    return None


def has_valid_coordinates(coords: object) -> bool:
    if coords is None:
        return False

    if isinstance(coords, (list, tuple)):
        return any(value is not None for value in coords)

    if isinstance(coords, dict):
        return any(value is not None for value in coords.values())

    return False


@dataclass
class DatasetMetrics:
    path: Path
    adapter_path: Optional[str] = None
    total_samples: int = 0
    false_negatives: int = 0
    false_positives: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    recall: Optional[float] = None
    precision: Optional[float] = None
    accuracy: Optional[float] = None
    valid_ids: Set[str] = field(default_factory=set)
    invalid_coordinate_ids: List[str] = field(default_factory=list)
    normalized_errors: Dict[str, float] = field(default_factory=dict)
    gaze_targets: Dict[str, Optional[str]] = field(default_factory=dict)
    in_out_labels: Dict[str, Optional[int]] = field(default_factory=dict)
    missing_intersection_errors: int = 0
    intersection_mean_error: Optional[float] = None
    intersection_size: int = 0
    mean_normalized_error: Optional[float] = None


def format_dataset_reference(dataset: DatasetMetrics) -> str:
    if dataset.adapter_path:
        return f"{dataset.path} [adapter: {dataset.adapter_path}]"
    return str(dataset.path)


def load_adapter_suffix(result_path: Path) -> Optional[str]:
    config_paths = [
        result_path.parent / "evaluation_config.json",
        result_path.parent.parent / "evaluation_config.json",
    ]
    adapter_value = None
    for config_path in config_paths:
        if not config_path.is_file():
            continue

        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse {config_path}: {exc}") from exc

        adapter_value = config.get("adapter_path")
        if adapter_value and isinstance(adapter_value, str):
            break

    if not adapter_value or not isinstance(adapter_value, str):
        return None

    parts = Path(adapter_value).parts
    if len(parts) >= 2:
        return "/".join(parts[-2:])

    return adapter_value if parts else adapter_value


def load_dataset_metrics(path: Path) -> DatasetMetrics:
    dataset = DatasetMetrics(path=path)
    dataset.adapter_path = load_adapter_suffix(path)

    try:
        entries = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse {path}: {exc}") from exc

    if not isinstance(entries, list):
        raise RuntimeError(f"Expected list at {path}, found {type(entries).__name__}")

    dataset.total_samples = len(entries)

    for index, entry in enumerate(entries):
        sample_id = entry.get("id") or entry.get("image_path") or f"index_{index}"
        gt_in_out_flag = normalize_in_out(entry.get("gt_in_out"))

        detection = entry.get("gaze_detections") or {}
        person = detection.get("person_1") or {}
        coords = person.get("gaze_coordinates")
        coords_valid = has_valid_coordinates(coords)
        gaze_target = person.get("gaze_target")

        dataset.in_out_labels[str(sample_id)] = gt_in_out_flag
        inferred_in_out = normalize_in_out(entry.get("pred_in_out"))
        if inferred_in_out is None:
            inferred_in_out = normalize_in_out(entry.get("predicted_in_out"))
        if inferred_in_out is None:
            inferred_in_out = normalize_in_out(infer_in_out_from_phrase(gaze_target))
        if inferred_in_out is None:
            inferred_in_out = 1 if coords_valid else 0
        if inferred_in_out == 0:
            person["gaze_coordinates"] = None
            person["gaze_normalized_l2_error"] = None
            person["gaze_l2_error"] = None
            coords = None
            coords_valid = False

        if gt_in_out_flag == 0:
            for error_key in (
                "gaze_normalized_l2_error",
                "gaze_l2_error",
                "gaze_modified_l2_error",
                "gaze_angular_error",
                "gaze_iou",
            ):
                if error_key in person:
                    person[error_key] = None

        if gt_in_out_flag == 1 and inferred_in_out == 0:
            dataset.false_negatives += 1

        if gt_in_out_flag == 0 and inferred_in_out == 1:
            dataset.false_positives += 1
        
        if gt_in_out_flag == 1 and inferred_in_out == 1:
            dataset.true_positives += 1
        
        if gt_in_out_flag == 0 and inferred_in_out == 0:
            dataset.true_negatives += 1

        dataset.gaze_targets[str(sample_id)] = (
            str(gaze_target) if gaze_target is not None else None
        )

        if coords_valid and inferred_in_out == 1:
            dataset.valid_ids.add(str(sample_id))
            normalized_error = person.get("gaze_normalized_l2_error")
            if isinstance(normalized_error, (int, float)):
                dataset.normalized_errors[str(sample_id)] = float(normalized_error)
        if not (coords_valid and inferred_in_out == 1):
            dataset.invalid_coordinate_ids.append(str(sample_id))

    if dataset.normalized_errors:
        dataset.mean_normalized_error = mean(dataset.normalized_errors.values())

    # Compute Recall Precision
    recall_denominator = dataset.true_positives + dataset.false_negatives
    precision_denominator = dataset.true_positives + dataset.false_positives
    dataset.recall = (
        dataset.true_positives / recall_denominator if recall_denominator else None
    )
    dataset.precision = (
        dataset.true_positives / precision_denominator if precision_denominator else None
    )
    dataset.accuracy = (
        (dataset.true_positives + dataset.true_negatives) / dataset.total_samples
        if dataset.total_samples
        else None
    )

    return dataset


def compute_intersection_metrics(
    datasets: Iterable[DatasetMetrics],
) -> Set[str]:
    datasets = list(datasets)
    if not datasets:
        return set()

    intersection = set.intersection(*(ds.valid_ids for ds in datasets)) if datasets else set()

    for ds in datasets:
        ds.intersection_size = len(intersection)
        errors = [
            ds.normalized_errors[sample_id]
            for sample_id in intersection
            if sample_id in ds.normalized_errors
        ]

        missing_count = len(intersection) - len(errors)
        ds.missing_intersection_errors = missing_count
        ds.intersection_mean_error = mean(errors) if errors else None

    return intersection


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6f}"


def select_top_by_intersection_error(
    datasets: Iterable[DatasetMetrics],
    baseline: Optional[DatasetMetrics],
    top_n: int = 5,
) -> List[DatasetMetrics]:
    ranked = sorted(
        (ds for ds in datasets if ds.intersection_mean_error is not None),
        key=lambda ds: ds.intersection_mean_error,
    )
    selected = ranked[:top_n]
    if baseline is not None and baseline not in selected:
        selected.append(baseline)

    unique_selected: List[DatasetMetrics] = []
    seen_paths: Set[str] = set()
    for ds in selected:
        path_key = str(ds.path)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        unique_selected.append(ds)

    return sorted(
        unique_selected,
        key=lambda ds: float("inf")
        if ds.intersection_mean_error is None
        else ds.intersection_mean_error,
    )


def print_top_table(top_results: List[DatasetMetrics], baseline: Optional[DatasetMetrics]) -> None:
    if not top_results:
        print("No datasets with intersection_mean_gaze_normalized_l2_error available for table.\n")
        return

    header = (
        f"{'adapter_path':40} "
        f"{'intersection_normalized_l2':>26} "
        f"{'intersection_size':>18} "
        f"{'recall':>8} "
        f"{'precision':>10} "
        f"{'accuracy':>10} "
        f"{'normalized_l2':>15}"
    )
    print("Top localization results (intersection_mean_gaze_normalized_l2_error):")
    print(header)
    print("-" * len(header))
    for ds in top_results:
        adapter_display = "baseline" if baseline is not None and ds is baseline else (ds.adapter_path or str(ds.path))
        print(
            f"{adapter_display:40.40} "
            f"{format_float(ds.intersection_mean_error):>26} "
            f"{ds.intersection_size:>18} "
            f"{format_float(ds.recall):>8} "
            f"{format_float(ds.precision):>10} "
            f"{format_float(ds.accuracy):>10} "
            f"{format_float(ds.mean_normalized_error):>15}"
        )
    print()


def select_baseline(
    datasets: Iterable[DatasetMetrics], substring: str
) -> DatasetMetrics:
    candidates = [
        ds for ds in datasets if substring in str(ds.path)
    ]

    if not candidates:
        raise RuntimeError(
            f"Could not find baseline dataset containing '{substring}' in path. Available files:\n"
            + "\n".join(str(ds.path) for ds in datasets)
        )

    if len(candidates) > 1:
        candidates.sort(key=lambda ds: len(str(ds.path)))

    return candidates[0]


def report_top_differences(
    baseline: Optional[DatasetMetrics],
    datasets: Iterable[DatasetMetrics],
    top_k: int,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    if baseline is None:
        return results
    baseline_reference = format_dataset_reference(baseline)

    for ds in datasets:
        if ds is baseline:
            continue

        shared_ids = baseline.valid_ids & ds.valid_ids
        ranked: List[Tuple[float, str, float, float]] = []

        for sample_id in shared_ids:
            baseline_error = baseline.normalized_errors.get(sample_id)
            other_error = ds.normalized_errors.get(sample_id)

            if baseline_error is None or other_error is None:
                continue

            diff = abs(other_error - baseline_error)
            ranked.append((diff, sample_id, baseline_error, other_error))

        ranked.sort(key=lambda entry: entry[0], reverse=True)
        top_entries = ranked[:top_k]
        dataset_key = str(ds.path)
        dataset_reference = format_dataset_reference(ds)
        dataset_result = {
            "path": str(ds.path),
            "adapter_path": ds.adapter_path,
            "total_shared_samples": len(shared_ids),
            "top_differences": [],
        }

        print(
            f"Top {len(top_entries)} normalized L2 error diffs vs baseline {baseline_reference} for {dataset_reference}:"
        )
        if not top_entries:
            print("  No shared samples with normalized L2 error available.\n")
            results[dataset_key] = dataset_result
            continue

        for diff, sample_id, baseline_error, other_error in top_entries:
            baseline_target = baseline.gaze_targets.get(sample_id)
            other_target = ds.gaze_targets.get(sample_id)
            in_out_value = baseline.in_out_labels.get(sample_id)
            if in_out_value is None:
                in_out_value = ds.in_out_labels.get(sample_id)
            print(
                "  id={id} diff={diff:.6f} baseline={baseline:.6f} current={current:.6f} "
                "baseline_target={baseline_target} current_target={current_target} gt_in_out={gt_in_out}".format(
                    id=sample_id,
                    diff=diff,
                    baseline=baseline_error,
                    current=other_error,
                    baseline_target=baseline_target if baseline_target is not None else "None",
                    current_target=other_target if other_target is not None else "None",
                    gt_in_out=in_out_value if in_out_value is not None else "None",
                )
            )
            dataset_result["top_differences"].append(
                {
                    "id": sample_id,
                    "abs_diff": diff,
                    "baseline_error": baseline_error,
                    "baseline_target": baseline_target,
                    "current_error": other_error,
                    "current_target": other_target,
                    "gt_in_out": in_out_value,
                }
            )
        if len(ranked) > top_k:
            print(f"  ... {len(ranked) - top_k} more samples truncated.")
        print()

        results[dataset_key] = dataset_result

    return results


def write_top_diff_json(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))


def write_metrics_csv(path: Path, datasets: Iterable[DatasetMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "adapter_path",
        "total_samples",
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "recall",
        "precision",
        "accuracy",
        "valid_gaze_samples",
        "intersection_size",
        "intersection_mean_gaze_normalized_l2_error",
        "missing_intersection_errors",
        "mean_gaze_normalized_l2_error",
    ]
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for ds in datasets:
            writer.writerow(
                {
                    "path": str(ds.path),
                    "adapter_path": ds.adapter_path or "",
                    "total_samples": ds.total_samples,
                    "true_positives": ds.true_positives,
                    "true_negatives": ds.true_negatives,
                    "false_positives": ds.false_positives,
                    "false_negatives": ds.false_negatives,
                    "recall": ds.recall,
                    "precision": ds.precision,
                    "accuracy": ds.accuracy,
                    "valid_gaze_samples": len(ds.valid_ids),
                    "intersection_size": ds.intersection_size,
                    "intersection_mean_gaze_normalized_l2_error": ds.intersection_mean_error,
                    "missing_intersection_errors": ds.missing_intersection_errors,
                    "mean_gaze_normalized_l2_error": ds.mean_normalized_error,
                }
            )


def write_invalid_coordinate_ids_json(path: Path, datasets: Iterable[DatasetMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        str(ds.path): {
            "adapter_path": ds.adapter_path,
            "count": len(ds.invalid_coordinate_ids),
            "image_ids": ds.invalid_coordinate_ids,
        }
        for ds in datasets
    }
    path.write_text(json.dumps(payload, indent=2))


def main() -> int:
    args = parse_args()
    files = find_localization_files(args.root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json_path = args.output_json or (args.root / f"localization_top_diffs_{timestamp}.json")
    output_csv_path = args.output_csv or (args.root / f"localization_total_metrics_{timestamp}.csv")
    output_invalid_coordinates_json_path = args.output_invalid_coordinates_json or (
        args.root / f"localization_invalid_coordinates_ids_{timestamp}.json"
    )

    print(f"Found {len(files)} localization result file(s) under {args.root}")

    if not files:
        print(f"No localization result JSON files found under {args.root}")
        return 1

    datasets = [load_dataset_metrics(path) for path in files]
    datasets = [ds for ds in datasets if ds.total_samples > 10]
    intersection = compute_intersection_metrics(datasets)
    baseline = None
    if args.baseline_substring:
        baseline = select_baseline(datasets, args.baseline_substring)

    print(f"Evaluated {len(datasets)} localization file(s) under {args.root}")
    print(f"Intersection of valid gaze samples: {len(intersection)}\n")

    dataset_summaries: Dict[str, Dict[str, Any]] = {}

    for ds in datasets:
        path_key = str(ds.path)
        print(format_dataset_reference(ds))
        print(f"  total_samples: {ds.total_samples}")
        print(f"  false_negatives (in_out=1 & missing gaze): {ds.false_negatives}")
        print(f"  false_positives (in_out=0 & gaze present): {ds.false_positives}")
        print(f"  valid_gaze_samples: {len(ds.valid_ids)}")
        print(f"  intersection_size: {ds.intersection_size}")
        print(
            f"  intersection_mean_gaze_normalized_l2_error: {format_float(ds.intersection_mean_error)}"
        )
        print(f"  accuracy: {format_float(ds.accuracy)}")
        if ds.missing_intersection_errors:
            print(
                f"  missing_intersection_errors: {ds.missing_intersection_errors} sample(s) without gaze_normalized_l2_error"
            )
        print()

        dataset_summaries[path_key] = {
            "path": path_key,
            "adapter_path": ds.adapter_path,
            "total_samples": ds.total_samples,
            "recall": ds.recall,
            "precision": ds.precision,
            "accuracy": ds.accuracy,
            "false_negatives": ds.false_negatives,
            "false_positives": ds.false_positives,
            "valid_gaze_samples": len(ds.valid_ids),
            "intersection_size": ds.intersection_size,
            "intersection_mean_gaze_normalized_l2_error": ds.intersection_mean_error,
            "missing_intersection_errors": ds.missing_intersection_errors,
            "mean_gaze_normalized_l2_error": ds.mean_normalized_error,
        }

    top_table_results = select_top_by_intersection_error(datasets, baseline, top_n=5)
    print_top_table(top_table_results, baseline)

    top_diff_results = report_top_differences(baseline, datasets, args.top_k)

    combined_results: Dict[str, Dict[str, Any]] = {}
    for path_key, metrics_summary in dataset_summaries.items():
        diff_summary = top_diff_results.get(path_key, {})
        combined_results[path_key] = {
            **metrics_summary,
            "total_shared_samples": diff_summary.get("total_shared_samples", 0),
            "top_differences": diff_summary.get("top_differences", []),
        }

    summary = {
        "root": str(args.root),
        "generated_at": timestamp,
        "top_k": args.top_k,
        "intersection_size": len(intersection),
        "baseline": {"path": str(baseline.path)} if baseline is not None else None,
        "top_by_intersection_mean": [
            {
                "path": str(ds.path),
                "adapter_path": "baseline" if baseline is not None and ds is baseline else ds.adapter_path,
                "intersection_normalized_l2": ds.intersection_mean_error,
                "intersection_size": ds.intersection_size,
                "recall": ds.recall,
                "precision": ds.precision,
                "accuracy": ds.accuracy,
                "normalized_l2": ds.mean_normalized_error,
            }
            for ds in top_table_results
        ],
        "datasets": combined_results,
    }
    write_top_diff_json(output_json_path, summary)
    write_metrics_csv(output_csv_path, datasets)
    write_invalid_coordinate_ids_json(output_invalid_coordinates_json_path, datasets)
    print(f"Top difference details saved to {output_json_path}")
    print(f"Total metrics CSV saved to {output_csv_path}")
    print(f"Invalid coordinate IDs JSON saved to {output_invalid_coordinates_json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
