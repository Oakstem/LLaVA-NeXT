#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from gazefollow.qwen3vl_utils import infer_in_out_from_phrase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare localization performance metrics across evaluation result files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("evaluation_results"),
        help="Root directory to search for localization result JSON files (default: evaluation_results).",
    )
    parser.add_argument(
        "--baseline-substring",
        type=str,
        default="baseline_llava",
        help="Substring to identify the baseline localization file (default: baseline_llava).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of samples with the highest normalized L2 error difference to report (default: 10).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("localization_top_diffs.json"),
        help="Path to write JSON file with top difference details (default: localization_top_diffs.json).",
    )
    return parser.parse_args()


def find_localization_files(root: Path) -> List[Path]:
    if not root.exists():
        return []

    candidates = [
        path
        for path in root.rglob("dataset_*localization*.json")
        if path.is_file()
    ]
    return sorted(candidates)


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
    total_samples: int = 0
    false_negatives: int = 0
    false_positives: int = 0
    valid_ids: Set[str] = field(default_factory=set)
    normalized_errors: Dict[str, float] = field(default_factory=dict)
    gaze_targets: Dict[str, Optional[str]] = field(default_factory=dict)
    in_out_labels: Dict[str, Optional[int]] = field(default_factory=dict)
    missing_intersection_errors: int = 0
    intersection_mean_error: Optional[float] = None


def load_dataset_metrics(path: Path) -> DatasetMetrics:
    dataset = DatasetMetrics(path=path)

    try:
        entries = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse {path}: {exc}") from exc

    if not isinstance(entries, list):
        raise RuntimeError(f"Expected list at {path}, found {type(entries).__name__}")

    dataset.total_samples = len(entries)

    for index, entry in enumerate(entries):
        sample_id = entry.get("id") or entry.get("image_path") or f"index_{index}"
        in_out_flag = normalize_in_out(entry.get("in_out"))

        detection = entry.get("gaze_detections") or {}
        person = detection.get("person_1") or {}
        coords = person.get("gaze_coordinates")
        coords_valid = has_valid_coordinates(coords)
        gaze_target = person.get("gaze_target")

        dataset.in_out_labels[str(sample_id)] = in_out_flag
        inferred_in_out = infer_in_out_from_phrase(gaze_target)
        if inferred_in_out == 0:
            person["gaze_coordinates"] = None
            person["gaze_normalized_l2_error"] = None
            person["gaze_l2_error"] = None
            coords = None
            coords_valid = False

        if in_out_flag == 0:
            for error_key in (
                "gaze_normalized_l2_error",
                "gaze_l2_error",
                "gaze_modified_l2_error",
                "gaze_angular_error",
                "gaze_iou",
            ):
                if error_key in person:
                    person[error_key] = None

        if in_out_flag == 1 and not coords_valid:
            dataset.false_negatives += 1

        if in_out_flag == 0 and coords_valid:
            dataset.false_positives += 1

        dataset.gaze_targets[str(sample_id)] = (
            str(gaze_target) if gaze_target is not None else None
        )

        if coords_valid:
            dataset.valid_ids.add(str(sample_id))
            normalized_error = person.get("gaze_normalized_l2_error")
            if isinstance(normalized_error, (int, float)):
                dataset.normalized_errors[str(sample_id)] = float(normalized_error)

    return dataset


def compute_intersection_metrics(
    datasets: Iterable[DatasetMetrics],
) -> Set[str]:
    datasets = list(datasets)
    if not datasets:
        return set()

    intersection = set.intersection(*(ds.valid_ids for ds in datasets)) if datasets else set()

    for ds in datasets:
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
    baseline: DatasetMetrics,
    datasets: Iterable[DatasetMetrics],
    top_k: int,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}

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
        dataset_result = {
            "total_shared_samples": len(shared_ids),
            "truncated_count": max(len(ranked) - len(top_entries), 0),
            "top_differences": [],
        }

        print(f"Top {len(top_entries)} normalized L2 error diffs vs baseline for {ds.path}:")
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


def main() -> int:
    args = parse_args()
    files = find_localization_files(args.root)

    if not files:
        print(f"No localization result JSON files found under {args.root}")
        return 1

    datasets = [load_dataset_metrics(path) for path in files]
    intersection = compute_intersection_metrics(datasets)
    baseline = select_baseline(datasets, args.baseline_substring)

    print(f"Evaluated {len(datasets)} localization file(s) under {args.root}")
    print(f"Intersection of valid gaze samples: {len(intersection)}\n")

    for ds in datasets:
        print(str(ds.path))
        print(f"  total_samples: {ds.total_samples}")
        print(f"  false_negatives (in_out=1 & missing gaze): {ds.false_negatives}")
        print(f"  false_positives (in_out=0 & gaze present): {ds.false_positives}")
        print(f"  valid_gaze_samples: {len(ds.valid_ids)}")
        print(
            f"  intersection_mean_gaze_normalized_l2_error: {format_float(ds.intersection_mean_error)}"
        )
        if ds.missing_intersection_errors:
            print(
                f"  missing_intersection_errors: {ds.missing_intersection_errors} sample(s) without gaze_normalized_l2_error"
            )
        print()

    top_diff_results = report_top_differences(baseline, datasets, args.top_k)

    summary = {
        "baseline_path": str(baseline.path),
        "top_k": args.top_k,
        "datasets": top_diff_results,
    }
    write_top_diff_json(args.output_json, summary)
    print(f"Top difference details saved to {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
