#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def _normalize_social_label(label: str | None) -> str | None:
    if label is None:
        return None
    if not isinstance(label, str):
        label = str(label)
    cleaned = label.strip().lower()
    if cleaned in {"", "null", "none"}:
        return None
    cleaned = cleaned.replace("-", " ").replace("_", " ")
    cleaned = " ".join(cleaned.split())
    if cleaned in {"mutualgaze", "mutual gaze", "mutual"}:
        return "MutualGaze"
    if cleaned in {
        "sharedobjectattention",
        "shared object attention",
        "shared attention",
        "joint attention",
        "joint att",
        "jointatt",
    }:
        return "SharedObjectAttention"
    if cleaned in {
        "onesidedgaze",
        "one sided gaze",
        "one sided",
        "single",
        "single gaze",
    }:
        return "OneSidedGaze"
    if cleaned in {
        "noncommmunicative",
        "noncommunicative",
        "non communicative",
        "non communicative gaze",
        "unclear",
    }:
        return "NonCommmunicative"
    return None


def _get_gt_label(entry: dict) -> str | None:
    if "atomic_attribute_combo_GT" in entry:
        return entry.get("atomic_attribute_combo_GT")
    return entry.get("atomic_attribute_combo")


def compute_metrics(results: list[dict]) -> dict:
    total_results = len(results)
    extracted_entries = [
        r for r in results if isinstance(r.get("extracted_gaze_info"), dict)
    ]
    total_extractions = len(extracted_entries)

    valid_labels = 0
    for entry in extracted_entries:
        pred = entry["extracted_gaze_info"].get("social_interaction_label")
        if _normalize_social_label(pred) is not None:
            valid_labels += 1

    valid_percent = (valid_labels / total_extractions) if total_extractions else 0.0

    evaluated = 0
    correct = 0
    skipped_missing_gt = 0
    skipped_missing_pred = 0
    for entry in extracted_entries:
        pred_norm = _normalize_social_label(
            entry["extracted_gaze_info"].get("social_interaction_label")
        )
        gt_norm = _normalize_social_label(_get_gt_label(entry))
        if pred_norm is None:
            skipped_missing_pred += 1
            continue
        if gt_norm is None:
            skipped_missing_gt += 1
            continue
        evaluated += 1
        if pred_norm == gt_norm:
            correct += 1

    accuracy = (correct / evaluated) if evaluated else 0.0

    response_count = 0
    response_length_sum = 0
    for entry in results:
        response = entry.get("response")
        if response is None:
            continue
        response_count += 1
        response_length_sum += len(str(response))

    avg_response_length = (
        response_length_sum / response_count if response_count else 0.0
    )

    return {
        "total_results": total_results,
        "total_extractions": total_extractions,
        "valid_social_interaction_label_percent": valid_percent,
        "accuracy_vs_atomic_attribute_combo": accuracy,
        "accuracy_evaluated_frames": evaluated,
        "accuracy_correct_nb": correct,
        "accuracy_skipped_missing_gt": skipped_missing_gt,
        "accuracy_skipped_missing_pred": skipped_missing_pred,
        "avg_response_length": avg_response_length,
        "response_count": response_count,
    }


def _load_results(path: Path) -> list[dict] | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if isinstance(results, list):
        return results
    return None


def _iter_result_files(results_dir: Path) -> list[Path]:
    return sorted(p for p in results_dir.rglob("*.json") if p.is_file())


def write_csv(rows: list[dict], output_csv: Path) -> None:
    if not rows:
        output_csv.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute vacation batch metrics for a results directory."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing vacation results JSON files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="CSV output path (defaults to <results_dir>/combined_metrics.csv).",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    output_csv = args.output_csv or (results_dir / "combined_metrics.csv")

    rows = []
    aggregate = {
        "total_results": 0,
        "total_extractions": 0,
        "valid_labels": 0,
        "accuracy_correct_nb": 0,
        "accuracy_evaluated_frames": 0,
        "accuracy_skipped_missing_gt": 0,
        "accuracy_skipped_missing_pred": 0,
        "response_length_sum": 0,
        "response_count": 0,
    }

    for json_path in _iter_result_files(results_dir):
        results = _load_results(json_path)
        if results is None:
            continue
        metrics = compute_metrics(results)
        rows.append(
            {
                "results_file": str(json_path),
                **metrics,
            }
        )

        aggregate["total_results"] += metrics["total_results"]
        aggregate["total_extractions"] += metrics["total_extractions"]
        aggregate["valid_labels"] += metrics["valid_social_interaction_label_percent"] * metrics["total_extractions"]
        aggregate["accuracy_correct_nb"] += metrics["accuracy_correct_nb"]
        aggregate["accuracy_evaluated_frames"] += metrics["accuracy_evaluated_frames"]
        aggregate["accuracy_skipped_missing_gt"] += metrics["accuracy_skipped_missing_gt"]
        aggregate["accuracy_skipped_missing_pred"] += metrics["accuracy_skipped_missing_pred"]
        aggregate["response_length_sum"] += metrics["avg_response_length"] * metrics["response_count"]
        aggregate["response_count"] += metrics["response_count"]

    if rows:
        total_extractions = aggregate["total_extractions"]
        evaluated = aggregate["accuracy_evaluated_frames"]
        response_count = aggregate["response_count"]
        rows.append(
            {
                "results_file": "ALL",
                "total_results": aggregate["total_results"],
                "total_extractions": total_extractions,
                "valid_social_interaction_label_percent": (
                    aggregate["valid_labels"] / total_extractions
                    if total_extractions
                    else 0.0
                ),
                "accuracy_vs_atomic_attribute_combo": (
                    aggregate["accuracy_correct_nb"] / evaluated if evaluated else 0.0
                ),
                "accuracy_evaluated_frames": evaluated,
                "accuracy_correct_nb": aggregate["accuracy_correct_nb"],
                "accuracy_skipped_missing_gt": aggregate["accuracy_skipped_missing_gt"],
                "accuracy_skipped_missing_pred": aggregate["accuracy_skipped_missing_pred"],
                "avg_response_length": (
                    aggregate["response_length_sum"] / response_count
                    if response_count
                    else 0.0
                ),
                "response_count": response_count,
            }
        )

    write_csv(rows, output_csv)
    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
