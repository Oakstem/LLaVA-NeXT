#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vacation.extraction_metrics_utils import compute_extraction_metrics


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
        description="Recompute extraction metrics for vacation batch results."
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
        help="CSV output path (defaults to <results_dir>/combined_metrics_<timestamp>.csv).",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = args.output_csv or (results_dir / f"combined_metrics_{timestamp}.csv")

    rows = []
    aggregate = {
        "total_results": 0,
        "total_extractions": 0,
        "valid_social_interaction_labels": 0,
        "null_social_interaction_labels": 0,
        "accuracy_correct_nb": 0,
        "accuracy_evaluated_frames": 0,
        "accuracy_skipped_missing_gt": 0,
        "accuracy_skipped_missing_pred": 0,
    }

    for json_path in tqdm(_iter_result_files(results_dir), desc="Computing metrics"):
        results = _load_results(json_path)
        if results is None:
            continue
        metrics = compute_extraction_metrics(results)
        rows.append(
            {
                "results_file": str(json_path),
                **metrics,
            }
        )
        aggregate["total_results"] += metrics["total_results"]
        aggregate["total_extractions"] += metrics["total_extractions"]
        aggregate["valid_social_interaction_labels"] += metrics[
            "valid_social_interaction_labels"
        ]
        aggregate["null_social_interaction_labels"] += metrics[
            "null_social_interaction_labels"
        ]
        aggregate["accuracy_correct_nb"] += metrics["accuracy_correct_nb"]
        aggregate["accuracy_evaluated_frames"] += metrics["accuracy_evaluated_frames"]
        aggregate["accuracy_skipped_missing_gt"] += metrics[
            "accuracy_skipped_missing_gt"
        ]
        aggregate["accuracy_skipped_missing_pred"] += metrics[
            "accuracy_skipped_missing_pred"
        ]

    if rows:
        total_extractions = aggregate["total_extractions"]
        evaluated = aggregate["accuracy_evaluated_frames"]
        rows.append(
            {
                "results_file": "ALL",
                "total_results": aggregate["total_results"],
                "total_extractions": total_extractions,
                "valid_social_interaction_labels": aggregate[
                    "valid_social_interaction_labels"
                ],
                "null_social_interaction_labels": aggregate[
                    "null_social_interaction_labels"
                ],
                "valid_social_interaction_label_percent": (
                    aggregate["valid_social_interaction_labels"] / total_extractions
                    if total_extractions
                    else 0.0
                ),
                "accuracy_vs_atomic_attribute_combo": (
                    aggregate["accuracy_correct_nb"] / evaluated if evaluated else 0.0
                ),
                "accuracy_evaluated_frames": evaluated,
                "accuracy_correct_nb": aggregate["accuracy_correct_nb"],
                "accuracy_skipped_missing_gt": aggregate[
                    "accuracy_skipped_missing_gt"
                ],
                "accuracy_skipped_missing_pred": aggregate[
                    "accuracy_skipped_missing_pred"
                ],
            }
        )

    write_csv(rows, output_csv)
    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
