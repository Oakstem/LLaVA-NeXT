#!/usr/bin/env python3
"""Analyze Vacation video scene-label CSV results against GT and baseline."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


GT_TO_LABEL = {
    "AvertGaze": "Gaze Aversion",
    "GazeFollow": "Gaze Following",
    "JointAtt": "Joint Attention",
    "MutualGaze": "Mutual Gaze",
    "SingleGaze": "SingleGaze",
}
LABELS = [
    "SingleGaze",
    "Mutual Gaze",
    "Gaze Aversion",
    "Gaze Following",
    "Joint Attention",
]
RESULT_PREFIX = "test_annotations_with_scene_results_"


@dataclass
class SceneRecord:
    video_id: str
    scene_id: str
    gt_event_attribute: str
    gt_label: str
    row_count: int
    scene_start_frame: str = ""
    scene_end_frame: str = ""
    scene_sampled_frame_indices: str = ""
    pred_label: str = ""
    generated_text: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return self.video_id, self.scene_id

    @property
    def has_prediction(self) -> bool:
        return bool(self.pred_label)

    @property
    def correct(self) -> bool:
        return self.has_prediction and self.pred_label == self.gt_label


def clean_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


LABEL_BY_CLEAN = {
    **{clean_label(label): label for label in LABELS},
    clean_label("Non-communicative"): "SingleGaze",
    clean_label("Non communicative"): "SingleGaze",
    clean_label("Single Gaze"): "SingleGaze",
}


def normalize_pred_label(text: str) -> str:
    return LABEL_BY_CLEAN.get(clean_label(text.strip()), "")


def first_nonempty(values: Iterable[str]) -> str:
    for value in values:
        if value:
            return value
    return ""


def run_name(path: Path) -> str:
    stem = path.stem
    if stem.startswith(RESULT_PREFIX):
        stem = stem[len(RESULT_PREFIX) :]
    return stem or path.stem


def sort_key(key: tuple[str, str]) -> tuple[int, int | str]:
    video_id, scene_id = key
    try:
        video_sort = int(video_id)
    except ValueError:
        video_sort = 0
    match = re.search(r"(\d+)$", scene_id)
    scene_sort: int | str = int(match.group(1)) if match else scene_id
    return video_sort, scene_sort


def load_scene_records(path: Path) -> dict[tuple[str, str], SceneRecord]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            grouped[(str(row.get("video_id", "")).strip(), str(row.get("scene_id", "")).strip())].append(row)

    records: dict[tuple[str, str], SceneRecord] = {}
    for (video_id, scene_id), rows in grouped.items():
        gt_counts = Counter((row.get("event_attribute") or "").strip() for row in rows)
        gt_counts.pop("", None)
        gt_event_attribute = gt_counts.most_common(1)[0][0] if gt_counts else ""
        gt_label = GT_TO_LABEL.get(gt_event_attribute, "")

        pred_values = [(row.get("scene_extracted_label") or "").strip() for row in rows]
        text_values = [(row.get("scene_generated_text") or "").strip() for row in rows]
        raw_pred = first_nonempty(pred_values)
        records[(video_id, scene_id)] = SceneRecord(
            video_id=video_id,
            scene_id=scene_id,
            gt_event_attribute=gt_event_attribute,
            gt_label=gt_label,
            row_count=len(rows),
            scene_start_frame=first_nonempty((row.get("scene_start_frame") or "").strip() for row in rows),
            scene_end_frame=first_nonempty((row.get("scene_end_frame") or "").strip() for row in rows),
            scene_sampled_frame_indices=first_nonempty(
                (row.get("scene_sampled_frame_indices") or "").strip() for row in rows
            ),
            pred_label=normalize_pred_label(raw_pred),
            generated_text=first_nonempty(text_values),
        )
    return records


def pct(numerator: int, denominator: int) -> str:
    return f"{(100.0 * numerator / denominator):.2f}%" if denominator else "n/a"


def summarize_run(records: dict[tuple[str, str], SceneRecord]) -> dict[str, object]:
    supported = [record for record in records.values() if record.gt_label]
    predicted = [record for record in supported if record.has_prediction]
    correct = sum(record.correct for record in predicted)
    by_label = {}
    for label in LABELS:
        label_records = [record for record in supported if record.gt_label == label]
        label_predicted = [record for record in label_records if record.has_prediction]
        label_correct = sum(record.correct for record in label_predicted)
        by_label[label] = {
            "support": len(label_records),
            "predicted": len(label_predicted),
            "correct": label_correct,
            "accuracy": label_correct / len(label_predicted) if label_predicted else None,
            "accuracy_missing_incorrect": label_correct / len(label_records) if label_records else None,
        }
    return {
        "scenes": len(records),
        "supported": len(supported),
        "predicted": len(predicted),
        "missing": len(supported) - len(predicted),
        "correct": correct,
        "accuracy": correct / len(predicted) if predicted else None,
        "accuracy_missing_incorrect": correct / len(supported) if supported else None,
        "by_label": by_label,
    }


def compare_to_baseline(
    records: dict[tuple[str, str], SceneRecord],
    baseline_records: dict[tuple[str, str], SceneRecord],
) -> dict[str, int]:
    counts = Counter()
    for key, record in records.items():
        baseline = baseline_records.get(key)
        if not record.gt_label or baseline is None or not baseline.gt_label:
            continue
        if not record.has_prediction and not baseline.has_prediction:
            counts["both_missing"] += 1
        elif not record.has_prediction:
            counts["model_missing"] += 1
        elif not baseline.has_prediction:
            counts["baseline_missing"] += 1
        elif record.correct and not baseline.correct:
            counts["model_win"] += 1
        elif baseline.correct and not record.correct:
            counts["model_loss"] += 1
        elif record.correct and baseline.correct:
            counts["same_correct"] += 1
        elif record.pred_label == baseline.pred_label:
            counts["same_wrong"] += 1
        else:
            counts["both_wrong_different"] += 1
        if record.has_prediction and baseline.has_prediction:
            counts["shared_predictions"] += 1
            if record.pred_label == baseline.pred_label:
                counts["prediction_agreement"] += 1
    return dict(counts)


def scene_vs_baseline(record: SceneRecord | None, baseline: SceneRecord | None) -> str:
    if record is None or baseline is None or not baseline.gt_label:
        return ""
    if not record.has_prediction and not baseline.has_prediction:
        return "both_missing"
    if not record.has_prediction:
        return "model_missing"
    if not baseline.has_prediction:
        return "baseline_missing"
    if record.correct and not baseline.correct:
        return "model_win"
    if baseline.correct and not record.correct:
        return "model_loss"
    if record.correct and baseline.correct:
        return "same_correct"
    if record.pred_label == baseline.pred_label:
        return "same_wrong"
    return "both_wrong_different"


def write_details_csv(
    path: Path,
    all_records: dict[str, dict[tuple[str, str], SceneRecord]],
    baseline_name: str,
    include_generated_text: bool,
) -> None:
    names = list(all_records)
    baseline_records = all_records[baseline_name]
    keys = sorted(set().union(*(records.keys() for records in all_records.values())), key=sort_key)
    fields = [
        "video_id",
        "scene_id",
        "gt_event_attribute",
        "gt_label",
        "annotation_rows",
        "scene_start_frame",
        "scene_end_frame",
        "scene_sampled_frame_indices",
    ]
    for name in names:
        fields.extend([f"{name}__pred_label", f"{name}__correct"])
        if name != baseline_name:
            fields.extend([f"{name}__vs_baseline", f"{name}__same_prediction_as_baseline"])
        if include_generated_text:
            fields.append(f"{name}__generated_text")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in keys:
            base = baseline_records.get(key)
            anchor = base or first_nonempty(records.get(key) for records in all_records.values())
            row = {
                "video_id": key[0],
                "scene_id": key[1],
                "gt_event_attribute": anchor.gt_event_attribute if anchor else "",
                "gt_label": anchor.gt_label if anchor else "",
                "annotation_rows": anchor.row_count if anchor else "",
                "scene_start_frame": anchor.scene_start_frame if anchor else "",
                "scene_end_frame": anchor.scene_end_frame if anchor else "",
                "scene_sampled_frame_indices": anchor.scene_sampled_frame_indices if anchor else "",
            }
            for name in names:
                record = all_records[name].get(key)
                row[f"{name}__pred_label"] = record.pred_label if record else ""
                row[f"{name}__correct"] = str(record.correct).lower() if record and record.has_prediction else ""
                if name != baseline_name:
                    row[f"{name}__vs_baseline"] = scene_vs_baseline(record, base)
                    row[f"{name}__same_prediction_as_baseline"] = (
                        str(record.has_prediction and base.has_prediction and record.pred_label == base.pred_label).lower()
                        if record and base
                        else ""
                    )
                if include_generated_text:
                    row[f"{name}__generated_text"] = record.generated_text if record else ""
            writer.writerow(row)


def format_summary(
    all_records: dict[str, dict[tuple[str, str], SceneRecord]],
    baseline_name: str,
    source_files: dict[str, Path],
) -> str:
    lines = ["Vacation Video Results Analysis", ""]
    lines.append(f"Baseline: {baseline_name} ({source_files[baseline_name]})")
    lines.append("Primary accuracy is computed over scenes with a prediction; coverage reports predicted / GT scenes.")
    lines.append("")
    lines.append("Overall")
    summaries = {name: summarize_run(records) for name, records in all_records.items()}
    for name, summary in summaries.items():
        correct = int(summary["correct"])
        predicted = int(summary["predicted"])
        supported = int(summary["supported"])
        scenes = int(summary["scenes"])
        lines.append(
            f"- {name}: {correct}/{predicted} correct, acc={pct(correct, predicted)}, "
            f"coverage={predicted}/{supported} ({pct(predicted, supported)}), "
            f"unsupported_gt={scenes - supported}, "
            f"missing-as-wrong={pct(correct, supported)}"
        )
        if name != baseline_name:
            comparison = compare_to_baseline(all_records[name], all_records[baseline_name])
            shared = comparison.get("shared_predictions", 0)
            agreement = comparison.get("prediction_agreement", 0)
            lines.append(
                f"  vs baseline: wins={comparison.get('model_win', 0)}, "
                f"losses={comparison.get('model_loss', 0)}, same_correct={comparison.get('same_correct', 0)}, "
                f"same_wrong={comparison.get('same_wrong', 0)}, both_wrong_different={comparison.get('both_wrong_different', 0)}, "
                f"prediction_agreement={agreement}/{shared} ({pct(agreement, shared)})"
            )

    for name, summary in summaries.items():
        lines.append("")
        lines.append(f"Per-label Accuracy: {name}")
        by_label = summary["by_label"]
        for label in LABELS:
            row = by_label[label]
            correct = int(row["correct"])
            predicted = int(row["predicted"])
            support = int(row["support"])
            lines.append(
                f"- {label}: {correct}/{predicted} correct, acc={pct(correct, predicted)}, "
                f"coverage={predicted}/{support} ({pct(predicted, support)}), "
                f"missing-as-wrong={pct(correct, support)}"
            )
    return "\n".join(lines) + "\n"


def find_baseline(files: list[Path], explicit_baseline: str | None) -> Path:
    if explicit_baseline:
        baseline = Path(explicit_baseline)
        if baseline.exists():
            return baseline
        for candidate in (Path.cwd() / baseline, files[0].parent / baseline):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Baseline CSV not found: {explicit_baseline}")
    candidates = [path for path in files if "baseline" in path.stem.casefold()]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one baseline CSV, found {len(candidates)}: {candidates}")
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("vacation_results/videos"))
    parser.add_argument("--pattern", default="test_annotations_with_scene_results*.csv", help="CSV glob under --results-dir.")
    parser.add_argument("--baseline-csv", default=None, help="Baseline CSV path. Defaults to the only CSV with 'baseline' in its name.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for analysis outputs. Defaults to <results-dir>/analysis.")
    parser.add_argument("--summary-txt", type=Path, default=None)
    parser.add_argument("--details-csv", type=Path, default=None)
    parser.add_argument("--include-generated-text", action="store_true", help="Include full model responses in the detailed CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(path for path in args.results_dir.glob(args.pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No CSV files found under {args.results_dir} matching {args.pattern}")

    baseline_path = find_baseline(files, args.baseline_csv).resolve()
    ordered_files = [baseline_path] + [path.resolve() for path in files if path.resolve() != baseline_path]
    source_files = {run_name(path): path for path in ordered_files}
    all_records = {name: load_scene_records(path) for name, path in source_files.items()}

    baseline_name = run_name(baseline_path)
    output_dir = args.output_dir or args.results_dir / "analysis"
    summary_txt = args.summary_txt or output_dir / "vacation_video_analysis_summary.txt"
    details_csv = args.details_csv or output_dir / "vacation_video_analysis_details.csv"

    summary = format_summary(all_records, baseline_name, source_files)
    print(summary, end="")
    summary_txt.parent.mkdir(parents=True, exist_ok=True)
    summary_txt.write_text(summary, encoding="utf-8")
    write_details_csv(details_csv, all_records, baseline_name, args.include_generated_text)
    print(f"\nWrote summary: {summary_txt}")
    print(f"Wrote detailed CSV: {details_csv}")


if __name__ == "__main__":
    main()
