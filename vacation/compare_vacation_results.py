#!/usr/bin/env python3
"""Compare Vacation model results against GT annotations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SKIP_EVENT_ATTRS = {"AvertGaze", "GazeFollow"}
GT_TO_COARSE = {
    "MutualGaze": "mutual gaze",
    "JointAtt": "joint attention toward a shared object",
    "SingleGaze": "non-communicative gaze",
}
VALID_PRED = set(GT_TO_COARSE.values())


def load_results(path: Path) -> Tuple[Dict[Tuple[str, str], dict], int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        results = data.get("results", [])
    elif isinstance(data, list):
        results = data
    else:
        raise ValueError(f"Unsupported results JSON shape in {path}")

    result_map: Dict[Tuple[str, str], dict] = {}
    dupes = 0
    for entry in results:
        try:
            key = (str(entry["video_id"]), str(entry["frame_id"]))
        except KeyError:
            continue
        if key in result_map:
            dupes += 1
            continue
        result_map[key] = entry
    return result_map, dupes


def load_gt_frames(path: Path) -> Dict[Tuple[str, str], List[str]]:
    gt_by_frame: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (str(row.get("video_id", "")).strip(), str(row.get("frame_id", "")).strip())
            event_attr = str(row.get("event_attribute", "")).strip()
            if key == ("", ""):
                continue
            if event_attr:
                gt_by_frame[key].append(event_attr)
    return gt_by_frame


def summarize_counts(counter: Counter, labels: Iterable[str]) -> str:
    return ", ".join(f"{label}={counter.get(label, 0)}" for label in labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Vacation results JSON against GT CSV.")
    parser.add_argument("--results", required=True, help="Path to results JSON file.")
    parser.add_argument("--gt", required=True, help="Path to test_annotations.csv.")
    parser.add_argument("--out", required=True, help="Output CSV path for per-frame results.")
    parser.add_argument("--summary-out", default=None, help="Optional JSON summary output path.")
    args = parser.parse_args()

    results_path = Path(args.results)
    gt_path = Path(args.gt)
    out_path = Path(args.out)

    result_map, dupes = load_results(results_path)
    gt_by_frame = load_gt_frames(gt_path)

    total_frames = len(gt_by_frame)
    skipped_multi = 0
    skipped_event = 0
    skipped_unsupported = 0
    evaluated = 0
    correct = 0
    incorrect = 0

    missing_result = 0
    missing_prediction = 0
    unknown_prediction = 0
    mismatches = 0

    confusion = Counter()  # (gt, pred) -> count

    out_rows = []
    multi_event_frames = []

    for key, attrs in gt_by_frame.items():
        video_id, frame_id = key
        uniq_attrs = sorted(set(attrs))

        included = True
        reason = ""
        gt_event_attr = ""
        gt_coarse = ""

        if len(uniq_attrs) == 0:
            included = False
            reason = "missing_event_attribute"
        elif len(uniq_attrs) > 1:
            included = False
            reason = f"multiple_event_attribute:{'|'.join(uniq_attrs)}"
            skipped_multi += 1
            multi_event_frames.append(key)
        else:
            gt_event_attr = uniq_attrs[0]
            if gt_event_attr in SKIP_EVENT_ATTRS:
                included = False
                reason = "skipped_event_attribute"
                skipped_event += 1
            elif gt_event_attr in GT_TO_COARSE:
                gt_coarse = GT_TO_COARSE[gt_event_attr]
            else:
                included = False
                reason = f"unsupported_event_attribute:{gt_event_attr}"
                skipped_unsupported += 1

        entry = result_map.get(key)
        response = ""
        pred_raw: Optional[str] = None
        pred_coarse: Optional[str] = None
        if entry is not None:
            response = entry.get("response", "") or ""
            pred_raw = (
                entry.get("extracted_gaze_info", {}) or {}
            ).get("inferred_gaze_interaction")
            if pred_raw in VALID_PRED:
                pred_coarse = pred_raw
        else:
            pred_raw = None

        if included and entry is None:
            included = False
            missing_result += 1
            reason = "missing_result"

        match = ""
        if included:
            evaluated += 1
            if pred_coarse is None:
                incorrect += 1
                if pred_raw is None:
                    missing_prediction += 1
                    reason = "missing_prediction"
                else:
                    unknown_prediction += 1
                    reason = "unknown_prediction"
                match = "false"
            else:
                is_match = pred_coarse == gt_coarse
                match = "true" if is_match else "false"
                if is_match:
                    correct += 1
                else:
                    incorrect += 1
                    mismatches += 1
                    reason = "mismatch"
                confusion[(gt_coarse, pred_coarse)] += 1
        else:
            match = ""

        out_rows.append(
            {
                "video_id": video_id,
                "frame_id": frame_id,
                "gt_event_attribute": gt_event_attr,
                "gt_coarse": gt_coarse,
                "pred_raw": pred_raw or "",
                "pred_coarse": pred_coarse or "",
                "match": match,
                "included_in_metrics": "true" if included else "false",
                "reason": reason,
                "response": response,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "frame_id",
                "gt_event_attribute",
                "gt_coarse",
                "pred_raw",
                "pred_coarse",
                "match",
                "included_in_metrics",
                "reason",
                "response",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    accuracy = (correct / evaluated) if evaluated else 0.0

    summary = {
        "total_gt_frames": total_frames,
        "evaluated_frames": evaluated,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": accuracy,
        "skipped_multiple_event_attribute": skipped_multi,
        "skipped_event_attribute": skipped_event,
        "skipped_unsupported_event_attribute": skipped_unsupported,
        "missing_result": missing_result,
        "missing_prediction": missing_prediction,
        "unknown_prediction": unknown_prediction,
        "mismatches": mismatches,
        "duplicate_results_entries": dupes,
        "confusion_matrix": {
            f"{gt} -> {pred}": count for (gt, pred), count in confusion.items()
        },
    }

    print("Evaluation summary:")
    print(f"- total GT frames: {total_frames}")
    print(f"- evaluated: {evaluated}, correct: {correct}, incorrect: {incorrect}, accuracy: {accuracy:.4f}")
    print(f"- skipped (multiple event_attribute): {skipped_multi}")
    print(f"- skipped (Avert/GazeFollow): {skipped_event}")
    print(f"- skipped (unsupported): {skipped_unsupported}")
    print(f"- missing result entries: {missing_result}")
    print(f"- missing predictions: {missing_prediction}")
    print(f"- unknown predictions: {unknown_prediction}")
    print(f"- mismatches: {mismatches}")
    if dupes:
        print(f"- duplicate result entries skipped: {dupes}")
    if confusion:
        labels = sorted(VALID_PRED)
        print("- confusion matrix counts:")
        for gt_label in labels:
            row = Counter({pred: confusion.get((gt_label, pred), 0) for pred in labels})
            print(f"  {gt_label}: {summarize_counts(row, labels)}")
    if multi_event_frames:
        sample = ", ".join(f"{vid}:{fid}" for vid, fid in multi_event_frames[:5])
        print(f"- sample multi-attribute frames: {sample}")

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
