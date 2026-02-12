#!/usr/bin/env python3
"""Build Stage-1b hard-negative ROI CSV using Qwen3VL detections."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gazefollow.auto_phrase_grounding.qwen3vl_grounding import load_qwen3vl_model, run_qwen3vl_grounding


DEFAULT_QUERY = (
    "Detect all visible people and objects in the image. "
    "Return ONLY a JSON array where each element is "
    "{\"label\": string, \"bbox\": [x1,y1,x2,y2], \"score\": number}. "
    "Use bbox coordinates normalized to [0,1000]."
)

PERSON_LABEL_HINTS = ("person", "man", "woman", "boy", "girl", "people", "child", "kid", "adult")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CSV of hard-negative ROIs for Stage 1b.")
    parser.add_argument("--dataset-json", required=True, help="Dataset JSON path.")
    parser.add_argument("--images-dir", required=True, help="Root image directory.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct", help="Qwen3VL model id.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Grounding query prompt.")
    parser.add_argument("--device-map", default="auto", help="Device map passed to model loader.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max generation tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of samples.")
    parser.add_argument("--start-index", type=int, default=0, help="Optional starting sample index.")
    parser.add_argument(
        "--remove-policy",
        choices=("smallest", "all"),
        default="smallest",
        help="How to remove detections containing GT point.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append to existing output CSV and skip already-seen sample ids.",
    )
    return parser.parse_args()


def load_dataset(dataset_json: Path) -> List[Dict[str, Any]]:
    with dataset_json.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list dataset, got {type(data)}")
    return data


def resolve_image_path(sample: Dict[str, Any], images_dir: Path) -> Optional[Path]:
    candidates: List[str] = []
    for key in ("image", "image_path", "id"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    for raw_value in candidates:
        value = raw_value.replace("\\", "/")
        path_value = Path(value)
        if path_value.is_absolute() and path_value.exists():
            return path_value

        joined = images_dir / value
        if joined.exists():
            return joined
        if joined.suffix:
            continue
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            alt = joined.with_suffix(ext)
            if alt.exists():
                return alt
    return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_ground_truth(sample: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    in_out = sample.get("in_out")
    if in_out is not None and int(in_out) == 0:
        return None

    gaze_x = _safe_float(sample.get("gaze_gt_x"))
    gaze_y = _safe_float(sample.get("gaze_gt_y"))
    gaze_w = _safe_float(sample.get("gaze_gt_width"))
    gaze_h = _safe_float(sample.get("gaze_gt_height"))
    if None not in (gaze_x, gaze_y, gaze_w, gaze_h):
        return gaze_x, gaze_y, gaze_w, gaze_h

    nested = sample.get("gaze_ground_truth")
    if isinstance(nested, dict):
        gaze_x = _safe_float(nested.get("x"))
        gaze_y = _safe_float(nested.get("y"))
        gaze_w = _safe_float(nested.get("image_width"))
        gaze_h = _safe_float(nested.get("image_height"))
        if None not in (gaze_x, gaze_y, gaze_w, gaze_h):
            return gaze_x, gaze_y, gaze_w, gaze_h

    return None


def _extract_label(detection: Dict[str, Any]) -> str:
    for key in ("label", "object", "name", "class", "category", "entity", "description", "text"):
        value = detection.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_score(detection: Dict[str, Any]) -> Optional[float]:
    for key in ("score", "confidence", "probability", "prob"):
        value = detection.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _normalize_bbox(bbox: Sequence[Any]) -> Optional[List[int]]:
    if len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def normalize_detection(detection: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    bbox_raw = detection.get("bbox")
    if not isinstance(bbox_raw, Sequence):
        return None
    bbox = _normalize_bbox(bbox_raw)
    if bbox is None:
        return None

    label = _extract_label(detection)
    label_lower = label.lower()
    category = "person" if any(hint in label_lower for hint in PERSON_LABEL_HINTS) else "object"
    center = [
        round((bbox[0] + bbox[2]) / 2.0, 3),
        round((bbox[1] + bbox[3]) / 2.0, 3),
    ]
    return {
        "label": label,
        "category": category,
        "score": _extract_score(detection),
        "bbox": bbox,
        "bbox_center": center,
    }


def bbox_contains_point(bbox: Sequence[int], x: float, y: float) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def split_positive_and_negatives(
    detections: List[Dict[str, Any]],
    gt_x: float,
    gt_y: float,
    remove_policy: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    positive_indices = [
        idx for idx, det in enumerate(detections) if bbox_contains_point(det["bbox"], gt_x, gt_y)
    ]
    if not positive_indices:
        return [], detections

    if remove_policy == "all":
        keep_idx = {idx for idx in range(len(detections)) if idx not in set(positive_indices)}
        positives = [detections[idx] for idx in positive_indices]
        negatives = [detections[idx] for idx in range(len(detections)) if idx in keep_idx]
        return positives, negatives

    best_idx = min(
        positive_indices,
        key=lambda idx: (detections[idx]["bbox"][2] - detections[idx]["bbox"][0])
        * (detections[idx]["bbox"][3] - detections[idx]["bbox"][1]),
    )
    positives = [detections[best_idx]]
    negatives = [detections[idx] for idx in range(len(detections)) if idx != best_idx]
    return positives, negatives


def load_seen_sample_ids(output_csv: Path) -> set[str]:
    seen: set[str] = set()
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = row.get("sample_id")
            if sample_id:
                seen.add(sample_id)
    return seen


def serialize_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, separators=(",", ":"))


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_json).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_path)
    if args.start_index > 0:
        dataset = dataset[args.start_index :]
    if args.limit is not None:
        dataset = dataset[: args.limit]

    seen_sample_ids: set[str] = set()
    write_mode = "w"
    if args.resume and output_csv.exists():
        seen_sample_ids = load_seen_sample_ids(output_csv)
        write_mode = "a"

    fieldnames = [
        "sample_index",
        "sample_id",
        "image_key",
        "image_path",
        "in_out",
        "gt_x",
        "gt_y",
        "gt_width",
        "gt_height",
        "status",
        "detections_total",
        "removed_positive_count",
        "removed_positive_json",
        "negative_count",
        "negative_detections_json",
    ]

    processor = None
    model = None
    detection_cache: Dict[str, List[Dict[str, Any]]] = {}
    error_cache: Dict[str, str] = {}

    with output_csv.open(write_mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_mode == "w":
            writer.writeheader()

        for local_idx, sample in enumerate(tqdm(dataset, desc="Building stage1b negatives")):
            sample_id = str(sample.get("id", args.start_index + local_idx))
            if sample_id in seen_sample_ids:
                continue

            image_key = str(sample.get("image") or sample.get("image_path") or sample_id)
            image_path = resolve_image_path(sample, images_dir)

            row = {
                "sample_index": args.start_index + local_idx,
                "sample_id": sample_id,
                "image_key": image_key,
                "image_path": str(image_path) if image_path else "",
                "in_out": sample.get("in_out", ""),
                "gt_x": "",
                "gt_y": "",
                "gt_width": "",
                "gt_height": "",
                "status": "ok",
                "detections_total": 0,
                "removed_positive_count": 0,
                "removed_positive_json": "[]",
                "negative_count": 0,
                "negative_detections_json": "[]",
            }

            if image_path is None:
                row["status"] = "missing_image"
                writer.writerow(row)
                continue

            cache_key = str(image_path)
            if cache_key not in detection_cache:
                try:
                    if processor is None or model is None:
                        processor, model = load_qwen3vl_model(args.model_id, device_map=args.device_map)
                    raw_detections, _ = run_qwen3vl_grounding(
                        image_path=cache_key,
                        query=args.query,
                        model_id=args.model_id,
                        max_new_tokens=args.max_new_tokens,
                        processor=processor,
                        model=model,
                        device_map=args.device_map,
                        temperature=args.temperature,
                    )
                    normalized = []
                    for det in raw_detections:
                        if isinstance(det, dict):
                            parsed = normalize_detection(det)
                            if parsed is not None:
                                normalized.append(parsed)
                    detection_cache[cache_key] = normalized
                except Exception as exc:  # noqa: BLE001
                    detection_cache[cache_key] = []
                    error_cache[cache_key] = str(exc)

            if cache_key in error_cache:
                row["status"] = f"inference_error:{error_cache[cache_key]}"

            detections = detection_cache[cache_key]
            row["detections_total"] = len(detections)

            gt = extract_ground_truth(sample)
            if gt is None:
                row["status"] = "no_gt" if row["status"] == "ok" else row["status"]
                negatives = detections
                positives: List[Dict[str, Any]] = []
            else:
                gt_x, gt_y, gt_w, gt_h = gt
                row["gt_x"] = gt_x
                row["gt_y"] = gt_y
                row["gt_width"] = gt_w
                row["gt_height"] = gt_h
                positives, negatives = split_positive_and_negatives(detections, gt_x, gt_y, args.remove_policy)

            row["removed_positive_count"] = len(positives)
            row["removed_positive_json"] = serialize_json(positives)
            row["negative_count"] = len(negatives)
            row["negative_detections_json"] = serialize_json(negatives)
            writer.writerow(row)

    print(f"Saved Stage-1b negatives CSV to {output_csv}")


if __name__ == "__main__":
    main()
