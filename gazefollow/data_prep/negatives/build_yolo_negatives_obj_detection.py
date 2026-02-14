#!/usr/bin/env python3
"""Build Stage-1b hard-negative ROI CSV using YOLO detections."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CSV of hard-negative ROIs for Stage 1b with YOLO.")
    parser.add_argument(
        "--dataset-json",
        "--dataset-path",
        dest="dataset_path",
        required=True,
        help="Dataset path (.json conversations or .csv annotations).",
    )
    parser.add_argument("--images-dir", required=True, help="Root image directory.")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path. Defaults to training_datasets/stage1b_negatives/<timestamp>.csv",
    )
    parser.add_argument(
        "--model-id",
        default="yolo11x.pt",
        help="YOLO model checkpoint name/path (e.g., yolo11x.pt, yolov9e.pt, /path/to/model.pt).",
    )
    parser.add_argument("--conf-thres", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou-thres", type=float, default=0.70, help="YOLO NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of samples.")
    parser.add_argument(
        "--start-index",
        "--start-from-index",
        dest="start_index",
        type=int,
        default=0,
        help="Optional starting sample index.",
    )
    parser.add_argument(
        "--remove-policy",
        choices=("smallest", "all"),
        default="smallest",
        help="How to remove detections containing GT point.",
    )
    parser.add_argument(
        "--gt-positive-radius-ratio",
        type=float,
        default=0.05,
        help="Radius ratio (w.r.t. min(image_w,image_h)) for GT positive box around gaze point.",
    )
    parser.add_argument(
        "--positive-center-distance-threshold",
        type=float,
        default=0.1,
        help="Normalized center-distance threshold for marking a detection as positive.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append to existing output CSV and skip already-seen sample ids.",
    )
    return parser.parse_args()


def load_json_dataset(dataset_json: Path) -> List[Dict[str, Any]]:
    with dataset_json.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list dataset, got {type(data)}")
    return data


def load_annotations_csv_dataset(dataset_csv: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with dataset_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            image_path = str(row.get("image_path", "")).strip()
            sample_id = str(row.get("id", index)).strip() or str(index)

            sample: Dict[str, Any] = {
                "id": sample_id,
                "image": image_path,
                "image_path": image_path,
            }

            in_or_out = _safe_float(row.get("in_or_out"))
            if in_or_out is not None:
                sample["in_out"] = 1 if int(in_or_out) >= 1 else 0

            gaze_x = _safe_float(row.get("gaze_x"))
            gaze_y = _safe_float(row.get("gaze_y"))
            if gaze_x is not None and gaze_y is not None:
                sample["gaze_gt_x"] = float(gaze_x) * 1000.0
                sample["gaze_gt_y"] = float(gaze_y) * 1000.0
                sample["gaze_gt_width"] = 1000.0
                sample["gaze_gt_height"] = 1000.0

            samples.append(sample)
    return samples


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        return load_annotations_csv_dataset(dataset_path)
    return load_json_dataset(dataset_path)


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


def should_skip_sample(sample: Dict[str, Any]) -> bool:
    in_out = _safe_float(sample.get("in_out"))
    if in_out is None:
        in_out = _safe_float(sample.get("in_or_out"))
    return in_out is not None and int(in_out) == 0


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


def _pixel_to_bbox(
    bbox_xyxy: Sequence[float],
    image_w: int,
    image_h: int,
) -> Optional[List[int]]:
    if image_w <= 0 or image_h <= 0:
        return None

    x1, y1, x2, y2 = bbox_xyxy
    x1 = min(float(image_w), max(0.0, float(x1)))
    y1 = min(float(image_h), max(0.0, float(y1)))
    x2 = min(float(image_w), max(0.0, float(x2)))
    y2 = min(float(image_h), max(0.0, float(y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return _normalize_bbox([x1, y1, x2, y2])


def detection_signature(detection: Dict[str, Any]) -> Tuple[str, Tuple[int, int, int, int]]:
    bbox = detection.get("bbox") or [0, 0, 0, 0]
    bbox_t = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    return (
        str(detection.get("label", "")).strip().lower(),
        bbox_t,
    )


def deduplicate_detections(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique: List[Dict[str, Any]] = []
    seen: set = set()
    for det in detections:
        sig = detection_signature(det)
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(det)
    return unique


def _bbox_center(bbox: Sequence[int]) -> Tuple[float, float]:
    return (float(bbox[0] + bbox[2]) / 2.0, float(bbox[1] + bbox[3]) / 2.0)


def _build_gt_radius_bbox(
    gt_x: float,
    gt_y: float,
    image_w: float,
    image_h: float,
    radius_ratio: float,
) -> List[float]:
    radius = max(1.0, float(radius_ratio) * float(min(image_w, image_h)))
    x1 = max(0.0, gt_x - radius)
    y1 = max(0.0, gt_y - radius)
    x2 = min(float(image_w) - 1.0, gt_x + radius)
    y2 = min(float(image_h) - 1.0, gt_y + radius)
    return [x1, y1, x2, y2]


def _bbox_intersects(box_a: Sequence[float], box_b: Sequence[float]) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _normalized_center_distance(
    det_bbox: Sequence[int],
    gt_x: float,
    gt_y: float,
    image_w: float,
    image_h: float,
) -> float:
    cx, cy = _bbox_center(det_bbox)
    diagonal = max((float(image_w) ** 2 + float(image_h) ** 2) ** 0.5, 1e-6)
    return (((cx - gt_x) ** 2 + (cy - gt_y) ** 2) ** 0.5) / diagonal


def split_positive_and_negatives(
    detections: List[Dict[str, Any]],
    gt_x: float,
    gt_y: float,
    image_w: float,
    image_h: float,
    gt_positive_radius_ratio: float,
    center_distance_threshold: float,
    remove_policy: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    gt_bbox = _build_gt_radius_bbox(
        gt_x=gt_x,
        gt_y=gt_y,
        image_w=image_w,
        image_h=image_h,
        radius_ratio=gt_positive_radius_ratio,
    )
    positive_indices: List[int] = []
    for idx, det in enumerate(detections):
        det_bbox = det["bbox"]
        overlap_positive = _bbox_intersects(det_bbox, gt_bbox)
        center_distance = _normalized_center_distance(
            det_bbox=det_bbox,
            gt_x=gt_x,
            gt_y=gt_y,
            image_w=image_w,
            image_h=image_h,
        )
        distance_positive = center_distance <= float(center_distance_threshold)
        if overlap_positive or distance_positive:
            positive_indices.append(idx)
    if not positive_indices:
        return [], detections

    if remove_policy == "all":
        positives = [detections[idx] for idx in positive_indices]
        positive_sigs = {detection_signature(det) for det in positives}
        negatives = [det for det in detections if detection_signature(det) not in positive_sigs]
        return positives, negatives

    best_idx = min(
        positive_indices,
        key=lambda idx: (detections[idx]["bbox"][2] - detections[idx]["bbox"][0])
        * (detections[idx]["bbox"][3] - detections[idx]["bbox"][1]),
    )
    selected_positive = detections[best_idx]
    selected_sig = detection_signature(selected_positive)
    positives = [det for det in detections if detection_signature(det) == selected_sig]
    negatives = [det for det in detections if detection_signature(det) != selected_sig]
    return positives, negatives


def maybe_convert_gt_to_pixel_space(
    gt: Tuple[float, float, float, float],
    image_w: int,
    image_h: int,
) -> Tuple[float, float, float, float]:
    gt_x, gt_y, gt_w, gt_h = gt

    # CSV annotations commonly come as gaze_x/gaze_y in [0,1], then scaled to [0,1000].
    if (
        abs(float(gt_w) - 1000.0) < 1e-6
        and abs(float(gt_h) - 1000.0) < 1e-6
        and 0.0 <= float(gt_x) <= 1000.0
        and 0.0 <= float(gt_y) <= 1000.0
    ):
        return (
            (float(gt_x) / 1000.0) * float(image_w),
            (float(gt_y) / 1000.0) * float(image_h),
            float(image_w),
            float(image_h),
        )

    # Handle normalized [0,1] GT if encountered.
    if (
        0.0 <= float(gt_w) <= 1.5
        and 0.0 <= float(gt_h) <= 1.5
        and 0.0 <= float(gt_x) <= 1.0
        and 0.0 <= float(gt_y) <= 1.0
    ):
        return (
            float(gt_x) * float(image_w),
            float(gt_y) * float(image_h),
            float(image_w),
            float(image_h),
        )

    return gt


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


def run_yolo_detection(
    image_path: str,
    model: Any,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    imgsz: int,
) -> List[Dict[str, Any]]:
    results = model.predict(
        source=image_path,
        conf=conf_thres,
        iou=iou_thres,
        max_det=max_det,
        imgsz=imgsz,
        verbose=False,
    )
    if not results:
        return []

    result = results[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    image_h, image_w = result.orig_shape
    names = result.names or {}
    xyxy_list = boxes.xyxy.tolist()
    cls_list = boxes.cls.tolist()

    detections: List[Dict[str, Any]] = []
    for xyxy, cls_id in zip(xyxy_list, cls_list):
        bbox = _pixel_to_bbox(xyxy, image_w=image_w, image_h=image_h)
        if bbox is None:
            continue
        cls_index = int(cls_id)
        label = str(names.get(cls_index, cls_index))
        detections.append(
            {
                "label": label,
                "bbox": bbox,
            }
        )
    return deduplicate_detections(detections)


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()
    if args.output_csv:
        output_csv = Path(args.output_csv).expanduser().resolve()
    else:
        output_dir = (REPO_ROOT / "training_datasets" / "stage1b_negatives").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = output_dir / f"stage1b_yolo_negatives_{time.strftime('%Y%m%d_%H%M%S')}.csv"
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

    from ultralytics import YOLO

    model = YOLO(args.model_id)
    detection_cache: Dict[str, List[Dict[str, Any]]] = {}
    error_cache: Dict[str, str] = {}

    with output_csv.open(write_mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_mode == "w":
            writer.writeheader()

        for local_idx, sample in enumerate(tqdm(dataset, desc="Building stage1b negatives (YOLO)")):
            # if should_skip_sample(sample):
            #     continue

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
                    detection_cache[cache_key] = run_yolo_detection(
                        image_path=cache_key,
                        model=model,
                        conf_thres=args.conf_thres,
                        iou_thres=args.iou_thres,
                        max_det=args.max_det,
                        imgsz=args.imgsz,
                    )
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
                with Image.open(image_path) as img:
                    image_w, image_h = img.size
                gt = maybe_convert_gt_to_pixel_space(gt, image_w=image_w, image_h=image_h)
                gt_x, gt_y, gt_w, gt_h = gt
                row["gt_x"] = gt_x
                row["gt_y"] = gt_y
                row["gt_width"] = gt_w
                row["gt_height"] = gt_h
                positives, negatives = split_positive_and_negatives(
                    detections=detections,
                    gt_x=gt_x,
                    gt_y=gt_y,
                    image_w=gt_w,
                    image_h=gt_h,
                    gt_positive_radius_ratio=args.gt_positive_radius_ratio,
                    center_distance_threshold=args.positive_center_distance_threshold,
                    remove_policy=args.remove_policy,
                )

            row["removed_positive_count"] = len(positives)
            row["removed_positive_json"] = serialize_json(positives)
            row["negative_count"] = len(negatives)
            row["negative_detections_json"] = serialize_json(negatives)
            writer.writerow(row)

    print(f"Saved Stage-1b negatives CSV to {output_csv}")


if __name__ == "__main__":
    main()
