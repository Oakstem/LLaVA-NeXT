#!/usr/bin/env python3
"""Visualize Stage-1b negative detections from CSV as bbox overlays."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gazefollow.generation_utils import fix_wsl_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw negative detection bboxes from Stage-1b CSV.")
    parser.add_argument(
        "--input-csv",
        default="training_datasets/stage1b_negatives/stage1b_yolo_negatives.csv",
        help="Path to Stage-1b negatives CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for overlay images. Defaults to training_datasets/stage1b_negatives_viz/<timestamp>/.",
    )
    parser.add_argument(
        "--filtered-csv",
        default=None,
        help="Optional path for filtered CSV. Defaults to training_datasets/stage1b_negatives_filtered/<input_stem>_filtered.csv",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to process.")
    parser.add_argument(
        "--images-root",
        default=None,
        help="Optional root for resolving relative image keys when image_path does not exist.",
    )
    parser.add_argument(
        "--only-status",
        default="ok",
        help="Only visualize rows with this status. Use empty string to disable filtering.",
    )
    parser.add_argument(
        "--gt-radius-ratio",
        type=float,
        default=0.05,
        help="Radius ratio (w.r.t min(gt_width,gt_height)) for drawing GT bbox around gaze point.",
    )
    parser.add_argument(
        "--recompute-positives",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute positives using GT overlap/distance logic instead of trusting removed_positive_json.",
    )
    parser.add_argument(
        "--positive-center-distance-threshold",
        type=float,
        default=0.1,
        help="Normalized center-distance threshold for recomputed positives.",
    )
    return parser.parse_args()


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _parse_bbox(value: Any) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(value, Sequence) or len(value) != 4:
        return None
    coords = [_safe_int(v) for v in value]
    if any(v is None for v in coords):
        return None
    x1, y1, x2, y2 = coords  # type: ignore[misc]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _parse_detections(raw_json: str) -> List[Dict[str, Any]]:
    if not raw_json:
        return []
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    out: List[Dict[str, Any]] = []
    for entry in parsed:
        if isinstance(entry, dict):
            out.append(entry)
    return out


def _detection_signature(det: Dict[str, Any]) -> Tuple[str, str, Tuple[int, int, int, int], Optional[float]]:
    bbox = _parse_bbox(det.get("bbox")) or (0, 0, 0, 0)
    score = det.get("score")
    score_sig = round(float(score), 4) if isinstance(score, (int, float)) else None
    return (
        str(det.get("label", "")).strip().lower(),
        str(det.get("category", "")).strip().lower(),
        bbox,
        score_sig,
    )


def _sanitize_negatives(
    negatives: List[Dict[str, Any]],
    removed: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    removed_sigs = {_detection_signature(det) for det in removed}
    overlap_count = 0
    filtered: List[Dict[str, Any]] = []
    seen: set = set()
    for det in negatives:
        sig = _detection_signature(det)
        if sig in removed_sigs:
            overlap_count += 1
            continue
        if sig in seen:
            continue
        seen.add(sig)
        filtered.append(det)
    return filtered, overlap_count


def _bbox_intersects(box_a: Sequence[float], box_b: Sequence[float]) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _bbox_center(bbox: Sequence[int]) -> Tuple[float, float]:
    return (float(bbox[0] + bbox[2]) / 2.0, float(bbox[1] + bbox[3]) / 2.0)


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


def _recompute_removed_and_negatives(
    all_detections: List[Dict[str, Any]],
    row: Dict[str, Any],
    gt_radius_ratio: float,
    center_distance_threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    gt_x = _safe_float(row.get("gt_x"))
    gt_y = _safe_float(row.get("gt_y"))
    gt_w = _safe_float(row.get("gt_width"))
    gt_h = _safe_float(row.get("gt_height"))
    if None in (gt_x, gt_y, gt_w, gt_h):
        return [], all_detections

    radius = max(1.0, float(gt_radius_ratio) * min(float(gt_w), float(gt_h)))
    gt_box = [
        max(0.0, float(gt_x) - radius),
        max(0.0, float(gt_y) - radius),
        min(float(gt_w) - 1.0, float(gt_x) + radius),
        min(float(gt_h) - 1.0, float(gt_y) + radius),
    ]
    positives: List[Dict[str, Any]] = []
    negatives: List[Dict[str, Any]] = []
    for det in all_detections:
        bbox = _parse_bbox(det.get("bbox"))
        if bbox is None:
            negatives.append(det)
            continue
        overlap_positive = _bbox_intersects(bbox, gt_box)
        center_distance = _normalized_center_distance(
            det_bbox=bbox,
            gt_x=float(gt_x),
            gt_y=float(gt_y),
            image_w=float(gt_w),
            image_h=float(gt_h),
        )
        distance_positive = center_distance <= float(center_distance_threshold)
        if overlap_positive or distance_positive:
            positives.append(det)
        else:
            negatives.append(det)
    return positives, negatives


def _make_caption(det: Dict[str, Any], idx: int) -> str:
    label = str(det.get("label") or det.get("category") or f"det_{idx}")
    score = det.get("score")
    if isinstance(score, (int, float)):
        return f"{idx}: {label} ({float(score):.2f})"
    return f"{idx}: {label}"


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_gt_bbox_from_row(row: Dict[str, Any], radius_ratio: float) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int]]]:
    gt_x = _safe_float(row.get("gt_x"))
    gt_y = _safe_float(row.get("gt_y"))
    gt_w = _safe_float(row.get("gt_width"))
    gt_h = _safe_float(row.get("gt_height"))
    if None in (gt_x, gt_y, gt_w, gt_h):
        return None, None

    radius = max(1.0, float(radius_ratio) * min(float(gt_w), float(gt_h)))
    x1 = _safe_int(max(0.0, float(gt_x) - radius))
    y1 = _safe_int(max(0.0, float(gt_y) - radius))
    x2 = _safe_int(min(float(gt_w) - 1.0, float(gt_x) + radius))
    y2 = _safe_int(min(float(gt_h) - 1.0, float(gt_y) + radius))
    cx = _safe_int(gt_x)
    cy = _safe_int(gt_y)
    if None in (x1, y1, x2, y2, cx, cy):
        return None, None
    return (x1, y1, x2, y2), (cx, cy)


def _path_with_optional_ext(path: Path) -> List[Path]:
    if path.suffix:
        return [path]
    return [path.with_suffix(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")]


def resolve_image_path(
    row: Dict[str, Any],
    images_root: Optional[Path],
) -> Optional[Path]:
    image_path_raw = str(row.get("image_path", "")).strip()
    image_key = str(row.get("image_key", "")).strip().replace("\\", "/")

    candidates: List[Path] = []
    if image_path_raw:
        candidates.append(Path(fix_wsl_paths(image_path_raw)).expanduser())

    if images_root is not None and image_key:
        key_rel = Path(image_key)
        candidates.append(images_root / key_rel)

        parts = list(key_rel.parts)
        if parts and parts[0].lower() == "train":
            candidates.append(images_root / Path(*parts[1:]))

    for candidate in candidates:
        for variant in _path_with_optional_ext(candidate):
            if variant.exists():
                return variant.resolve()
    return None


def draw_overlay(
    image_path: Path,
    detections: List[Dict[str, Any]],
    gt_bbox: Optional[Tuple[int, int, int, int]],
    gt_point: Optional[Tuple[int, int]],
    output_path: Path,
) -> bool:
    if not image_path.exists():
        return False

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    colors = [
        "#ff6b6b",
        "#4ecdc4",
        "#ffa600",
        "#1f78b4",
        "#2ec4b6",
        "#e76f51",
        "#8ab17d",
        "#f4a261",
    ]
    width, height = image.size

    if gt_bbox is not None:
        gx1, gy1, gx2, gy2 = gt_bbox
        gx1 = max(0, min(width - 1, gx1))
        gy1 = max(0, min(height - 1, gy1))
        gx2 = max(0, min(width - 1, max(gx2, gx1 + 1)))
        gy2 = max(0, min(height - 1, max(gy2, gy1 + 1)))
        draw.rectangle([(gx1, gy1), (gx2, gy2)], outline="#ffffff", width=2)
        draw.text((gx1 + 2, max(0, gy1 - 12)), "GT", fill="white", font=font)
    if gt_point is not None:
        px, py = gt_point
        r = 3
        draw.ellipse([(px - r, py - r), (px + r, py + r)], fill="#ffffff")

    for idx, det in enumerate(detections, start=1):
        bbox = _parse_bbox(det.get("bbox"))
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, max(x2, x1 + 1)))
        y2 = max(0, min(height - 1, max(y2, y1 + 1)))

        color = colors[(idx - 1) % len(colors)]
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        caption = _make_caption(det, idx)

        if hasattr(draw, "textbbox"):
            tb = draw.textbbox((x1, y1), caption, font=font)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]
        else:
            tw, th = draw.textsize(caption, font=font)

        tx = x1
        ty = max(0, y1 - th - 4)
        draw.rectangle([(tx, ty), (tx + tw + 4, ty + th + 4)], fill=(0, 0, 0, 160))
        draw.text((tx + 2, ty + 2), caption, fill="white", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    image.close()
    return True


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    images_root = Path(fix_wsl_paths(args.images_root)) if args.images_root else None
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = (input_csv.parent / "stage1b_negatives_viz" / time.strftime("%Y%m%d_%H%M%S")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.filtered_csv:
        filtered_csv_path = Path(args.filtered_csv).expanduser().resolve()
    else:
        filtered_csv_path = (
            input_csv.parent
            / "stage1b_negatives_filtered"
            / f"{input_csv.stem}_filtered.csv"
        ).resolve()
    filtered_csv_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    rendered = 0
    missing_images = 0
    rows_with_overlap = 0
    total_removed_from_negatives = 0
    filtered_rows: List[Dict[str, Any]] = []

    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if args.limit is not None:
            rows = rows[: args.limit]

        for row in tqdm(rows, desc="Rendering negatives"):
            total_rows += 1
            removed = _parse_detections(str(row.get("removed_positive_json", "")))
            negatives = _parse_detections(str(row.get("negative_detections_json", "")))
            if args.recompute_positives:
                all_candidates = removed + negatives
                removed, negatives = _recompute_removed_and_negatives(
                    all_detections=all_candidates,
                    row=row,
                    gt_radius_ratio=args.gt_radius_ratio,
                    center_distance_threshold=args.positive_center_distance_threshold,
                )
                row["removed_positive_json"] = json.dumps(
                    removed, ensure_ascii=True, separators=(",", ":")
                )
                row["removed_positive_count"] = str(len(removed))
            sanitized_negatives, overlap_count = _sanitize_negatives(negatives, removed)
            if overlap_count > 0:
                rows_with_overlap += 1
                total_removed_from_negatives += overlap_count
            row["negative_detections_json"] = json.dumps(
                sanitized_negatives, ensure_ascii=True, separators=(",", ":")
            )
            row["negative_count"] = str(len(sanitized_negatives))
            row["removed_overlap_count"] = str(overlap_count)
            row["removed_overlap_found"] = "1" if overlap_count > 0 else "0"
            filtered_rows.append(row)

            status = str(row.get("status", "")).strip()
            if args.only_status and status != args.only_status:
                continue

            image_path = resolve_image_path(row, images_root)
            if image_path is None:
                missing_images += 1
                continue

            sample_id = str(row.get("sample_id", "")).strip() or str(total_rows)
            safe_sample_id = sample_id.replace("/", "__")
            out_path = output_dir / f"{safe_sample_id}.jpg"
            gt_bbox, gt_point = _build_gt_bbox_from_row(row, args.gt_radius_ratio)

            if draw_overlay(image_path, sanitized_negatives, gt_bbox, gt_point, out_path):
                rendered += 1

    fieldnames = list(filtered_rows[0].keys()) if filtered_rows else []
    if filtered_rows:
        with filtered_csv_path.open("w", encoding="utf-8", newline="") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(filtered_rows)

    print(f"Input CSV: {input_csv}")
    print(f"Output dir: {output_dir}")
    print(f"Filtered CSV: {filtered_csv_path}")
    print(f"Rows read: {total_rows}")
    print(f"Overlays rendered: {rendered}")
    print(f"Missing images: {missing_images}")
    print(f"Rows with removed-overlap: {rows_with_overlap}")
    print(f"Detections removed from negatives: {total_removed_from_negatives}")


if __name__ == "__main__":
    main()
    
