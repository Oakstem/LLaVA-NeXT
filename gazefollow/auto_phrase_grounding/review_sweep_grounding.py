#!/usr/bin/env python3
"""Ground every sweep description and score it against a region-mask center."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gazefollow.auto_phrase_grounding.qwen3vl_grounding import (
    load_qwen3vl_model,
    run_qwen3vl_grounding,
)
from gazefollow.gaze_metrics import compute_gaze_errors
from gazefollow.qwen3vl_utils import build_qwen_query, extract_bbox, extract_score


DEFAULT_RESULTS_FILE = "results/steered_generation/repr_layer_sweep_results_20260403_183355.json"
DEFAULT_MASK_META = "region_masks/active_region_mask2.json"
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_OUTPUT_DIR = "results/steered_generation/review_grounding"


@dataclass(frozen=True)
class SweepEntry:
    combo_key: str
    text: str
    raw_text: str
    source_layer: Optional[int]
    target_layer: Optional[int]


def _normalize_path(path_str: str) -> Path:
    path = str(path_str or "").strip()
    if not path:
        raise ValueError("Expected a non-empty path.")
    path = path.replace("\\", "/")
    drive_match = re.match(r"^([A-Za-z]):/(.*)$", path)
    if drive_match:
        drive = drive_match.group(1).lower()
        rest = drive_match.group(2)
        path = f"/mnt/{drive}/{rest}"
    return Path(path).expanduser().resolve()


def _combo_sort_key(combo_key: str) -> Tuple[int, int, str]:
    match = re.fullmatch(r"src(\d+)_tgt(\d+)", combo_key)
    if not match:
        return (10**9, 10**9, combo_key)
    return (int(match.group(1)), int(match.group(2)), combo_key)


def _parse_combo_layers(combo_key: str) -> Tuple[Optional[int], Optional[int]]:
    match = re.fullmatch(r"src(\d+)_tgt(\d+)", combo_key)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _extract_description_parts(text: str) -> Tuple[str, str]:
    if "→" in text:
        left, right = text.split("→", 1)
        return left.strip(), right.strip()
    if "->" in text:
        left, right = text.split("->", 1)
        return left.strip(), right.strip()
    cleaned = text.strip()
    return cleaned, cleaned


def load_sweep_entries(results_file: Path) -> List[SweepEntry]:
    payload = json.loads(results_file.read_text(encoding="utf-8"))
    generated = payload.get("generated_text_by_layer_combination")
    if not isinstance(generated, dict):
        raise ValueError("Results JSON does not contain 'generated_text_by_layer_combination'.")

    entries: List[SweepEntry] = []
    for combo_key in sorted(generated.keys(), key=_combo_sort_key):
        raw_text = str(generated[combo_key] or "").strip()
        _, target_text = _extract_description_parts(raw_text)
        src_layer, tgt_layer = _parse_combo_layers(combo_key)
        entries.append(
            SweepEntry(
                combo_key=combo_key,
                text=target_text,
                raw_text=raw_text,
                source_layer=src_layer,
                target_layer=tgt_layer,
            )
        )
    return entries


def load_mask_center(
    mask_meta_path: Path,
    image_path_override: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any], Tuple[float, float], Tuple[int, int], List[int]]:
    metadata: Dict[str, Any] = {}
    if mask_meta_path.suffix.lower() == ".json":
        metadata = json.loads(mask_meta_path.read_text(encoding="utf-8"))
        npy_path: Optional[Path] = None
    elif mask_meta_path.suffix.lower() == ".npy":
        npy_path = mask_meta_path
    else:
        raise ValueError(f"Expected a .json or .npy mask input, got: {mask_meta_path}")

    if image_path_override is not None:
        image_path = image_path_override
    else:
        image_path_value = str(metadata.get("image_path") or "").strip()
        if not image_path_value:
            raise ValueError(
                "Mask input does not provide an image path. Pass --image-path or use a JSON file that includes image_path."
            )
        image_path = _normalize_path(image_path_value)

    image_width = int(metadata.get("image_width") or 0)
    image_height = int(metadata.get("image_height") or 0)
    if npy_path is not None:
        mask = np.load(npy_path, allow_pickle=True)
        if isinstance(mask, np.ndarray) and mask.ndim > 2:
            mask = np.squeeze(mask)
        mask = np.asarray(mask).astype(bool)
        if mask.ndim != 2:
            raise ValueError(f"Mask array must be 2D after squeeze: {npy_path}")
        if image_width <= 0 or image_height <= 0:
            image_height, image_width = mask.shape
        ys, xs = np.nonzero(mask)
        if xs.size == 0 or ys.size == 0:
            raise ValueError(f"Mask file is empty: {npy_path}")
        center = (float(xs.mean()), float(ys.mean()))
        bounds = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        return image_path, metadata, center, (image_width, image_height), bounds

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Mask metadata must include positive image_width and image_height.")
    pixel_bounds = metadata.get("pixel_bounds") or {}
    x_min = int(pixel_bounds.get("x_min", 0))
    y_min = int(pixel_bounds.get("y_min", 0))
    x_max = int(pixel_bounds.get("x_max", 0))
    y_max = int(pixel_bounds.get("y_max", 0))
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Mask metadata does not contain usable bounds: {mask_meta_path}")
    center = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
    return image_path, metadata, center, (image_width, image_height), [x_min, y_min, x_max, y_max]


def serialize_detection_metrics(
    detections: Sequence[Dict[str, Any]],
    mask_center: Tuple[float, float],
    image_size: Tuple[int, int],
) -> List[Dict[str, Any]]:
    width, height = image_size
    serialized: List[Dict[str, Any]] = []
    for idx, detection in enumerate(detections):
        bbox = detection.get("bbox")
        metrics = compute_gaze_errors(
            predicted_box=bbox,
            person_box=None,
            ground_truth_point=mask_center,
            image_width=width,
            image_height=height,
        )
        serialized.append(
            {
                "index": idx,
                "bbox": bbox,
                "bbox_center": detection.get("bbox_center"),
                "score": detection.get("score"),
                "label": detection.get("label") or detection.get("text") or detection.get("description"),
                "metrics": metrics,
            }
        )
    return serialized


def choose_best_detection_by_score(
    detections: Sequence[Dict[str, Any]],
    ground_truth_point: Tuple[float, float],
    image_width: int,
    image_height: int,
    score_threshold: float,
    iou_radius_ratio: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    best_detection: Optional[Dict[str, Any]] = None
    best_metrics: Optional[Dict[str, Any]] = None
    best_score = float("-inf")
    best_modified_l2 = float("inf")

    for detection in detections:
        bbox = extract_bbox(detection)
        if not bbox:
            continue

        score = extract_score(detection)
        if score is not None and score < score_threshold:
            continue

        metrics = compute_gaze_errors(
            predicted_box=bbox,
            person_box=None,
            ground_truth_point=ground_truth_point,
            image_width=image_width,
            image_height=image_height,
            iou_radius_ratio=iou_radius_ratio,
        )
        modified_l2 = metrics.get("gaze_modified_l2_error")
        comparable_score = score if score is not None else float("-inf")
        if comparable_score > best_score or (
            comparable_score == best_score
            and modified_l2 is not None
            and modified_l2 < best_modified_l2
        ):
            best_score = comparable_score
            best_modified_l2 = modified_l2 if modified_l2 is not None else float("inf")
            best_detection = detection
            best_metrics = metrics

    return best_detection, best_metrics


def draw_overlay(
    image_path: Path,
    mask_bounds: Sequence[int],
    mask_center: Tuple[float, float],
    best_detection: Optional[Dict[str, Any]],
    combo_key: str,
    description: str,
    output_path: Path,
) -> Path:
    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")

    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()

    mx1, my1, mx2, my2 = [int(v) for v in mask_bounds]
    draw.rectangle([(mx1, my1), (mx2, my2)], outline="#18a999", width=3)
    cx, cy = mask_center
    draw.ellipse([(cx - 5, cy - 5), (cx + 5, cy + 5)], fill="#18a999")

    caption_lines = [combo_key, description]
    if best_detection and best_detection.get("bbox"):
        x1, y1, x2, y2 = [int(round(float(v))) for v in best_detection["bbox"]]
        draw.rectangle([(x1, y1), (x2, y2)], outline="#ff6b35", width=4)
        caption_lines.append(f"bbox=[{x1}, {y1}, {x2}, {y2}]")
        pred_center = best_detection.get("bbox_center")
        if pred_center:
            px, py = pred_center
            draw.ellipse([(px - 5, py - 5), (px + 5, py + 5)], fill="#ff6b35")
        metrics = best_detection.get("metrics") or {}
        l2 = metrics.get("gaze_l2_error")
        norm_l2 = metrics.get("gaze_normalized_l2_error")
        if l2 is not None:
            caption_lines.append(f"l2={l2:.2f}")
        if norm_l2 is not None:
            caption_lines.append(f"norm_l2={norm_l2:.4f}")
    else:
        caption_lines.append("bbox=<none>")

    draw.multiline_text(
        (12, 12),
        "\n".join(caption_lines),
        fill="white",
        font=font,
        stroke_width=2,
        stroke_fill="black",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stringify_markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    text = str(value)
    text = text.replace("\n", " ").replace("|", "\\|")
    return text


def write_markdown_table(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return

    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join("---" for _ in headers) + " |\n")
        for row in rows:
            values = [_stringify_markdown_cell(row.get(header)) for header in headers]
            handle.write("| " + " | ".join(values) + " |\n")


def _safe_filename(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
    return cleaned[:120] if cleaned else fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ground each sweep description and score it against a region-mask center.")
    parser.add_argument("--results-file", default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--mask-meta-file", default=DEFAULT_MASK_META)
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-overlays", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print-top-k", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_path = _normalize_path(args.results_file)
    mask_meta_path = _normalize_path(args.mask_meta_file)
    output_root = _normalize_path(args.output_dir)
    image_path_override = _normalize_path(args.image_path) if args.image_path else None

    entries = load_sweep_entries(results_path)
    if args.limit is not None:
        entries = entries[: args.limit]
    image_path, mask_meta, mask_center, image_size, mask_bounds = load_mask_center(
        mask_meta_path,
        image_path_override=image_path_override,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"grounding_review_{timestamp}"
    overlays_dir = run_dir / "overlays"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Qwen3-VL model: {args.model_id}")
    processor, model = load_qwen3vl_model(args.model_id, device_map=args.device_map)

    width, height = image_size
    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    unique_texts = list(dict.fromkeys(entry.text for entry in entries))
    print(f"Loaded {len(entries)} entries with {len(unique_texts)} unique text values.")

    text_results: Dict[str, Dict[str, Any]] = {}
    for idx, text in enumerate(unique_texts, start=1):
        print(f"[unique {idx}/{len(unique_texts)}] {text}")
        query = build_qwen_query(text)
        detections, raw_response = run_qwen3vl_grounding(
            image_path=str(image_path),
            query=query,
            model_id=args.model_id,
            max_new_tokens=args.max_new_tokens,
            processor=processor,
            model=model,
            device_map=args.device_map,
            temperature=args.temperature,
        )

        best_detection, best_metrics = choose_best_detection_by_score(
            detections=detections,
            ground_truth_point=mask_center,
            image_width=width,
            image_height=height,
            score_threshold=0.0,
            iou_radius_ratio=0.05,
        )
        if best_detection is not None and best_metrics is not None:
            best_detection = dict(best_detection)
            best_detection["metrics"] = best_metrics

        overlay_path = None
        if args.save_overlays:
            overlay_path = draw_overlay(
                image_path=image_path,
                mask_bounds=mask_bounds,
                mask_center=mask_center,
                best_detection=best_detection,
                combo_key=_safe_filename(text, f"text_{idx}"),
                description=text,
                output_path=overlays_dir / f"{idx:04d}_{_safe_filename(text, f'text_{idx}')}.png",
            )

        text_results[text] = {
            "query": query,
            "detections": detections,
            "best_detection": best_detection,
            "best_metrics": best_metrics,
            "raw_response": raw_response,
            "overlay_path": None if overlay_path is None else str(overlay_path),
            "num_detections": len(detections),
            "all_detections": serialize_detection_metrics(detections, mask_center, image_size),
        }

    for entry in entries:
        cached = text_results[entry.text]
        best_detection = cached["best_detection"]
        best_metrics = cached["best_metrics"]
        best_bbox = best_detection.get("bbox") if best_detection else None
        best_center = best_detection.get("bbox_center") if best_detection else None
        score = best_detection.get("score") if best_detection else None

        summary_rows.append(
            {
                "combo_key": entry.combo_key,
                "src_layer": entry.source_layer,
                "tgt_layer": entry.target_layer,
                "text": entry.text,
                "score": score,
                "best_bbox": json.dumps(best_bbox),
                "best_center": json.dumps(best_center),
                "l2": None if best_metrics is None else best_metrics.get("gaze_l2_error"),
                "norm_l2": None if best_metrics is None else best_metrics.get("gaze_normalized_l2_error"),
                "modified_l2": None if best_metrics is None else best_metrics.get("gaze_modified_l2_error"),
                "gaze_iou": None if best_metrics is None else best_metrics.get("gaze_iou"),
                "num_detections": cached["num_detections"],
                "overlay_path": "" if cached["overlay_path"] is None else cached["overlay_path"],
            }
        )
        detail_rows.append(
            {
                "combo_key": entry.combo_key,
                "source_layer": entry.source_layer,
                "target_layer": entry.target_layer,
                "raw_text": entry.raw_text,
                "text": entry.text,
                "query": cached["query"],
                "best_detection": best_detection,
                "best_metrics": best_metrics,
                "all_detections": cached["all_detections"],
                "raw_response": cached["raw_response"],
                "overlay_path": cached["overlay_path"],
            }
        )

    summary_rows.sort(key=lambda row: math.inf if row["norm_l2"] is None else float(row["norm_l2"]))
    summary_path = run_dir / "summary.json"
    csv_path = run_dir / "summary.csv"
    md_path = run_dir / "summary.md"
    details_path = run_dir / "details.json"

    summary_payload = {
        "results_file": str(results_path),
        "mask_meta_file": str(mask_meta_path),
        "image_path": str(image_path),
        "image_size": {"width": width, "height": height},
        "mask_center": {"x": mask_center[0], "y": mask_center[1]},
        "mask_bounds": mask_bounds,
        "mask_metadata": mask_meta,
        "num_entries": len(entries),
        "num_unique_texts": len(unique_texts),
        "summary_rows": summary_rows,
        "details": detail_rows,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    details_path.write_text(json.dumps(detail_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(summary_rows, csv_path)
    write_markdown_table(summary_rows, md_path)

    print("")
    print(f"Saved summary JSON: {summary_path}")
    print(f"Saved summary CSV:  {csv_path}")
    print(f"Saved summary MD:   {md_path}")
    print(f"Saved details JSON: {details_path}")
    if args.save_overlays:
        print(f"Saved overlays dir: {overlays_dir}")

    top_k = min(args.print_top_k, len(summary_rows))
    if top_k > 0:
        print("")
        print("Top results by normalized L2:")
        for row in summary_rows[:top_k]:
            norm_l2 = row["norm_l2"]
            print(
                f"{row['combo_key']:<12} "
                f"norm_l2={norm_l2 if norm_l2 is not None else 'NA':<12} "
                f"l2={row['l2'] if row['l2'] is not None else 'NA':<12} "
                f"text={row['text']}"
            )


if __name__ == "__main__":
    main()
