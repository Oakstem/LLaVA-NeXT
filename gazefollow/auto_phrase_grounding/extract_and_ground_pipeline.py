#!/usr/bin/env python3
"""Chain person description extraction with Qwen3-VL grounding."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gazefollow.extract_attention_direct_gt_masks import run_generation_with_attention
from generation_utils import (
    DEFAULT_GT_GAZE_CSV,
    enable_inference_optimizations,
    fix_wsl_paths,
    load_mask_from_file,
    load_model_and_setup,
    _mask_array_to_binary,
)
from gazefollow.auto_phrase_grounding.qwen3vl_grounding import (
    load_qwen3vl_model,
    run_qwen3vl_grounding,
)
from gazefollow.gaze_metrics import compute_gaze_errors
import traceback


DEFAULT_MASK_DIR = Path(r"D:\Projects\data\gazefollow\train_gaze_segmentations\small_masks")
# DEFAULT_PROMPT = """Complete the sentence in the following format, for example:
#     a tired guy in a hoodie → a guy in a gray hoodie and ripped jeans sitting on a worn wooden bench.
#     a chic woman in a beige coat → a woman in a beige coat and ankle boots holding a phone.
#     a techy man in a leather jacket → a man in a black leather jacket and glasses.
#     a relaxed woman in a green sweater → a woman in a dark green sweater and black jeans carrying a tan shoulder bag.
#     The sentence: a _ → """
DEFAULT_PROMPT = """Complete the sentence in the following format, for example:
a tired guy in a hoodie → a guy in a gray hoodie and ripped jeans.
a chic woman in a beige coat → a woman in a beige coat and ankle boots.
a techy man in a leather jacket → a man in a black leather jacket and glasses.
a relaxed woman in a green sweater → a woman in a dark green sweater and black jeans.

The sentence: a _ →"""
DEFAULT_QUERY_TEMPLATE = (
    "Locate the person described as: {description}. "
    "Return JSON with a `bbox` field using pixel coordinates."
)


@dataclass(frozen=True)
class ImageTask:
    image_path: Path
    mask_path: Path
    image_id: str


def _flag_provided(flag: str) -> bool:
    for arg in sys.argv[1:]:
        if arg == flag or arg.startswith(f"{flag}="):
            return True
    return False


def _ensure_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("entries", "images", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]
    raise ValueError("Expected list or dict when loading image list JSON.")


def _iter_entries_from_list_file(list_path: Path) -> Iterable[Dict[str, Any]]:
    suffix = list_path.suffix.lower()
    if suffix in {".json", ".jsonc"}:
        with list_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        for item in _ensure_list(payload):
            if isinstance(item, dict):
                yield item
    elif suffix == ".jsonl":
        with list_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        with list_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue
                if "," in raw:
                    parts = [segment.strip() for segment in raw.split(",")]
                else:
                    parts = raw.split()
                if not parts:
                    continue
                entry: Dict[str, Any] = {"image_path": parts[0]}
                if len(parts) > 1:
                    entry["mask_path"] = parts[1]
                yield entry


def _normalize_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    normalized = Path(fix_wsl_paths(value)).expanduser()
    return normalized


def _resolve_mask_path(
    image_path: Path,
    explicit_mask: Optional[str],
    mask_dir: Optional[Path],
    mask_template: str,
    fallback_template: Optional[str],
) -> Path:
    if explicit_mask:
        resolved = _normalize_path(explicit_mask)
        if resolved and resolved.exists():
            return resolved
        raise FileNotFoundError(f"Mask path not found: {explicit_mask}")

    if mask_dir is None:
        raise ValueError("Mask directory is required when mask_path is not provided.")

    candidates: List[Tuple[str, Path]] = []
    mask_dir = mask_dir.expanduser()
    mask_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    candidates.append((mask_template, mask_dir / mask_template.format(stem=stem)))
    if fallback_template:
        candidates.append((fallback_template, mask_dir / fallback_template.format(stem=stem)))

    for template_name, candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not locate mask for {image_path.name} using templates "
        f"{', '.join(template for template, _ in candidates)} in {mask_dir}"
    )


def load_image_tasks(
    *,
    mode: str,
    image_path: Optional[str],
    mask_path: Optional[str],
    image_list: Optional[str],
    mask_dir: Optional[Path],
    mask_template: str,
    fallback_mask_template: Optional[str],
    limit: Optional[int],
    image_id: Optional[str],
) -> List[ImageTask]:
    tasks: List[ImageTask] = []
    mask_base = mask_dir or DEFAULT_MASK_DIR

    if mode == "single":
        if not image_path:
            raise ValueError("--image-path is required for single mode.")
        img_path = _normalize_path(image_path)
        if not img_path or not img_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        resolved_mask = _resolve_mask_path(
            img_path,
            mask_path,
            mask_base,
            mask_template,
            fallback_mask_template,
        )
        derived_id = image_id or img_path.stem
        tasks.append(ImageTask(img_path, resolved_mask, derived_id))
        return tasks

    if not image_list:
        raise ValueError("--image-list is required when mode is 'list'.")

    list_path = _normalize_path(image_list)
    if not list_path or not list_path.exists():
        raise FileNotFoundError(f"Image list not found: {image_list}")

    for entry in _iter_entries_from_list_file(list_path):
        image_val = entry.get("image_path") or entry.get("image") or entry.get("path")
        entry_mask = entry.get("mask_path") or entry.get("mask") or entry.get("mask_file")
        entry_id = entry.get("image_id") or entry.get("id")
        if not image_val:
            continue
        img_path = _normalize_path(str(image_val))
        if not img_path or not img_path.exists():
            raise FileNotFoundError(f"Image not found: {image_val}")
        resolved_mask = _resolve_mask_path(
            img_path,
            entry_mask,
            mask_base,
            mask_template,
            fallback_mask_template,
        )
        tasks.append(ImageTask(img_path, resolved_mask, entry_id or img_path.stem))
        if limit and len(tasks) >= limit:
            break

    if not tasks:
        raise ValueError("No image entries were loaded from the provided list.")
    return tasks


def _bbox_from_binary_mask(mask_array: Optional[np.ndarray]) -> Optional[List[float]]:
    if mask_array is None:
        return None
    binary = np.asarray(mask_array).astype(bool)
    if binary.size == 0:
        return None
    ys, xs = np.nonzero(binary)
    if ys.size == 0 or xs.size == 0:
        return None
    y_min = int(ys.min())
    y_max = int(ys.max())
    x_min = int(xs.min())
    x_max = int(xs.max())
    return [float(x_min), float(y_min), float(x_max + 1), float(y_max + 1)]


def load_person_bbox_from_masks(mask_path: Path, image_size: Tuple[int, int]) -> Optional[List[float]]:
    candidates: List[Path] = []
    mask_path = Path(mask_path)
    person_name = mask_path.name.replace("gaze__", "person__")
    if person_name != mask_path.name:
        candidates.append(mask_path.with_name(person_name))
    candidates.append(mask_path)

    for candidate in candidates:
        candidate = Path(candidate)
        if not candidate.exists():
            continue
        raw_mask = load_mask_from_file(candidate)
        binary = _mask_array_to_binary(raw_mask, image_size)
        bbox = _bbox_from_binary_mask(binary)
        if bbox:
            return bbox
    return None


def compute_bbox_iou(box_a: Optional[Sequence[float]], box_b: Optional[Sequence[float]]) -> Optional[float]:
    if not box_a or not box_b:
        return None
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return None
    return inter_area / denom


def annotate_grounding_metrics(
    detections: List[Dict[str, Any]],
    image_path: Path,
    mask_path: Path,
    person_mask_bbox: Optional[Sequence[float]],
) -> Optional[Dict[str, Any]]:
    """
    Attach gaze error metrics (L2/IoU) to each detection using the person mask bounding box.
    """
    with Image.open(image_path) as img:
        width, height = img.size

    image_size = (width, height)
    bbox_source = "generation_results" if person_mask_bbox else "person_mask_file"
    gt_bbox = list(person_mask_bbox) if person_mask_bbox else load_person_bbox_from_masks(mask_path, image_size)
    if not gt_bbox:
        return None

    gt_point = ((gt_bbox[0] + gt_bbox[2]) / 2.0, (gt_bbox[1] + gt_bbox[3]) / 2.0)

    best_idx: Optional[int] = None
    best_iou: Optional[float] = None
    for idx, detection in enumerate(detections):
        bbox = detection.get("bbox")
        metrics = compute_gaze_errors(bbox, None, gt_point, width, height)
        bbox_iou = compute_bbox_iou(bbox, gt_bbox)
        metrics["bbox_iou_vs_person"] = bbox_iou
        detection["metrics"] = metrics
        if bbox_iou is None:
            continue
        if best_iou is None or bbox_iou > best_iou:
            best_iou = bbox_iou
            best_idx = idx

    return {
        "image_size": [width, height],
        "person_bbox": gt_bbox,
        "person_bbox_source": bbox_source,
        "ground_truth_point": {"x": gt_point[0], "y": gt_point[1]},
        "best_detection_index": best_idx,
        "best_detection_iou": best_iou,
    }


def draw_grounding_overlay(
    image_path: Path,
    detections: Sequence[Dict[str, Any]],
    save_path: Path,
) -> Optional[Path]:
    valid_entries: List[Tuple[Tuple[int, int, int, int], str]] = []
    for idx, detection in enumerate(detections, start=1):
        bbox = detection.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(coord))) for coord in bbox]
        label = (
            detection.get("label")
            or detection.get("description")
            or detection.get("object")
            or detection.get("text")
            or detection.get("name")
            or f"detection_{idx}"
        )
        valid_entries.append(((x1, y1, x2, y2), str(label)))

    if not valid_entries:
        return None

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    colors = ["#ff6b6b", "#4ecdc4", "#ffa600", "#1f78b4", "#995dd8", "#2ec4b6"]
    font = ImageFont.load_default()

    for idx, (bbox, label) in enumerate(valid_entries, start=1):
        x1 = max(0, min(width - 1, bbox[0]))
        y1 = max(0, min(height - 1, bbox[1]))
        x2 = max(0, min(width - 1, max(bbox[0] + 1, bbox[2])))
        y2 = max(0, min(height - 1, max(bbox[1] + 1, bbox[3])))
        color = colors[(idx - 1) % len(colors)]
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        caption = f"{idx}: {label}"
        if hasattr(draw, "textbbox"):
            text_bbox = draw.textbbox((x1, y1), caption, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        else:
            text_w, text_h = draw.textsize(caption, font=font)
        text_x = x1
        text_y = max(0, y1 - text_h - 2)
        draw.rectangle(
            [(text_x, text_y), (text_x + text_w + 4, text_y + text_h + 4)],
            fill=(0, 0, 0, 160),
        )
        draw.text((text_x + 2, text_y + 2), caption, fill="white", font=font)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)
    return save_path


def _json_default(value: Any) -> Any:
    """Coerce numpy and pathlib objects into json-serializable primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(payload: Dict[str, Any], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False, default=_json_default)
    return destination


def build_generation_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "bias_strength": args.llava_bias_strength,
        "max_new_tokens": args.llava_max_new_tokens,
        "temperature": args.llava_temperature,
        "do_sample": args.llava_do_sample,
        "top_k": args.llava_top_k,
        "output_hidden_states": True,
        "save_debug_files": args.save_debug_files,
    }


def build_attention_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "attn_threshold": args.attn_threshold,
        "opening_kernel_size": args.attn_opening_kernel,
        "min_blob_area": args.attn_min_blob_area,
        "min_avg_attention": args.attn_min_avg_attention,
        "show_highest_attn_blob": args.attn_show_highest_blob,
        "dilate_kernel_size": args.attn_dilate_kernel,
        "create_collage": args.attn_create_collage,
        "query_indices": {
            "gaze_source": args.gaze_source_query_indices,
            "gaze_target": args.gaze_target_query_indices,
        },
        "layer_idx": args.attn_layer_index,
        "save_tensors": args.attn_save_tensors,
        "repr_capture_layer_idx": args.attn_capture_layer_idx,
        "repr_inject_layer_idx": args.attn_inject_layer_idx,
    }


def build_guidance_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "top_k": args.guidance_top_k,
        "similarity_weight": args.guidance_similarity_weight,
        "probability_weight": args.guidance_probability_weight,
        "enable_after_step": args.guidance_enable_after_step,
        "enable_after_keyword": args.guidance_enable_after_keyword,
    }


def _should_capture_repr(attention_config: Optional[Dict[str, Any]]) -> bool:
    if not attention_config:
        return False
    return any(
        attention_config.get(key) is not None for key in ("repr_capture_layer_idx", "repr_inject_layer_idx")
    )


def capture_initial_repr_state(
    *,
    task: ImageTask,
    args: argparse.Namespace,
    tokenizer,
    model,
    image_processor,
    generation_config: Dict[str, Any],
    attention_config: Optional[Dict[str, Any]],
    guidance_config: Dict[str, Any],
    image_output_dir: Path,
    attention_mask_viz_dir: Optional[Union[str, Path, bool]],
) -> Optional[Any]:
    """
    Run a short initial pass to cache the representation tensors that will be injected in the real pass.
    """
    if not _should_capture_repr(attention_config):
        return None

    cache_generation_config = copy.deepcopy(generation_config)
    cache_attention_config = copy.deepcopy(attention_config or {})
    cache_attention_config["return_full_hidden_states"] = True
    cache_output_dir = image_output_dir / "repr_cache"
    cache_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Capturing representation tokens for {task.image_id} prior to main generation run...")
    cache_results = run_generation_with_attention(
        image_path=str(task.image_path),
        mask_path=str(task.mask_path),
        prompt="",
        output_dir=str(cache_output_dir),
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        generation_config=cache_generation_config,
        attention_config=cache_attention_config,
        bias_strength=cache_generation_config.get("bias_strength", 0.0),
        prev_run_last_hidden_state=None,
        break_after_first_step=True,
        use_gaze_guidance=args.use_gaze_guidance,
        guidance_config=guidance_config,
        save_debug_files=args.save_debug_files,
        use_gt_gaze_csv=args.use_gt_gaze_csv,
        gt_gaze_csv_path=args.gt_gaze_csv_path,
        gt_gaze_mask_radius=args.gt_gaze_mask_radius,
        gt_gaze_mask_radius_ratio=args.gt_gaze_mask_radius_ratio,
        save_mask_overlays=args.save_mask_overlays,
        mask_overlay_alpha=args.mask_overlay_alpha,
        include_image_inputs=not args.exclude_image_inputs,
        use_body_bbox=args.use_body_bbox,
        attention_mask_viz_dir=attention_mask_viz_dir,
    )
    prev_hidden_state = cache_results.get("first_step_hidden_state")
    if prev_hidden_state is None:
        print(f"  Warning: Initial representation capture for {task.image_id} did not return a hidden state payload.")
    return prev_hidden_state


def process_image_task(
    task: ImageTask,
    *,
    args: argparse.Namespace,
    tokenizer,
    model,
    image_processor,
    generation_config: Dict[str, Any],
    attention_config: Dict[str, Any],
    guidance_config: Dict[str, Any],
    llava_output_dir: Path,
    qwen_processor,
    qwen_model,
    visualization_dir: Optional[Path],
    qwen_query_template: str,
    attention_mask_viz_dir: Optional[Union[str, Path, bool]] = None,
) -> Dict[str, Any]:
    image_output_dir = llava_output_dir / task.image_id
    image_output_dir.mkdir(parents=True, exist_ok=True)
    prev_hidden_state = capture_initial_repr_state(
        task=task,
        args=args,
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        generation_config=generation_config,
        attention_config=attention_config,
        guidance_config=guidance_config,
        image_output_dir=image_output_dir,
        attention_mask_viz_dir=attention_mask_viz_dir,
    )
    description_results = run_generation_with_attention(
        image_path=str(task.image_path),
        mask_path=str(task.mask_path),
        prompt=args.prompt,
        output_dir=str(image_output_dir),
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        generation_config=generation_config,
        attention_config=attention_config,
        bias_strength=generation_config.get("bias_strength", 0.0),
        prev_run_last_hidden_state=prev_hidden_state,
        use_gaze_guidance=args.use_gaze_guidance,
        guidance_config=guidance_config,
        save_debug_files=args.save_debug_files,
        use_gt_gaze_csv=args.use_gt_gaze_csv,
        gt_gaze_csv_path=args.gt_gaze_csv_path,
        gt_gaze_mask_radius=args.gt_gaze_mask_radius,
        gt_gaze_mask_radius_ratio=args.gt_gaze_mask_radius_ratio,
        save_mask_overlays=args.save_mask_overlays,
        mask_overlay_alpha=args.mask_overlay_alpha,
        include_image_inputs=not args.exclude_image_inputs,
        attention_mask_viz_dir=attention_mask_viz_dir,
        use_body_bbox=args.use_body_bbox,
    )
    description_text = (description_results.get("generated_text") or "").strip()
    if not description_text:
        description_text = args.query_fallback_description

    query_description = description_text
    arrow_index = query_description.rfind("→")
    if arrow_index != -1:
        query_description = query_description[arrow_index + 1 :].strip()
    if not query_description:
        query_description = args.query_fallback_description
    truncation_index = min(
        (i for i in (query_description.find(","), query_description.find(".")) if i != -1),
        default=None,
    )
    if truncation_index is not None:
        query_description = query_description[:truncation_index].strip()
        if not query_description:
            query_description = args.query_fallback_description

    grounding_query = qwen_query_template.format(description=query_description)
    detections, raw_response = run_qwen3vl_grounding(
        image_path=str(task.image_path),
        query=grounding_query,
        model_id=args.qwen_model_id,
        max_new_tokens=args.qwen_max_new_tokens,
        processor=qwen_processor,
        model=qwen_model,
        device_map=args.qwen_device_map,
        temperature=args.qwen_temperature,
    )

    overlay_path: Optional[Path] = None
    overlay_output_in_llava_dir = image_output_dir / "grounding_visualization.png"
    if args.save_visualization:
        overlay_path = draw_grounding_overlay(task.image_path, detections, overlay_output_in_llava_dir)
        if overlay_path and visualization_dir:
            viz_copy_path = visualization_dir / f"{task.image_id}.png"
            viz_copy_path.parent.mkdir(parents=True, exist_ok=True)
            if viz_copy_path != overlay_path:
                shutil.copy2(overlay_path, viz_copy_path)

    metrics_summary = annotate_grounding_metrics(
        detections=detections,
        image_path=task.image_path,
        mask_path=task.mask_path,
        person_mask_bbox=description_results.get("person_mask_bbox"),
    )

    description_summary = {
        "text": description_text,
        "num_tokens": description_results.get("num_tokens"),
        "evaluation_summary": description_results.get("evaluation_summary"),
        "quality_analysis": description_results.get("quality_analysis"),
        "attention_correlation": description_results.get("attention_correlation"),
        "person_mask_correlation": description_results.get("person_mask_correlation"),
        "target_mask_correlation": description_results.get("target_mask_correlation"),
        "mask_overlay_path": description_results.get("mask_overlay_path"),
        "person_mask_bbox": description_results.get("person_mask_bbox"),
    }

    result_payload = {
        "image_id": task.image_id,
        "image_path": str(task.image_path),
        "mask_path": str(task.mask_path),
        "person_description": description_summary,
        "grounding": {
            "query": query_description,
            "detections": detections,
            "raw_response": raw_response,
            "metrics_summary": metrics_summary,
        },
        "artifacts": {
            "llava_output_dir": str(image_output_dir),
            "visualization_path": str(overlay_path) if overlay_path else None,
        },
    }
    return result_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract person descriptions with LLaVA-NeXT attention and ground them with Qwen3-VL."
    )
    parser.add_argument("--mode", choices=["single", "list"], default="single")
    parser.add_argument("--image-path", default=r"D:\Projects\data\gazefollow\train\00000000\00000032.jpg", help="Path to a single image to process.")
    parser.add_argument("--mask-path", help="Optional explicit mask path for the single image.")
    parser.add_argument("--image-id", help="Override identifier for the single image.")
    parser.add_argument("--image-list", help="Path to a JSON/JSONL/txt list of images for list mode.")
    parser.add_argument("--list-limit", type=int, default=None, help="Optional cap on entries from --image-list.")

    parser.add_argument(
        "--mask-dir",
        default=str(DEFAULT_MASK_DIR),
        help="Directory containing gaze masks (used when mask_path is not provided).",
    )
    parser.add_argument(
        "--mask-template",
        default="gaze__{stem}_results.npy",
        help="Primary template for mask filenames (use '{stem}').",
    )
    parser.add_argument(
        "--mask-fallback-template",
        default="gaze__{stem}_masks.npy",
        help="Fallback template for mask filenames.",
    )

    parser.add_argument(
        "--output-dir",
        default=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help=("Output directory or name. When omitted, results are written to "
              "<dataset_root>/results/steered_generation/<output-dir-name>."),
    )
    parser.add_argument(
        "--llava-output-dir",
        default=None,
        help="Directory for intermediate LLaVA outputs (defaults to <output-dir>/llava_runs).",
    )
    parser.add_argument(
        "--visualization-dir",
        default=None,
        help="Directory for grounding overlay images (defaults to <output-dir>/visualizations).",
    )
    parser.add_argument(
        "--save-visualization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable saving overlay images.",
    )
    parser.add_argument(
        "--query-template",
        default=DEFAULT_QUERY_TEMPLATE,
        help="Template for Qwen grounding prompts (use '{description}').",
    )
    parser.add_argument(
        "--query-fallback-description",
        default="person of interest",
        help="Substitute text when the description step returns empty output.",
    )

    parser.add_argument("--model-path", default="lmms-lab/llava-onevision-qwen2-7b-ov-chat")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--attn-layer-index", type=int, default=23)
    parser.add_argument("--exclude-image-inputs", action="store_true")
    parser.add_argument("--save-debug-files", action="store_true")
    parser.add_argument("--use-gaze-guidance", action="store_true")

    parser.add_argument("--llava-bias-strength", type=float, default=2.5)
    parser.add_argument("--llava-max-new-tokens", type=int, default=128)
    parser.add_argument("--llava-temperature", type=float, default=0.1)
    parser.add_argument("--llava-top-k", type=int, default=50)
    parser.add_argument("--llava-do-sample", action="store_true")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt used when extracting the person description.",
    )

    parser.add_argument("--attn-threshold", type=float, default=0.4)
    parser.add_argument("--attn-opening-kernel", type=int, default=5)
    parser.add_argument("--attn-min-blob-area", type=int, default=50)
    parser.add_argument("--attn-min-avg-attention", type=float, default=0.2)
    parser.add_argument("--attn-show-highest-blob", action="store_true")
    parser.add_argument("--attn-dilate-kernel", type=int, default=0)
    parser.add_argument("--attn-create-collage", action="store_true")
    parser.add_argument("--attn-save-tensors", action="store_true")
    parser.add_argument(
        "--gaze-source-query-indices",
        type=int,
        nargs=2,
        default=(-6, -3),
        metavar=("START", "END"),
        help="Range of prompt tokens that describe the person.",
    )
    parser.add_argument(
        "--gaze-target-query-indices",
        type=int,
        nargs=2,
        default=(-3, 0),
        metavar=("START", "END"),
        help="Range of prompt tokens describing the gaze target.",
    )

    parser.add_argument("--use-gt-gaze-csv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gt-gaze-csv-path", default=str(DEFAULT_GT_GAZE_CSV))
    parser.add_argument("--gt-gaze-mask-radius", type=int, default=None)
    parser.add_argument("--gt-gaze-mask-radius-ratio", type=float, default=0.02)
    parser.add_argument("--save-mask-overlays", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mask-overlay-alpha", type=float, default=0.6)

    parser.add_argument("--guidance-top-k", type=int, default=10)
    parser.add_argument("--guidance-similarity-weight", type=float, default=0.7)
    parser.add_argument("--guidance-probability-weight", type=float, default=0.3)
    parser.add_argument("--guidance-enable-after-step", type=int, default=0)
    parser.add_argument("--guidance-enable-after-keyword", default="looking")

    parser.add_argument("--qwen-model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=300)
    parser.add_argument("--qwen-temperature", type=float, default=0.0)
    parser.add_argument("--qwen-device-map", default="auto")
    parser.add_argument("--attn-capture-layer-idx", type=int, default=20)
    parser.add_argument("--attn-inject-layer-idx", type=int, default=0)
    parser.add_argument("--use-body-bbox", action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    user_specified_output = _flag_provided("--output-dir")

    mask_dir = _normalize_path(args.mask_dir) if args.mask_dir else DEFAULT_MASK_DIR
    tasks = load_image_tasks(
        mode=args.mode,
        image_path=args.image_path,
        mask_path=args.mask_path,
        image_list=args.image_list,
        mask_dir=mask_dir,
        mask_template=args.mask_template,
        fallback_mask_template=args.mask_fallback_template,
        limit=args.list_limit,
        image_id=args.image_id,
    )

    if user_specified_output:
        output_dir = _normalize_path(args.output_dir)
    else:
        first_image = tasks[0].image_path.resolve()
        dataset_dir = first_image.parent.parent if len(first_image.parents) >= 2 else first_image.parent
        dataset_root = dataset_dir.parent if dataset_dir and dataset_dir.parent else dataset_dir
        output_name = Path(args.output_dir).name
        if dataset_root is None:
            dataset_root = Path.cwd()
        output_dir = dataset_root / "results" / "steered_generation" / output_name
    if output_dir is None:
        raise ValueError("Unable to resolve --output-dir.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    llava_output_dir = (
        _normalize_path(args.llava_output_dir) if args.llava_output_dir else output_dir / "llava_runs"
    )
    llava_output_dir = Path(llava_output_dir)
    llava_output_dir.mkdir(parents=True, exist_ok=True)

    visualization_dir = None
    if args.save_visualization:
        viz_dir = _normalize_path(args.visualization_dir) if args.visualization_dir else output_dir / "visualizations"
        visualization_dir = Path(viz_dir)
        visualization_dir.mkdir(parents=True, exist_ok=True)

    enable_inference_optimizations()
    adapter_path = _normalize_path(args.adapter_path)
    model_base = _normalize_path(args.model_base)
    tokenizer, model, image_processor, _ = load_model_and_setup(
        model_path=fix_wsl_paths(args.model_path),
        attn_implementation=args.attn_implementation,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        attn_layer_ind=args.attn_layer_index,
        model_base=fix_wsl_paths(str(model_base)) if model_base else None,
        adapter_path=fix_wsl_paths(str(adapter_path)) if adapter_path else None,
    )

    qwen_processor, qwen_model = load_qwen3vl_model(args.qwen_model_id, device_map=args.qwen_device_map)
    generation_config = build_generation_config(args)
    attention_config = build_attention_config(args)
    guidance_config = build_guidance_config(args)
    qwen_query_template = args.query_template if "{description}" in args.query_template else (
        args.query_template + " {description}"
    )

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    total = len(tasks)

    for idx, task in enumerate(tasks, start=1):
        print(f"[{idx}/{total}] Processing {task.image_id}")
        try:
            payload = process_image_task(
                task,
                args=args,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                generation_config=generation_config,
                attention_config=attention_config,
                guidance_config=guidance_config,
                llava_output_dir=llava_output_dir,
                qwen_processor=qwen_processor,
                qwen_model=qwen_model,
                visualization_dir=visualization_dir,
                qwen_query_template=qwen_query_template,
            )
            per_image_path = output_dir / f"{task.image_id}.json"
            save_json(payload, per_image_path)
            payload.setdefault("artifacts", {})["result_json"] = str(per_image_path)
            results.append(payload)
        except Exception as exc:  # noqa: BLE001
            print(f"  Failed to process {task.image_id}: {exc}", file=sys.stderr)
            traceback.print_exc()
            failures.append(
                {
                    "image_id": task.image_id,
                    "image_path": str(task.image_path),
                    "error": str(exc),
                }
            )

    summary = {
        "results": results,
        "failures": failures,
        "num_results": len(results),
        "num_failures": len(failures),
    }
    summary_path = output_dir / "pipeline_results.json"
    save_json(summary, summary_path)
    print(f"Saved combined summary to {summary_path}")


if __name__ == "__main__":
    main()
