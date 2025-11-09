#!/usr/bin/env python3
"""
Describe gaze source and target coordinates using Qwen3-VL.

This utility reads normalized gaze annotations from a CSV file, converts them to the
expected 0-1000 coordinate system used by Qwen3-VL, queries the model for per-region
descriptions, and stores the responses as JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TextIO

from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gazefollow.auto_phrase_grounding.qwen3vl_grounding import (
    load_qwen3vl_model,
    parse_grounding_predictions,
    run_qwen3vl_grounding,
)
from gazefollow.auto_phrase_grounding.build_conversation_json import (
    build_conversation_entry,
    save_dataset as save_conversation_dataset,
    DEFAULT_SAVE_INTERVAL as DEFAULT_CONVO_SAVE_INTERVAL,
    DEFAULT_L2_TARGET_THRESHOLD as DEFAULT_CONVO_L2_TARGET_THRESHOLD,
    DEFAULT_L2_SOURCE_THRESHOLD as DEFAULT_CONVO_L2_SOURCE_THRESHOLD,
    OUTSIDE_FRAME_TARGET_DESCRIPTION,
    is_outside_frame,
)
from gazefollow.gaze_metrics import compute_gaze_errors


DEFAULT_CSV = Path("gazefollow/data/train_annotations_release.csv")
DEFAULT_IMAGES_ROOT = Path("/mnt/d/Projects/data/gazefollow")
DEFAULT_OUTPUT = Path("gazefollow/auto_phrase_grounding/gaze_region_descriptions.json")
DEFAULT_SAVE_INTERVAL = 1
NUMERIC_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


class SkipLogger:
    def __init__(self, path: Optional[Path]) -> None:
        self.path = Path(path) if path else None
        self.handle: Optional[TextIO]
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("a")
        else:
            self.handle = None

    def log(self, message: str) -> None:
        if self.handle:
            self.handle.write(message + "\n")
            self.handle.flush()
        else:
            print(message)

    def close(self) -> None:
        if self.handle:
            self.handle.close()
            self.handle = None


@dataclass(frozen=True)
class RegionRequest:
    role: str
    bbox: List[int]
    point: Optional[List[int]] = None
    use_point: bool = False


def normalize_key_component(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value_str = str(value).strip()
    if not value_str:
        return None
    return value_str.replace("\\", "/")


def build_sample_key(primary: Optional[str], secondary: Optional[str], annotation_id: Optional[str]) -> Tuple[str, str]:
    """Backward-compatible wrapper that returns the first generated sample key."""
    variants = generate_sample_key_variants(
        primary_candidates=[primary],
        secondary_candidates=[secondary],
        annotation_candidates=[annotation_id],
    )
    return variants[0] if variants else ("", "")


def normalize_annotation_component(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value_str = str(value).strip()
    return value_str or None


def strip_key_extension(component: str) -> Optional[str]:
    if "." not in component:
        return None
    slash_index = component.rfind("/")
    after_slash = component[slash_index + 1 :] if slash_index >= 0 else component
    if "." not in after_slash:
        return None
    base = after_slash.rsplit(".", 1)[0]
    prefix = component[: slash_index + 1] if slash_index >= 0 else ""
    stripped = f"{prefix}{base}"
    return stripped or None


def generate_key_component_variants(value: Optional[str]) -> List[str]:
    normalized = normalize_key_component(value)
    if not normalized:
        return []
    variants = [normalized]
    stripped = strip_key_extension(normalized)
    if stripped and stripped not in variants:
        variants.append(stripped)
    return variants


def generate_sample_key_variants(
    *,
    primary_candidates: Sequence[Optional[str]] = (),
    secondary_candidates: Sequence[Optional[str]] = (),
    annotation_candidates: Sequence[Optional[str]] = (),
) -> List[Tuple[str, str]]:
    components: List[str] = []
    for candidate in list(primary_candidates) + list(secondary_candidates):
        for variant in generate_key_component_variants(candidate):
            if variant not in components:
                components.append(variant)
    if not components:
        components = [""]

    annotations: List[str] = []
    for candidate in annotation_candidates:
        normalized = normalize_annotation_component(candidate)
        if normalized and normalized not in annotations:
            annotations.append(normalized)
    if not annotations:
        annotations = [""]

    return [(component, annotation) for component in components for annotation in annotations]


def load_resume_state(resume_path: Optional[Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], set[Tuple[str, str]]]:
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    seen_keys: set[Tuple[str, str]] = set()

    if resume_path is None:
        return results, failures, seen_keys

    resume_path = Path(resume_path)
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume JSON not found: {resume_path}")

    with resume_path.open("r") as fp:
        summary = json.load(fp)

    if isinstance(summary, list):
        previous_results = summary
        previous_failures: list[dict[str, Any]] = []
    elif isinstance(summary, dict):
        previous_results = summary.get("results") or []
        previous_failures = summary.get("failures") or []
    else:
        raise ValueError(f"Resume JSON must be an object or list: {resume_path}")

    if not isinstance(previous_results, list) or not isinstance(previous_failures, list):
        raise ValueError(f"Resume JSON does not contain list-based results/failures: {resume_path}")

    results = list(previous_results)
    failures = list(previous_failures)

    for entry in results:
        variants = generate_sample_key_variants(
            primary_candidates=[
                entry.get("relative_path"),
                entry.get("image_path"),
                entry.get("image_id"),
                entry.get("image"),
            ],
            secondary_candidates=[
                entry.get("image_id"),
                entry.get("image_path"),
                entry.get("relative_path"),
                entry.get("image"),
            ],
            annotation_candidates=[
                entry.get("annotation_id"),
                entry.get("id"),
            ],
        )
        seen_keys.update(variants)

    print(
        f"[resume] Loaded {len(results)} results and {len(failures)} failures from {resume_path}; derived {len(seen_keys)} unique sample keys"
    )
    return results, failures, seen_keys


def coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    value_str = str(value).strip()
    if not value_str or not NUMERIC_PATTERN.match(value_str):
        return None
    return float(value_str)


def parse_float(row: Dict[str, Any], key: str) -> Optional[float]:
    return coerce_float(row.get(key))


def clamp_0_1000(value: float) -> int:
    return int(max(0.0, min(1000.0, round(value))))


def normalized_bbox_to_absolute(x: float, y: float, width: float, height: float) -> List[int]:
    x1 = clamp_0_1000(x * 1000.0)
    y1 = clamp_0_1000(y * 1000.0)
    x2 = clamp_0_1000((x + width) * 1000.0)
    y2 = clamp_0_1000((y + height) * 1000.0)
    return [x1, y1, x2, y2]


def normalized_point_to_bbox(x: float, y: float, box_size: float) -> Tuple[List[int], List[int]]:
    center_x = clamp_0_1000(x * 1000.0)
    center_y = clamp_0_1000(y * 1000.0)
    half = box_size / 2.0
    x1 = clamp_0_1000(center_x - half)
    y1 = clamp_0_1000(center_y - half)
    x2 = clamp_0_1000(center_x + half)
    y2 = clamp_0_1000(center_y + half)
    return [x1, y1, x2, y2], [center_x, center_y]


def pixel_bbox_to_absolute(bbox: Sequence[float], image_width: int, image_height: int) -> List[int]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Invalid image dimensions for head bounding box conversion.")
    if len(bbox) != 4:
        raise ValueError("Head bbox must contain four coordinates.")
    x_min, y_min, x_max, y_max = bbox
    return [
        clamp_0_1000((x_min / image_width) * 1000.0),
        clamp_0_1000((y_min / image_height) * 1000.0),
        clamp_0_1000((x_max / image_width) * 1000.0),
        clamp_0_1000((y_max / image_height) * 1000.0),
    ]


def extract_relative_path(row: Dict[str, Any]) -> Optional[str]:
    relative_path = row.get("image_path.1") or row.get("relative_path") or row.get("image_path")
    if relative_path is None:
        return None
    relative_path = str(relative_path).strip()
    return relative_path or None


def load_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            return img.convert("RGB")
        return img.copy()


def bbox_1000_to_pixels(bbox: Sequence[int], image_width: int, image_height: int) -> List[float]:
    if len(bbox) != 4:
        raise ValueError("Expected bbox with four coordinates.")
    x1, y1, x2, y2 = bbox
    scale_x = image_width / 1000.0
    scale_y = image_height / 1000.0
    return [
        x1 * scale_x,
        y1 * scale_y,
        x2 * scale_x,
        y2 * scale_y,
    ]


def point_1000_to_pixels(point: Sequence[float], image_width: int, image_height: int) -> Tuple[float, float]:
    if len(point) != 2:
        raise ValueError("Expected point with two coordinates.")
    scale_x = image_width / 1000.0
    scale_y = image_height / 1000.0
    return point[0] * scale_x, point[1] * scale_y


def bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    if len(bbox) != 4:
        raise ValueError("Expected bbox with four coordinates.")
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def build_source_prompt(region: RegionRequest) -> str:
    return (
        "You are provided with a bounding box defined in a 0-1000 coordinate system where (0,0) is the top-left "
        "corner of the image. Describe the person located inside the bounding box in one concise phrase." \
        "Under any circumstances, do NOT describe objects in the bounding box; only describe people. " \
        "Return ONLY JSON following this schema: [{\"role\": \"source\", \"bbox\": [x1,y1,x2,y2], \"description\": \"...\"}].\n"
        f"Bounding box: {region.bbox}"
    )


def build_target_prompt(region: RegionRequest) -> str:
    if region.use_point and region.point is not None:
        region_desc = f"Point: {region.point}"
        instruction = "point location"
    else:
        region_desc = f"Bounding box: {region.bbox}"
        instruction = "bounding box"
    return (
        "You are provided with a region defined in a 0-1000 coordinate system where (0,0) is the top-left corner of the image. "
        f"The {instruction} corresponds to the primary entity (person, object, or location). "
        "Describe this primary entity in one concise phrase. Return ONLY JSON following this schema: "
        "[{\"role\": \"target\", \"bbox\": [x1,y1,x2,y2], \"description\": \"...\"}].\n"
        f"{region_desc}"
    )


def extract_bbox_from_entry(entry: Dict[str, Any]) -> Optional[List[int]]:
    for key, value in entry.items():
        if not isinstance(key, str):
            continue
        if "bbox" not in key.lower() and key.lower() not in {"box", "region"}:
            continue
        if isinstance(value, Sequence) and len(value) == 4:
            converted: List[int] = []
            for coord in value:
                numeric = coerce_float(coord)
                if numeric is None:
                    converted = []
                    break
                converted.append(clamp_0_1000(numeric))
            if converted:
                return converted
    return None


def align_region_predictions(
    parsed: List[Dict[str, Any]],
    regions: Sequence[RegionRequest],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Align model predictions with requested regions; return (aligned, extras)."""
    aligned: List[Dict[str, Any]] = []
    extras: List[Dict[str, Any]] = []
    remaining = {region.role.lower(): region for region in regions}
    matched_roles: set[str] = set()

    for entry in parsed:
        role_value = entry.get("role") or entry.get("region") or entry.get("name")
        role_str = str(role_value).strip() if role_value is not None else ""
        role_key = role_str.lower()
        if role_key in remaining and role_key not in matched_roles:
            region = remaining[role_key]
            aligned.append(
                {
                    "role": region.role,
                    "bbox": extract_bbox_from_entry(entry) or region.bbox,
                    "description": entry.get("description") or entry.get("text") or entry.get("label"),
                    "raw": entry,
                }
            )
            matched_roles.add(role_key)
        else:
            extras.append(entry)

    for region in regions:
        if region.role.lower() in matched_roles:
            continue
        aligned.append(
            {
                "role": region.role,
                "bbox": region.bbox,
                "description": None,
                "raw": None,
            }
        )

    return aligned, extras


def describe_region(
    pil_image: Image.Image,
    region: RegionRequest,
    *,
    processor,
    model,
    max_new_tokens: int,
    temperature: float,
) -> Tuple[Dict[str, Any], str]:
    if region.role.lower() == "source":
        prompt = build_source_prompt(region)
    else:
        prompt = build_target_prompt(region)
    messages = [
        {"role": "system", "content": "You are a precise vision assistant. Reply strictly in JSON."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_image], return_tensors="pt", padding=True).to(model.device)
    generation = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
    )
    decoded = processor.batch_decode(generation, skip_special_tokens=True)[0]
    parsed = parse_grounding_predictions(decoded)
    if parsed:
        mapped = parsed[0]
        bbox = extract_bbox_from_entry(mapped) or region.bbox
        description = mapped.get("description") or mapped.get("text") or mapped.get("label")
    else:
        bbox = region.bbox
        description = None
    return {"role": region.role, "bbox": bbox, "description": description, "raw": parsed[0] if parsed else None}, decoded


def locate_with_qwen_grounding(
    image_path: Path,
    description: Optional[str],
    *,
    processor,
    model,
    model_id: str,
    max_new_tokens: int,
    temperature: float,
) -> Tuple[Optional[List[int]], List[Dict[str, Any]], Optional[str]]:
    if not description:
        return None, [], None

    query = (
        "Locate the region described as "
        f"'{description}'. Return a JSON array with a single object "
        "following the schema: [{\"bbox\": [x1, y1, x2, y2]}]."
    )
    detections, raw = run_qwen3vl_grounding(
        image_path=str(image_path),
        query=query,
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        processor=processor,
        model=model,
        temperature=temperature,
    )
    best = detections[0].get("bbox") if detections else None
    return best if isinstance(best, list) else None, detections, raw


def find_region(regions: Sequence[Dict[str, Any]], role: str) -> Optional[Dict[str, Any]]:
    role_lower = role.lower()
    for region in regions:
        if region.get("role", "").lower() == role_lower:
            return region
    return None


def iter_rows(csv_path: Path, *, image_id: Optional[str]) -> Iterable[Dict[str, Any]]:
    with csv_path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            identifier = str(row.get("image_path") or "").strip()
            identifier = Path(identifier).stem
            if image_id and identifier != image_id:
                continue
            yield row


def process_csv(args: argparse.Namespace) -> Tuple[Dict[str, Any], Path, Optional[Path]]:
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if args.start_row < 0:
        raise ValueError("--start-row must be >= 0")

    processor, model = load_qwen3vl_model(args.model_id, device_map=args.device_map)
    images_root = Path(args.images_root)
    output_path = Path(args.output_json)
    results, failures, seen_keys = load_resume_state(args.resume_from)
    processed = len(results)
    handled_rows = 0
    conversation_entries: List[Dict[str, Any]] = []
    conversation_output_path: Optional[Path] = None
    conversation_interval = (
        args.conversation_save_interval if args.build_conversations and args.conversation_save_interval > 0 else None
    )
    conversation_source_l2_threshold = (
        args.conversation_max_source_normalized_l2
        if args.conversation_max_source_normalized_l2 is not None
        else args.conversation_max_normalized_l2
    )
    conversation_target_l2_threshold = (
        args.conversation_max_target_normalized_l2
        if args.conversation_max_target_normalized_l2 is not None
        else args.conversation_max_normalized_l2
    )
    if args.build_conversations:
        conversation_output_path = (
            Path(args.conversation_output_json)
            if args.conversation_output_json
            else output_path.with_name(f"{output_path.stem}_conversation.json")
        )
        if args.resume_from and conversation_output_path.exists():
            with conversation_output_path.open("r") as fp:
                existing_conversations = json.load(fp)
            if isinstance(existing_conversations, list):
                conversation_entries = existing_conversations
                print(
                    f"[resume] Loaded {len(conversation_entries)} conversation entries from {conversation_output_path}"
                )
            else:
                print(
                    f"[warn] Conversation resume file is not a list; starting fresh: {conversation_output_path}"
                )
                conversation_entries = []
    skip_logger = SkipLogger(args.log_file)

    def build_summary() -> Dict[str, Any]:
        return {
            "csv_path": str(csv_path),
            "images_root": str(args.images_root),
            "model_id": args.model_id,
            "image_id_filter": args.image_id,
            "start_row": args.start_row,
            "limit": args.limit,
            "target_box_size": args.target_box_size,
            "use_head_bbox": args.use_head_bbox,
            "grounding_eval": args.grounding_eval,
            "results": results,
            "failures": failures,
            "conversation_source_l2_threshold": conversation_source_l2_threshold,
            "conversation_target_l2_threshold": conversation_target_l2_threshold,
        }

    def persist(reason: str) -> None:
        summary_snapshot = build_summary()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as fp:
            json.dump(summary_snapshot, fp, indent=2)
        print(f"[save] {reason} -> {output_path} ({len(results)} results, {len(failures)} failures)")

    save_interval = args.save_interval if args.save_interval and args.save_interval > 0 else None

    rows_iterator = list(iter_rows(csv_path, image_id=args.image_id))
    if args.start_row:
        if args.start_row >= len(rows_iterator):
            rows_iterator = []
        else:
            rows_iterator = rows_iterator[args.start_row :]
        print(f"[start] Skipping first {args.start_row} rows before processing")
    progress_total = args.limit if args.limit is not None else len(rows_iterator)
    with ExitStack() as stack:
        progress_bar = stack.enter_context(
            tqdm(rows_iterator, desc=f"Processing samples (total={progress_total})", unit="sample", total=progress_total)
        )
        stack.callback(skip_logger.close)
        for row in progress_bar:
            if args.limit is not None and processed >= args.limit:
                break

            relative_key = extract_relative_path(row)
            image_identifier = relative_key or str(row.get("image_path") or "").strip()
            annotation_id = str(row.get("id") or "")
            handled_rows += 1
            sample_keys = generate_sample_key_variants(
                primary_candidates=[relative_key, image_identifier],
                secondary_candidates=[image_identifier, relative_key],
                annotation_candidates=[annotation_id],
            )
            if any(key in seen_keys for key in sample_keys):
                skip_logger.log(
                    f"[{handled_rows}] {image_identifier or 'N/A'}#{annotation_id or '-'} SKIP: already processed (resume)"
                )
                print(f"[skip] {image_identifier or 'N/A'}#{annotation_id or '-'}: already processed (resume)")
                continue
            else:
                print(f"[process] {image_identifier or 'N/A'}#{annotation_id or '-'}")
            if not relative_key:
                reason = "missing image path columns"
                failures.append({"image_id": image_identifier, "annotation_id": annotation_id, "reason": reason})
                skip_logger.log(f"[{handled_rows}] {image_identifier or 'N/A'}#{annotation_id or '-'} FAILED: {reason}")
                if save_interval and handled_rows % save_interval == 0:
                    persist(f"autosave after {handled_rows} rows")
                continue
            image_path = images_root / relative_key
            if not image_path.exists():
                reason = f"image not found on disk ({image_path})"
                failures.append({"image_id": image_identifier, "annotation_id": annotation_id, "reason": reason})
                skip_logger.log(f"[{handled_rows}] {image_identifier or 'N/A'}#{annotation_id or '-'} FAILED: {reason}")
                if save_interval and handled_rows % save_interval == 0:
                    persist(f"autosave after {handled_rows} rows")
                continue

            source_x = parse_float(row, "body_bbox_x")
            source_y = parse_float(row, "body_bbox_y")
            source_w = parse_float(row, "body_bbox_width")
            source_h = parse_float(row, "body_bbox_height")
            gaze_x = parse_float(row, "gaze_x")
            gaze_y = parse_float(row, "gaze_y")

            if gaze_x is None or gaze_y is None:
                failures.append(
                    {"image_id": image_identifier, "annotation_id": annotation_id, "reason": "missing gaze target coordinates"}
                )
                skip_logger.log(
                    f"[{handled_rows}] {image_identifier or 'N/A'}#{annotation_id or '-'} FAILED: missing gaze target coordinates"
                )
                if save_interval and handled_rows % save_interval == 0:
                    persist(f"autosave after {handled_rows} rows")
                continue

            if not args.use_head_bbox and None in (source_x, source_y, source_w, source_h):
                failures.append(
                    {"image_id": image_identifier, "annotation_id": annotation_id, "reason": "missing body bbox coordinates"}
                )
                skip_logger.log(
                    f"[{handled_rows}] {image_identifier or 'N/A'}#{annotation_id or '-'} FAILED: missing body bbox coordinates"
                )
                if save_interval and handled_rows % save_interval == 0:
                    persist(f"autosave after {handled_rows} rows")
                continue

            pil_image = load_rgb_image(image_path)

            img_width, img_height = pil_image.size

            if args.use_head_bbox:
                head_x_min = parse_float(row, "head_bbox_x_min")
                head_y_min = parse_float(row, "head_bbox_y_min")
                head_x_max = parse_float(row, "head_bbox_x_max")
                head_y_max = parse_float(row, "head_bbox_y_max")
                if None in (head_x_min, head_y_min, head_x_max, head_y_max):
                    reason = "missing head bbox coordinates"
                    failures.append({"image_id": image_identifier, "annotation_id": annotation_id, "reason": reason})
                    skip_logger.log(f"[{handled_rows}] {image_identifier or 'N/A'}#{annotation_id or '-'} FAILED: {reason}")
                    if save_interval and handled_rows % save_interval == 0:
                        persist(f"autosave after {handled_rows} rows")
                    continue
                head_bbox_pixels = [head_x_min, head_y_min, head_x_max, head_y_max]
                source_bbox = pixel_bbox_to_absolute(head_bbox_pixels, img_width, img_height)
                source_bbox_type = "head"
            else:
                source_bbox = normalized_bbox_to_absolute(source_x, source_y, source_w, source_h)  # type: ignore[arg-type]
                source_bbox_type = "body"
            target_bbox, target_point = normalized_point_to_bbox(gaze_x, gaze_y, args.target_box_size)
            source_bbox_pixels = bbox_1000_to_pixels(source_bbox, img_width, img_height)
            target_bbox_pixels = bbox_1000_to_pixels(target_bbox, img_width, img_height)
            target_point_pixels = point_1000_to_pixels(target_point, img_width, img_height)
            gaze_point_pixels = (gaze_x * img_width, gaze_y * img_height)
            source_center_pixels = bbox_center(source_bbox_pixels)

            regions = [
                RegionRequest(role="source", bbox=source_bbox),
                RegionRequest(
                    role="target",
                    bbox=target_bbox,
                    point=target_point,
                    use_point=args.use_target_point,
                ),
            ]

            try:
                region_descriptions: List[Dict[str, Any]] = []
                for region in regions:
                    described, _ = describe_region(
                        pil_image,
                        region,
                        processor=processor,
                        model=model,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
                    region_descriptions.append(described)
            except Exception as exc:  # pragma: no cover - runtime inference errors
                failures.append({"image_id": image_identifier, "annotation_id": annotation_id, "reason": f"inference failed: {exc}"})
                skip_logger.log(f"[{handled_rows}] {image_identifier or 'N/A'}#{annotation_id or '-'} FAILED: inference failed ({exc})")
                if save_interval and handled_rows % save_interval == 0:
                    persist(f"autosave after {handled_rows} rows")
                continue

            source_region = find_region(region_descriptions, "source")
            target_region = find_region(region_descriptions, "target")

            grounding_eval: Optional[Dict[str, Any]] = None
            if args.grounding_eval:
                grounding_eval = {}

                source_pred_box: Optional[List[int]] = None
                if source_region and source_region.get("description"):
                    source_pred_box, source_detections, source_raw = locate_with_qwen_grounding(
                        image_path,
                        source_region["description"],
                        processor=processor,
                        model=model,
                        model_id=args.model_id,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
                else:
                    source_detections = []
                    source_raw = None

                source_errors = None
                if source_pred_box:
                    source_errors = compute_gaze_errors(
                        predicted_box=source_pred_box,
                        person_box=source_bbox_pixels,
                        ground_truth_point=source_center_pixels,
                        image_width=img_width,
                        image_height=img_height,
                    )

                target_pred_box: Optional[List[int]] = None
                if target_region and target_region.get("description"):
                    target_pred_box, target_detections, target_raw = locate_with_qwen_grounding(
                        image_path,
                        target_region["description"],
                        processor=processor,
                        model=model,
                        model_id=args.model_id,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
                else:
                    target_detections = []
                    target_raw = None

                target_errors = None
                if target_pred_box:
                    target_errors = compute_gaze_errors(
                        predicted_box=target_pred_box,
                        person_box=source_bbox_pixels,
                        ground_truth_point=gaze_point_pixels,
                        image_width=img_width,
                        image_height=img_height,
                    )

                grounding_eval["source"] = {
                    "description": source_region.get("description") if source_region else None,
                    "predicted_bbox": source_pred_box,
                    "detections": source_detections,
                    "raw_response": source_raw,
                    "errors": source_errors,
                }
                grounding_eval["target"] = {
                    "description": target_region.get("description") if target_region else None,
                    "predicted_bbox": target_pred_box,
                    "detections": target_detections,
                    "raw_response": target_raw,
                    "errors": target_errors,
                }

            processed += 1
            if images_root in image_path.parents:
                relative_path = str(image_path.relative_to(images_root))
            else:
                relative_path = image_path.name
            record = {
                "image_id": image_identifier,
                "annotation_id": annotation_id,
                "relative_path": relative_path,
                "image_path": str(image_path),
                "source": {
                    "bbox": source_bbox_pixels,
                    "bbox_type": source_bbox_type,
                    "description": source_region.get("description") if source_region else None,
                },
                "target": {
                    "bbox": target_bbox_pixels,
                    "point": target_point_pixels,
                    "description": target_region.get("description") if target_region else None,
                },
                "grounding_eval": grounding_eval,
                "in_or_out": row.get("in_or_out"),
            }
            if is_outside_frame(record.get("in_or_out")):
                record["target"]["description"] = OUTSIDE_FRAME_TARGET_DESCRIPTION
            results.append(record)
            seen_keys.update(sample_keys)
            seen_keys.update(
                generate_sample_key_variants(
                    primary_candidates=[relative_path],
                    secondary_candidates=[image_identifier, relative_path],
                    annotation_candidates=[annotation_id],
                )
            )
            if args.build_conversations and conversation_output_path is not None:
                convo_entry, reason = build_conversation_entry(
                    record,
                    images_root,
                    precomputed_size=(img_width, img_height),
                    source_l2_threshold=conversation_source_l2_threshold,
                    target_l2_threshold=conversation_target_l2_threshold,
                )
                if convo_entry:
                    conversation_entries.append(convo_entry)
                    if conversation_interval and len(conversation_entries) % conversation_interval == 0:
                        save_conversation_dataset(conversation_entries, conversation_output_path)
                        print(f"[save] conversation autosave -> {conversation_output_path} ({len(conversation_entries)} entries)")
                else:
                    skip_logger.log(
                        f"[skip] conversation entry {image_identifier or 'N/A'}#{annotation_id or '-'}: {reason}"
                    )
                    print(f"[skip] conversation entry {image_identifier or 'N/A'}#{annotation_id or '-'}: {reason}")
            region_summary = ", ".join(
                f"{entry['role']}={entry.get('description') or 'N/A'}" for entry in region_descriptions
            ) or "no regions parsed"
            print(f"[{handled_rows}] {image_identifier or 'N/A'}#{annotation_id or '-'} OK: {region_summary}")

            if save_interval and handled_rows % save_interval == 0:
                persist(f"autosave after {handled_rows} rows")

    final_summary = build_summary()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fp:
        json.dump(final_summary, fp, indent=2)
    print(f"[save] final -> {output_path} ({len(results)} results, {len(failures)} failures)")

    conversation_path = None
    if args.build_conversations and conversation_output_path is not None:
        save_conversation_dataset(conversation_entries, conversation_output_path)
        print(f"[save] conversation final -> {conversation_output_path} ({len(conversation_entries)} entries)")
        conversation_path = conversation_output_path

    return final_summary, output_path, conversation_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Describe gaze source and target coordinates with Qwen3-VL.")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV, help="CSV file with gaze annotations.")
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT, help="Directory containing gaze images.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-4B-Instruct", help="Qwen3-VL model identifier.")
    parser.add_argument("--device-map", type=str, default="auto", help="Device map passed to transformers.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT, help="Destination JSON file for the results.")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to a prior JSON summary; when provided, already processed samples are skipped and appended to the new output.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="0-based row index in the CSV to begin processing from (rows before this index are skipped).",
    )
    parser.add_argument("--target-box-size", type=float, default=60.0, help="Bounding box size (in 0-1000 space) for gaze target points.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of rows to process.")
    parser.add_argument("--image-id", type=str, default=None, help="Process a single image_id (matches image_path column).")
    parser.add_argument(
        "--use-target-point",
        action="store_true",
        help="When set, describe the gaze target using its point instead of a bounding box.",
    )
    parser.add_argument(
        "--use-head-bbox",
        action="store_true",
        help="Use head bounding boxes from the annotation CSV for the gaze source instead of body boxes.",
    )
    parser.add_argument(
        "--grounding-eval",
        action="store_true",
        help="Run Qwen3-VL grounding on the generated descriptions and compute normalized L2 errors.",
    )
    parser.add_argument(
        "--build-conversations",
        action="store_true",
        help="After producing gaze descriptions, build a conversational JSON dataset.",
    )
    parser.add_argument(
        "--conversation-output-json",
        type=Path,
        default=None,
        help="Optional override for the generated conversational JSON path.",
    )
    parser.add_argument(
        "--conversation-save-interval",
        type=int,
        default=DEFAULT_CONVO_SAVE_INTERVAL,
        help="Autosave interval while building conversational JSON (only used when --build-conversations is set).",
    )
    parser.add_argument(
        "--conversation-max-normalized-l2",
        type=float,
        default=DEFAULT_CONVO_L2_TARGET_THRESHOLD,
        help="Maximum normalized L2 error required for both source and target when adding to conversations (deprecated; use source/target specific flags).",
    )
    parser.add_argument(
        "--conversation-max-source-normalized-l2",
        type=float,
        default=DEFAULT_CONVO_L2_SOURCE_THRESHOLD,
        help="Maximum normalized L2 error for the gaze source when adding to conversations (defaults to --conversation-max-normalized-l2).",
    )
    parser.add_argument(
        "--conversation-max-target-normalized-l2",
        type=float,
        default=DEFAULT_CONVO_L2_TARGET_THRESHOLD,
        help="Maximum normalized L2 error for the gaze target when adding to conversations (defaults to --conversation-max-normalized-l2).",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=DEFAULT_SAVE_INTERVAL,
        help="Autosave summary JSON every N processed rows (set <=0 to disable).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("gazefollow/auto_phrase_grounding/describe_gaze_targets.log"),
        help="Optional path to append logs about skipped samples and failures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.build_conversations and not args.grounding_eval:
        print("[warn] --build-conversations requires --grounding-eval to produce bounding boxes; enabling conversations without grounding data will result in zero entries.")
    summary, output_path, conversation_path = process_csv(args)
    print(f"Completed run: {len(summary['results'])} successes, {len(summary['failures'])} failures. Output -> {output_path}")
    if args.build_conversations and conversation_path:
        print(f"Conversation dataset saved to {conversation_path}")


if __name__ == "__main__":
    main()
