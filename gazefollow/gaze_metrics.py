from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


COMBINED_CSV_PATH = Path("/galitylab/students/alonmardi/projects/LLaVA-NeXT/combined_description_results.csv")
_combined_dataset_cache: Optional[Dict[str, Dict[str, str]]] = None


def load_combined_description_cache(csv_path: Path = COMBINED_CSV_PATH) -> Dict[str, Dict[str, str]]:
    """
    Load the combined description CSV into a dictionary keyed by relative image path.
    Data is cached for subsequent calls.
    """
    global _combined_dataset_cache

    if _combined_dataset_cache is not None and csv_path == COMBINED_CSV_PATH:
        return _combined_dataset_cache

    mapping: Dict[str, Dict[str, str]] = {}
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                rel_path = row.get("image_path.1")
                if not rel_path:
                    continue
                key = rel_path.strip()
                if key:
                    mapping[key] = row
    else:
        print(f"Warning: Combined description CSV not found at {csv_path}")

    if csv_path == COMBINED_CSV_PATH:
        _combined_dataset_cache = mapping

    return mapping


def ensure_ground_truth_gaze(
    sample: Dict[str, Any],
    relative_image_path: str,
    image_width: int,
    image_height: int,
    combined_cache: Dict[str, Dict[str, str]],
) -> Tuple[Optional[Tuple[float, float, int, int]], bool]:
    """
    Ensure ground-truth gaze coordinates exist for a sample.

    Returns:
        ((gt_x, gt_y, width, height), updated_flag) or (None, False) if unavailable.
    """
    if "gaze_gt_x" in sample and "gaze_gt_y" in sample:
        try:
            width = int(sample.get("gaze_gt_width", image_width))
            height = int(sample.get("gaze_gt_height", image_height))
            gt_x = float(sample["gaze_gt_x"])
            gt_y = float(sample["gaze_gt_y"])
            return (gt_x, gt_y, width, height), False
        except (TypeError, ValueError):
            pass  # Fall through and recompute

    mapping = combined_cache.get(relative_image_path)
    if not mapping:
        return None, False

    try:
        gaze_x = float(mapping["gaze_x"])
        gaze_y = float(mapping["gaze_y"])
    except (TypeError, ValueError, KeyError):
        return None, False

    gt_x = gaze_x * image_width
    gt_y = gaze_y * image_height

    sample["gaze_gt_x"] = gt_x
    sample["gaze_gt_y"] = gt_y
    sample["gaze_gt_width"] = image_width
    sample["gaze_gt_height"] = image_height

    return (gt_x, gt_y, image_width, image_height), True


def _box_center(box: Optional[List[float]]) -> Optional[Tuple[float, float]]:
    if not box or len(box) != 4:
        return None
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def compute_gaze_errors(
    predicted_box: Optional[List[float]],
    person_box: Optional[List[float]],
    ground_truth_point: Tuple[float, float],
    image_width: int,
    image_height: int,
) -> Dict[str, Optional[float]]:
    """Compute L2, normalized L2, and angular errors for gaze predictions."""
    errors: Dict[str, Optional[float]] = {
        "gaze_l2_error": None,
        "gaze_normalized_l2_error": None,
        "gaze_angular_error": None,
    }

    if not predicted_box:
        return errors

    predicted_center = _box_center(predicted_box)
    if not predicted_center:
        return errors

    gt_x, gt_y = ground_truth_point
    dx = predicted_center[0] - gt_x
    dy = predicted_center[1] - gt_y
    l2 = math.hypot(dx, dy)
    errors["gaze_l2_error"] = l2

    diagonal = math.hypot(image_width, image_height)
    if diagonal > 0:
        errors["gaze_normalized_l2_error"] = l2 / diagonal

    person_center = _box_center(person_box)
    if person_center:
        pred_vec = (predicted_center[0] - person_center[0], predicted_center[1] - person_center[1])
        gt_vec = (gt_x - person_center[0], gt_y - person_center[1])
        pred_norm = math.hypot(pred_vec[0], pred_vec[1])
        gt_norm = math.hypot(gt_vec[0], gt_vec[1])
        if pred_norm > 0 and gt_norm > 0:
            dot = pred_vec[0] * gt_vec[0] + pred_vec[1] * gt_vec[1]
            denom = pred_norm * gt_norm
            if denom > 0:
                cosine = max(-1.0, min(1.0, dot / denom))
                errors["gaze_angular_error"] = math.degrees(math.acos(cosine))

    return errors


def persist_ground_truth_updates(dataset_path: Path, updates: List[Dict[str, Any]]) -> int:
    """
    Persist gaze ground-truth updates back into the dataset JSON.

    Args:
        dataset_path: Path to the dataset JSON file.
        updates: List of dictionaries containing id/image/path and values to update.

    Returns:
        Number of entries updated.
    """
    if not updates:
        return 0

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset JSON not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as dataset_file:
        full_dataset = json.load(dataset_file)

    updates_by_id: Dict[str, Dict[str, Any]] = {}
    updates_by_image: Dict[str, Dict[str, Any]] = {}
    updates_by_relative: Dict[str, Dict[str, Any]] = {}

    for payload in updates:
        values = payload["values"]
        if payload.get("id"):
            updates_by_id[payload["id"]] = values
        if payload.get("image"):
            updates_by_image[payload["image"]] = values
        if payload.get("relative_path"):
            updates_by_relative[payload["relative_path"]] = values

    persisted = 0
    for entry in full_dataset:
        entry_id = entry.get("id")
        entry_image = entry.get("image")

        update_values = None
        if entry_id and entry_id in updates_by_id:
            update_values = updates_by_id[entry_id]
        elif entry_image and entry_image in updates_by_image:
            update_values = updates_by_image[entry_image]
        else:
            image_value = entry.get("image")
            if image_value:
                candidate_path = updates_by_relative.get(image_value)
                if not candidate_path:
                    for extension in (".jpg", ".png", ".jpeg"):
                        candidate = f"{image_value}{extension}"
                        if candidate in updates_by_relative:
                            candidate_path = updates_by_relative[candidate]
                            break
                if candidate_path:
                    update_values = candidate_path

        if update_values:
            entry.update(update_values)
            persisted += 1

    if persisted:
        with open(dataset_path, "w", encoding="utf-8") as dataset_file:
            json.dump(full_dataset, dataset_file, indent=2, ensure_ascii=False)

    return persisted
