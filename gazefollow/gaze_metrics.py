from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from shutil import copy2
from typing import Any, Dict, List, Optional, Tuple



COMBINED_CSV_PATH_TRAIN = Path("gazefollow/data/combined_description_results.csv")
COMBINED_CSV_PATH_TEST = Path("gazefollow/data/test2_combined_description_results.csv")
_combined_dataset_cache: Optional[Dict[str, Dict[str, str]]] = None


def load_combined_description_cache(csv_path: Path = COMBINED_CSV_PATH_TRAIN) -> Dict[str, Dict[str, str]]:
    """
    Load the combined description CSV into a dictionary keyed by relative image path.
    Data is cached for subsequent calls.
    """
    global _combined_dataset_cache

    # if _combined_dataset_cache is not None and csv_path == COMBINED_CSV_PATH_TRAIN:
    #     return _combined_dataset_cache
    print(f"Loading combined description CSV from {csv_path}...")
    mapping: Dict[str, Dict[str, str]] = {}
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                rel_path = (
                    row.get("image_path.1")
                    or row.get("image_path")
                    or row.get("image")
                    or row.get("id")
                )
                if not rel_path:
                    continue
                key = str(rel_path).strip().replace("\\", "/")
                if key.startswith("./"):
                    key = key[2:]
                if key:
                    # Keep first annotation per image path when multiple rows exist.
                    mapping.setdefault(key, row)
    else:
        print(f"Warning: Combined description CSV not found at {csv_path}")

    # if csv_path == COMBINED_CSV_PATH_TRAIN:
    #     _combined_dataset_cache = mapping

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


def compute_gaze_iou(
    predicted_box: Optional[List[float]],
    ground_truth_point: Tuple[float, float],
    image_width: int,
    image_height: int,
    radius_ratio: float = 0.05,
) -> Optional[float]:
    """
    Compute IoU between predicted bounding box and ground-truth gaze point treated as a circle.
    
    Args:
        predicted_box: [x1, y1, x2, y2] predicted bounding box coordinates
        ground_truth_point: (x, y) ground-truth gaze point
        image_width: Width of the image
        image_height: Height of the image
        radius_ratio: Ratio of image diagonal to use as ground-truth circle radius (default: 0.05)
    
    Returns:
        IoU value between 0 and 1, or None if computation fails
    """
    if not predicted_box or len(predicted_box) != 4:
        return None
    
    gt_x, gt_y = ground_truth_point
    diagonal = math.hypot(image_width, image_height)
    gt_radius = diagonal * radius_ratio
    
    # Ground-truth circle as a square bounding box
    gt_x1 = gt_x - gt_radius
    gt_y1 = gt_y - gt_radius
    gt_x2 = gt_x + gt_radius
    gt_y2 = gt_y + gt_radius
    
    # Predicted box
    pred_x1, pred_y1, pred_x2, pred_y2 = predicted_box
    
    # Compute intersection
    inter_x1 = max(pred_x1, gt_x1)
    inter_y1 = max(pred_y1, gt_y1)
    inter_x2 = min(pred_x2, gt_x2)
    inter_y2 = min(pred_y2, gt_y2)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0  # No overlap
    
    intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
    # Compute union
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    union_area = pred_area + gt_area - intersection_area
    
    if union_area <= 0:
        return None
    
    return intersection_area / union_area


def compute_modified_l2_error(
    predicted_box: Optional[List[float]],
    ground_truth_point: Tuple[float, float],
    image_width: int,
    image_height: int,
    ratio_factor: float = 0.25,
) -> Optional[float]:
    """
    Compute modified L2 error that considers both containment and bbox size.
    
    If GT point is inside the predicted bbox:
        error = A_bbox / A_image
    If GT point is outside the predicted bbox:
        error = (distance_to_nearest_edge / diagonal) + (A_bbox / A_image)
    
    This metric penalizes large boxes and adds distance penalty when GT is outside.
    
    Args:
        predicted_box: [x1, y1, x2, y2] predicted bounding box coordinates
        ground_truth_point: (x, y) ground-truth gaze point
        image_width: Width of the image
        image_height: Height of the image
    
    Returns:
        Modified L2 error value, or None if computation fails
    """
    if not predicted_box or len(predicted_box) != 4:
        return None
    
    pred_x1, pred_y1, pred_x2, pred_y2 = predicted_box
    gt_x, gt_y = ground_truth_point
    
    # Compute area ratio
    bbox_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    image_area = image_width * image_height
    area_ratio = bbox_area / image_area if image_area > 0 else 0.0
    area_ratio *= ratio_factor  # Scale factor to control influence
    
    # Check if GT point is inside the bbox
    inside = (pred_x1 <= gt_x <= pred_x2) and (pred_y1 <= gt_y <= pred_y2)
    
    if inside:
        # GT is inside: error is only the area ratio
        return area_ratio
    else:
        # GT is outside: compute distance to nearest edge
        # Distance to nearest vertical edge
        dx = 0.0
        if gt_x < pred_x1:
            dx = pred_x1 - gt_x
        elif gt_x > pred_x2:
            dx = gt_x - pred_x2
        
        # Distance to nearest horizontal edge
        dy = 0.0
        if gt_y < pred_y1:
            dy = pred_y1 - gt_y
        elif gt_y > pred_y2:
            dy = gt_y - pred_y2
        
        # Euclidean distance to nearest point on bbox
        distance_to_bbox = math.hypot(dx, dy)
        
        # Normalize by image diagonal
        diagonal = math.hypot(image_width, image_height)
        normalized_distance = distance_to_bbox / diagonal if diagonal > 0 else 0.0
        
        # Combined error: distance + area ratio
        return normalized_distance + area_ratio


def compute_gaze_errors(
    predicted_box: Optional[List[float]],
    person_box: Optional[List[float]],
    ground_truth_point: Tuple[float, float],
    image_width: int,
    image_height: int,
    iou_radius_ratio: float = 0.05,
) -> Dict[str, Optional[float]]:
    """
    Compute L2, normalized L2, angular errors, and IoU for gaze predictions.
    
    Args:
        predicted_box: [x1, y1, x2, y2] predicted bounding box coordinates
        person_box: [x1, y1, x2, y2] person bounding box coordinates
        ground_truth_point: (x, y) ground-truth gaze point
        image_width: Width of the image
        image_height: Height of the image
        iou_radius_ratio: Ratio of image diagonal to use as ground-truth circle radius for IoU (default: 0.05)
    
    Returns:
        Dictionary with error metrics including gaze_iou
    """
    errors: Dict[str, Optional[float]] = {
        "gaze_l2_error": None,
        "gaze_normalized_l2_error": None,
        "gaze_angular_error": None,
        "gaze_iou": None,
        "gaze_modified_l2_error": None,
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

    # Compute IoU
    iou = compute_gaze_iou(predicted_box, ground_truth_point, image_width, image_height, iou_radius_ratio)
    if iou is not None:
        errors["gaze_iou"] = iou
    
    # Compute modified L2 error
    modified_l2 = compute_modified_l2_error(predicted_box, ground_truth_point, image_width, image_height)
    if modified_l2 is not None:
        errors["gaze_modified_l2_error"] = modified_l2

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_suffix = dataset_path.suffix + f".bak_{timestamp}"
        backup_path = dataset_path.with_suffix(backup_suffix)

        counter = 1
        while backup_path.exists():
            backup_path = dataset_path.with_suffix(
                dataset_path.suffix + f".bak_{timestamp}_{counter}"
            )
            counter += 1

        try:
            copy2(dataset_path, backup_path)
            print(f"Created backup of dataset at {backup_path}")
        except Exception as exc:
            print(f"Warning: Failed to create dataset backup at {backup_path}: {exc}")

        with open(dataset_path, "w", encoding="utf-8") as dataset_file:
            json.dump(full_dataset, dataset_file, indent=2, ensure_ascii=False)

    return persisted
