import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


def _is_rank0() -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def roi_debug_enabled(step: int, debug_steps: int) -> bool:
    return int(debug_steps) > 0 and int(step) <= int(debug_steps) and _is_rank0()


def roi_debug_log(step: int, debug_steps: int, message: str) -> None:
    if roi_debug_enabled(step=step, debug_steps=debug_steps):
        print(f"[ROI debug step={step}] {message}")


class ROIContrastivePreviewBuffer:
    """Collect and materialize fixed-size ROI contrastive preview tensors."""

    def __init__(self, preview_samples: int):
        self.preview_samples = max(1, int(preview_samples))
        self.sample_indices: List[int] = []
        self.pred_candidate_slots: List[int] = []
        self.candidate_slot_scores: List[torch.Tensor] = []
        self.oof_scores: List[torch.Tensor] = []

    def append(
        self,
        sample_index: int,
        pred_candidate_slot: int,
        candidate_slot_scores: torch.Tensor,
        oof_scores: Optional[torch.Tensor] = None,
    ) -> None:
        if len(self.sample_indices) >= self.preview_samples:
            return
        self.sample_indices.append(int(sample_index))
        self.pred_candidate_slots.append(int(pred_candidate_slot))
        if candidate_slot_scores.ndim == 1:
            self.candidate_slot_scores.append(candidate_slot_scores.detach().float())
        else:
            self.candidate_slot_scores.append(torch.empty((0,), dtype=torch.float32))
        if oof_scores is not None and oof_scores.ndim == 1:
            self.oof_scores.append(oof_scores.detach().float())
        else:
            self.oof_scores.append(torch.empty((0,), dtype=torch.float32))

    def to_tensors(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_index_tensor = torch.full((self.preview_samples,), -1, device=device, dtype=torch.long)
        pred_candidate_slot_tensor = torch.full((self.preview_samples,), -1, device=device, dtype=torch.long)

        max_candidate_slots = 0
        for slot_scores in self.candidate_slot_scores:
            max_candidate_slots = max(max_candidate_slots, int(slot_scores.numel()))
        max_oof_scores = 0
        for sample_oof_scores in self.oof_scores:
            max_oof_scores = max(max_oof_scores, int(sample_oof_scores.numel()))

        candidate_slot_score_tensor = torch.full(
            (self.preview_samples, max_candidate_slots),
            float("nan"),
            device=device,
            dtype=torch.float32,
        )
        oof_score_tensor = torch.full(
            (self.preview_samples, max_oof_scores),
            float("nan"),
            device=device,
            dtype=torch.float32,
        )

        limit = min(self.preview_samples, len(self.sample_indices))
        for idx in range(limit):
            sample_index_tensor[idx] = int(self.sample_indices[idx])
            pred_candidate_slot_tensor[idx] = int(self.pred_candidate_slots[idx])
            if idx < len(self.candidate_slot_scores):
                slot_scores = self.candidate_slot_scores[idx]
                if slot_scores.numel() > 0:
                    candidate_slot_score_tensor[idx, : int(slot_scores.numel())] = slot_scores.to(
                        device=device, dtype=torch.float32
                    )
            if idx < len(self.oof_scores):
                sample_oof_scores = self.oof_scores[idx]
                if sample_oof_scores.numel() > 0:
                    oof_score_tensor[idx, : int(sample_oof_scores.numel())] = sample_oof_scores.to(
                        device=device, dtype=torch.float32
                    )
        return (
            sample_index_tensor,
            pred_candidate_slot_tensor,
            candidate_slot_score_tensor,
            oof_score_tensor,
        )


def build_circular_roi_patch_indices(x_norm: float, y_norm: float, grid_side: int, radius: int) -> List[int]:
    center_x = int(round(x_norm * (grid_side - 1)))
    center_y = int(round(y_norm * (grid_side - 1)))
    valid_indices: List[int] = []
    r2 = radius * radius
    min_x = max(0, center_x - radius)
    max_x = min(grid_side - 1, center_x + radius)
    min_y = max(0, center_y - radius)
    max_y = min(grid_side - 1, center_y + radius)
    for y_coord in range(min_y, max_y + 1):
        dy = y_coord - center_y
        for x_coord in range(min_x, max_x + 1):
            dx = x_coord - center_x
            if dx * dx + dy * dy <= r2:
                valid_indices.append(y_coord * grid_side + x_coord)
    return valid_indices


def build_focus_phrase_token_ids(tokenizer, phrase: str) -> List[List[int]]:
    if not phrase:
        return []

    stripped = phrase.strip()
    candidates = {phrase}
    if stripped:
        candidates.add(stripped)
        candidates.add(stripped.lower())
        candidates.add(stripped.capitalize())
        candidates.add(f" {stripped}")
        candidates.add(f" {stripped.lower()}")

    token_sequences: List[List[int]] = []
    seen: set = set()
    for candidate in candidates:
        if not candidate:
            continue
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if not ids:
            continue
        key = tuple(ids)
        if key in seen:
            continue
        seen.add(key)
        token_sequences.append(ids)

    return token_sequences


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _parse_detection_bbox(value: Any) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(value, list) or len(value) != 4:
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


def _detection_signature(det: Dict[str, Any]) -> Tuple[str, str, Tuple[int, int, int, int], Optional[float]]:
    bbox = _parse_detection_bbox(det.get("bbox")) or (0, 0, 0, 0)
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
) -> List[Dict[str, Any]]:
    removed_sigs = {_detection_signature(det) for det in removed}
    filtered: List[Dict[str, Any]] = []
    seen: set = set()
    for det in negatives:
        sig = _detection_signature(det)
        if sig in removed_sigs:
            continue
        if sig in seen:
            continue
        seen.add(sig)
        filtered.append(det)
    return filtered


def _bbox_intersects(box_a: Tuple[int, int, int, int], box_b: List[float]) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    return (float(bbox[0] + bbox[2]) / 2.0, float(bbox[1] + bbox[3]) / 2.0)


def _normalized_center_distance(
    det_bbox: Tuple[int, int, int, int],
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
    gt_x: float,
    gt_y: float,
    gt_w: float,
    gt_h: float,
    gt_radius_ratio: float,
    center_distance_threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
        bbox = _parse_detection_bbox(det.get("bbox"))
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


def build_roi_gaze_metadata(
    sample_dict: Dict[str, Any],
    roi_entry: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    default_xy = torch.tensor([-1.0, -1.0], dtype=torch.float32)
    gaze_x = sample_dict.get("gaze_gt_x", sample_dict.get("gt_x"))
    gaze_y = sample_dict.get("gaze_gt_y", sample_dict.get("gt_y"))
    gaze_w = sample_dict.get("gaze_gt_width", sample_dict.get("gt_width"))
    gaze_h = sample_dict.get("gaze_gt_height", sample_dict.get("gt_height"))
    in_out = sample_dict.get("in_out")

    if (gaze_x is None or gaze_y is None or gaze_w is None or gaze_h is None) and roi_entry:
        gaze_x = roi_entry.get("gt_x", gaze_x)
        gaze_y = roi_entry.get("gt_y", gaze_y)
        gaze_w = roi_entry.get("gt_width", gaze_w)
        gaze_h = roi_entry.get("gt_height", gaze_h)
        if in_out is None:
            in_out = roi_entry.get("in_out")

    if gaze_x is None or gaze_y is None or gaze_w is None or gaze_h is None:
        return default_xy, torch.tensor(False, dtype=torch.bool)

    gaze_x = float(gaze_x)
    gaze_y = float(gaze_y)
    gaze_w = float(gaze_w)
    gaze_h = float(gaze_h)
    if not all(math.isfinite(v) for v in (gaze_x, gaze_y, gaze_w, gaze_h)):
        return default_xy, torch.tensor(False, dtype=torch.bool)
    if gaze_w <= 0 or gaze_h <= 0:
        return default_xy, torch.tensor(False, dtype=torch.bool)
    if in_out is not None:
        try:
            if int(in_out) == 0:
                return default_xy, torch.tensor(False, dtype=torch.bool)
        except (TypeError, ValueError):
            pass

    x_norm = max(0.0, min(1.0, gaze_x / gaze_w))
    y_norm = max(0.0, min(1.0, gaze_y / gaze_h))
    return torch.tensor([x_norm, y_norm], dtype=torch.float32), torch.tensor(True, dtype=torch.bool)


def build_bbox_patch_indices(
    x1_norm: float,
    y1_norm: float,
    x2_norm: float,
    y2_norm: float,
    grid_side: int,
) -> List[int]:
    x1 = int(math.floor(max(0.0, min(1.0, x1_norm)) * (grid_side - 1)))
    y1 = int(math.floor(max(0.0, min(1.0, y1_norm)) * (grid_side - 1)))
    x2 = int(math.ceil(max(0.0, min(1.0, x2_norm)) * (grid_side - 1)))
    y2 = int(math.ceil(max(0.0, min(1.0, y2_norm)) * (grid_side - 1)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    patch_indices: List[int] = []
    for y_coord in range(y1, y2 + 1):
        for x_coord in range(x1, x2 + 1):
            patch_indices.append(y_coord * grid_side + x_coord)
    return patch_indices


def build_roi_position_vector(x1_norm: float, y1_norm: float, x2_norm: float, y2_norm: float) -> List[float]:
    x1 = max(0.0, min(1.0, x1_norm))
    y1 = max(0.0, min(1.0, y1_norm))
    x2 = max(0.0, min(1.0, x2_norm))
    y2 = max(0.0, min(1.0, y2_norm))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    area = width * height
    aspect = min(width / max(height, 1e-6), 4.0) / 4.0
    return [cx, cy, width, height, area, aspect]


def _normalize_candidate_bbox(
    detection: Dict[str, Any],
    image_width: float,
    image_height: float,
) -> Optional[List[float]]:
    bbox = detection.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    if image_width <= 0 or image_height <= 0:
        return None
    x1 = _safe_float(bbox[0])
    y1 = _safe_float(bbox[1])
    x2 = _safe_float(bbox[2])
    y2 = _safe_float(bbox[3])
    if None in (x1, y1, x2, y2):
        return None
    x1n = max(0.0, min(1.0, float(x1) / image_width))
    y1n = max(0.0, min(1.0, float(y1) / image_height))
    x2n = max(0.0, min(1.0, float(x2) / image_width))
    y2n = max(0.0, min(1.0, float(y2) / image_height))
    if x2n < x1n:
        x1n, x2n = x2n, x1n
    if y2n < y1n:
        y1n, y2n = y2n, y1n
    return [x1n, y1n, x2n, y2n]


def _csv_entry_lookup_keys(csv_entry: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    for key_name in ("sample_id", "image_key", "image_path"):
        value = csv_entry.get(key_name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            keys.append(text)
            keys.append(text.replace("\\", "/"))
            path = Path(text)
            keys.append(path.name)
            keys.append(path.stem)
    deduped: List[str] = []
    seen: set = set()
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def _sample_lookup_keys(sample: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    for key_name in ("id", "image", "image_path"):
        value = sample.get(key_name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            keys.append(text)
            keys.append(text.replace("\\", "/"))
            path = Path(text)
            keys.append(path.name)
            keys.append(path.stem)
    deduped: List[str] = []
    seen: set = set()
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def load_roi_candidate_lookup(
    csv_path: Path,
    logger: Optional[Callable[[str], None]] = None,
    recompute_positives: bool = False,
    positive_center_distance_threshold: float = 0.1,
    gt_radius_ratio: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    csv_path = csv_path.expanduser()
    if not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()
    if not csv_path.exists():
        if logger:
            logger(f"ROI candidates CSV not found: {csv_path}")
        return {}

    lookup: Dict[str, Dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for csv_entry in reader:
            image_width = _safe_float(csv_entry.get("gt_width"))
            image_height = _safe_float(csv_entry.get("gt_height"))
            if image_width is None or image_height is None or image_width <= 0 or image_height <= 0:
                continue
            negatives_json = str(csv_entry.get("negative_detections_json") or "[]")
            try:
                negatives_raw = json.loads(negatives_json)
            except json.JSONDecodeError:
                continue
            if not isinstance(negatives_raw, list):
                continue
            negatives_raw = [det for det in negatives_raw if isinstance(det, dict)]

            removed_json = str(csv_entry.get("removed_positive_json") or "[]")
            try:
                removed_raw = json.loads(removed_json)
            except json.JSONDecodeError:
                removed_raw = []
            if not isinstance(removed_raw, list):
                removed_raw = []
            removed_raw = [det for det in removed_raw if isinstance(det, dict)]

            negatives_norm: List[List[float]] = []
            seen_neg: set = set()
            gt_x = _safe_float(csv_entry.get("gt_x"))
            gt_y = _safe_float(csv_entry.get("gt_y"))
            if (
                recompute_positives
                and gt_x is not None
                and gt_y is not None
            ):
                recomputed_removed, recomputed_negatives = _recompute_removed_and_negatives(
                    all_detections=(removed_raw + negatives_raw),
                    gt_x=gt_x,
                    gt_y=gt_y,
                    gt_w=image_width,
                    gt_h=image_height,
                    gt_radius_ratio=gt_radius_ratio,
                    center_distance_threshold=positive_center_distance_threshold,
                )
                removed_raw = recomputed_removed
                negatives_raw = recomputed_negatives

            negatives_raw = _sanitize_negatives(negatives_raw, removed_raw)
            for det in negatives_raw:
                norm_bbox = _normalize_candidate_bbox(det, image_width=image_width, image_height=image_height)
                if norm_bbox is None:
                    continue
                key = tuple(round(val, 6) for val in norm_bbox)
                if key in seen_neg:
                    continue
                seen_neg.add(key)
                negatives_norm.append(norm_bbox)

            in_out_raw = csv_entry.get("in_out")
            in_out = None
            if in_out_raw is not None and str(in_out_raw).strip() != "":
                try:
                    in_out = int(float(in_out_raw))
                except (TypeError, ValueError):
                    in_out = None

            if not negatives_norm:
                continue
            entry = {
                "negatives": negatives_norm,
                "gt_x": gt_x,
                "gt_y": gt_y,
                "gt_width": image_width,
                "gt_height": image_height,
                "in_out": in_out,
            }
            for key in _csv_entry_lookup_keys(csv_entry):
                lookup[key] = entry
    return lookup


def find_roi_candidate_entry(
    sample_dict: Dict[str, Any],
    roi_candidate_lookup: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    for key in _sample_lookup_keys(sample_dict):
        if key in roi_candidate_lookup:
            return roi_candidate_lookup[key]
    return None


def build_gaze_positive_bbox(
    sample_dict: Dict[str, Any],
    radius_ratio: float,
    roi_entry: Optional[Dict[str, Any]] = None,
) -> Optional[List[float]]:
    gaze_xy, gaze_valid = build_roi_gaze_metadata(sample_dict, roi_entry=roi_entry)
    if not bool(gaze_valid.item()):
        return None
    x = float(gaze_xy[0].item())
    y = float(gaze_xy[1].item())
    radius = max(0.0, min(0.5, float(radius_ratio)))
    return [
        max(0.0, x - radius),
        max(0.0, y - radius),
        min(1.0, x + radius),
        min(1.0, y + radius),
    ]


def build_roi_candidate_metadata(
    sample_dict: Dict[str, Any],
    roi_candidate_lookup: Dict[str, Dict[str, Any]],
    roi_max_positives: int,
    roi_max_negatives: int,
    roi_candidate_slots: int,
    roi_positive_radius_ratio: float,
    roi_entry: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boxes = torch.zeros((roi_candidate_slots, 4), dtype=torch.float32)
    is_positive = torch.zeros((roi_candidate_slots,), dtype=torch.bool)
    valid = torch.zeros((roi_candidate_slots,), dtype=torch.bool)
    if roi_candidate_slots <= 0:
        return boxes, is_positive, valid

    entry = roi_entry if roi_entry is not None else find_roi_candidate_entry(sample_dict, roi_candidate_lookup)

    idx = 0
    if roi_max_positives > 0:
        positive_bbox = build_gaze_positive_bbox(sample_dict, radius_ratio=roi_positive_radius_ratio, roi_entry=entry)
        if positive_bbox is not None and idx < roi_candidate_slots:
            boxes[idx] = torch.tensor(positive_bbox, dtype=torch.float32)
            is_positive[idx] = True
            valid[idx] = True
            idx += 1

    if entry:
        for bbox in entry.get("negatives", [])[: max(0, roi_max_negatives)]:
            if idx >= roi_candidate_slots:
                break
            boxes[idx] = torch.tensor(bbox, dtype=torch.float32)
            valid[idx] = True
            idx += 1
    return boxes, is_positive, valid
