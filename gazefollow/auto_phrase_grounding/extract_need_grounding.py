#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gazefollow.gaze_metrics import compute_gaze_iou, compute_modified_l2_error

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def load_samples(path: Path) -> Sequence[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of samples in {path}, found {type(data).__name__}")
    return data




def _extract_image_dimensions(
    sample: dict[str, Any],
    images_dir: Optional[Path] = None,
    dimensions_cache: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Optional[Tuple[int, int]]:
    """Extract image width and height from sample metadata or by loading the image."""
    # Try to get from gaze ground truth metadata
    gaze_detections = sample.get("gaze_detections", {})
    if isinstance(gaze_detections, dict):
        for person_data in gaze_detections.values():
            gt_data = person_data.get("gaze_ground_truth")
            if isinstance(gt_data, dict):
                # Some samples may have stored dimensions
                width = gt_data.get("width") or sample.get("gaze_gt_width")
                height = gt_data.get("height") or sample.get("gaze_gt_height")
                if width and height:
                    return int(width), int(height)
    
    # Fallback to sample-level metadata
    width = sample.get("gaze_gt_width") or sample.get("width")
    height = sample.get("gaze_gt_height") or sample.get("height")
    if width and height:
        return int(width), int(height)
    
    # Try to load from image if images_dir is provided
    if images_dir and PIL_AVAILABLE:
        image_path_str = sample.get("image_path") or sample.get("image")
        if not image_path_str:
            return None
        
        # Check cache first
        if dimensions_cache is not None and image_path_str in dimensions_cache:
            return dimensions_cache[image_path_str]
        
        # Try to resolve and load the image
        image_path = images_dir / image_path_str
        if not image_path.exists():
            # Try with different extensions
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = images_dir / (image_path_str + ext)
                if candidate.exists():
                    image_path = candidate
                    break
        
        if image_path.exists():
            try:
                with Image.open(image_path) as img:
                    dims = (img.width, img.height)
                    if dimensions_cache is not None:
                        dimensions_cache[image_path_str] = dims
                    return dims
            except Exception:
                pass
    
    return None


def _compute_missing_iou(
    person_data: dict[str, Any],
    image_dims: Optional[Tuple[int, int]],
    iou_radius_ratio: float = 0.05,
) -> Optional[float]:
    """Compute IoU if it's missing from the person data."""
    if image_dims is None:
        return None
    
    gaze_coords = person_data.get("gaze_coordinates")
    gt_data = person_data.get("gaze_ground_truth")
    
    if not gaze_coords or not isinstance(gt_data, dict):
        return None
    
    gt_x = gt_data.get("x")
    gt_y = gt_data.get("y")
    
    if gt_x is None or gt_y is None:
        return None
    
    width, height = image_dims
    return compute_gaze_iou(
        predicted_box=gaze_coords,
        ground_truth_point=(float(gt_x), float(gt_y)),
        image_width=width,
        image_height=height,
        radius_ratio=iou_radius_ratio,
    )


def _compute_missing_modified_l2(
    person_data: dict[str, Any],
    image_dims: Optional[Tuple[int, int]],
) -> Optional[float]:
    """Compute modified L2 error if it's missing from the person data."""
    if image_dims is None:
        return None
    
    gaze_coords = person_data.get("gaze_coordinates")
    gt_data = person_data.get("gaze_ground_truth")
    
    if not gaze_coords or not isinstance(gt_data, dict):
        return None
    
    gt_x = gt_data.get("x")
    gt_y = gt_data.get("y")
    
    if gt_x is None or gt_y is None:
        return None
    
    width, height = image_dims
    return compute_modified_l2_error(
        predicted_box=gaze_coords,
        ground_truth_point=(float(gt_x), float(gt_y)),
        image_width=width,
        image_height=height,
    )


def _is_gaze_outside_image(gaze_target: Optional[str]) -> bool:
    """Check if the gaze target description indicates it's outside the image."""
    if not gaze_target:
        return False
    
    gaze_target_lower = gaze_target.lower()
    
    # Check for common phrases indicating the target is outside the image
    outside_phrases = [
        "outside the image",
        "outside image",
        "outside of the image",
        "outside of image",
        "out of the image",
        "out of image",
        "beyond the image",
        "beyond image",
        "off screen",
        "off-screen",
        "not in the image",
        "not in image",
        "not visible",
    ]
    
    return any(phrase in gaze_target_lower for phrase in outside_phrases)


def filter_samples(
    samples: Iterable[dict[str, Any]],
    threshold: float,
    metric: str = "l2",
    iou_radius_ratio: float = 0.05,
    images_dir: Optional[Path] = None,
) -> List[dict[str, Any]]:
    """
    Filter samples based on a specified metric threshold.
    
    Args:
        samples: Iterable of sample dictionaries
        threshold: Threshold value for filtering
        metric: Metric to use for filtering ('l2', 'iou', 'modified_l2')
            - 'l2': Filter samples where gaze_normalized_l2_error is null or > threshold
            - 'iou': Filter samples where gaze_iou is null or < threshold
            - 'modified_l2': Filter samples where gaze_modified_l2_error is null or > threshold
        iou_radius_ratio: Ratio of image diagonal for IoU computation (default: 0.05)
        images_dir: Optional directory containing images (for loading dimensions)
    
    Returns:
        List of filtered samples (excluding those with gaze target outside image)
    """
    filtered: List[dict[str, Any]] = []
    computed_iou_count = 0
    computed_modified_l2_count = 0
    missing_dims_count = 0
    skipped_outside_image_count = 0
    dimensions_cache: Dict[str, Tuple[int, int]] = {}
    
    for sample in samples:
        gaze_detections = sample.get("gaze_detections")
        if not isinstance(gaze_detections, dict):
            continue
        
        # Extract image dimensions once per sample if needed for iou or modified_l2
        image_dims = None
        if metric in ("iou", "modified_l2"):
            image_dims = _extract_image_dimensions(sample, images_dir, dimensions_cache)
            if image_dims is None:
                missing_dims_count += 1
            
        for key, value in gaze_detections.items():
            # Check if gaze target is outside the image - skip if so
            gaze_target = value.get("gaze_target")
            if _is_gaze_outside_image(gaze_target):
                skipped_outside_image_count += 1
                break  # Skip this entire sample
            
            if metric == "iou":
                metric_value = value.get("gaze_iou")
                
                # Compute IoU if missing
                if metric_value is None and image_dims is not None:
                    metric_value = _compute_missing_iou(value, image_dims, iou_radius_ratio)
                    if metric_value is not None:
                        # Store computed value back in the data
                        value["gaze_iou"] = metric_value
                        computed_iou_count += 1
                
                # For IoU, we want samples where IoU is null or below threshold (low overlap = needs grounding)
                if metric_value is None or (isinstance(metric_value, (int, float)) and metric_value < threshold):
                    filtered.append(sample)
                    break
            
            elif metric == "modified_l2":
                metric_value = value.get("gaze_modified_l2_error")
                
                # Compute modified L2 if missing
                if metric_value is None and image_dims is not None:
                    metric_value = _compute_missing_modified_l2(value, image_dims)
                    if metric_value is not None:
                        # Store computed value back in the data
                        value["gaze_modified_l2_error"] = metric_value
                        computed_modified_l2_count += 1
                
                # For modified L2, we want samples where error is null or above threshold (high error = needs grounding)
                if metric_value is None or (isinstance(metric_value, (int, float)) and metric_value > threshold):
                    filtered.append(sample)
                    break
            
            else:  # default to l2
                metric_value = value.get("gaze_normalized_l2_error")
                # For L2, we want samples where error is null or above threshold (high error = needs grounding)
                if metric_value is None or (isinstance(metric_value, (int, float)) and metric_value > threshold):
                    filtered.append(sample)
                    break
    
    if metric == "iou":
        if computed_iou_count > 0:
            print(f"Computed IoU for {computed_iou_count} samples that were missing this metric.")
        if missing_dims_count > 0:
            print(f"Warning: {missing_dims_count} samples missing image dimensions (could not compute IoU).")
            if images_dir is None and PIL_AVAILABLE:
                print("  Tip: Use --images-dir to load dimensions from image files.")
            elif not PIL_AVAILABLE:
                print("  Tip: Install PIL/Pillow to load dimensions from image files.")
    
    elif metric == "modified_l2":
        if computed_modified_l2_count > 0:
            print(f"Computed modified L2 for {computed_modified_l2_count} samples that were missing this metric.")
        if missing_dims_count > 0:
            print(f"Warning: {missing_dims_count} samples missing image dimensions (could not compute modified L2).")
            if images_dir is None and PIL_AVAILABLE:
                print("  Tip: Use --images-dir to load dimensions from image files.")
            elif not PIL_AVAILABLE:
                print("  Tip: Install PIL/Pillow to load dimensions from image files.")
    
    if skipped_outside_image_count > 0:
        print(f"Skipped {skipped_outside_image_count} samples with gaze target outside the image.")
                    
    return filtered


def write_output(samples: Sequence[dict[str, Any]], source_path: Path) -> Path:
    output_path = source_path.parent / "need_grounding.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(samples, fh, indent=2)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract samples whose gaze metric is null or outside acceptable threshold. "
            "Writes results to 'need_grounding.json' in the source file's parent directory."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to the dataset evaluation JSON file (e.g. dataset_evaluation_results/.../person_level_results.json)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default="/mnt/d/Projects/data/gazefollow",
        help="Optional directory containing images (used to extract dimensions for IoU computation when not in metadata)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Threshold for the selected metric (default: 0.1)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["l2", "iou", "modified_l2"],
        default="modified_l2",
        help=(
            "Metric to use for filtering (default: l2). "
            "'l2': normalized L2 error (samples with error > threshold). "
            "'iou': IoU metric (samples with IoU < threshold). "
            "'modified_l2': modified L2 error (samples with error > threshold)."
        ),
    )
    parser.add_argument(
        "--iou-radius-ratio",
        type=float,
        default=0.1,
        help="Ratio of image diagonal to use as ground-truth circle radius for IoU computation (default: 0.05).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.source)
    filtered = filter_samples(
        samples, 
        args.threshold, 
        args.metric, 
        args.iou_radius_ratio,
        args.images_dir,
    )
    output_path = write_output(filtered, args.source)
    print(f"Wrote {len(filtered)} samples to {output_path} (metric: {args.metric}, threshold: {args.threshold})")
    if args.metric == "iou":
        print(f"IoU radius ratio: {args.iou_radius_ratio}")


if __name__ == "__main__":
    main()
