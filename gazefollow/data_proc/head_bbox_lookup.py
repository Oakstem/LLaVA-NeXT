from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class HeadBBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_head_bbox_map(csv_path: Path) -> Dict[str, HeadBBox]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Head annotation CSV not found: {csv_path}")

    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"image_path", "head_bbox_x_min", "head_bbox_y_min", "head_bbox_x_max", "head_bbox_y_max"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Head annotation CSV {csv_path} missing columns: {sorted(missing)}")

        mapping: Dict[str, HeadBBox] = {}
        for row in reader:
            image_path = (row.get("image_path") or "").strip()
            if not image_path:
                continue
            x_min = _parse_float(row.get("head_bbox_x_min"))
            y_min = _parse_float(row.get("head_bbox_y_min"))
            x_max = _parse_float(row.get("head_bbox_x_max"))
            y_max = _parse_float(row.get("head_bbox_y_max"))
            if None in (x_min, y_min, x_max, y_max):
                continue
            # Preserve first occurrence per image_path
            mapping.setdefault(image_path, HeadBBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max))

    return mapping


__all__ = ["HeadBBox", "load_head_bbox_map"]
