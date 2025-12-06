#!/usr/bin/env python3
"""Helpers for initializing the inject-and-ground queue from annotation dumps."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from gazefollow.data_proc import convert_gazefollow_annotations as annotations_utils
from gazefollow.generation_utils import fix_wsl_paths

DEFAULT_ANNOTATION_SOURCES: Tuple[Tuple[str, str], ...] = (
    ("gazefollow/data/train_annotations_release.csv", "gazefollow/data/train_annotations_release.txt"),
    ("gazefollow/data/test_annotations_release.csv", "gazefollow/data/test_annotations_release.txt"),
)
QUEUE_COLUMN_ORDER: Tuple[str, ...] = (
    "image_path.1",
    "image_path",
    "eye_x",
    "eye_y",
    "gaze_x",
    "gaze_y",
    "body_bbox_x",
    "body_bbox_y",
    "body_bbox_width",
    "body_bbox_height",
    "in_or_out",
    "image_key",
    "index",
    "llava_matched_person_id",
    "llava_person_bbox",
    "llava_gaze_points",
    "gaze_error",
    "angular_gaze_error",
    "llava_person_bb_iou",
    "num_people",
    "error_reason",
    "source_description",
    "target_description",
    "full_description",
    "steered_source_description",
    "steered_target_description",
    "source_grounding_normalized_l2_error",
    "source_grounding_bbox_iou",
)
TEXT_DEFAULT_COLUMNS: Tuple[str, ...] = (
    "llava_person_bbox",
    "llava_gaze_points",
    "error_reason",
    "source_description",
    "target_description",
    "full_description",
    "steered_source_description",
    "steered_target_description",
)
NUMERIC_DEFAULT_COLUMNS: Tuple[str, ...] = (
    "llava_matched_person_id",
    "gaze_error",
    "angular_gaze_error",
    "llava_person_bb_iou",
    "num_people",
    "source_grounding_normalized_l2_error",
    "source_grounding_bbox_iou",
)
FLOAT_COLUMNS: Tuple[str, ...] = (
    "eye_x",
    "eye_y",
    "gaze_x",
    "gaze_y",
    "body_bbox_x",
    "body_bbox_y",
    "body_bbox_width",
    "body_bbox_height",
    "head_bbox_x_min",
    "head_bbox_y_min",
    "head_bbox_x_max",
    "head_bbox_y_max",
)
HEAD_BBOX_COLUMNS: Tuple[str, ...] = (
    "head_bbox_x_min",
    "head_bbox_y_min",
    "head_bbox_x_max",
    "head_bbox_y_max",
)


def _discover_default_annotation_paths() -> List[Path]:
    discovered: List[Path] = []
    for csv_candidate, txt_candidate in DEFAULT_ANNOTATION_SOURCES:
        for raw_path in (csv_candidate, txt_candidate):
            resolved = Path(fix_wsl_paths(raw_path)).expanduser()
            if resolved.exists():
                discovered.append(resolved)
                break
    return discovered


def _count_annotation_columns(txt_path: Path) -> int:
    with txt_path.open("r", newline="") as src:
        reader = csv.reader(src, delimiter=",")
        for row in reader:
            cleaned = [value.strip() for value in row]
            if any(cleaned):
                return len(cleaned)
    raise ValueError(f"Annotation file {txt_path} is empty.")


def _read_annotation_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".txt":
        column_count = _count_annotation_columns(path)
        headers = annotations_utils.determine_headers(column_count)
        return pd.read_csv(path, names=headers, header=None, skip_blank_lines=True)
    raise ValueError(f"Unsupported annotation file extension: {path.suffix}")


def _extract_image_key_from_path(rel_path: str) -> Optional[int]:
    if not rel_path:
        return None
    stem = Path(str(rel_path)).stem
    if not stem:
        return None
    if stem.isdigit():
        return int(stem)
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return int(digits)
    return None


def _aggregate_gaze_by_head_bbox(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["image_key", *HEAD_BBOX_COLUMNS]
    if any(column not in frame.columns for column in required_columns):
        return frame

    head_bbox_values = frame.loc[:, HEAD_BBOX_COLUMNS]
    valid_mask = head_bbox_values.notna().all(axis=1)
    if not valid_mask.any():
        return frame

    group_columns = ["image_key", *HEAD_BBOX_COLUMNS]
    aggregation_map = {}
    for column in frame.columns:
        if column in group_columns:
            continue
        if column in {"gaze_x", "gaze_y"}:
            aggregation_map[column] = "mean"
        else:
            aggregation_map[column] = "first"

    aggregated = (
        frame.loc[valid_mask]
        .groupby(group_columns, as_index=False)
        .agg(aggregation_map)
    )
    if valid_mask.all():
        return aggregated
    return pd.concat([frame.loc[~valid_mask], aggregated], ignore_index=True)


def build_queue_from_annotation(annotation_file: Path, logger: logging.Logger) -> pd.DataFrame:
    try:
        annotations = _read_annotation_file(annotation_file)
    except Exception as error:  # noqa: BLE001
        logger.warning("Failed to parse annotation file %s: %s", annotation_file, error)
        raise
    logger.info("Loaded %d annotation row(s) from %s", len(annotations), annotation_file)
    if annotations.empty:
        raise FileNotFoundError("No valid annotation files were provided to initialize the queue.")

    if "image_path" not in annotations.columns:
        raise ValueError("Annotation frames are missing the required 'image_path' column.")

    annotations.rename(columns={"image_path": "image_path.1"}, inplace=True)
    annotations["image_path.1"] = annotations["image_path.1"].astype(str).str.strip()
    annotations = annotations[annotations["image_path.1"].astype(bool)]
    if annotations.empty:
        raise ValueError("Annotation files did not contain any valid image paths.")

    for column in FLOAT_COLUMNS:
        if column in annotations.columns:
            annotations[column] = pd.to_numeric(annotations[column], errors="coerce")
        else:
            annotations[column] = pd.NA

    annotations["image_key"] = annotations["image_path.1"].map(_extract_image_key_from_path, na_action="ignore")
    existing_keys = annotations["image_key"].dropna()
    next_key = int(existing_keys.max()) + 1 if not existing_keys.empty else 1
    missing_mask = annotations["image_key"].isna()
    if missing_mask.any():
        filler = range(next_key, next_key + missing_mask.sum())
        annotations.loc[missing_mask, "image_key"] = list(filler)
    annotations["image_key"] = annotations["image_key"].astype("Int64")
    annotations["image_path"] = annotations["image_key"]

    for column in TEXT_DEFAULT_COLUMNS:
        annotations[column] = ""
    for column in NUMERIC_DEFAULT_COLUMNS:
        annotations[column] = pd.NA

    queue_df = _aggregate_gaze_by_head_bbox(annotations)
    queue_df["index"] = range(len(queue_df))
    queue_df["index"] = queue_df["index"].astype("Int64")
    queue_df = queue_df.reindex(columns=QUEUE_COLUMN_ORDER, copy=False)
    return queue_df


def set_output_path_name(
    queue_csv: Optional[Path],
    *,
    annotation_file: Optional[str],
    logger: logging.Logger,
) -> Optional[Path]:
    if queue_csv is not None:
        return queue_csv
    first_name = Path(annotation_file).stem
    prefix = first_name.split("_", 1)[0].strip().lower()
    if not prefix:
        return queue_csv
    output_filename = f"gazefollow/data/{prefix}_inject_and_ground_queue.csv"
    output_filename = Path(fix_wsl_paths(output_filename)).expanduser().resolve()
    logger.info(
        "Queue CSV %s was missing; redirecting bootstrap output to %s based on annotation file %s.",
        queue_csv,
        output_filename,
        annotation_file,
    )
    return output_filename


def load_or_initialize_queue(
    *,
    queue_csv: Optional[Path],
    rebuild_queue: bool,
    annotation_file: Optional[str],
    logger: logging.Logger,
    write_callback: Callable[[pd.DataFrame, Path, logging.Logger], bool],
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    needs_bootstrap = rebuild_queue or queue_csv is None or not queue_csv.exists()
    if not needs_bootstrap:
        logger.info("Loading queue entries from %s", queue_csv)
        return pd.read_csv(queue_csv)

    annotation_path = Path(fix_wsl_paths(annotation_file)).expanduser().resolve() if annotation_file else None
    if not annotation_path or not annotation_path.exists():
        raise FileNotFoundError(
            "Queue CSV is missing and no annotation files were provided. "
            "Supply --annotation-file or place the default train/test annotations in gazefollow/data/."
        )
    logger.info(
        "Initializing queue CSV at %s using annotation file %s.",
        queue_csv,
        annotation_path,
    )
    queue_df = build_queue_from_annotation(annotation_path, logger)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not write_callback(queue_df, output_csv, logger):
        raise RuntimeError(f"Failed to write initialized queue CSV to {output_csv}")
    return queue_df


__all__ = [
    "build_queue_from_annotation",
    "load_or_initialize_queue",
    "set_output_path_name",
]
