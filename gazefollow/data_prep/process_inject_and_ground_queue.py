#!/usr/bin/env python3
"""Process the inject_and_ground_queue.csv file by running the phrase grounding pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from gazefollow.generation_utils import fix_wsl_paths
from gazefollow.auto_phrase_grounding import extract_and_ground_pipeline as pipeline
from gazefollow.data_prep import queue_bootstrap_utils as queue_utils


LOGGER_NAME = "inject_and_ground_queue"
DEFAULT_QUEUE_PATH = Path("gazefollow/data/combined_source_extract_patchscope_valid.csv")
# DEFAULT_QUEUE_PATH = None
DEFAULT_DATA_ROOT = Path(r"D:\Projects\data\gazefollow")
DEFAULT_OUTPUT_ROOT = Path("results/steered_generation")
DEFAULT_THRESHOLD = 0.2
TARGET_DESCRIPTION = False
DEFAULT_ANNOTATION_FILE = 'gazefollow/data/test_annotations_release.csv'
# DEFAULT_ANNOTATION_FILE = 'gazefollow/data/train_annotations_release.csv'


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterate over inject_and_ground_queue entries and store successful grounding queries."
    )
    parser.add_argument(
        "--queue-csv",
        default=DEFAULT_QUEUE_PATH,
        help="Path to gazefollow/data/inject_and_ground_queue.csv (default: %(default)s).",
    )
    parser.add_argument(
        "--annotation-file",
        default=DEFAULT_ANNOTATION_FILE,
        help="Annotation CSV/TXT file used to bootstrap the queue when it is missing or when --rebuild-queue is set. "
    )
    parser.add_argument(
        "--rebuild-queue",
        action="store_true",
        help="Force rebuilding the queue CSV from annotations prior to processing.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional output CSV path. When omitted the input file is updated in-place.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Normalized L2 threshold for accepting a grounding result.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional upper bound on the number of rows to process.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Row index to begin processing from (0-based).",
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Root directory of the gazefollow dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where intermediate pipeline artifacts are stored. "
        "Defaults to results/steered_generation/inject_and_ground_<timestamp>.",
    )
    parser.add_argument(
        "--mask-dir",
        default=str(pipeline.DEFAULT_MASK_DIR),
        help="Directory containing gaze masks for the dataset.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path. Defaults to <output-dir>/inject_and_ground.log.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rows that already have the selected grounding metrics populated.",
    )
    parser.add_argument(
        "--use-body-bbox",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use body bounding boxes when available during person description extraction.",
    )
    parser.add_argument(
        "--use-target-insert-for-source",
        action=argparse.BooleanOptionalAction,
        default=TARGET_DESCRIPTION,
        help="Write outputs to target_* columns instead of source_* columns.",
    )
    return parser.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)
    return logger


def _resolve_output_dir(cli_args: argparse.Namespace) -> Path:
    if cli_args.output_dir:
        return Path(fix_wsl_paths(cli_args.output_dir)).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_dir = DEFAULT_OUTPUT_ROOT / f"inject_and_ground_{timestamp}"
    return Path(fix_wsl_paths(str(default_dir))).resolve()


def _build_pipeline_args(
    *,
    base_output_dir: Path,
    llava_output_dir: Path,
    visualization_dir: Path,
    mask_dir: Path,
    use_body_bbox: bool,
    use_target_insert_for_source: bool,
    gt_gaze_csv_path: Path,
) -> argparse.Namespace:
    saved_argv = sys.argv
    sys.argv = ["extract_and_ground_pipeline"]
    pipeline_args = pipeline.parse_args()
    sys.argv = saved_argv

    default_prompt = pipeline.DEFAULT_TARGET_PROMPT if use_target_insert_for_source else pipeline.DEFAULT_SOURCE_PROMPT

    pipeline_args.mode = "single"
    pipeline_args.mask_dir = str(mask_dir)
    pipeline_args.llava_output_dir = str(llava_output_dir)
    pipeline_args.visualization_dir = str(visualization_dir)
    pipeline_args.output_dir = str(base_output_dir)
    pipeline_args.save_visualization = True
    pipeline_args.save_debug_files = False
    pipeline_args.save_mask_overlays = False
    pipeline_args.attn_show_highest_blob = False
    pipeline_args.attn_create_collage = False
    pipeline_args.attn_save_tensors = False
    pipeline_args.attn_capture_layer_idx = 20
    pipeline_args.attn_inject_layer_idx = 0
    pipeline_args.prompt = default_prompt
    pipeline_args.use_body_bbox = use_body_bbox
    pipeline_args.use_target_insert_for_source = use_target_insert_for_source
    pipeline_args.gt_gaze_csv_path = str(gt_gaze_csv_path)
    return pipeline_args


def _initialize_pipeline_components(
    args: argparse.Namespace,
) -> Tuple[Any, Any, Any, Any, Any, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    pipeline.enable_inference_optimizations()
    adapter_path = pipeline._normalize_path(args.adapter_path)  # noqa: SLF001
    model_base = pipeline._normalize_path(args.model_base)  # noqa: SLF001
    tokenizer, model, image_processor, _ = pipeline.load_model_and_setup(
        model_path=fix_wsl_paths(args.model_path),
        attn_implementation=args.attn_implementation,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        attn_layer_ind=args.attn_layer_index,
        model_base=fix_wsl_paths(str(model_base)) if model_base else None,
        adapter_path=fix_wsl_paths(str(adapter_path)) if adapter_path else None,
    )
    qwen_processor, qwen_model = pipeline.load_qwen3vl_model(
        args.qwen_model_id,
        device_map=args.qwen_device_map,
    )
    generation_config = pipeline.build_generation_config(args)
    attention_config = pipeline.build_attention_config(args)
    guidance_config = pipeline.build_guidance_config(args)
    return tokenizer, model, image_processor, qwen_processor, qwen_model, generation_config, attention_config, guidance_config


def _select_image_path(row: pd.Series) -> Optional[str]:
    candidates = (
        row.get("image_path.1"),
        row.get("image_full_path"),
        row.get("image"),
        row.get("image_path"),
    )
    for candidate in candidates:
        if candidate is None or (isinstance(candidate, float) and pd.isna(candidate)):
            continue
        value = str(candidate).strip()
        if value:
            return value
    image_key = row.get("image_key")
    if image_key is None or (isinstance(image_key, float) and pd.isna(image_key)):
        return None
    try:
        image_index = int(image_key)
    except (TypeError, ValueError):
        return None
    file_name = f"{image_index:08d}.jpg"
    folder_name = f"{image_index // 1000:08d}"
    return f"train/{folder_name}/{file_name}"


def _create_image_task(
    *,
    rel_image_path: str,
    data_root: Path,
    mask_dir: Path,
    mask_template: str,
    fallback_template: Optional[str],
) -> pipeline.ImageTask:
    image_path = Path(fix_wsl_paths(str(data_root / rel_image_path))).expanduser()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")
    # The pipeline now relies on GT gaze CSV entries for masks, so simply format the
    # expected mask path without verifying it exists (it may not be generated).
    mask_candidates: List[Path] = []
    mask_dir = mask_dir.expanduser()
    primary = mask_dir / mask_template.format(stem=image_path.stem)
    mask_candidates.append(primary)
    if fallback_template:
        mask_candidates.append(mask_dir / fallback_template.format(stem=image_path.stem))
    # Prefer a candidate that contains the gaze prefix to maximize GT CSV lookup hits.
    mask_path = next((candidate for candidate in mask_candidates if "gaze__" in candidate.name), mask_candidates[0])
    image_id = image_path.stem
    return pipeline.ImageTask(image_path=image_path, mask_path=mask_path, image_id=image_id)


def _safe_write_csv(df: pd.DataFrame, output_csv: Path, logger: logging.Logger, reason: str) -> bool:
    try:
        df.to_csv(output_csv, index=False)
    except OSError as error:
        logger.warning("Failed to write CSV (%s) at %s due to %s. Will retry later.", reason, output_csv, error)
        return False
    except Exception as error:  # noqa: BLE001
        logger.warning("Unexpected error while writing CSV (%s) at %s: %s", reason, output_csv, error)
        return False
    return True


def _has_valid_metric(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return not pd.isna(value)


def _get_insert_columns(use_target_insert_for_source: bool) -> Dict[str, str]:
    prefix = "target" if use_target_insert_for_source else "source"
    return {
        "error": f"{prefix}_grounding_normalized_l2_error",
        "iou": f"{prefix}_grounding_bbox_iou",
        "iou_over_source": f"{prefix}_grounding_bbox_iou_over_source",
        "bbox": f"{prefix}_grounding_bbox",
        "description": f"patchscope_{prefix}_description",
    }


def _ensure_output_columns(df: pd.DataFrame, insert_columns: Dict[str, str], use_target_insert_for_source: bool) -> None:
    required = [
        insert_columns["error"],
        insert_columns["iou"],
        insert_columns["bbox"],
        insert_columns["description"],
    ]
    if use_target_insert_for_source:
        required.append(insert_columns["iou_over_source"])
    for column in required:
        if column not in df.columns:
            df[column] = pd.NA


def _format_bbox_list(bbox: Any) -> Optional[List[Any]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    normalized: List[Any] = []
    for coord in bbox:
        try:
            value = float(coord)
        except (TypeError, ValueError):
            return None
        normalized.append(int(value) if value.is_integer() else value)
    return normalized


def _source_bbox_from_body_row(row: pd.Series, image_size: Any) -> Optional[List[float]]:
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        return None
    try:
        width = float(image_size[0])
        height = float(image_size[1])
        x = float(row.get("body_bbox_x"))
        y = float(row.get("body_bbox_y"))
        w = float(row.get("body_bbox_width"))
        h = float(row.get("body_bbox_height"))
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0 or w <= 0 or h <= 0:
        return None

    x1 = x * width
    y1 = y * height
    x2 = (x + w) * width
    y2 = (y + h) * height
    return [x1, y1, x2, y2]


def main() -> None:
    cli_args = parse_cli_args()
    queue_csv = Path(fix_wsl_paths(cli_args.queue_csv)).expanduser() if cli_args.queue_csv is not None else None

    output_dir = _resolve_output_dir(cli_args)
    llava_output_dir = output_dir / "llava_runs"
    visualization_dir = output_dir / "grounding_visualizations"
    llava_output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(fix_wsl_paths(cli_args.log_file)).expanduser() if cli_args.log_file else output_dir / "inject_and_ground.log"
    logger = setup_logger(log_file)
    cli_args.output_csv = queue_utils.set_output_path_name(
        queue_csv,
        annotation_file=cli_args.annotation_file,
        logger=logger,
    )

    mask_dir = Path(fix_wsl_paths(cli_args.mask_dir)).expanduser()
    mask_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(fix_wsl_paths(cli_args.data_root)).expanduser()

    df = queue_utils.load_or_initialize_queue(
        queue_csv=queue_csv,
        rebuild_queue=cli_args.rebuild_queue,
        annotation_file=cli_args.annotation_file,
        logger=logger,
        write_callback=lambda frame, path, log: _safe_write_csv(frame, path, log, "init"),
        output_csv=cli_args.output_csv,
    )
    annotation_snapshot_path = output_dir / "queue_snapshot_annotations.csv"
    df.to_csv(annotation_snapshot_path, index=False)
    logger.info("Queue snapshot saved to %s for pipeline annotations.", annotation_snapshot_path)

    pipeline_args = _build_pipeline_args(
        base_output_dir=output_dir,
        llava_output_dir=llava_output_dir,
        visualization_dir=visualization_dir,
        mask_dir=mask_dir,
        use_body_bbox=cli_args.use_body_bbox,
        use_target_insert_for_source=cli_args.use_target_insert_for_source,
        gt_gaze_csv_path=annotation_snapshot_path,
    )

    logger.info("Loading models and processors...")
    tokenizer, model, image_processor, qwen_processor, qwen_model, generation_config, attention_config, guidance_config = (
        _initialize_pipeline_components(pipeline_args)
    )
    logger.info("Models ready. Starting queue processing from %s", queue_csv)

    total_rows = len(df)
    start_index = max(cli_args.start_index, 0)
    if total_rows == 0:
        logger.info("Queue CSV is empty. Nothing to process.")
        return
    if start_index >= total_rows:
        logger.warning("Start index %d is beyond available rows (%d). Nothing to process.", start_index, total_rows)
        return
    if cli_args.limit is not None:
        requested_limit = max(cli_args.limit, 0)
        stop_index = min(total_rows, start_index + requested_limit)
    else:
        stop_index = total_rows
    rows_to_process = stop_index - start_index
    if rows_to_process <= 0:
        logger.warning(
            "No rows remaining to process after applying start index (%d) and limit (%s).",
            start_index,
            cli_args.limit,
        )
        return
    logger.info(
        "Processing %d row(s) from indices %d (inclusive) to %d (exclusive).",
        rows_to_process,
        start_index,
        stop_index,
    )
    success_count = 0
    skipped_count = 0

    output_csv = Path(fix_wsl_paths(cli_args.output_csv)).expanduser() if cli_args.output_csv else queue_csv
    results: List[Dict[str, Any]] = []
    insert_columns = _get_insert_columns(cli_args.use_target_insert_for_source)
    _ensure_output_columns(df, insert_columns, cli_args.use_target_insert_for_source)

    for idx, row in df.iterrows():
        if idx < start_index:
            continue
        if idx >= stop_index:
            break
        progress_prefix = f"[{idx - start_index + 1}/{rows_to_process}]"
        has_iou = _has_valid_metric(row.get(insert_columns["iou"]))
        has_error = _has_valid_metric(row.get(insert_columns["error"]))
        if cli_args.skip_existing and has_iou and has_error:
            skipped_count += 1
            logger.info("%s Row %d already has grounding metrics. Skipping.", progress_prefix, idx)
            continue

        if cli_args.use_target_insert_for_source:
            in_or_out_value = row.get("in_or_out")
            is_zero = False
            if in_or_out_value is not None and not (isinstance(in_or_out_value, float) and pd.isna(in_or_out_value)):
                try:
                    is_zero = float(in_or_out_value) == 0
                except (TypeError, ValueError):
                    is_zero = False
            if is_zero:
                skipped_count += 1
                logger.info(
                    "%s Row %d has in_or_out=0 while using target insertion for source. Skipping.",
                    progress_prefix,
                    idx,
                )
                continue

        rel_image_path = _select_image_path(row)
        if not rel_image_path:
            skipped_count += 1
            logger.warning("%s Missing image path for row %d.", progress_prefix, idx)
            continue
        task = _create_image_task(
            rel_image_path=rel_image_path,
            data_root=data_root,
            mask_dir=mask_dir,
            mask_template=pipeline_args.mask_template,
            fallback_template=pipeline_args.mask_fallback_template,
        )

        logger.info("%s Processing %s", progress_prefix, task.image_id)
        payload, best_error, best_iou, best_detection, used_retry, used_scaled_retry = pipeline.process_image_task_with_body_bbox_retry(
            task,
            args=pipeline_args,
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
            qwen_query_template=pipeline_args.query_template,
            attention_mask_viz_dir=False,
            retry_threshold=cli_args.threshold,
            retry_iou_threshold=0.2,
            retry_use_body_bbox=False,
        )
        if used_retry:
            logger.info(
                "%s Retried %s without body bounding boxes after exceeding threshold %.4f",
                progress_prefix,
                task.image_id,
                cli_args.threshold,
            )
        if used_scaled_retry:
            if cli_args.use_target_insert_for_source:
                final_radius_ratio = (payload.get("artifacts") or {}).get("gt_gaze_mask_radius_ratio")
                logger.info(
                    "%s Applied target gaze-radius retries for %s after repeated threshold failures (final ratio=%s)",
                    progress_prefix,
                    task.image_id,
                    final_radius_ratio,
                )
            else:
                logger.info(
                    "%s Applied scaled person mask retry (scale=0.5) for %s after repeated threshold failures",
                    progress_prefix,
                    task.image_id,
                )
        per_image_json = output_dir / f"{task.image_id}.json"
        pipeline.save_json(payload, per_image_json)
        artifacts = payload.setdefault("artifacts", {})
        artifacts["result_json"] = str(per_image_json)
        artifacts["body_bbox_retry_used"] = used_retry
        artifacts["scaled_person_mask_retry_used"] = used_scaled_retry
        artifacts["scaled_target_radius_retry_used"] = bool(used_scaled_retry and cli_args.use_target_insert_for_source)
        results.append(payload)

        if best_error is None:
            skipped_count += 1
            logger.info("%s %s returned no detections. Skipping update.", progress_prefix, task.image_id)
            continue
        df.at[idx, insert_columns["error"]] = best_error
        df.at[idx, insert_columns["iou"]] = best_iou
        grounding_data = payload.get("grounding") or {}
        query_value = grounding_data.get("query")
        if isinstance(query_value, str):
            query_text = query_value.strip()
        elif query_value is None:
            query_text = ""
        else:
            query_text = str(query_value).strip()
        df.at[idx, insert_columns["description"]] = query_text
        best_bbox = _format_bbox_list(best_detection.get("bbox") if best_detection else None)
        df.at[idx, insert_columns["bbox"]] = best_bbox if best_bbox is not None else ""
        if cli_args.use_target_insert_for_source:
            source_gt_bbox = _source_bbox_from_body_row(
                row,
                (payload.get("grounding", {}) or {}).get("metrics_summary", {}).get("image_size"),
            )
            df.at[idx, insert_columns["iou_over_source"]] = pipeline.compute_bbox_iou(best_bbox, source_gt_bbox)

        if best_error <= cli_args.threshold:
            # Clear any previous error reason if it mentioned 'person'
            if 'person' in str(df.at[idx, 'error_reason']).lower():
                df.at[idx, 'error_reason'] = ''

            success_count += 1
            logger.info(
                "%s ACCEPTED %s (normalized L2=%.4f, IoU=%s). Stored query: %s",
                progress_prefix,
                task.image_id,
                best_error,
                f"{best_iou:.4f}" if best_iou is not None else "None",
                query_text,
            )
            if success_count % 5 == 0:
                _safe_write_csv(df, output_csv, logger, "checkpoint")
                logger.info("Checkpoint save after %d successes to %s", success_count, output_csv)
        else:
            skipped_count += 1
            logger.info(
                "%s REJECTED %s (normalized L2=%.4f >= %.4f, IoU=%s).",
                progress_prefix,
                task.image_id,
                best_error,
                cli_args.threshold,
                f"{best_iou:.4f}" if best_iou is not None else "None",
            )
            if best_detection:
                metrics_summary = payload.get("grounding", {}).get("metrics_summary", {}) or {}
                gt_bbox = metrics_summary.get("person_bbox")
                logger.info(
                    "  Best detection bbox: %s | GT bbox: %s",
                    best_detection.get("bbox"),
                    gt_bbox,
                )

    if _safe_write_csv(df, output_csv, logger, "final"):
        logger.info("Saved updated CSV to %s", output_csv)
    else:
        logger.warning("Failed to save final CSV to %s. Results will be retried on the next run.", output_csv)
    summary_path = output_dir / "queue_processing_summary.json"
    pipeline.save_json(
        {
            "results": results,
            "num_results": len(results),
            "accepted": success_count,
            "skipped": skipped_count,
            "threshold": cli_args.threshold,
            "queue_csv": str(queue_csv),
            "output_csv": str(output_csv),
        },
        summary_path,
    )
    logger.info(
        "Processing complete. Accepted=%d, skipped=%d. Summary saved to %s",
        success_count,
        skipped_count,
        summary_path,
    )


if __name__ == "__main__":
    main()
