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


LOGGER_NAME = "inject_and_ground_queue"
DEFAULT_QUEUE_PATH = Path("gazefollow/data/inject_and_ground_queue.csv")
DEFAULT_DATA_ROOT = Path(r"D:\Projects\data\gazefollow")
DEFAULT_OUTPUT_ROOT = Path("results/steered_generation")
DEFAULT_THRESHOLD = 0.16


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterate over inject_and_ground_queue entries and store successful grounding queries."
    )
    parser.add_argument(
        "--queue-csv",
        default=str(DEFAULT_QUEUE_PATH),
        help="Path to gazefollow/data/inject_and_ground_queue.csv (default: %(default)s).",
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
        help="Skip rows that already have both source_description fields populated.",
    )
    parser.add_argument(
        "--use-body-bbox",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use body bounding boxes when available during person description extraction.",
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
) -> argparse.Namespace:
    saved_argv = sys.argv
    sys.argv = ["extract_and_ground_pipeline"]
    pipeline_args = pipeline.parse_args()
    sys.argv = saved_argv

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
    pipeline_args.attn_capture_layer_idx = None
    pipeline_args.attn_inject_layer_idx = None
    pipeline_args.prompt = pipeline.DEFAULT_PROMPT
    pipeline_args.use_body_bbox = use_body_bbox
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
    mask_path = pipeline._resolve_mask_path(  # noqa: SLF001
        image_path=image_path,
        explicit_mask=None,
        mask_dir=mask_dir,
        mask_template=mask_template,
        fallback_template=fallback_template,
    )
    image_id = image_path.stem
    return pipeline.ImageTask(image_path=image_path, mask_path=mask_path, image_id=image_id)


def _extract_best_error(payload: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[Dict[str, Any]]]:
    detections: List[Dict[str, Any]] = payload.get("grounding", {}).get("detections", []) or []
    best_error: Optional[float] = None
    best_iou: Optional[float] = None
    best_detection: Optional[Dict[str, Any]] = None
    for detection in detections:
        metrics = detection.get("metrics") or {}
        error = metrics.get("gaze_normalized_l2_error")
        if error is None:
            continue
        if best_error is None or error < best_error:
            best_error = error
            best_iou = metrics.get("bbox_iou_vs_person")
            best_detection = detection
    return best_error, best_iou, best_detection


def main() -> None:
    cli_args = parse_cli_args()
    queue_csv = Path(fix_wsl_paths(cli_args.queue_csv)).expanduser()
    if not queue_csv.exists():
        raise FileNotFoundError(f"Queue CSV not found: {queue_csv}")

    output_dir = _resolve_output_dir(cli_args)
    llava_output_dir = output_dir / "llava_runs"
    visualization_dir = output_dir / "grounding_visualizations"
    llava_output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(fix_wsl_paths(cli_args.log_file)).expanduser() if cli_args.log_file else output_dir / "inject_and_ground.log"
    logger = setup_logger(log_file)

    mask_dir = Path(fix_wsl_paths(cli_args.mask_dir)).expanduser()
    mask_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(fix_wsl_paths(cli_args.data_root)).expanduser()

    pipeline_args = _build_pipeline_args(
        base_output_dir=output_dir,
        llava_output_dir=llava_output_dir,
        visualization_dir=visualization_dir,
        mask_dir=mask_dir,
        use_body_bbox=cli_args.use_body_bbox,
    )

    logger.info("Loading models and processors...")
    tokenizer, model, image_processor, qwen_processor, qwen_model, generation_config, attention_config, guidance_config = (
        _initialize_pipeline_components(pipeline_args)
    )
    logger.info("Models ready. Starting queue processing from %s", queue_csv)

    df = pd.read_csv(queue_csv)
    total_rows = len(df)
    limit = min(cli_args.limit, total_rows) if cli_args.limit else total_rows
    success_count = 0
    skipped_count = 0

    output_csv = Path(fix_wsl_paths(cli_args.output_csv)).expanduser() if cli_args.output_csv else queue_csv
    results: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        if idx >= limit:
            break
        source_desc = row.get("source_description")
        steered_desc = row.get("steered_source_description")
        if cli_args.skip_existing and isinstance(source_desc, str) and source_desc.strip() and isinstance(steered_desc, str) and steered_desc.strip():
            skipped_count += 1
            logger.info("[%d/%d] Row %d already populated. Skipping.", idx + 1, limit, idx)
            continue

        rel_image_path = _select_image_path(row)
        if not rel_image_path:
            skipped_count += 1
            logger.warning("[%d/%d] Missing image path for row %d.", idx + 1, limit, idx)
            continue
        task = _create_image_task(
            rel_image_path=rel_image_path,
            data_root=data_root,
            mask_dir=mask_dir,
            mask_template=pipeline_args.mask_template,
            fallback_template=pipeline_args.mask_fallback_template,
        )

        logger.info("[%d/%d] Processing %s", idx + 1, limit, task.image_id)
        payload = pipeline.process_image_task(
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
        )
        per_image_json = output_dir / f"{task.image_id}.json"
        pipeline.save_json(payload, per_image_json)
        payload.setdefault("artifacts", {})["result_json"] = str(per_image_json)
        results.append(payload)

        best_error, best_iou, best_detection = _extract_best_error(payload)
        if best_error is None:
            skipped_count += 1
            logger.info(
                "[%d/%d] %s returned no detections. Skipping update.",
                idx + 1,
                limit,
                task.image_id,
            )
            continue
        df.at[idx, "source_grounding_normalized_l2_error"] = best_error
        df.at[idx, "source_grounding_bbox_iou"] = best_iou

        if best_error <= cli_args.threshold:
            desc_result = payload['grounding']['query']
            query_text = desc_result.strip()
            df.at[idx, "source_description"] = query_text
            df.at[idx, "steered_source_description"] = query_text
            # Clear any previous error reason if it mentioned 'person'
            if 'person' in str(df.at[idx, 'error_reason']).lower():
                df.at[idx, 'error_reason'] = ''

            success_count += 1
            logger.info(
                "[%d/%d] ACCEPTED %s (normalized L2=%.4f, IoU=%s). Stored query: %s",
                idx + 1,
                limit,
                task.image_id,
                best_error,
                f"{best_iou:.4f}" if best_iou is not None else "None",
                query_text,
            )
            if success_count % 5 == 0:
                df.to_csv(output_csv, index=False)
                logger.info("Checkpoint save after %d successes to %s", success_count, output_csv)
        else:
            skipped_count += 1
            logger.info(
                "[%d/%d] REJECTED %s (normalized L2=%.4f >= %.4f, IoU=%s).",
                idx + 1,
                limit,
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

    df.to_csv(output_csv, index=False)
    logger.info("Saved updated CSV to %s", output_csv)
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
