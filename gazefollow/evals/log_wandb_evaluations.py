#!/usr/bin/env python
"""Utility to backfill evaluation metrics into Weights & Biases."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, MutableMapping, Sequence, Tuple

# from gazefollow.evals.wandb_dedup import WandbImportError, prune_wandb_run_duplicates


DEFAULT_PROJECT = "llava-model-eval"
MIN_TOTAL_SAMPLES = 1000
MIN_SUCCESS_RATE = 0.5
MODEL_GENERATION_RESULTS_FILENAME = "model_generation_results.json"
GENERATION_TABLE_COLUMNS = [
    "id",
    "person_description",
    "ground_truth",
    "model_prediction",
    "gaze_predicted_target",
    "gaze_ground_truth",
    "predicted_gaze",
    "gaze_normalized_l2_error",
    "loss",
    "image_width",
    "image_height",
    "in_out",
]


@dataclass
class EvaluationRun:
    """Represents a single evaluation directory with metrics ready for logging."""

    run_name: str
    adapter_path: str
    metrics: Dict[str, object]
    total_samples: int
    eval_dir: Path
    metrics_path: Path
    config_path: Path
    mtime: float
    generation_samples: List[Dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log evaluation metrics from evaluation_results/ into W&B."
    )
    parser.add_argument(
        "--base-dir",
        default="evaluation_results",
        type=Path,
        help="Root directory to search for evaluation result folders.",
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="Optional Weights & Biases entity/organization name.",
    )
    parser.add_argument(
        "--min-total-samples",
        default=MIN_TOTAL_SAMPLES,
        type=int,
        help="Minimum number of samples required to log a run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not upload to W&B; instead write the planned logs to a CSV file.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("wandb_eval_dry_run.csv"),
        help="CSV path used when --dry-run is supplied.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase logging verbosity.",
    )
    parser.add_argument(
        "--prune-wandb-duplicates",
        action="store_true",
        help="Delete older W&B runs that share the same name before uploading new logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    # if args.prune_wandb_duplicates:
    #     try:
    #         prune_wandb_run_duplicates(
    #             project=args.project,
    #             entity=args.entity,
    #             dry_run=args.dry_run,
    #         )
    #     except WandbImportError as exc:
    #         logging.error("%s", exc)
    #         return
    #     except RuntimeError as exc:
    #         logging.error("Failed to prune W&B duplicates: %s", exc)
    #         return

    runs = collect_evaluation_runs(
        base_dir=args.base_dir,
        min_total_samples=args.min_total_samples,
    )
    if not runs:
        logging.info("No evaluation runs matched the provided filters.")
        return

    deduped_runs = deduplicate_runs(runs)
    logging.info(
        "Prepared %d runs for logging after filtering %d initial candidates.",
        len(deduped_runs),
        len(runs),
    )

    if args.dry_run:
        write_dry_run_csv(deduped_runs, args.csv_path)
        logging.info(
            "Dry run complete. Review CSV output at %s before removing --dry-run.",
            args.csv_path,
        )
        return

    log_runs_to_wandb(
        runs=deduped_runs,
        project=args.project,
        entity=args.entity,
    )


def collect_evaluation_runs(base_dir: Path, min_total_samples: int) -> List[EvaluationRun]:
    """Find evaluation directories that contain both metrics and config files."""
    if not base_dir.exists():
        logging.warning("Base directory %s does not exist. Nothing to log.", base_dir)
        return []

    runs: List[EvaluationRun] = []
    for metrics_path in base_dir.rglob("metrics.json"):
        eval_dir = metrics_path.parent
        run = prepare_evaluation_run(eval_dir, min_total_samples=min_total_samples)
        if run is not None:
            runs.append(run)
            logging.debug(
                "Queued %s for logging (total_samples=%d).",
                eval_dir,
                run.total_samples,
            )
    return runs


def prepare_evaluation_run(
    eval_dir: Path,
    *,
    min_total_samples: int = MIN_TOTAL_SAMPLES,
) -> EvaluationRun | None:
    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.is_file():
        logging.debug("Skipping %s because metrics.json is missing.", eval_dir)
        return None

    config_path = eval_dir / "evaluation_config.json"
    if not config_path.is_file():
        logging.debug(
            "Skipping %s because evaluation_config.json is missing.",
            eval_dir,
        )
        return None

    raw_metrics = load_json(metrics_path)
    if raw_metrics is None:
        return None
    metrics = dict(raw_metrics)

    total_samples = coerce_int(metrics.get("total_samples"))
    if total_samples is None:
        logging.warning(
            "Skipping %s: metrics.json lacks a valid total_samples field.",
            eval_dir,
        )
        return None
    if total_samples < min_total_samples:
        logging.debug(
            "Skipping %s: total_samples %d is below threshold %d.",
            eval_dir,
            total_samples,
            min_total_samples,
        )
        return None

    gaze_mean = coerce_float(metrics.get("gaze_l2_normalized_mean"))
    if gaze_mean is None:
        logging.debug(
            "Skipping %s: gaze_l2_normalized_mean is missing or invalid.",
            eval_dir,
        )
        return None

    gaze_count = coerce_int(metrics.get("gaze_l2_normalized_count"))
    if gaze_count is None:
        logging.debug(
            "Skipping %s: gaze_l2_normalized_count is missing or invalid.",
            eval_dir,
        )
        return None

    success_rate = gaze_count / total_samples if total_samples else 0.0
    if success_rate < MIN_SUCCESS_RATE:
        logging.debug(
            (
                "Skipping %s: gaze success rate %.3f below threshold %.2f "
                "(gaze_l2_normalized_count=%s, total_samples=%s)."
            ),
            eval_dir,
            success_rate,
            MIN_SUCCESS_RATE,
            gaze_count,
            total_samples,
        )
        return None

    metrics["gaze_l2_success_rate"] = success_rate

    config = load_json(config_path)
    if config is None:
        return None
    adapter_path = config.get("adapter_path")
    if not adapter_path:
        adapter_path = "baseline"
        # logging.warning(
        #     "Skipping %s: evaluation_config.json does not define adapter_path.",
        #     eval_dir,
        # )
        # return None

    run_name = build_run_name_from_adapter(adapter_path)
    mtime = max(metrics_path.stat().st_mtime, config_path.stat().st_mtime)
    generation_samples = load_generation_samples(eval_dir)
    return EvaluationRun(
        run_name=run_name,
        adapter_path=str(adapter_path),
        metrics=metrics,
        total_samples=total_samples,
        eval_dir=eval_dir,
        metrics_path=metrics_path,
        config_path=config_path,
        mtime=mtime,
        generation_samples=generation_samples,
    )


def load_generation_samples(eval_dir: Path) -> List[Dict[str, object]]:
    results_path = eval_dir / MODEL_GENERATION_RESULTS_FILENAME
    if not results_path.is_file():
        logging.debug(
            "No %s found for %s; skipping generation table logging.",
            MODEL_GENERATION_RESULTS_FILENAME,
            eval_dir,
        )
        return []

    raw_samples = load_json(results_path)
    if not isinstance(raw_samples, list):
        logging.warning(
            "Expected %s to contain a list, but found %s. Skipping.",
            results_path,
            type(raw_samples).__name__,
        )
        return []

    formatted: List[Dict[str, object]] = []
    for sample in raw_samples:
        if not isinstance(sample, MutableMapping):
            logging.debug(
                "Skipping malformed sample in %s: expected mapping, found %s.",
                results_path,
                type(sample).__name__,
            )
            continue
        formatted.extend(format_generation_sample(sample))
    logging.debug(
        "Prepared %d generation samples from %s.",
        len(formatted),
        results_path,
    )
    return formatted


def format_generation_sample(sample: MutableMapping[str, object]) -> List[Dict[str, object]]:
    base_ground_truth = sample.get("gaze_ground_truth")
    gaze_ground_truth: Dict[str, object] | None = None
    image_width: object | None = None
    image_height: object | None = None
    if isinstance(base_ground_truth, MutableMapping):
        gaze_ground_truth = {
            "x": base_ground_truth.get("x"),
            "y": base_ground_truth.get("y"),
        }
        image_width = base_ground_truth.get("image_width")
        image_height = base_ground_truth.get("image_height")

    gaze_detections = sample.get("gaze_detections")
    detection_entries: List[MutableMapping[str, object] | None] = []
    if isinstance(gaze_detections, MutableMapping):
        detection_entries = [
            detection
            for detection in gaze_detections.values()
            if isinstance(detection, MutableMapping)
        ]
    if not detection_entries:
        detection_entries = [None]

    rows: List[Dict[str, object]] = []
    for detection in detection_entries:
        row = {
            "id": sample.get("id"),
            "person_description": detection.get("person_description") if detection else None,
            "ground_truth": sample.get("ground_truth"),
            "model_prediction": sample.get("model_prediction"),
            "gaze_predicted_target": detection.get("gaze_target") if detection else None,
            "gaze_ground_truth": gaze_ground_truth,
            "predicted_gaze": extract_bbox_center(detection) if detection else None,
            "gaze_normalized_l2_error": detection.get("gaze_normalized_l2_error") if detection else None,
            "loss": sample.get("loss"),
            "image_width": image_width,
            "image_height": image_height,
            "in_out": sample.get("in_out"),
        }
        rows.append(row)
    return rows


def extract_bbox_center(detection: MutableMapping[str, object]) -> object:
    best_detection = detection.get("best_detection")
    if isinstance(best_detection, MutableMapping):
        center = best_detection.get("bbox_center")
        if isinstance(center, Sequence) and not isinstance(center, (str, bytes)):
            return list(center)

    detections = detection.get("detections")
    if isinstance(detections, Sequence) and detections:
        first_detection = detections[0]
        if isinstance(first_detection, MutableMapping):
            center = first_detection.get("bbox_center")
            if isinstance(center, Sequence) and not isinstance(center, (str, bytes)):
                return list(center)

    center = detection.get("bbox_center")
    if isinstance(center, Sequence) and not isinstance(center, (str, bytes)):
        return list(center)
    return None


def load_json(path: Path) -> MutableMapping[str, object] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        logging.error("Expected JSON file %s is missing.", path)
    except json.JSONDecodeError as exc:
        logging.error("Failed to parse JSON file %s: %s", path, exc)
    except OSError as exc:
        logging.error("Unable to read JSON file %s: %s", path, exc)
    return None


def coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def build_run_name_from_adapter(adapter_path: str) -> str:
    """Generate a human-readable run name from the adapter path."""
    path = PurePosixPath(adapter_path)
    parts = [part for part in path.parts if part not in ("", ".")]

    try:
        reverse_index = len(parts) - 1 - parts[::-1].index("training_outputs")
    except ValueError:
        reverse_index = -1

    if reverse_index >= 0:
        parts = parts[reverse_index + 1 :]

    if not parts:
        fallback = adapter_path.strip("/").replace("/", "_")
        return fallback or "unknown_adapter"

    return "_".join(parts)


def deduplicate_runs(runs: Sequence[EvaluationRun]) -> List[EvaluationRun]:
    """Keep the newest run when multiple directories map to the same run name."""
    latest_by_name: Dict[str, EvaluationRun] = {}
    for run in runs:
        existing = latest_by_name.get(run.run_name)
        if existing is None or run.mtime > existing.mtime:
            if existing is not None:
                logging.info(
                    "Replacing run %s from %s with newer data from %s.",
                    run.run_name,
                    existing.eval_dir,
                    run.eval_dir,
                )
            latest_by_name[run.run_name] = run
        else:
            logging.info(
                "Skipping %s in favor of newer data already queued for %s.",
                run.eval_dir,
                run.run_name,
            )
    return sorted(latest_by_name.values(), key=lambda item: item.mtime)


def log_evaluation_directory(
    eval_dir: Path | str,
    *,
    project: str = DEFAULT_PROJECT,
    entity: str | None = None,
    min_total_samples: int = MIN_TOTAL_SAMPLES,
    dry_run: bool = False,
    csv_path: Path | None = None,
) -> EvaluationRun | None:
    """Public helper to log a single evaluation directory in the standard format."""
    run = prepare_evaluation_run(
        Path(eval_dir),
        min_total_samples=min_total_samples,
    )
    if run is None:
        logging.info(
            "Skipping logging for %s because it did not meet logging requirements.",
            eval_dir,
        )
        return None

    if dry_run:
        target_csv = csv_path if csv_path is not None else Path("wandb_eval_dry_run.csv")
        write_dry_run_csv([run], target_csv)
        logging.info(
            "Dry run complete for %s. Preview available at %s.",
            eval_dir,
            target_csv,
        )
    else:
        log_runs_to_wandb(
            runs=[run],
            project=project,
            entity=entity,
        )
    return run


def split_metrics(metrics: MutableMapping[str, object]) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Separate metrics into scalar and non-scalar values for logging."""
    scalar: Dict[str, object] = {}
    non_scalar: Dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, bool)):
            scalar[key] = value
        else:
            non_scalar[key] = value
    return scalar, non_scalar


def log_runs_to_wandb(
    runs: Sequence[EvaluationRun],
    project: str,
    entity: str | None,
) -> None:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is required to upload metrics. Install wandb in the active environment."
        ) from exc

    for run in runs:
        scalar_metrics, non_scalar_metrics = split_metrics(run.metrics)
        init_config: Dict[str, object] = {
            "adapter_path": run.adapter_path,
            "evaluation_dir": str(run.eval_dir),
            "metrics_path": str(run.metrics_path),
            "config_path": str(run.config_path),
            "total_samples": run.total_samples,
        }
        init_config.update(non_scalar_metrics)
        init_kwargs = {
            "project": project,
            "name": run.run_name,
            "config": init_config,
        }
        if entity:
            init_kwargs["entity"] = entity

        logging.info("Logging run %s to project %s.", run.run_name, project)
        wb_run = wandb.init(**init_kwargs)
        try:
            wandb.log(scalar_metrics or {"_placeholder": run.total_samples})
            if run.generation_samples:
                table = wandb.Table(columns=GENERATION_TABLE_COLUMNS)
                for sample in run.generation_samples:
                    table.add_data(*(sample.get(column) for column in GENERATION_TABLE_COLUMNS))
                wb_run.log({"generation_results": table})
        finally:
            wb_run.finish()


def write_dry_run_csv(runs: Sequence[EvaluationRun], csv_path: Path) -> None:
    metrics_keys = sorted({key for run in runs for key in run.metrics.keys()})
    fieldnames = [
        "run_name",
        "adapter_path",
        "evaluation_dir",
        "metrics_path",
        "config_path",
        "total_samples",
    ] + metrics_keys

    csv_path = csv_path.resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            row = {
                "run_name": run.run_name,
                "adapter_path": run.adapter_path,
                "evaluation_dir": str(run.eval_dir),
                "metrics_path": str(run.metrics_path),
                "config_path": str(run.config_path),
                "total_samples": run.total_samples,
            }
            for key in metrics_keys:
                value = run.metrics.get(key)
                row[key] = format_csv_value(value)
            writer.writerow(row)
    logging.info("Wrote dry-run preview for %d runs to %s.", len(runs), csv_path)


def format_csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=True)


if __name__ == "__main__":
    main()
