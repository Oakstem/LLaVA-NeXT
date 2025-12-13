#!/usr/bin/env python3
"""Recompute in/out labels and gaze metrics for a finished eval run.

This script reloads saved generation tables, re-derives in/out labels from the
raw text predictions, and recomputes gaze errors so we can compare results even
after changing label logic or filtering rules without rerunning the expensive
vision-language model. In contrast to `evaluate_model_qwen3vl.py`, which assigns
in/out during live evaluation with simple outside-phrase heuristics, this
offline pass re-runs the full person parsing plus `resolve_in_out_label` on both
ground truth and predictions (overriding stored `predicted_in_out`) so metrics
reflect the latest label resolution rules."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gazefollow.auto_phrase_grounding.detect_gaze_targets import (  # noqa: E402
    PersonDescription,
    parse_person_descriptions,
)
from gazefollow.data_proc.add_in_out_labels import load_in_out_lookup  # noqa: E402
from gazefollow.evals.in_out_metrics import compute_binary_precision, serialize_binary_precision  # noqa: E402
from gazefollow.evals.log_wandb_evaluations import (  # noqa: E402
    DEFAULT_PROJECT,
    GENERATION_TABLE_COLUMNS,
    build_run_name_from_adapter,
    format_generation_sample,
    split_metrics,
)
from gazefollow.qwen3vl_utils import coerce_in_out_value, resolve_in_out_label  # noqa: E402


METRIC_FIELDS = (
    "gaze_l2_error",
    "gaze_normalized_l2_error",
    "gaze_angular_error",
    "gaze_iou",
    "gaze_modified_l2_error",
)

METRIC_NAME_ALIASES: Dict[str, str] = {
    "gaze_l2_error": "gaze_l2_error",
    "gaze_normalized_l2_error": "gaze_l2_normalized",
    "gaze_angular_error": "gaze_angular_error",
    "gaze_iou": "gaze_iou",
    "gaze_modified_l2_error": "gaze_modified_l2",
}


def parse_people(text: Optional[str]) -> List[PersonDescription]:
    people = parse_person_descriptions(text or "")
    if not people and text:
        people = parse_person_descriptions(f"Person 1: {text}")
    return people


def build_sample_lookup(entry: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "id": entry.get("id"),
        "image": entry.get("image"),
        "image_path": entry.get("image_path"),
        "relative_path": entry.get("relative_path") or entry.get("image_path"),
        "image_relative_path": entry.get("image_relative_path") or entry.get("image_path"),
    }


def infer_in_out_flags(
    entry: Mapping[str, Any],
    lookup: Mapping[str, Any],
    *,
    prediction_override: Optional[str] = None,
) -> tuple[Optional[int], Optional[int]]:
    gt_people = parse_people(entry.get("ground_truth"))
    pred_text = prediction_override if prediction_override is not None else entry.get("prediction")
    if pred_text is None:
        pred_text = entry.get("model_prediction")
    pred_people = parse_people(pred_text)
    sample_lookup = build_sample_lookup(entry)

    gt_sample = dict(sample_lookup)
    gt_sample["in_out"] = entry.get("in_out")
    gt_flag = resolve_in_out_label(gt_sample, lookup, gt_people)
    if gt_flag is None:
        gt_flag = coerce_in_out_value(entry.get("in_out"))

    pred_flag = resolve_in_out_label(sample_lookup, {}, pred_people)
    if pred_flag is None:
        pred_flag = coerce_in_out_value(entry.get("predicted_in_out"))

    return gt_flag, pred_flag


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected {path} to contain a list.")
    return payload


def summarize_metrics(
    generations: List[Dict[str, Any]],
    binary_predictions: List[int],
    binary_labels: List[int],
    total_counts: Dict[str, int],
    filtered_counts: Dict[str, int],
) -> Dict[str, Any]:
    metric_summary: Dict[str, Dict[str, Any]] = {}

    for field in METRIC_FIELDS:
        values: List[float] = []
        for entry in generations:
            for person in entry.get("gaze_detections", {}).values():
                value = person.get(field)
                if value is None:
                    continue
                if isinstance(value, float) and math.isnan(value):
                    continue
                values.append(value)

        if values:
            metric_summary[field] = {
                "mean": sum(values) / len(values),
                "median": statistics.median(values),
                "count": len(values),
                "filtered_out": filtered_counts.get(field, 0),
                "total": total_counts.get(field, 0),
            }
        else:
            metric_summary[field] = {
                "mean": None,
                "median": None,
                "count": 0,
                "filtered_out": filtered_counts.get(field, 0),
                "total": total_counts.get(field, 0),
            }

    binary_result = compute_binary_precision(binary_predictions, binary_labels) if binary_labels else None

    confusion: Dict[str, Any] = {}
    if binary_labels:
        tp = fp = fn = tn = 0
        for predicted, label in zip(binary_predictions, binary_labels):
            if predicted == 1 and label == 1:
                tp += 1
            elif predicted == 1 and label == 0:
                fp += 1
            elif predicted == 0 and label == 1:
                fn += 1
            else:
                tn += 1
        total = tp + fp + fn + tn
        recall = tp / (tp + fn) if (tp + fn) else None
        precision = tp / (tp + fp) if (tp + fp) else None
        accuracy = (tp + tn) / total if total else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall)
            else None
        )
        confusion = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "total": total,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "f1": f1,
        }

    summary: Dict[str, Any] = {
        "gaze_metrics": metric_summary,
        "samples_evaluated": len(binary_labels),
    }
    if binary_result is not None:
        summary["in_out_precision"] = serialize_binary_precision(binary_result)
        summary["in_out_confusion"] = confusion

    return summary


def flatten_recomputed_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    gaze_metrics = summary.get("gaze_metrics", {})
    for field, payload in gaze_metrics.items():
        if not isinstance(payload, Mapping):
            continue
        base_name = METRIC_NAME_ALIASES.get(field, field)
        for key_name, metric_key in (
            ("mean", f"{base_name}_mean"),
            ("median", f"{base_name}_median"),
            ("count", f"{base_name}_count"),
            ("filtered_out", f"{base_name}_filtered_out"),
            ("total", f"{base_name}_total"),
        ):
            value = payload.get(key_name)
            if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
                flat[metric_key] = float(value)

    precision_block = summary.get("in_out_precision", {})
    if isinstance(precision_block, Mapping):
        for key, value in precision_block.items():
            if isinstance(value, (int, float)):
                metric_key = "in_out_precision" if key == "precision" else f"in_out_precision_{key}"
                flat[metric_key] = float(value)

    confusion = summary.get("in_out_confusion", {})
    if isinstance(confusion, Mapping):
        for key, value in confusion.items():
            if isinstance(value, (int, float)):
                flat[f"in_out_confusion_{key}"] = float(value)

    samples = summary.get("samples_evaluated")
    if isinstance(samples, (int, float)):
        flat["samples_evaluated"] = float(samples)

    return flat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline recomputation of in/out and gaze metrics.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("evaluation_results/eval_qwen3vl_qwen3vl_20251210_230318"),
        help="Evaluation results directory (contains model_generation_results.json).",
    )
    parser.add_argument(
        "--gt-csv",
        type=Path,
        default=Path("gazefollow/data/test2_combined_description_results.csv"),
        help="CSV with ground-truth in/out labels.",
    )
    parser.add_argument("--log-to-wandb", action="store_true", help="Log recomputed metrics to wandb.")
    parser.add_argument("--wandb-project", default=DEFAULT_PROJECT, help="wandb project name.")
    parser.add_argument("--wandb-entity", default=None, help="Optional wandb entity.")
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Override wandb run name (defaults to adapter-based name).",
    )
    parser.add_argument(
        "--wandb-run-id",
        default=None,
        help="Optional wandb run id to resume/update an existing run.",
    )
    parser.add_argument(
        "--log-generation-table",
        action="store_true",
        help="Log the recomputed generation table to wandb.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir: Path = args.results_dir
    generations_path = results_dir / "model_generation_results.json"
    if not generations_path.is_file():
        alt_path = results_dir / "dataset_localization_results.json"
        if alt_path.is_file():
            generations_path = alt_path
    generations_list = load_json_list(generations_path)
    in_out_lookup = load_in_out_lookup(args.gt_csv)

    total_counts = {field: 0 for field in METRIC_FIELDS}
    filtered_counts = {field: 0 for field in METRIC_FIELDS}
    binary_predictions: List[int] = []
    binary_labels: List[int] = []

    for entry in generations_list:
        gt_flag, pred_flag = infer_in_out_flags(
            entry,
            in_out_lookup,
            prediction_override=entry.get("model_prediction"),
        )
        entry["gt_in_out"] = gt_flag
        entry["pred_in_out"] = pred_flag
        entry.pop("in_out", None)
        entry.pop("predicted_in_out", None)

        keep_metric = gt_flag == 1 and pred_flag == 1
        for person in entry.get("gaze_detections", {}).values():
            for field in METRIC_FIELDS:
                value = person.get(field)
                if value is None:
                    continue
                total_counts[field] += 1
                if not keep_metric:
                    filtered_counts[field] += 1
                    person[field] = None

        if gt_flag is None or pred_flag is None:
            continue
        binary_predictions.append(pred_flag)
        binary_labels.append(gt_flag)

    summary = summarize_metrics(generations_list, binary_predictions, binary_labels, total_counts, filtered_counts)
    print(json.dumps(summary, indent=2))

    output_dir = results_dir / "offline_recompute"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "recomputed_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "model_generation_results_recomputed.json").write_text(
        json.dumps(generations_list, indent=2),
        encoding="utf-8",
    )

    if args.log_to_wandb:
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - optional dependency
            print(f"\n⚠️  wandb logging skipped (wandb not installed): {exc}")
            return

        config_path = results_dir / "evaluation_config.json"
        eval_config: Dict[str, Any] = {}
        if config_path.is_file():
            try:
                eval_config = json.loads(config_path.read_text(encoding="utf-8"))
            except OSError as exc:
                print(f"\n⚠️  wandb logging skipped (unable to read {config_path}): {exc}")
                return

        adapter_path = eval_config.get("adapter_path") if eval_config else None
        if not adapter_path:
            adapter_path = "baseline"
        run_name = args.wandb_run_name or (
            build_run_name_from_adapter(str(adapter_path))
            if eval_config
            else results_dir.name
        )
        init_config: Dict[str, Any] = {
            "adapter_path": adapter_path,
            "evaluation_dir": str(results_dir),
            "recomputed_dir": str(output_dir),
            "recomputed_metrics_path": str(output_dir / "recomputed_metrics.json"),
            "recomputed_generation_path": str(output_dir / "model_generation_results_recomputed.json"),
            "config_path": str(config_path),
        }
        init_config.update(eval_config)
        scalar_metrics, non_scalar_metrics = split_metrics(summary)
        flat_metrics = flatten_recomputed_metrics(summary)
        init_config.update(non_scalar_metrics)

        init_kwargs: Dict[str, Any] = {
            "project": args.wandb_project,
            "name": run_name,
            "config": init_config,
        }
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        if args.wandb_run_id:
            init_kwargs["id"] = args.wandb_run_id
            init_kwargs["resume"] = "allow"

        print(f"\nLogging recomputed metrics to wandb run '{run_name}' in project '{args.wandb_project}'...")
        wb_run = wandb.init(**init_kwargs)
        try:
            metrics_payload = dict(flat_metrics)
            if not metrics_payload:
                metrics_payload = {"_placeholder": summary.get("samples_evaluated", 0)}
            wandb.log(metrics_payload)

            if args.log_generation_table:
                table_rows: List[Dict[str, Any]] = []
                for sample in generations_list:
                    sample_for_table = dict(sample)
                    if "in_out" not in sample_for_table:
                        sample_for_table["in_out"] = sample_for_table.get("gt_in_out")
                    rows = format_generation_sample(sample_for_table)
                    for row in rows:
                        row["gt_in_out"] = sample_for_table.get("gt_in_out")
                        row["pred_in_out"] = sample_for_table.get("pred_in_out")
                    table_rows.extend(rows)
                if table_rows:
                    table_columns = list(GENERATION_TABLE_COLUMNS) + ["gt_in_out", "pred_in_out"]
                    table = wandb.Table(columns=table_columns)
                    for row in table_rows:
                        table.add_data(*(row.get(column) for column in table_columns))
                    wb_run.log({"generation_results": table})
        finally:
            wb_run.finish()


if __name__ == "__main__":
    main()
