from __future__ import annotations

import math
import statistics
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Sequence, Tuple

from gazefollow.evals.in_out_metrics import compute_binary_precision, serialize_binary_precision


METRIC_FIELDS: Tuple[str, ...] = (
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


def filter_gaze_metrics(
    generations: Sequence[Mapping[str, Any]],
    keep_metric: Callable[[Mapping[str, Any]], bool] | None = None,
    *,
    mutate: bool = False,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Count and optionally clear gaze metric values that should be filtered out."""

    total_counts = {field: 0 for field in METRIC_FIELDS}
    filtered_counts = {field: 0 for field in METRIC_FIELDS}

    for entry in generations:
        keep = True if keep_metric is None else bool(keep_metric(entry))
        detections = entry.get("gaze_detections", {}) or {}

        for person in detections.values():
            for field in METRIC_FIELDS:
                value = person.get(field)
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    continue

                total_counts[field] += 1
                if keep:
                    continue

                filtered_counts[field] += 1
                if mutate:
                    person[field] = None

    return total_counts, filtered_counts


def summarize_metrics(
    generations: Sequence[Mapping[str, Any]],
    binary_predictions: Sequence[int],
    binary_labels: Sequence[int],
    total_counts: Mapping[str, int],
    filtered_counts: Mapping[str, int],
) -> Dict[str, Any]:
    """Summarize gaze metrics and binary in/out precision."""

    metric_summary: Dict[str, Dict[str, Any]] = {}

    for field in METRIC_FIELDS:
        values: List[float] = []
        for entry in generations:
            for person in entry.get("gaze_detections", {}).values():
                value = person.get(field)
                if value is None or (isinstance(value, float) and math.isnan(value)):
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
        summary["inout_precision"] = serialize_binary_precision(binary_result)
        summary["inout_confusion"] = confusion

    return summary


def flatten_recomputed_metrics(summary: Mapping[str, Any]) -> Dict[str, float]:
    """Flatten nested metric summary into scalar-friendly keys."""

    flat: Dict[str, float] = {}
    gaze_metrics = summary.get("gaze_metrics", {})
    for field, payload in gaze_metrics.items():
        if not isinstance(payload, MutableMapping):
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
            if isinstance(value, (int, float)) and not (
                isinstance(value, float) and math.isnan(value)
            ):
                flat[metric_key] = float(value)

    precision_block = summary.get("inout_precision", {})
    if isinstance(precision_block, Mapping):
        precision_key_map = {
            "precision": "inout_precision",
            "true_positives": "inout_true_positives",
            "false_positives": "inout_false_positives",
            "predicted_positives": "inout_predicted_positives",
            "actual_positives": "inout_actual_positives",
            "positive_label": "inout_positive_label",
            "total_samples": "inout_total_samples",
        }
        for key, value in precision_block.items():
            if not isinstance(value, (int, float)):
                continue
            metric_key = precision_key_map.get(key)
            if metric_key is None:
                continue
            flat[metric_key] = float(value)

    confusion = summary.get("inout_confusion", {})
    if isinstance(confusion, Mapping):
        confusion_key_map = {
            "true_positives": "inout_true_positives",
            "false_positives": "inout_false_positives",
            "false_negatives": "inout_false_negatives",
            "true_negatives": "inout_true_negatives",
            "total": "inout_total",
            "precision": "inout_precision_confusion",
            "recall": "inout_recall",
            "accuracy": "inout_accuracy",
            "f1": "inout_f1",
        }
        for key, value in confusion.items():
            if not isinstance(value, (int, float)):
                continue
            metric_key = confusion_key_map.get(key)
            if metric_key is None or metric_key in flat:
                continue
            flat[metric_key] = float(value)

    samples = summary.get("samples_evaluated")
    if isinstance(samples, (int, float)):
        flat["samples_evaluated"] = float(samples)

    processed = summary.get("successfully_processed_samples")
    if isinstance(processed, (int, float)):
        flat["successfully_processed_samples"] = float(processed)

    return flat
