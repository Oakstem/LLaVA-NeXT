from __future__ import annotations


def _normalize_social_label(label: str | None) -> str | None:
    if label is None:
        return None
    if not isinstance(label, str):
        label = str(label)
    cleaned = label.strip().lower()
    if cleaned in {"", "null", "none"}:
        return None
    cleaned = cleaned.replace("-", " ").replace("_", " ")
    cleaned = " ".join(cleaned.split())
    if cleaned in {"mutualgaze", "mutual gaze", "mutual"}:
        return "MutualGaze"
    if cleaned in {
        "sharedobjectattention",
        "shared object attention",
        "shared attention",
        "joint attention",
        "joint att",
        "jointatt",
    }:
        return "SharedObjectAttention"
    if cleaned in {
        "onesidedgaze",
        "one sided gaze",
        "one sided",
        "single",
        "single gaze",
    }:
        return "OneSidedGaze"
    if cleaned in {
        "noncommmunicative",
        "noncommunicative",
        "non communicative",
        "non communicative gaze",
        "unclear",
    }:
        return "NonCommmunicative"
    return None


def compute_extraction_metrics(results: list[dict]) -> dict:
    total_results = len(results)
    extracted_entries = [
        r for r in results if isinstance(r.get("extracted_gaze_info"), dict)
    ]
    total_extractions = len(extracted_entries)
    valid_labels = 0
    null_labels = 0

    for entry in extracted_entries:
        label = entry["extracted_gaze_info"].get("social_interaction_label")
        if _normalize_social_label(label) is None:
            null_labels += 1
        else:
            valid_labels += 1

    valid_percent = (valid_labels / total_extractions) if total_extractions else 0.0

    evaluated = 0
    correct = 0
    skipped_missing_gt = 0
    skipped_missing_pred = 0
    for entry in extracted_entries:
        pred_norm = _normalize_social_label(
            entry["extracted_gaze_info"].get("social_interaction_label")
        )
        gt_val = entry.get("atomic_attribute_combo") or entry.get(
            "atomic_attribute_combo_GT"
        )
        gt_norm = _normalize_social_label(gt_val)
        if pred_norm is None:
            skipped_missing_pred += 1
            continue
        if gt_norm is None:
            skipped_missing_gt += 1
            continue
        evaluated += 1
        if pred_norm == gt_norm:
            correct += 1

    accuracy = (correct / evaluated) if evaluated else 0.0

    return {
        "total_results": total_results,
        "total_extractions": total_extractions,
        "valid_social_interaction_labels": valid_labels,
        "null_social_interaction_labels": null_labels,
        "valid_social_interaction_label_percent": valid_percent,
        "accuracy_vs_atomic_attribute_combo": accuracy,
        "accuracy_evaluated_frames": evaluated,
        "accuracy_correct_nb": correct,
        "accuracy_skipped_missing_gt": skipped_missing_gt,
        "accuracy_skipped_missing_pred": skipped_missing_pred,
    }
