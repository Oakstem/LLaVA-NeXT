from __future__ import annotations

import re


_LABEL_PATTERNS: list[tuple[str, str]] = [
    ("MutualGaze", r"mutual\s*gaze|mutualgaze"),
    (
        "SharedObjectAttention",
        r"shared\s*object\s*attention|sharedobjectattention|joint\s*attention|jointatt",
    ),
    ("OneSidedGaze", r"one\s*sided\s*gaze|onesidedgaze"),
    ("NonCommmunicative", r"non\s*comm+unicative"),
]


def _label_from_text(text: str) -> str | None:
    normalized = text.strip().lower()
    if normalized in {"", "null", "none"}:
        return None

    compact = " ".join(normalized.replace("-", " ").replace("_", " ").split())
    for canonical, pattern in _LABEL_PATTERNS:
        if re.fullmatch(pattern, compact):
            return canonical
    return None


def _extract_decision_label(text: str) -> str | None:
    decision_patterns = [
        r"(?:^|\b)label\s*:\s*(?P<label>[^\n\r.]+)",
        r"best fits the description is\s*[\"']?(?P<label>[^\"'\n\r.]+)",
        r"most appropriate label(?: based on [^.]*)? is\s*[\"']?(?P<label>[^\"'\n\r.]+)",
        r"the social interaction label(?: that best fits the description)? is\s*[\"']?(?P<label>[^\"'\n\r.]+)",
    ]
    for pattern in decision_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        if normalized := _label_from_text(match.group("label")):
            return normalized
    return None


def _extract_first_mentioned_label(text: str) -> str | None:
    normalized = text.strip().lower()
    if normalized in {"", "null", "none"}:
        return None

    compact = " ".join(normalized.replace("-", " ").replace("_", " ").split())
    earliest: tuple[int, str] | None = None
    for canonical, pattern in _LABEL_PATTERNS:
        match = re.search(pattern, compact)
        if not match:
            continue
        start = match.start()
        if earliest is None or start < earliest[0]:
            earliest = (start, canonical)
    return earliest[1] if earliest is not None else None


def _normalize_social_label(label: str | None) -> str | None:
    if label is None:
        return None
    if not isinstance(label, str):
        label = str(label)
    if normalized := _label_from_text(label):
        return normalized
    if normalized := _extract_decision_label(label):
        return normalized
    return _extract_first_mentioned_label(label)


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
