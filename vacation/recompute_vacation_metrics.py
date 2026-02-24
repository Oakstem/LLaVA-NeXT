#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from datetime import datetime


def _extract_explicit_label(label: str) -> str | None:
    match = re.search(r"label\s*:\s*([^\n\r]+)", label, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def _normalize_social_label(label: str | None) -> str | None:
    if label is None:
        return None
    if not isinstance(label, str):
        label = str(label)
    explicit_label = _extract_explicit_label(label)
    if explicit_label is not None:
        explicit_cleaned = explicit_label.strip().lower()
        if explicit_cleaned in {"", "null", "none"}:
            return "NonCommmunicative"
        label = explicit_label
    cleaned = label.strip().lower()
    if cleaned in {"", "null", "none"}:
        return None
    cleaned = cleaned.replace("-", " ").replace("_", " ")
    cleaned = " ".join(cleaned.split())
    if any(
        option in cleaned for option in {"mutualgaze", "mutual gaze", "mutual"}
    ):
        return "MutualGaze"
    if any(
        option in cleaned
        for option in {
            "sharedobjectattention",
            "shared object attention",
            "shared attention",
            "joint attention",
            "joint att",
            "jointatt",
        }
    ):
        return "SharedObjectAttention"
    if any(
        option in cleaned
        for option in {
            "onesidedgaze",
            "one sided gaze",
            "one sided",
            "single",
            "single gaze",
        }
    ):
        return "OneSidedGaze"
    if any(
        option in cleaned
        for option in {
            "noncommmunicative",
            "noncommunicative",
            "non communicative",
            "non communicative gaze",
            "unclear",
        }
    ):
        return "NonCommmunicative"
    return None


def _get_gt_labels(entry: dict) -> list[str]:
    if "atomic_attribute_combo_GT_multi" in entry:
        multi = entry.get("atomic_attribute_combo_GT_multi")
        if isinstance(multi, list):
            return [str(label) for label in multi]
        if multi is not None:
            return [str(multi)]
        return []
    if "atomic_attribute_combo_GT" in entry:
        label = entry.get("atomic_attribute_combo_GT")
        return [str(label)] if label is not None else []
    label = entry.get("atomic_attribute_combo")
    return [str(label)] if label is not None else []


def _get_prediction_label(entry: dict) -> str | None:
    if isinstance(entry.get("extracted_gaze_info"), dict):
        return entry.get("extracted_gaze_info").get("social_interaction_label")
    return entry.get("response")


def _get_frame_path(entry: dict) -> str | None:
    for key in ("frame_path", "image_path", "img_path", "frame", "image", "file_path"):
        value = entry.get(key)
        if value is not None:
            return str(value)
    return None


def _build_per_sample_rows(results: list[dict]) -> list[dict]:
    rows = []
    for idx, entry in enumerate(results):
        pred_raw = _get_prediction_label(entry)
        pred_norm = _normalize_social_label(pred_raw)
        gt_labels = _get_gt_labels(entry)
        gt_norms = [
            normalized
            for label in gt_labels
            if (normalized := _normalize_social_label(label)) is not None
        ]
        is_correct = int(pred_norm is not None and bool(gt_norms) and pred_norm in gt_norms)
        rows.append(
            {
                "sample_idx": idx,
                "frame_path": _get_frame_path(entry),
                "prediction": pred_raw,
                "gt": " | ".join(gt_labels),
                "inferred_label": pred_norm,
                "correct": is_correct,
                "total_correct_rows": "",
            }
        )
    if rows:
        total_correct = sum(int(row["correct"]) for row in rows)
        rows[0]["total_correct_rows"] = total_correct
    return rows


def compute_metrics(results: list[dict]) -> dict:
    total_results = len(results)
    extracted_entries = []
    valid_labels = 0
    for r in results:
        pred_label = _get_prediction_label(r)
        extracted_entries.append(pred_label)
        if _normalize_social_label(pred_label) is not None:
            valid_labels += 1

    total_extractions = len(extracted_entries)


    valid_percent = (valid_labels / total_extractions) if total_extractions else 0.0

    evaluated = 0
    correct = 0
    skipped_missing_gt = 0
    skipped_missing_pred = 0
    for entry in results:
        pred_raw = _get_prediction_label(entry)
        pred_norm = _normalize_social_label(pred_raw)
        gt_norms = [
            normalized
            for label in _get_gt_labels(entry)
            if (normalized := _normalize_social_label(label)) is not None
        ]
        if pred_norm is None:
            skipped_missing_pred += 1
            continue
        if not gt_norms:
            skipped_missing_gt += 1
            continue
        evaluated += 1
        if pred_norm in gt_norms:
            correct += 1

    accuracy = (correct / evaluated) if evaluated else 0.0

    response_count = 0
    response_length_sum = 0
    for entry in results:
        response = entry.get("response")
        if response is None:
            continue
        response_count += 1
        response_length_sum += len(str(response))

    avg_response_length = (
        response_length_sum / response_count if response_count else 0.0
    )

    return {
        "total_results": total_results,
        "total_extractions": total_extractions,
        "valid_social_interaction_label_percent": valid_percent,
        "accuracy_vs_atomic_attribute_combo": accuracy,
        "accuracy_evaluated_frames": evaluated,
        "accuracy_correct_nb": correct,
        "accuracy_skipped_missing_gt": skipped_missing_gt,
        "accuracy_skipped_missing_pred": skipped_missing_pred,
        "avg_response_length": avg_response_length,
        "response_count": response_count,
    }


def _load_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return {}


def _extract_run_config(payload: dict) -> dict:
    config = payload.get("config")
    if not isinstance(config, dict):
        config = {}

    keep_adapter_key = "keep_adapter" if "keep_adapter" in config else "second_step_keep_adapter"
    keep_adapter = config.get(keep_adapter_key) if keep_adapter_key in config else None

    return {
        "temperature": config.get("temperature"),
        "keep_adapter": (keep_adapter is True) if keep_adapter is not None else None,
        "second_step_uses_image": (config.get("second_step_uses_image") is True)
        if "second_step_uses_image" in config
        else None,
        "gpt_extraction_enabled": (config.get("gpt_extraction_enabled") is True)
        if "gpt_extraction_enabled" in config
        else None,
    }


def _extract_prompt_columns(payload: dict) -> dict:
    config = payload.get("config")
    if not isinstance(config, dict):
        return {"prompt_step1": None, "prompt_step2": None}

    two_step = config.get("two_step_inference") is True
    if two_step:
        return {
            "prompt_step1": config.get("prompt_a"),
            "prompt_step2": config.get("prompt_b"),
        }

    return {
        "prompt_step1": config.get("prompt"),
        "prompt_step2": None,
    }


def _extract_adapter_name(json_path: Path) -> str | None:
    # Extract adapter name from the JSON file path
    name = json_path.name.replace(".json", "")
    return name



def _iter_result_files(results_dir: Path) -> list[Path]:
    return sorted(p for p in results_dir.rglob("*.json") if p.is_file())


def write_csv(rows: list[dict], output_csv: Path) -> None:
    if not rows:
        output_csv.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute vacation batch metrics for a results directory."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing vacation results JSON files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="CSV output path (defaults to <results_dir>/combined_metrics.csv).",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    output_csv = args.output_csv or (
        results_dir / f"combined_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    rows = []
    aggregate = {
        "total_results": 0,
        "total_extractions": 0,
        "valid_labels": 0,
        "accuracy_correct_nb": 0,
        "accuracy_evaluated_frames": 0,
        "accuracy_skipped_missing_gt": 0,
        "accuracy_skipped_missing_pred": 0,
        "response_length_sum": 0,
        "response_count": 0,
    }

    for json_path in _iter_result_files(results_dir):
        payload = _load_payload(json_path)
        results = payload.get("results")
        if not isinstance(results, list):
            continue
        metrics = compute_metrics(results)
        run_config = _extract_run_config(payload)
        prompt_columns = _extract_prompt_columns(payload)
        adapter_name = _extract_adapter_name(json_path)
        rows.append(
            {
                "results_file": str(json_path),
                "adapter_name": adapter_name,
                **run_config,
                **metrics,
                **prompt_columns,
            }
        )
        per_sample_csv = json_path.with_name(f"{json_path.stem}_samples.csv")
        write_csv(_build_per_sample_rows(results), per_sample_csv)

        aggregate["total_results"] += metrics["total_results"]
        aggregate["total_extractions"] += metrics["total_extractions"]
        aggregate["valid_labels"] += metrics["valid_social_interaction_label_percent"] * metrics["total_extractions"]
        aggregate["accuracy_correct_nb"] += metrics["accuracy_correct_nb"]
        aggregate["accuracy_evaluated_frames"] += metrics["accuracy_evaluated_frames"]
        aggregate["accuracy_skipped_missing_gt"] += metrics["accuracy_skipped_missing_gt"]
        aggregate["accuracy_skipped_missing_pred"] += metrics["accuracy_skipped_missing_pred"]
        aggregate["response_length_sum"] += metrics["avg_response_length"] * metrics["response_count"]
        aggregate["response_count"] += metrics["response_count"]

    write_csv(rows, output_csv)
    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
