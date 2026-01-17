#!/usr/bin/env python3
"""Combine conversation JSON files into a single dataset with simple metrics."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from gazefollow.auto_phrase_grounding.build_conversation_json import (  # noqa: E402
    DEFAULT_L2_SOURCE_THRESHOLD as DEFAULT_CONVO_L2_SOURCE_THRESHOLD,
)

from gazefollow.auto_phrase_grounding.conversation_utils import (
    OUTSIDE_FRAME_TARGET_DESCRIPTION,
    build_conversation_entry,
    build_outside_frame_conversation,
    extract_question_subject,
    has_looking_at_phrase,
    strip_helper_keys,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine gaze conversation JSON files into a single dataset."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing conversation JSON files (each holding a list of samples).",
    )
    parser.add_argument(
        "--output-root",
        default="training_datasets/qwen_sets",
        help="Root directory where the timestamped dataset directory is created.",
    )
    parser.add_argument(
        "--timestamp",
        help="Optional timestamp override for the output directory (format YYYYMMDD_HHMMSS).",
    )
    parser.add_argument(
        "--output-name",
        default="combined_conversations.json",
        help="Filename for the combined dataset JSON.",
    )
    parser.add_argument(
        "--pattern",
        default="*_conversation.json",
        help="Glob pattern (relative to input_dir) for selecting JSON conversation files to combine.",
    )
    parser.add_argument(
        "--non-conversation-pattern",
        default="*.json",
        help="Glob pattern for non-conversation JSON files. Matches already selected via --pattern are skipped.",
    )
    parser.add_argument(
        "--non-conversation-output",
        default="combined_non_conversations.json",
        help="Filename for the combined non-conversation dataset JSON.",
    )
    parser.add_argument(
        "--source-l2-threshold",
        type=float,
        default=DEFAULT_CONVO_L2_SOURCE_THRESHOLD,
        help="Maximum normalized L2 error for the gaze source when overriding outside-frame answers.",
    )
    return parser.parse_args()


def load_samples(files: Iterable[Path]) -> List[dict]:
    combined: List[dict] = []
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            print(
                f"JSON decode error in {path}: {exc}. Attempting to recover partial data."
            )
            recovered = recover_truncated_json(path)
            if not recovered:
                print(f"Skipping {path}: unable to recover valid samples.")
                continue
            combined.extend(recovered)
            continue
        if isinstance(data, list):
            combined.extend(data)
            continue
        if isinstance(data, dict):
            results = data.get("results")
            if isinstance(results, list):
                images_root = data.get("images_root")
                for record in results:
                    if images_root and "_images_root" not in record:
                        record["_images_root"] = images_root
                combined.extend(results)
                continue
        print(f"Skipping {path}: unsupported top-level JSON type {type(data).__name__}")
    return combined


def recover_truncated_json(path: Path) -> List[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Failed to read {path}: {exc}")
        return []

    images_root = None
    match = re.search(r'"images_root"\\s*:\\s*"([^"]+)"', text)
    if match:
        images_root = match.group(1)

    array_start = locate_array_start(text)
    if array_start is None:
        return []

    decoder = json.JSONDecoder()
    index = array_start + 1
    recovered: List[dict] = []
    while index < len(text):
        while index < len(text) and text[index] in " \r\n\t,":
            index += 1
        if index >= len(text) or text[index] == "]":
            break
        try:
            obj, offset = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            # Hit a truncated object; stop recovering.
            break
        if isinstance(obj, dict) and images_root and "_images_root" not in obj:
            obj["_images_root"] = images_root
        recovered.append(obj)
        index = offset

    if recovered:
        print(f"Recovered {len(recovered)} samples from truncated JSON {path}.")
    return recovered


def locate_array_start(text: str) -> Optional[int]:
    results_key = text.find('"results"')
    if results_key != -1:
        bracket_index = text.find("[", results_key)
        if bracket_index != -1:
            return bracket_index
    # Fall back to the first array in the document if "results" is missing.
    return text.find("[")


def get_identifier(
    record: Dict[str, Any],
    *,
    preferred_keys: tuple[str, ...] = ("annotation_id", "id", "image_id", "image"),
) -> Optional[str]:
    for key in preferred_keys:
        value = record.get(key)
        if value is not None:
            return str(value)
    return None


def get_source_normalized_l2(record: Dict[str, Any]) -> Optional[float]:
    try:
        grounding_eval = record.get("grounding_eval") or {}
        source_eval = grounding_eval.get("source") or {}
        errors = source_eval.get("errors") or {}
        value = errors.get("gaze_normalized_l2_error")
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def override_with_outside_answer(entry: Dict[str, Any], outside_answer: str) -> bool:
    for turn in entry.get("conversations", []):
        if (turn.get("from") or "").lower() == "gpt":
            if turn.get("value") == outside_answer:
                return False
            turn["value"] = outside_answer
            return True
    return False


def normalize_gpt_answers(conversation_samples: List[Dict[str, Any]]) -> Dict[str, int]:
    stats: Dict[str, int] = {
        "processed_gpt_turns": 0,
        "updated_answers": 0,
        "missing_subject": 0,
        "missing_gpt_value": 0,
    }
    for sample in conversation_samples:
        conversation = sample.get("conversations")
        if not isinstance(conversation, list):
            continue
        last_subject: Optional[str] = None
        for turn in conversation:
            speaker = (turn.get("from") or "").lower()
            if speaker == "human":
                question_value = turn.get("value")
                if isinstance(question_value, str):
                    last_subject = extract_question_subject(question_value)
                else:
                    last_subject = None
                continue
            if speaker != "gpt":
                continue
            answer_value = turn.get("value")
            if not isinstance(answer_value, str):
                stats["missing_gpt_value"] += 1
                continue
            stats["processed_gpt_turns"] += 1
            if has_looking_at_phrase(answer_value):
                continue
            if not last_subject:
                stats["missing_subject"] += 1
                continue
            target_desc = answer_value.strip()
            if not target_desc:
                stats["missing_gpt_value"] += 1
                continue
            try:
                formatted = build_conversation_entry(last_subject, target_desc)
            except ValueError:
                stats["missing_subject"] += 1
                continue
            turn["value"] = formatted["value"]
            stats["updated_answers"] += 1
    return stats


def apply_outside_frame_overrides(
    conversation_samples: List[Dict[str, Any]],
    reference_records: List[Dict[str, Any]],
    *,
    source_l2_threshold: float,
    outside_answer: str = OUTSIDE_FRAME_TARGET_DESCRIPTION,
) -> Dict[str, Any]:
    index: Dict[str, Dict[str, Any]] = {}
    for sample in conversation_samples:
        identifier = get_identifier(sample, preferred_keys=("id", "annotation_id", "image"))
        if identifier:
            index[identifier] = sample

    stats: Dict[str, Any] = {
        "eligible_records": 0,
        "updated_conversations": 0,
        "missing_conversation": 0,
        "missing_identifier": 0,
        "missing_conversation_ids": [],
        "created_conversations": 0,
        "created_conversation_ids": [],
        "creation_failures": [],
    }

    for record in reference_records:
        in_out = str(record.get("in_or_out", "0"))
        if in_out != "0":
            continue
        source_l2 = get_source_normalized_l2(record)
        if source_l2 is None or source_l2 > source_l2_threshold:
            continue
        stats["eligible_records"] += 1
        identifier = get_identifier(record)
        if identifier is None:
            stats["missing_identifier"] += 1
            continue
        entry = index.get(identifier)
        if entry is None:
            new_entry, reason = build_outside_frame_conversation(
                record, outside_answer=outside_answer
            )
            if new_entry is None:
                stats["missing_conversation"] += 1
                stats["missing_conversation_ids"].append(identifier)
                stats["creation_failures"].append({"id": identifier, "reason": reason})
                continue
            conversation_samples.append(new_entry)
            index[identifier] = new_entry
            stats["created_conversations"] += 1
            stats["created_conversation_ids"].append(identifier)
            continue
        entry["in_or_out"] = "0"
        if override_with_outside_answer(entry, outside_answer):
            stats["updated_conversations"] += 1
    return stats


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    conversation_files = sorted(path for path in input_dir.glob(args.pattern) if path.is_file())
    if not conversation_files:
        raise FileNotFoundError(
            f"No files matching pattern '{args.pattern}' found in {input_dir}"
        )

    conversation_set = {path.resolve() for path in conversation_files}
    non_conversation_candidates = sorted(
        path for path in input_dir.glob(args.non_conversation_pattern) if path.is_file()
    )
    non_conversation_files = [
        path for path in non_conversation_candidates if path.resolve() not in conversation_set
    ]

    combined_conversations = load_samples(conversation_files)
    if not combined_conversations:
        raise ValueError(
            "No samples were loaded. Check the input files or adjust --pattern."
        )
    combined_non_conversations: List[Dict[str, Any]] = []
    if non_conversation_files:
        combined_non_conversations = load_samples(non_conversation_files)

    override_stats = {
        "eligible_records": 0,
        "updated_conversations": 0,
        "missing_conversation": 0,
        "missing_identifier": 0,
        "missing_conversation_ids": [],
        "created_conversations": 0,
        "created_conversation_ids": [],
        "creation_failures": [],
    }
    if combined_non_conversations:
        override_stats = apply_outside_frame_overrides(
            combined_conversations,
            combined_non_conversations,
            source_l2_threshold=args.source_l2_threshold,
        )
    format_stats = normalize_gpt_answers(combined_conversations)
    total_samples = len(combined_conversations)
    in_samples = sum(
        1 for sample in combined_conversations if str(sample.get("in_or_out", "0")) == "1"
    )

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root).expanduser() / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    combined_path = output_dir / args.output_name
    with combined_path.open("w", encoding="utf-8") as fh:
        json.dump(combined_conversations, fh, ensure_ascii=False, indent=2)

    non_convo_path = output_dir / args.non_conversation_output
    if non_conversation_files:
        sanitized_non_conversations = strip_helper_keys(
            combined_non_conversations, helper_keys=("_images_root",)
        )
        with non_convo_path.open("w", encoding="utf-8") as fh:
            json.dump(sanitized_non_conversations, fh, ensure_ascii=False, indent=2)

    metrics = {
        "total_samples": total_samples,
        "in_or_out_positive": in_samples,
        "source_dir": str(input_dir),
        "file_count": len(conversation_files),
        "non_conversation_file_count": len(non_conversation_files),
        "non_conversation_samples": len(combined_non_conversations),
        "outside_source_threshold": args.source_l2_threshold,
        "outside_eligible_records": override_stats["eligible_records"],
        "outside_updated_conversations": override_stats["updated_conversations"],
        "outside_missing_conversation": override_stats["missing_conversation"],
        "outside_missing_identifier": override_stats["missing_identifier"],
        "outside_created_conversations": override_stats["created_conversations"],
        "outside_creation_failures": len(override_stats["creation_failures"]),
        "looking_at_checked_turns": format_stats["processed_gpt_turns"],
        "looking_at_updated_answers": format_stats["updated_answers"],
        "looking_at_missing_subjects": format_stats["missing_subject"],
        "looking_at_missing_gpt_values": format_stats["missing_gpt_value"],
    }
    metrics_path = output_dir / "set_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    print(f"Combined {total_samples} samples from {len(conversation_files)} files.")
    print(f"In-out positives: {in_samples}")
    print(f"Saved dataset: {combined_path}")
    print(f"Saved metrics: {metrics_path}")
    if non_conversation_files:
        print(
            f"Combined {len(combined_non_conversations)} non-conversation samples "
            f"from {len(non_conversation_files)} files -> {non_convo_path}"
        )
        print(
            "Outside-frame overrides: "
            f"{override_stats['updated_conversations']} updates, "
            f"{override_stats['eligible_records']} eligible, "
            f"{override_stats['created_conversations']} created, "
            f"{override_stats['missing_conversation']} missing conversations."
        )
        if override_stats["created_conversation_ids"]:
            print(
                f"Added conversation sample identifiers to {len(override_stats['created_conversation_ids'])} samples"
            )
        if override_stats["missing_conversation_ids"]:
            missing_ids = ", ".join(str(sample_id) for sample_id in override_stats["missing_conversation_ids"])
            print(f"Missing conversation sample identifiers: {missing_ids}")
        if override_stats["creation_failures"]:
            for failure in override_stats["creation_failures"]:
                print(
                    f"Failed to create conversation for {failure.get('id')}: "
                    f"{failure.get('reason', 'unknown reason')}"
                )
    else:
        print("No non-conversation files were combined.")

    if format_stats["updated_answers"]:
        print(
            "Normalized GPT answers to include 'looking at': "
            f"{format_stats['updated_answers']} updates out of {format_stats['processed_gpt_turns']} checked turns."
        )
    if format_stats["missing_subject"] or format_stats["missing_gpt_value"]:
        print(
            "Formatting skips — "
            f"missing subject: {format_stats['missing_subject']}, "
            f"missing GPT value: {format_stats['missing_gpt_value']}"
        )


if __name__ == "__main__":
    main()
