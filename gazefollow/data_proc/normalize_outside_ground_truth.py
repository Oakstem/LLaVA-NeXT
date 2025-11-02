#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Match
import sys

TARGET_PHRASE = "something or someone outside the image"
LOOKING_AT_PATTERN = re.compile(r"(?i)(looking at)([^.?!]*)")
LOOKING_OUTSIDE_PATTERN = re.compile(
    r"(?i)(looking)(\s+)(?:outside(?:\s+of)?|out of)\s+(?:the|the\s+frame|frame|scene|photo|picture|shot)"
)
LOOKING_OUTSIDE_BARE_PATTERN = re.compile(r"(?i)(looking)\s+outside([.,;:!?]|$)")
LOOKING_BARE_PATTERN = re.compile(r"(?i)(looking)(\s*)([.,;:!?]|$)")


def normalize_ground_truth(text: str) -> tuple[str, bool]:
    def replace_looking_at(match: Match[str]) -> str:
        prefix = match.group(1)
        return f"{prefix} {TARGET_PHRASE}"

    updated_text, replacements = LOOKING_AT_PATTERN.subn(replace_looking_at, text)
    if replacements:
        return updated_text, True

    def replace_outside(match: Match[str]) -> str:
        prefix = match.group(1)
        spacing = match.group(2) or " "
        return f"{prefix}{spacing}at {TARGET_PHRASE}"

    updated_text, replacements = LOOKING_OUTSIDE_PATTERN.subn(replace_outside, text)
    if not replacements:
        def replace_outside_bare(match: Match[str]) -> str:
            punctuation = match.group(2)
            suffix = punctuation if punctuation else ""
            return f"{match.group(1)} at {TARGET_PHRASE}{suffix}"

        updated_text, bare_outside_replacements = LOOKING_OUTSIDE_BARE_PATTERN.subn(
            replace_outside_bare, text
        )
        if bare_outside_replacements:
            return updated_text, True

        def replace_bare(match: Match[str]) -> str:
            spacing = match.group(2)
            punctuation = match.group(3)
            suffix = f"{spacing}{punctuation}" if punctuation else spacing
            return f"{match.group(1)} at {TARGET_PHRASE}{suffix}"

        updated_text, bare_replacements = LOOKING_BARE_PATTERN.subn(replace_bare, text)
        if bare_replacements:
            return updated_text, True

        if "looking" not in text.lower():
            trimmed = text.rstrip()
            stripped = trimmed.rstrip(".,;:!?")
            removed = trimmed[len(stripped) :]
            base = stripped or trimmed or text.strip()
            if not base:
                return f"Looking at {TARGET_PHRASE}{removed}", True
            updated_text = f"{base}, looking at {TARGET_PHRASE}{removed}"
            return updated_text, True
        return text, False

    updated_text, _ = LOOKING_AT_PATTERN.subn(replace_looking_at, updated_text)
    return updated_text, True


def adjust_records(records: list[dict]) -> tuple[int, int]:
    total_out = 0
    adjusted = 0
    for record in records:
        if record.get("in_out") != 0:
            continue
        total_out += 1
        ground_truth = record.get("ground_truth")
        if not isinstance(ground_truth, str):
            continue
        normalized, changed = normalize_ground_truth(ground_truth)
        if changed:
            record["ground_truth"] = normalized
            adjusted += 1
        record.pop("gaze_ground_truth", None)
        record.pop("gaze_detections", None)
    return total_out, adjusted


def _try_parse_partial_list(raw_text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    idx = 0
    length = len(raw_text)

    def advance(pos: int) -> int:
        while pos < length and raw_text[pos].isspace():
            pos += 1
        return pos

    idx = advance(idx)
    if idx >= length or raw_text[idx] != "[":
        return []
    idx += 1

    parsed: list[dict] = []
    while idx < length:
        idx = advance(idx)
        if idx >= length or raw_text[idx] == "]":
            break
        try:
            item, next_idx = decoder.raw_decode(raw_text, idx)
        except json.JSONDecodeError:
            break
        parsed.append(item)
        idx = advance(next_idx)
        if idx < length and raw_text[idx] == ",":
            idx += 1
            continue
        if idx < length and raw_text[idx] == "]":
            break
        idx = advance(idx)
    return parsed


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        raw_text = handle.read()
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        data = _try_parse_partial_list(raw_text)
        if not data:
            raise
        print(
            "Warning: input JSON appears truncated; processed "
            f"{len(data)} records from {path.name}.",
            file=sys.stderr,
        )
    if not isinstance(data, list):
        raise ValueError("Expected the input JSON to be a list of samples")
    return data


def write_json(path: Path, data: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def derive_default_output_path(path: Path) -> Path:
    suffix = path.suffix or ".json"
    return path.with_name(f"{path.stem}_normalized{suffix}")


def derive_default_conversation_path(path: Path) -> Path:
    suffix = path.suffix or ".json"
    return path.with_name(f"{path.stem}_conversations{suffix}")


def _image_relative_path(image_path: str | None) -> str | None:
    if not image_path:
        return None
    path = Path(image_path)
    if "gazefollow" in path.parts:
        idx = path.parts.index("gazefollow")
        rel = Path(*path.parts[idx + 1 :])
    else:
        rel = path
    rel = rel.with_suffix("")
    return rel.as_posix()


def build_conversation_records(records: list[dict]) -> list[dict]:
    conversation_records: list[dict] = []
    for record in records:
        raw_prompt = record.get("dataset_prompt") or ""
        human_prompt = raw_prompt if raw_prompt.lstrip().startswith("<image>") else f"<image>\n{raw_prompt}"
        ground_truth_value = record.get("ground_truth")
        if isinstance(ground_truth_value, str):
            ground_truth = ground_truth_value
        elif ground_truth_value is None:
            ground_truth = ""
        else:
            ground_truth = str(ground_truth_value)
        image_rel = _image_relative_path(record.get("image_path"))

        conversation: list[dict[str, str]] = [
            {"from": "human", "value": human_prompt},
            {"from": "gpt", "value": ground_truth},
        ]

        conv_record: dict[str, object] = {
            "id": record.get("id"),
            "image": image_rel,
            "num_people": record.get("num_people"),
            "conversations": conversation,
            "in_out": record.get("in_out"),
        }

        gaze_gt = record.get("gaze_ground_truth")
        if isinstance(gaze_gt, dict):
            conv_record.update(
                {
                    "gaze_gt_x": gaze_gt.get("x"),
                    "gaze_gt_y": gaze_gt.get("y"),
                    "gaze_gt_width": gaze_gt.get("image_width"),
                    "gaze_gt_height": gaze_gt.get("image_height"),
                }
            )

        conversation_records.append(conv_record)
    return conversation_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize ground truth text for out-of-image samples so that they "
            'read "looking at something or someone outside the image".'
        )
    )
    parser.add_argument("input_path", type=Path, help="Path to the input JSON file")
    parser.add_argument(
        "--output-path",
        type=Path,
        help=(
            "Optional path to write the updated JSON. Defaults to "
            '<input>_normalized.json in the same directory.'
        ),
    )
    parser.add_argument(
        "--conversation-output",
        type=Path,
        help=(
            "Optional path to write the conversation-style dataset. Defaults to "
            '<input>_conversations.json in the same directory.'
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_json(args.input_path)
    total_out, adjusted = adjust_records(records)

    output_path = args.output_path or derive_default_output_path(args.input_path)
    write_json(output_path, records)

    conversation_path = (
        args.conversation_output or derive_default_conversation_path(args.input_path)
    )
    conversation_records = build_conversation_records(records)
    write_json(conversation_path, conversation_records)

    print(
        f"Adjusted {adjusted} of {total_out} out-of-image samples "
        f"(total records: {len(records)})."
    )
    print(f"Wrote updated records to {output_path}")
    print(f"Wrote conversation dataset to {conversation_path}")


if __name__ == "__main__":
    main()
