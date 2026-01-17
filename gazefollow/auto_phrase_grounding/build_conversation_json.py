#!/usr/bin/env python3
"""
Convert gaze_region_descriptions.json into a conversational dataset.

Each entry becomes a (<image>, question, answer) pair modeled after
training_datasets/.../val_rephrased_*.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from gazefollow.auto_phrase_grounding.conversation_utils import (  # noqa: E402
    OUTSIDE_FRAME_TARGET_DESCRIPTION,
    clean_question_phrase,
    is_outside_frame,
    load_image_dims,
    sanitize_person_description,
)


DEFAULT_INPUT = Path("gazefollow/auto_phrase_grounding/gaze_region_descriptions.json")
DEFAULT_OUTPUT = DEFAULT_INPUT.with_name("gaze_region_conversations.json")
DEFAULT_IMAGES_ROOT = Path("/mnt/d/Projects/data/gazefollow")
DEFAULT_SAVE_INTERVAL = 2
DEFAULT_L2_TARGET_THRESHOLD = 0.24
DEFAULT_L2_SOURCE_THRESHOLD = 0.4
def build_conversation_entry(
    record: Dict[str, Any],
    images_root: Path,
    *,
    precomputed_size: Optional[Tuple[int, int]] = None,
    source_l2_threshold: float = DEFAULT_L2_SOURCE_THRESHOLD,
    target_l2_threshold: float = DEFAULT_L2_TARGET_THRESHOLD,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    image_rel = record.get("relative_path") or record.get("image_id")
    if not image_rel:
        return None, "missing image path"
    source = record.get("source") or {}
    target = record.get("target") or {}
    source_desc = sanitize_person_description(source.get("description"))
    target_point = target.get("point")
    outside_frame = is_outside_frame(record.get("in_or_out"))

    if not source_desc:
        return None, "missing source description"
    if not target_point:
        return None, "missing target point"
    if outside_frame:
        target_desc = OUTSIDE_FRAME_TARGET_DESCRIPTION
    else:
        target_desc = target.get("description")
        if not target_desc:
            return None, "missing target description"
    grounding_eval = record.get("grounding_eval") or {}
    source_eval = grounding_eval.get("source") or {}
    target_eval = grounding_eval.get("target") or {}
    source_bbox = source_eval.get("predicted_bbox")
    target_bbox = target_eval.get("predicted_bbox")
    source_l2 = (source_eval.get("errors") or {}).get("gaze_normalized_l2_error")
    target_l2 = (target_eval.get("errors") or {}).get("gaze_normalized_l2_error")
    if not source_bbox or (not outside_frame and not target_bbox):
        return None, "missing grounding bounding boxes"
    if source_l2 is None or (not outside_frame and target_l2 is None):
        return None, "missing grounding L2 errors"
    if source_l2 > source_l2_threshold:
        return None, f"source normalized L2 too high ({source_l2:.3f}>{source_l2_threshold:.3f})"
    if not outside_frame and target_l2 > target_l2_threshold:
        return None, f"target normalized L2 too high ({target_l2:.3f}>{target_l2_threshold:.3f})"

    if precomputed_size is None:
        image_width, image_height = load_image_dims(images_root, image_rel)
    else:
        image_width, image_height = precomputed_size
    gaze_x, gaze_y = target_point

    question_subject = clean_question_phrase(source_desc)
    conversation = [
        {
            "from": "human",
            "value": f"<image>\nDescribe where the {question_subject} is looking at",
        },
        {
            "from": "gpt",
            "value": f"The {question_subject} is looking at {target_desc}",
        },
    ]

    return {
        "id": record.get("annotation_id") or record.get("image_id"),
        "image": image_rel.rsplit(".", 1)[0],
        "num_people": 1,
        "conversations": conversation,
        "gaze_gt_x": gaze_x,
        "gaze_gt_y": gaze_y,
        "gaze_gt_width": image_width,
        "gaze_gt_height": image_height,
        "in_or_out": record.get("in_or_out"),
    }, None


def iter_records(input_path: Path) -> List[Dict[str, Any]]:
    with input_path.open("r") as fp:
        payload = json.load(fp)
    if isinstance(payload, dict) and "results" in payload:
        return payload["results"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Unsupported input JSON format.")


def save_dataset(entries: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fp:
        json.dump(entries, fp, indent=2)


def build_conversation_dataset(
    input_json: Path,
    output_json: Path,
    images_root: Path,
    save_interval: Optional[int] = DEFAULT_SAVE_INTERVAL,
    source_l2_threshold: float = DEFAULT_L2_SOURCE_THRESHOLD,
    target_l2_threshold: float = DEFAULT_L2_TARGET_THRESHOLD,
) -> int:
    records = iter_records(input_json)
    entries: List[Dict[str, Any]] = []
    interval = save_interval if save_interval and save_interval > 0 else None

    for idx, record in enumerate(records, start=1):
        entry, reason = build_conversation_entry(
            record,
            images_root,
            source_l2_threshold=source_l2_threshold,
            target_l2_threshold=target_l2_threshold,
        )
        if entry is None:
            identifier = record.get("annotation_id") or record.get("image_id")
            print(f"[skip] conversation entry {identifier}: {reason}")
            continue
        entries.append(entry)
        if interval and idx % interval == 0:
            save_dataset(entries, output_json)
            print(f"[save] autosaved {len(entries)} entries -> {output_json}")

    save_dataset(entries, output_json)
    print(f"[done] wrote {len(entries)} entries to {output_json}")
    return len(entries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build conversational dataset from gaze region descriptions.")
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT, help="Path to gaze_region_descriptions.json")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT, help="Path to save conversational JSON")
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT, help="Root directory containing images")
    parser.add_argument("--save-interval", type=int, default=DEFAULT_SAVE_INTERVAL, help="Autosave every N entries (<=0 disables)")
    parser.add_argument(
        "--max-normalized-l2",
        type=float,
        default=DEFAULT_L2_TARGET_THRESHOLD,
        help="Maximum normalized L2 error for both source and target to keep a sample (deprecated; use --max-source-normalized-l2 / --max-target-normalized-l2).",
    )
    parser.add_argument(
        "--max-source-normalized-l2",
        type=float,
        default=None,
        help="Maximum normalized L2 error for the gaze source (defaults to --max-normalized-l2).",
    )
    parser.add_argument(
        "--max-target-normalized-l2",
        type=float,
        default=None,
        help="Maximum normalized L2 error for the gaze target (defaults to --max-normalized-l2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_l2_threshold = (
        args.max_source_normalized_l2 if args.max_source_normalized_l2 is not None else args.max_normalized_l2
    )
    target_l2_threshold = (
        args.max_target_normalized_l2 if args.max_target_normalized_l2 is not None else args.max_normalized_l2
    )
    build_conversation_dataset(
        input_json=args.input_json,
        output_json=args.output_json,
        images_root=Path(args.images_root),
        save_interval=args.save_interval,
        source_l2_threshold=source_l2_threshold,
        target_l2_threshold=target_l2_threshold,
    )


if __name__ == "__main__":
    main()
