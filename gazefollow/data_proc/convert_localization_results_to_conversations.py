#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_INPUT = Path(
    "report_finals/datasets/train/qwen3vl_train_20251213_223625/train_dataset_localization_results_normalized_filtered_20251224_203622.json"
)
DEFAULT_OUTPUT = Path(
    "training_datasets/20251205_174741_sgl_conversation_data_20251214_204738/"
    "train_filtered_20251224_203622_69k.json"
)


def _relative_image_path(image_path: Optional[str]) -> Optional[str]:
    if not image_path:
        return None
    path = Path(image_path)
    if "gazefollow" in path.parts:
        idx = path.parts.index("gazefollow")
        rel = Path(*path.parts[idx + 1 :])
    else:
        rel = path
    return rel.as_posix()


def _normalize_prompt(prompt: str) -> str:
    if prompt.lstrip().startswith("<image>"):
        return prompt
    return f"<image>\n{prompt}"


def load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as fp:
        payload = json.load(fp)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    raise ValueError("Unsupported input JSON format.")


def build_conversation_record(record: Dict[str, Any]) -> Dict[str, Any]:
    raw_prompt = record.get("dataset_prompt") or ""
    ground_truth_value = record.get("ground_truth")
    if isinstance(ground_truth_value, str):
        ground_truth = ground_truth_value
    elif ground_truth_value is None:
        ground_truth = ""
    else:
        ground_truth = str(ground_truth_value)

    image_rel = (
        record.get("id")
        or record.get("image")
        or _relative_image_path(record.get("image_path"))
    )
    if not image_rel:
        raise ValueError("Missing image path or id in record.")

    conversation = [
        {"from": "human", "value": _normalize_prompt(raw_prompt)},
        {"from": "gpt", "value": ground_truth},
    ]

    return {
        "id": record.get("id") or image_rel,
        "image": image_rel,
        "num_people": record.get("num_people"),
        "conversations": conversation,
        "image_id": Path(image_rel).name,
    }


def convert_dataset(input_path: Path, output_path: Path) -> int:
    records = load_records(input_path)
    converted = [build_conversation_record(record) for record in records]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fp:
        json.dump(converted, fp, indent=2)
    return len(converted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert localization results into SGL-style conversation JSON."
        )
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the localization results JSON.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the output conversation JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = convert_dataset(args.input_json, args.output_json)
    print(f"Wrote {total} records to {args.output_json}")


if __name__ == "__main__":
    main()
