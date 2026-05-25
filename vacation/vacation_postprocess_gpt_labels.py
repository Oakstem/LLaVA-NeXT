#!/usr/bin/env python3
"""Post-process Vacation results with GPT to deduce social interaction labels.

This script reads Vacation result JSON files, uses GPT to infer the
social interaction label from extracted_gaze_info.persons only, and writes
back a new label field per sample.
"""

import argparse
import json
from pathlib import Path

from openai import OpenAI


DEFAULT_DEDUCTION_PROMPT = """You are given structured gaze info for people in an image.
Use ONLY the provided persons list (description + attention_focus) to deduce the social interaction label.

Choose exactly one:
MutualGaze: at least two people are looking at each other (A looks at B and B looks at A).
SharedObjectAttention: at least two people are looking at the same external object or place (not a person), including one person following another person's reference to that external target.
OneSidedGaze: one person looks at another person but the other looks away or elsewhere (not reciprocated).
NonCommmunicative: no clear gaze interaction or gaze is unclear.

Rules:
Do not guess MutualGaze. If reciprocity is not explicit, it is not MutualGaze.
If attention_focus is missing or unclear for most people, choose null.

Return JSON with key "social_interaction_label" only.
"""

GPT_DEDUCTION_RESPONSE_FORMAT = {"type": "json_object"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduce social interaction labels using GPT from extracted_gaze_info.persons."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Result JSON file or directory containing JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output file or directory. If omitted, writes <file>_gpt_deduced.json "
            "or <dir>_gpt_deduced/."
        ),
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Update files in place instead of writing to a new location.",
    )
    parser.add_argument(
        "--gpt-model",
        type=str,
        default="gpt-5-nano",
        help="OpenAI model to use for deduction (default: gpt-5-nano).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_DEDUCTION_PROMPT,
        help="Instruction prompt for GPT deduction.",
    )
    parser.add_argument(
        "--output-field",
        type=str,
        default="social_interaction_label_gpt_deduced",
        help="Field name to store the GPT-deduced label.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute the label even if the output field already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples per file (useful for quick tests).",
    )
    return parser.parse_args()


def _iter_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.rglob("*.json") if p.is_file())


def _resolve_output_root(input_path: Path, output: Path | None, in_place: bool) -> Path:
    if in_place:
        return input_path
    if output is None:
        if input_path.is_file():
            return input_path.with_name(f"{input_path.stem}_gpt_deduced.json")
        return Path(f"{input_path}_gpt_deduced")
    return output


def _map_output_path(input_path: Path, output_root: Path, file_path: Path, in_place: bool) -> Path:
    if in_place:
        return file_path
    if input_path.is_file():
        if output_root.suffix.lower() == ".json":
            return output_root
        if output_root.exists() and output_root.is_dir():
            return output_root / file_path.name
        if output_root.suffix == "":
            return output_root / file_path.name
        return output_root
    return output_root / file_path.relative_to(input_path)


def _extract_persons(entry: dict) -> list | None:
    extracted = entry.get("extracted_gaze_info")
    if not isinstance(extracted, dict):
        return None
    persons = extracted.get("persons")
    if not isinstance(persons, list) or not persons:
        return None
    return persons


def _deduce_label_with_gpt(
    client: OpenAI,
    persons: list,
    gpt_model: str,
    prompt: str,
) -> str:
    user_payload = json.dumps({"persons": persons})
    completion = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_payload},
        ],
        response_format=GPT_DEDUCTION_RESPONSE_FORMAT,
    )
    content = completion.choices[0].message.content
    return json.loads(content)["social_interaction_label"]


def _upsert_label(entry: dict, label_key: str, label_value: str | None) -> dict:
    after_keys = {"social_interaction_label", "extracted_gaze_info"}
    updated = {}
    inserted = False
    for key, value in entry.items():
        if key == label_key:
            continue
        updated[key] = value
        if not inserted and key in after_keys:
            updated[label_key] = label_value
            inserted = True
    if not inserted:
        updated[label_key] = label_value
    return updated


def _process_file(
    json_path: Path,
    output_path: Path,
    client: OpenAI,
    gpt_model: str,
    prompt: str,
    output_field: str,
    overwrite: bool,
    limit: int | None,
) -> None:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Invalid results list in {json_path}")

    processed = 0
    skipped_existing = 0
    skipped_missing = 0
    gpt_calls = 0

    updated_results = []
    for idx, entry in enumerate(results):
        if limit is not None and idx >= limit:
            updated_results.append(entry)
            continue
        if not overwrite and entry.get(output_field) not in (None, ""):
            skipped_existing += 1
            updated_results.append(entry)
            continue

        persons = _extract_persons(entry)
        if persons is None:
            skipped_missing += 1
            updated_results.append(_upsert_label(entry, output_field, None))
            continue

        label = _deduce_label_with_gpt(client, persons, gpt_model, prompt)
        gpt_calls += 1
        processed += 1
        updated_results.append(_upsert_label(entry, output_field, label))

    payload["results"] = updated_results

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"Processed {json_path} -> {output_path} | "
        f"gpt_calls={gpt_calls}, updated={processed}, "
        f"skipped_existing={skipped_existing}, skipped_missing={skipped_missing}"
    )


def main() -> None:
    args = parse_args()

    input_path = args.input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    output_root = _resolve_output_root(input_path, args.output, args.in_place)
    if input_path.is_dir() and output_root.suffix.lower() == ".json":
        raise ValueError("Output must be a directory when input_path is a directory.")

    files = _iter_input_files(input_path)
    if not files:
        raise ValueError(f"No JSON files found under {input_path}")

    client = OpenAI()

    for file_path in files:
        output_path = _map_output_path(input_path, output_root, file_path, args.in_place)
        _process_file(
            json_path=file_path,
            output_path=output_path,
            client=client,
            gpt_model=args.gpt_model,
            prompt=args.prompt,
            output_field=args.output_field,
            overwrite=args.overwrite,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
