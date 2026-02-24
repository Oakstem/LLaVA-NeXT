#!/usr/bin/env python
"""Annotate a conversation dataset with in/out labels from a gaze summary CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


CSV_KEY_CANDIDATES: Sequence[str] = ("image_path", "image_path.1", "image", "id")
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Insert an `in_out` field into a JSON dataset by looking up values from "
            "a CSV file that contains an `in_or_out` column."
        )
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to the JSON dataset to update.",
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the CSV file that provides the `in_or_out` labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the updated JSON. Defaults to overwriting the input file.",
    )
    return parser.parse_args()


def iter_normalized_keys(value: Any, *, include_basename: bool = True) -> Iterable[str]:
    """Produce lookup keys for different representations of the same identifier."""
    if value is None:
        return
    text = str(value).strip()
    if not text:
        return

    text = text.replace("\\", "/")
    lowered = text.lower()
    for ext in IMAGE_EXTENSIONS:
        if lowered.endswith(ext):
            text = text[: -len(ext)]
            break

    yield text
    if include_basename and "/" in text:
        yield text.rsplit("/", 1)[-1]


def load_in_out_lookup_with_conflicts(
    csv_path: Path,
    *,
    include_basename: bool = True,
) -> Tuple[Dict[str, Any], Set[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "in_or_out" not in reader.fieldnames:
            raise ValueError("CSV must contain an 'in_or_out' column.")

        lookup: Dict[str, Any] = {}
        conflicting_keys: Set[str] = set()
        for row in reader:
            raw_value = row["in_or_out"]
            label: Any
            if raw_value is None:
                continue
            raw_value = raw_value.strip()
            if not raw_value:
                continue
            if raw_value.isdigit():
                label = int(raw_value)
            else:
                try:
                    label = float(raw_value)
                except ValueError:
                    label = raw_value

            for column in CSV_KEY_CANDIDATES:
                if column not in row or not row[column]:
                    continue
                for key in iter_normalized_keys(row[column], include_basename=include_basename):
                    if key in conflicting_keys:
                        continue
                    existing = lookup.get(key)
                    if existing is not None and existing != label:
                        conflicting_keys.add(key)
                        lookup.pop(key, None)
                        continue
                    lookup[key] = label
        return lookup, conflicting_keys


def load_in_out_lookup(
    csv_path: Path,
    *,
    drop_conflicts: bool = False,
    include_basename: bool = True,
) -> Dict[str, Any]:
    lookup, conflicting_keys = load_in_out_lookup_with_conflicts(
        csv_path,
        include_basename=include_basename,
    )
    if conflicting_keys and not drop_conflicts:
        key = sorted(conflicting_keys)[0]
        raise ValueError(f"Conflicting 'in_or_out' values found for key '{key}'.")
    return lookup


def annotate_json_samples(dataset: List[Dict[str, Any]], lookup: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    for sample in dataset:
        sample_keys = []
        for field in ("id", "image"):
            sample_keys.extend(iter_normalized_keys(sample.get(field)))

        in_out = None
        for key in sample_keys:
            if key in lookup:
                in_out = lookup[key]
                break

        if in_out is None:
            sample_id = sample.get("id") or sample.get("image") or "<unknown>"
            missing.append(str(sample_id))
            continue

        sample["in_out"] = in_out

    return missing


def main() -> None:
    args = parse_args()
    json_path: Path = args.json_path
    csv_path: Path = args.csv_path
    output_path: Path = args.output or json_path

    dataset = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, list):
        raise ValueError("Expected the JSON dataset to be a list of samples.")

    lookup = load_in_out_lookup(csv_path)
    missing = annotate_json_samples(dataset, lookup)

    output_path.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote annotated dataset to {output_path}")

    print(f"Annotated {len(dataset) - len(missing)} / {len(dataset)} samples with in_out labels.")
    if missing:
        print(f"Missing labels for {len(missing)} samples. Example IDs: {', '.join(missing[:5])}")


if __name__ == "__main__":
    main()
