#!/usr/bin/env python3
"""Convert GazeFollow annotation TXT dumps into structured CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Sequence


DEFAULT_INPUT = Path("gazefollow/data/test_annotations_release.txt")
DEFAULT_OUTPUT = DEFAULT_INPUT.with_suffix(".csv")

IN_OR_OUT_COLUMN = "in_or_out"
ORIGINAL_PATH_COLUMN = "original_path"
DEFAULT_IN_OR_OUT_VALUE = "1"

BASE_COLUMNS: Sequence[str] = (
    "image_path",
    "id",
    "body_bbox_x",
    "body_bbox_y",
    "body_bbox_width",
    "body_bbox_height",
    "eye_x",
    "eye_y",
    "gaze_x",
    "gaze_y",
    "head_bbox_x_min",
    "head_bbox_y_min",
    "head_bbox_x_max",
    "head_bbox_y_max",
    IN_OR_OUT_COLUMN,
    "dataset_source",
    "meta",
)

IN_OR_OUT_INDEX = BASE_COLUMNS.index(IN_OR_OUT_COLUMN)


def determine_headers(first_row: Sequence[str]) -> tuple[List[str], bool]:
    base_headers = list(BASE_COLUMNS)
    column_count = len(first_row)
    base_len = len(base_headers)
    if column_count == base_len:
        if _looks_like_in_or_out_value(first_row[IN_OR_OUT_INDEX]):
            return base_headers, False
        return [*base_headers, ORIGINAL_PATH_COLUMN], True
    if column_count == base_len - 1:
        return base_headers, True
    if column_count == base_len + 1:
        return [*base_headers, ORIGINAL_PATH_COLUMN], False
    raise ValueError(
        f"Unexpected column count {column_count}; expected between {base_len - 1} and {base_len + 1}."
    )


def _looks_like_in_or_out_value(value: str) -> bool:
    try:
        numeric_value = float(value)
    except ValueError:
        return False
    return numeric_value in (0.0, 1.0)


def insert_in_or_out_value(row: Sequence[str], value: str = DEFAULT_IN_OR_OUT_VALUE) -> List[str]:
    result = list(row)
    result.insert(IN_OR_OUT_INDEX, value)
    return result


def iter_clean_rows(reader: Iterable[List[str]]) -> Iterable[List[str]]:
    for raw in reader:
        if not raw:
            continue
        cleaned = [value.strip() for value in raw]
        if not any(cleaned):
            continue
        yield cleaned


def convert_annotations(input_path: Path, output_path: Path) -> tuple[int, List[str]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {input_path}")

    with input_path.open("r", newline="") as src:
        reader = csv.reader(src, delimiter=",")
        rows_iter = iter_clean_rows(reader)
        try:
            first_row = next(rows_iter)
        except StopIteration:
            raise ValueError(f"Annotation file {input_path} is empty.")
        headers, needs_in_or_out = determine_headers(first_row)
        if needs_in_or_out:
            first_row = insert_in_or_out_value(first_row)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        row_count = 0
        with output_path.open("w", newline="") as dst:
            writer = csv.writer(dst)
            writer.writerow(headers)
            writer.writerow(first_row + [""] * (len(headers) - len(first_row)))
            row_count = 1
            for row_idx, row in enumerate(rows_iter, start=2):
                if needs_in_or_out:
                    row = insert_in_or_out_value(row)
                if len(row) > len(headers):
                    raise ValueError(
                        f"Row {row_idx} has {len(row)} columns (expected <= {len(headers)}). "
                        "Check for stray commas in meta/original_path fields."
                    )
                padded_row = row + [""] * (len(headers) - len(row))
                writer.writerow(padded_row)
                row_count += 1

    return row_count, headers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert GazeFollow annotation TXT file to CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to train/test_annotations_release.txt.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count, headers = convert_annotations(args.input, args.output)
    print(f"Wrote {count} rows with columns {headers} to {args.output}")


if __name__ == "__main__":
    main()
