#!/usr/bin/env python3
"""Combine all *combined_metrics*.csv files under a base directory.

- Adds a subdir column (relative parent path)
- Removes rows where any column equals "ALL"
- Prints top 5 rows by accuracy_correct_nb
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _find_csvs(base_dir: Path, out_csv: Path) -> list[Path]:
    csvs = sorted(base_dir.rglob("*combined_metrics*.csv"))
    out_csv_resolved = out_csv.resolve()
    return [p for p in csvs if p.resolve() != out_csv_resolved]


def _add_subdir(df: pd.DataFrame, csv_path: Path, base_dir: Path) -> pd.DataFrame:
    try:
        rel_parent = csv_path.parent.relative_to(base_dir)
        subdir = str(rel_parent)
    except ValueError:
        subdir = csv_path.parent.name
    df = df.copy()
    df["subdir"] = subdir
    return df


def _drop_all_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "results_file" in df.columns:
        df = df[df["results_file"] != "ALL"]
    mask_all = (df == "ALL").any(axis=1)
    return df[~mask_all]


def combine_metrics(base_dir: Path, out_csv: Path) -> pd.DataFrame:
    csvs = _find_csvs(base_dir, out_csv)
    if not csvs:
        raise FileNotFoundError(f"No *combined_metrics*.csv files under {base_dir}")

    frames: list[pd.DataFrame] = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        df = _drop_all_rows(df)
        df = _add_subdir(df, csv_path, base_dir)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    if "accuracy_correct_nb" in combined.columns:
        combined = combined.sort_values("accuracy_correct_nb", ascending=False).reset_index(drop=True)
    combined.to_csv(out_csv, index=False)
    return combined


def _print_top5(df: pd.DataFrame) -> None:
    if "accuracy_correct_nb" not in df.columns:
        print("Column 'accuracy_correct_nb' not found; skipping top-5 output.")
        return

    top5 = (
        df.sort_values("accuracy_correct_nb", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )

    cols = [c for c in ["results_file", "accuracy_correct_nb", "subdir"] if c in top5.columns]
    print("Top 5 by accuracy_correct_nb:")
    print(top5[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine *combined_metrics*.csv files.")
    parser.add_argument(
        "base_dir",
        nargs="?",
        default=r"/mnt/d/Projects/LLaVA-NeXT/vacation_results",
        help="Base directory containing result subdirs",
    )
    parser.add_argument(
        "--out",
        default="combined_metrics_all.csv",
        help="Output CSV filename (written under base_dir if relative)",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = base_dir / out_path

    combined = combine_metrics(base_dir, out_path)
    print(f"Wrote {len(combined)} rows to {out_path}")
    _print_top5(combined)


if __name__ == "__main__":
    main()
