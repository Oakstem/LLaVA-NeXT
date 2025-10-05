"""Utility to combine many JSON files into a single JSON.

Behavior:
- Walks a given directory (non-recursive by default) for files matching a glob (default: "*.json").
- Loads each JSON; expects the file to be a mapping from id-str -> object.
- Merges all mappings. If duplicate ids occur, last-seen wins by default, but there's an option to error.
- Sorts the combined mapping by numeric value of the id when possible, otherwise lexicographically.
- Writes the combined JSON to output file (creates parent dirs if needed).

Usage example:
    python gazefollow/combine_jsons.py /path/to/jsons -o combined.json

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple
from attn_utils import fix_wsl_paths
import sys



def find_json_files(directory: Path, pattern: str = "*metrics.json", recursive: bool = False) -> Iterable[Path]:
    if recursive:
        yield from directory.rglob(pattern)
    else:
        yield from directory.glob(pattern)


def load_json_file(p: Path) -> Dict[str, Any]:
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Accept either a top-level object (mapping) or a list of mappings.
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            # Expect list of small mappings like [{id: {...}}, {id2: {...}}, ...]
            merged: Dict[str, Any] = {}
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    raise ValueError(f"Item {i} in array from {p} is not an object/dict")
                # Merge shallowly — later items in the list override earlier ones.
                for k, v in item.items():
                    merged[k] = v
            return merged

        raise ValueError(f"JSON file {p} has unsupported top-level type: {type(data)}")
    except Exception as e:
        raise RuntimeError(f"Failed to load JSON from {p}: {e}")


def merge_dicts(dicts: Iterable[Dict[str, Any]], on_duplicate: str = "overwrite") -> Dict[str, Any]:
    # on_duplicate: overwrite | error | keep-first
    combined: Dict[str, Any] = {}
    for dk, dd in dicts.items():
        combined[dk] = dd
    return combined


def sort_keys(keys: Iterable[str]) -> list[str]:
    def key_fn(k: str) -> Tuple[int, str]:
        # try numeric
        try:
            return (0, int(k))  # numeric sorts first
        except Exception:
            return (1, k)

    # We return a list of keys sorted primarily by numeric value when possible
    return sorted(keys, key=lambda x: key_fn(x))


def write_json_atomic(data: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine many JSON files (mapping id->obj) into one file.")
    p.add_argument("--dir", type=Path, default="/mnt/d/Projects/LLaVA-NeXT/evaluation_results/multi_checkpoint",
                    help="Directory containing JSON files to combine")
    p.add_argument("-o", "--output", default=None, type=Path, help="Output JSON file path")
    p.add_argument("-p", "--pattern", default="*metrics.json", help="Glob pattern to match files (default: *metrics.json)")
    p.add_argument("-r", "--recursive", default=True, help="Search recursively")
    p.add_argument("--on-duplicate", choices=["overwrite", "error", "keep-first"], default="overwrite",
                   help="How to handle duplicate ids across files")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # fix WSL paths if needed
    args.dir = Path(fix_wsl_paths(str(args.dir)))
    if args.output is None:
        args.output = args.dir / "combined.json"
    else:
        args.output = Path(fix_wsl_paths(str(args.output)))
    print(f"Using directory: {args.dir}")
    print(f"Using output file: {args.output}")
    if not args.dir.exists() or not args.dir.is_dir():
        print(f"Error: directory {args.dir} does not exist or is not a directory")
        return 2

    files = list(find_json_files(args.dir, args.pattern, args.recursive))
    if args.verbose:
        print(f"Found {len(files)} files matching pattern '{args.pattern}' in {args.dir} (recursive={args.recursive})")

    dicts = {}
    for p in sorted(files):
        if args.verbose:
            print(f"Loading {p}")
        try:
            file_name = p.parent.name  # use parent directory name as key
            d = load_json_file(p)
            total_samples = d.get("total_samples", 0)
            if total_samples < 100:
                print(f"Warning: file {p} has only {total_samples} samples, skipping.")
                continue
            dicts[file_name] = d
        except Exception as e:
            print(e)

    combined = merge_dicts(dicts, on_duplicate=args.on_duplicate)

    # Sort keys
    ordered_keys = sort_keys(combined.keys())
    ordered = {k: combined[k] for k in ordered_keys}

    if args.verbose:
        print(f"Writing combined JSON ({len(ordered)} entries) to {args.output}")
    write_json_atomic(ordered, args.output)
    if args.verbose:
        print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
