"""Split combined conversation datasets into train and validation files.

By default, the script splits a single combined JSON file (``--combined-path``). When a
``--datasets-root`` is provided, it iterates over each subdirectory and looks for the
combined JSON relative to that directory. Splits can be derived from a reference
``val.json`` or a ratio-based sample.

Usage examples:
    python split_combined_conversations.py --combined-path sgl_conversation_data.json --val-ratio 0.15
    python split_combined_conversations.py --datasets-root training_datasets/qwen_sets --combined-path combined_conversations.json --reference-val path/to/val.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split combined_conversations.json files into train/val splits."
    )
    parser.add_argument(
        "--combined-path",
        type=Path,
        default=Path("sgl_conversation_data.json"),
        help=(
            "Path to the combined dataset JSON when splitting a single dataset. "
            "If --datasets-root is set and this path is relative, it is resolved inside each dataset directory."
        ),
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=None,
        help="Optional root containing per-run folders. When set, the script iterates each subdirectory and looks for the combined JSON inside it.",
    )
    parser.add_argument(
        "--reference-val",
        type=Path,
        default=None,
        help="Optional path to an existing val.json file used as ground truth for the split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Validation ratio used when reference-val is not provided (e.g., 0.1 for 10%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for the ratio-based split.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=None,
        help="Name of the subdirectory (per dataset) where train/val files will be saved. "
        "Defaults to split_<timestamp>.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of dataset directory names to process. "
        "Defaults to processing every subdirectory in datasets-root.",
    )
    return parser.parse_args()


def load_json_list(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, found {type(data)}")
    return data


def ensure_ids(samples: Sequence[dict]) -> Sequence[dict]:
    for sample in samples:
        if "id" not in sample:
            raise KeyError("Every sample must contain an 'id' field")
    return samples


def ensure_image_ids(samples: Sequence[dict]) -> Sequence[dict]:
    modified = 0
    for sample in samples:
        if "image_id" in sample:
            continue
        image_path = sample.get("image")
        if not image_path:
            raise KeyError("Sample missing 'image_id' and 'image' fields")
        sample["image_id"] = Path(image_path).name
        modified += 1
    if modified > 0:
        print(f"[INFO] Added 'image_id' to {modified} samples based on 'image' field")
    return samples


def build_image_id_map(samples: Sequence[dict]) -> dict:
    image_map = {}
    for sample in samples:
        image_id = sample.get("image_id")
        if image_id is None:
            raise KeyError("Sample missing image_id after ensure_image_ids call")
        image_id = Path(image_id).stem
        image_map.setdefault(image_id, []).append(sample)
    return image_map


def split_from_reference(
    combined_samples: Sequence[dict], reference_val_path: Path
) -> Tuple[List[dict], List[dict]]:
    reference_samples = ensure_ids(load_json_list(reference_val_path))
    combined_map = build_image_id_map(ensure_ids(combined_samples))
    val_samples: List[dict] = []
    missing_ids: List[str] = []
    for ref_sample in reference_samples:
        sample_id = ref_sample["id"]
        matches = combined_map.get(sample_id)
        if not matches:
            missing_ids.append(sample_id)
            continue
        val_samples.extend(matches)
    if missing_ids:
        print(
            f"[WARN] {len(missing_ids)} reference ids were missing in the combined set. "
            f"Examples: {missing_ids[:5]}",
            file=sys.stderr,
        )
    val_image_ids = {sample["image_id"] for sample in val_samples}
    train_samples = [
        sample for sample in combined_samples if sample["image_id"] not in val_image_ids
    ]
    return train_samples, val_samples


def split_with_ratio(
    combined_samples: Sequence[dict], val_ratio: float, seed: int
) -> Tuple[List[dict], List[dict]]:
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError("val-ratio must be between 0 and 1")
    if not combined_samples:
        return [], []
    rng = random.Random(seed)
    indices = list(range(len(combined_samples)))
    rng.shuffle(indices)
    val_count = int(round(len(combined_samples) * val_ratio))
    val_indices = set(indices[:val_count])
    train_samples = [sample for idx, sample in enumerate(combined_samples) if idx not in val_indices]
    val_samples = [sample for idx, sample in enumerate(combined_samples) if idx in val_indices]
    return train_samples, val_samples


def split_and_save(
    combined_path: Path,
    target_dir: Path,
    reference_val: Path | None,
    val_ratio: float | None,
    seed: int,
) -> Tuple[int, int, int]:
    combined_samples = ensure_image_ids(ensure_ids(load_json_list(combined_path)))
    if reference_val:
        train_samples, val_samples = split_from_reference(combined_samples, reference_val)
    else:
        train_samples, val_samples = split_with_ratio(combined_samples, val_ratio, seed)

    target_dir.mkdir(parents=True, exist_ok=True)
    train_path = target_dir / "train.json"
    val_path = target_dir / "val.json"

    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=2)
    with val_path.open("w", encoding="utf-8") as f:
        json.dump(val_samples, f, indent=2)

    metrics_path = target_dir / "split_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
                "total_samples": len(combined_samples),
            },
            f,
            indent=2,
        )
    return len(train_samples), len(val_samples), len(combined_samples)


def iter_dataset_dirs(root: Path, datasets: Iterable[str] | None) -> Iterable[Path]:
    if datasets:
        for dataset_name in datasets:
            yield root / dataset_name
        return
    for path in sorted(root.iterdir()):
        if path.is_dir():
            yield path


def split_datasets_root(
    datasets_root: Path,
    combined_path: Path,
    output_subdir: str,
    datasets: Iterable[str] | None,
    reference_val: Path | None,
    val_ratio: float | None,
    seed: int,
) -> int:
    if not datasets_root.exists():
        raise FileNotFoundError(f"Datasets root {datasets_root} does not exist")

    processed = 0
    for dataset_dir in iter_dataset_dirs(datasets_root, datasets):
        resolved_combined = (
            combined_path if combined_path.is_absolute() else dataset_dir / combined_path
        )
        if not resolved_combined.exists():
            continue

        target_dir = dataset_dir / output_subdir
        train_count, val_count, total = split_and_save(
            resolved_combined,
            target_dir,
            reference_val,
            val_ratio,
            seed,
        )
        print(
            f"[INFO] {dataset_dir.name}: wrote {train_count} train and {val_count} val samples to {target_dir} "
            f"(total {total})"
        )
        processed += 1

    if processed == 0:
        print(
            f"[WARN] No combined files found under {datasets_root} using {combined_path}",
            file=sys.stderr,
        )
    return processed


def main() -> None:
    args = parse_args()
    if args.reference_val is None and args.val_ratio is None:
        raise ValueError("Either --reference-val or --val-ratio must be provided.")

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    combined_stem = args.combined_path.stem
    output_subdir = (
        args.output_subdir if args.output_subdir else f"{combined_stem}_{timestamp}"
    )

    if args.datasets_root:
        split_datasets_root(
            args.datasets_root,
            args.combined_path,
            output_subdir,
            args.datasets,
            args.reference_val,
            args.val_ratio,
            args.seed,
        )
        return

    combined_path = args.combined_path
    if not combined_path.exists():
        raise FileNotFoundError(f"Combined dataset file {combined_path} does not exist")
    target_dir = (
        combined_path.parent / output_subdir if combined_path.parent != Path("") else Path(output_subdir)
    )
    train_count, val_count, total = split_and_save(
        combined_path, target_dir, args.reference_val, args.val_ratio, args.seed
    )
    print(
        f"[INFO] Single dataset: wrote {train_count} train and {val_count} val samples to {target_dir} (total {total})"
    )


if __name__ == "__main__":
    main()
