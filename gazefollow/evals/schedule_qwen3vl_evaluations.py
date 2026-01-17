#!/usr/bin/env python3
"""
Scan evaluation outputs and enqueue Qwen3-VL localization runs for datasets that
still need localization metrics. The scheduler looks for evaluation directories
under ``evaluation_results`` whose name starts with ``eval_2025``, verifies that
their ``model_generation_results.json`` file has more than the requested minimum
number of samples, and skips any directory that already contains a
``dataset_localization_results_inout.json`` artifact. Matching runs are executed
sequentially using ``gazefollow/evaluate_dataset_qwen3vl.py``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

DATASET_RESULTS_NAME = "dataset_localization_results_inout.json"
DATASET_RESULTS_NAME2 = "dataset_localization_results.json"
MODEL_GENERATION_NAME = "model_generation_results.json"


@dataclass(frozen=True)
class EvaluationJob:
    directory: Path
    dataset_json: Path
    sample_count: int
    job_name: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Queue pending Qwen3-VL localization evaluations."
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("evaluation_results"),
        help="Directory that stores evaluation run folders (default: evaluation_results).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/mnt/d/Projects/data/gazefollow"),
        help="Base directory containing dataset images.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Only enqueue datasets with more than this many samples (default: 100).",
    )
    parser.add_argument(
        "--python-exec",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used to spawn the evaluator (default: current interpreter).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of evaluations to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the queued commands without executing them.",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=None,
        help="Additional arguments appended to each evaluator invocation.",
    )
    return parser.parse_args(argv)


def has_existing_localization(directory: Path) -> bool:
    return any(directory.rglob(DATASET_RESULTS_NAME)) or any(directory.rglob(DATASET_RESULTS_NAME2))


def count_samples(dataset_json: Path) -> int:
    with dataset_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        samples = payload.get("samples")
        if isinstance(samples, list):
            return len(samples)
        return len(payload)
    raise ValueError(f"Unexpected content in {dataset_json}")


def collect_jobs(evaluation_root: Path, min_samples: int) -> List[EvaluationJob]:
    jobs: List[EvaluationJob] = []
    if not evaluation_root.is_dir():
        raise FileNotFoundError(f"Evaluation root not found: {evaluation_root}")

    for directory in sorted(evaluation_root.iterdir()):
        if not directory.is_dir() or not directory.name.startswith("eval_2025"):
            continue
        dataset_json = directory / MODEL_GENERATION_NAME
        if not dataset_json.is_file():
            continue
        if has_existing_localization(directory):
            continue
        try:
            sample_count = count_samples(dataset_json)
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[skip] {directory.name}: failed to read {MODEL_GENERATION_NAME} ({exc})")
            continue
        if sample_count <= min_samples:
            continue
        config_file = directory / "evaluation_config.json"
        if not config_file.is_file():
            print(f"[warn] {directory.name}: missing evaluation_config.json")
            continue
        with config_file.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        adapter_path = config.get("adapter_path")
        if adapter_path is not None:
            job_name = "_".join(Path(adapter_path).parts[-2:])
        else:
            job_name = "baseline"
        jobs.append(EvaluationJob(directory=directory, dataset_json=dataset_json, sample_count=sample_count, job_name=job_name))
    return jobs


def build_command(python_exec: Path, evaluator: Path, job: EvaluationJob, images_dir: Path, extra_args: Sequence[str] | None) -> List[str]:
    command = [
        str(python_exec),
        str(evaluator),
        "--dataset-json",
        str(job.dataset_json),
        "--images-dir",
        str(images_dir),
        "--wandb-run-name",
        job.job_name,

    ]
    if extra_args:
        command.extend(extra_args)
    return command


def run_jobs(
    jobs: Iterable[EvaluationJob],
    python_exec: Path,
    evaluator: Path,
    images_dir: Path,
    dry_run: bool,
    extra_args: Sequence[str] | None,
) -> int:
    executed = 0
    failures = 0
    job_list = list(jobs)
    total = len(job_list)

    for index, job in enumerate(job_list, start=1):
        command = build_command(python_exec, evaluator, job, images_dir, extra_args)
        print(f"[{index}/{total}] {job.directory.name} -> {job.dataset_json.name} ({job.sample_count} samples)")
        print("       ", " ".join(command))
        if dry_run:
            continue
        result = subprocess.run(command, check=False)
        executed += 1
        if result.returncode != 0:
            failures += 1
            print(f"[fail] Command returned non-zero exit status ({result.returncode}).")

    if dry_run:
        print(f"Dry run: {total} evaluations queued.")
        return 0

    print(f"Completed {executed - failures}/{executed} evaluations.")
    return 0 if failures == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    evaluator = Path(__file__).resolve().parents[1] / "evaluate_dataset_qwen3vl.py"
    if not evaluator.is_file():
        raise FileNotFoundError(f"Could not locate evaluator script at {evaluator}")

    jobs = collect_jobs(args.evaluation_root, args.min_samples)
    if not jobs:
        print("No evaluation directories matched the scheduling criteria.")
        return 0

    if args.limit is not None:
        jobs = jobs[: args.limit]

    return run_jobs(
        jobs=jobs,
        python_exec=args.python_exec,
        evaluator=evaluator,
        images_dir=args.images_dir,
        dry_run=args.dry_run,
        extra_args=args.extra_args,
    )


if __name__ == "__main__":
    sys.exit(main())
