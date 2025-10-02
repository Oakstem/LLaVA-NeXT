#!/usr/bin/env python3
"""Batch evaluation helper that runs evaluate_model.py across multiple checkpoints."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

DEFAULT_PYTHON = Path.home() / "llava/bin/python"
REPO_ROOT = Path(__file__).resolve().parent
EVALUATE_SCRIPT = REPO_ROOT / "evaluate_model.py"


class EvaluationError(Exception):
    """Raised when an individual checkpoint evaluation fails."""


def parse_arguments() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluate a collection of checkpoints and gather their metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoints-dir", type=Path, required=True, help="Directory containing checkpoint-* subdirectories.")
    parser.add_argument("--dataset-json", type=Path, required=True, help="Path to the evaluation dataset JSON file.")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory containing evaluation images.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("evaluation_results") / "multi_checkpoint",
        help="Directory where per-checkpoint outputs will be stored.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Optional path for the aggregated summary JSON (defaults to <output-root>/summary.json).",
    )
    parser.add_argument("--model-base", type=str, default=None, help="Optional base model path passed to evaluate_model.py.")
    parser.add_argument("--adapter-path", type=str, default=None, help="Optional adapter path passed to evaluate_model.py.")
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help="Optional attention implementation forwarded to evaluate_model.py.",
    )
    parser.add_argument("--load-4bit", action="store_true", help="Forward the --load-4bit flag.")
    parser.add_argument("--load-8bit", action="store_true", help="Forward the --load-8bit flag.")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Forward the --save-predictions flag to keep per-sample predictions.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip checkpoints whose metrics.json already exists in the output directory.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failures and continue with the remaining checkpoints instead of exiting immediately.",
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=DEFAULT_PYTHON,
        help="Python executable used to invoke evaluate_model.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without launching evaluations.",
    )

    args, unknown = parser.parse_known_args()
    return args, unknown


def checkpoint_sort_key(path: Path) -> Tuple[int, Any]:
    name = path.name
    if name.startswith("checkpoint-"):
        step_text = name[len("checkpoint-") :]
        if step_text.isdigit():
            return (0, int(step_text))
        try:
            return (0, float(step_text))
        except ValueError:
            return (0, step_text)
    return (1, name)


def discover_checkpoints(checkpoints_dir: Path) -> List[Path]:
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoints_dir}")
    candidates = [path for path in checkpoints_dir.iterdir() if path.is_dir()]
    filtered = [path for path in candidates if path.name.startswith("checkpoint-")]
    if not filtered:
        raise FileNotFoundError(f"No checkpoint-* subdirectories found under {checkpoints_dir}")
    return sorted(filtered, key=checkpoint_sort_key)


def build_command(args: argparse.Namespace, checkpoint: Path, extra_args: List[str], output_dir: Path) -> List[str]:
    python_executable = args.python_executable
    if not python_executable.exists():
        raise FileNotFoundError(f"Python executable not found: {python_executable}")
    print(f"Using Python executable: {python_executable}")
    cmd = [
        str(python_executable),
        str(EVALUATE_SCRIPT),
        "--adapter-path",
        str(checkpoint.resolve()),
        "--dataset-json",
        str(args.dataset_json.resolve()),
        "--images-dir",
        str(args.images_dir.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--focus-loss-after-looking",
        "--limit", "10"
    ]
    if args.model_base:
        cmd.extend(["--model-base", args.model_base])
    if args.adapter_path:
        cmd.extend(["--adapter-path", args.adapter_path])
    if args.attn_implementation:
        cmd.extend(["--attn-implementation", args.attn_implementation])
    if args.load_4bit:
        cmd.append("--load-4bit")
    if args.load_8bit:
        cmd.append("--load-8bit")
    if args.save_predictions:
        cmd.append("--save-predictions")
    cmd.extend(extra_args)
    return cmd


def run_command(cmd: List[str]) -> None:
    print(f"\nRunning: {' '.join(shlex.quote(part) for part in cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise EvaluationError(f"Command failed with exit code {result.returncode}")


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics.json not found at {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def summarize_results(
    args: argparse.Namespace,
    extra_args: List[str],
    evaluations: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> Path:
    summary_dir = args.output_root
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_file or (summary_dir / "summary.json")
    payload: Dict[str, Any] = {
        "checkpoints_dir": str(args.checkpoints_dir.resolve()),
        "dataset_json": str(args.dataset_json.resolve()),
        "images_dir": str(args.images_dir.resolve()),
        "evaluate_script": str(EVALUATE_SCRIPT),
        "python_executable": str(args.python_executable.expanduser().resolve()),
        "forwarded_args": extra_args,
        "evaluations": evaluations,
    }
    if args.model_base:
        payload["model_base"] = args.model_base
    if args.adapter_path:
        payload["adapter_path"] = args.adapter_path
    if args.attn_implementation:
        payload["attn_implementation"] = args.attn_implementation
    if args.load_4bit:
        payload["load_4bit"] = True
    if args.load_8bit:
        payload["load_8bit"] = True
    if args.save_predictions:
        payload["save_predictions"] = True
    if failures:
        payload["failures"] = failures

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return summary_path


def main() -> int:
    args, extra_args = parse_arguments()

    checkpoints = discover_checkpoints(args.checkpoints_dir.resolve())
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    print(f"Discovered {len(checkpoints)} checkpoints under {args.checkpoints_dir}.")
    evaluations: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for checkpoint in checkpoints:
        output_dir = args.output_root / checkpoint.name
        metrics_path = output_dir / "metrics.json"
        if args.skip_existing and metrics_path.is_file():
            print(f"Skipping {checkpoint.name}: metrics.json already exists.")
            try:
                metrics = load_metrics(metrics_path)
            except Exception as exc:
                failures.append({
                    "checkpoint": str(checkpoint.resolve()),
                    "reason": f"Failed to load existing metrics: {exc}",
                })
                if not args.continue_on_error:
                    break
                continue
            evaluations.append(
                {
                    "checkpoint": checkpoint.name,
                    "checkpoint_path": str(checkpoint.resolve()),
                    "output_dir": str(output_dir.resolve()),
                    "metrics_file": str(metrics_path.resolve()),
                    "metrics": metrics,
                }
            )
            continue

        cmd = build_command(args, checkpoint, extra_args, output_dir)
        print(f"\nRunning command: {' '.join(shlex.quote(part) for part in cmd)}")
        if args.dry_run:
            print(f"DRY RUN: {' '.join(shlex.quote(part) for part in cmd)}")
            continue

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            run_command(cmd)
            metrics = load_metrics(metrics_path)
            evaluations.append(
                {
                    "checkpoint": checkpoint.name,
                    "checkpoint_path": str(checkpoint.resolve()),
                    "output_dir": str(output_dir.resolve()),
                    "metrics_file": str(metrics_path.resolve()),
                    "metrics": metrics,
                }
            )
        except EvaluationError as exc:
            failures.append({
                "checkpoint": str(checkpoint.resolve()),
                "reason": str(exc),
            })
            if not args.continue_on_error:
                break
        except Exception as exc:
            failures.append({
                "checkpoint": str(checkpoint.resolve()),
                "reason": f"Post-processing error: {exc}",
            })
            if not args.continue_on_error:
                break

    if args.dry_run:
        print("Dry run completed without executing evaluations.")
        return 0

    summary_path = summarize_results(args, extra_args, evaluations, failures)
    print(f"\nSummary written to {summary_path}")

    if failures:
        print(f"Encountered {len(failures)} failure(s). See summary for details.")
        return 1 if not args.continue_on_error else 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
