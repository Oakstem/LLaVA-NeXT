#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from gazefollow.evals.log_wandb_evaluations import (
    DEFAULT_PROJECT,
    build_run_name_from_adapter,
)
from vacation.recompute_vacation_metrics import (
    _extract_prompt_columns,
    _extract_run_config,
    compute_metrics,
)
from vacation.wandb_utils import infer_wandb_run_id, log_prefixed_metrics_to_wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline Vacation results uploader: scan JSON results and update W&B runs."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Root results directory to scan recursively.",
    )
    parser.add_argument(
        "--wandb-project",
        default=DEFAULT_PROJECT,
        help="Default W&B project when missing in result config.",
    )
    parser.add_argument(
        "--wandb-entity",
        default='gylab',
        help="Optional W&B entity override.",
    )
    parser.add_argument(
        "--wandb-resume",
        type=str,
        default="allow",
        choices=["allow", "must", "never", "auto"],
        help="W&B resume mode when updating an existing run.",
    )
    parser.add_argument(
        "--metric-prefix",
        type=str,
        default="vacation",
        help="Prefix for logged metrics.",
    )
    parser.add_argument(
        "--allow-create",
        action="store_true",
        help="Create a new run if no existing run id can be inferred.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned updates; do not log to W&B.",
    )
    return parser.parse_args()


def iter_result_files(results_dir: Path) -> Iterable[Path]:
    return sorted(path for path in results_dir.rglob("*.json") if path.is_file())


def load_payload(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("results"), list):
        return None
    return payload


def build_metric_row(results_file: Path, payload: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    config = payload.get("config")
    if not isinstance(config, dict):
        config = {}
    wrapped = {"config": config}
    row: Dict[str, Any] = {
        "results_file": str(results_file),
        "adapter_name": results_file.stem,
    }
    row.update(_extract_run_config(wrapped))
    row.update(metrics)
    row.update(_extract_prompt_columns(wrapped))
    return row


def main() -> None:
    args = parse_args()
    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

    candidates = list(iter_result_files(args.results_dir))
    print(f"Discovered {len(candidates)} JSON files under {args.results_dir}")

    processed = 0
    logged = 0
    skipped = 0

    for result_file in candidates:
        payload = load_payload(result_file)
        if payload is None:
            continue

        results = payload.get("results")
        if not isinstance(results, list) or not results:
            continue

        config = payload.get("config")
        if not isinstance(config, dict):
            config = {}

        model_path = config.get("adapter_path") or config.get("model_path")
        if not isinstance(model_path, str) or not model_path.strip():
            print(f"Skipping {result_file}: missing adapter_path/model_path in config.")
            skipped += 1
            continue

        project = args.wandb_project
        if isinstance(config.get("wandb_project"), str) and config.get("wandb_project"):
            project = str(config["wandb_project"])
        entity = args.wandb_entity
        if entity is None and isinstance(config.get("wandb_entity"), str) and config.get("wandb_entity"):
            entity = str(config["wandb_entity"])

        run_name = build_run_name_from_adapter(model_path)
        run_id = None
        if isinstance(config.get("wandb_run_id"), str) and config.get("wandb_run_id"):
            run_id = str(config["wandb_run_id"])
        if run_id is None:
            inferred_id, inferred_entity = infer_wandb_run_id(
                wandb_project=project,
                wandb_entity=entity,
                inferred_run_name=run_name,
                model_path=model_path,
            )
            run_id = inferred_id
            if entity is None and inferred_entity:
                entity = inferred_entity

        if run_id is None and not args.allow_create:
            print(
                f"Skipping {result_file}: could not infer existing run id "
                f"(run_name={run_name}, project={project}, entity={entity or 'auto'})."
            )
            skipped += 1
            continue

        metrics = compute_metrics(results)
        row = build_metric_row(result_file, payload, metrics)
        processed += 1

        print(
            f"Prepared {result_file} -> project={project}, entity={entity or 'default'}, "
            f"run_name={run_name}, run_id={run_id or 'NEW'}"
        )

        if args.dry_run:
            continue

        updated_run_id, error = log_prefixed_metrics_to_wandb(
            wandb_project=project,
            wandb_entity=entity,
            wandb_run_name=run_name,
            wandb_run_id=run_id,
            wandb_resume=args.wandb_resume,
            base_wandb_config={
                "offline_vacation_update": True,
                "source_results_file": str(result_file),
                "model_path": config.get("model_path"),
                "adapter_path": config.get("adapter_path"),
            },
            metric_row=row,
            metric_prefix=args.metric_prefix,
            total_results=metrics.get("total_results", 0),
            extra_config_updates={
                f"{args.metric_prefix}/results_file": str(result_file),
                f"{args.metric_prefix}/offline_update": True,
                f"{args.metric_prefix}/wandb_run_id": run_id,
            },
        )
        if error:
            print(f"Failed to log {result_file}: {error}")
            skipped += 1
            continue
        logged += 1
        print(f"Logged {result_file} to run id={updated_run_id or run_id}")

    print(
        f"\nDone. processed={processed}, logged={logged}, skipped={skipped}, dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
