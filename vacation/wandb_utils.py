from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from gazefollow.evals.log_wandb_evaluations import split_metrics


def _normalize_wandb_path(value: Optional[str]) -> str:
    if not value:
        return ""
    return str(value).replace("\\", "/").rstrip("/")


def infer_wandb_run_id(
    *,
    wandb_project: str,
    wandb_entity: Optional[str],
    inferred_run_name: str,
    model_path: str,
) -> Tuple[Optional[str], Optional[str]]:
    try:
        import wandb as wandb_lib
    except ImportError:
        return None, wandb_entity

    api = wandb_lib.Api()
    viewer = None
    try:
        viewer = api.viewer
    except Exception:
        viewer = None

    entity_candidates: List[str] = []
    for candidate in [
        wandb_entity,
        os.getenv("WANDB_ENTITY"),
        getattr(wandb_lib.api, "default_entity", None),
        getattr(viewer, "entity", None),
    ]:
        if isinstance(candidate, str) and candidate and candidate not in entity_candidates:
            entity_candidates.append(candidate)
    viewer_teams = getattr(viewer, "teams", None)
    if isinstance(viewer_teams, list):
        for team in viewer_teams:
            if isinstance(team, str) and team and team not in entity_candidates:
                entity_candidates.append(team)

    model_path_norm = _normalize_wandb_path(model_path)
    for entity in entity_candidates:
        api_path = f"{entity}/{wandb_project}"
        try:
            runs = list(api.runs(api_path, filters={"display_name": inferred_run_name}))
            if not runs:
                runs = list(api.runs(api_path, filters={"name": inferred_run_name}))
        except Exception:
            continue
        if not runs:
            continue

        exact_config_matches = []
        for run in runs:
            cfg_adapter = _normalize_wandb_path(run.config.get("adapter_path"))
            cfg_model = _normalize_wandb_path(run.config.get("model_path"))
            if model_path_norm and model_path_norm in {cfg_adapter, cfg_model}:
                exact_config_matches.append(run)

        candidates = exact_config_matches or runs
        candidates.sort(key=lambda run: getattr(run, "updated_at", "") or "")
        selected = candidates[-1]
        return str(selected.id), entity

    return None, wandb_entity


def log_prefixed_metrics_to_wandb(
    *,
    wandb_project: str,
    wandb_entity: Optional[str],
    wandb_run_name: str,
    wandb_run_id: Optional[str],
    wandb_resume: str,
    base_wandb_config: Dict[str, Any],
    metric_row: Dict[str, Any],
    metric_prefix: str,
    total_results: int,
    extra_config_updates: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    try:
        import wandb as wandb_lib
    except ImportError as exc:
        return wandb_run_id, f"wandb import failed: {exc}"

    init_kwargs: Dict[str, Any] = {
        "project": wandb_project,
        "config": dict(base_wandb_config),
    }
    if wandb_entity:
        init_kwargs["entity"] = wandb_entity
    if wandb_run_id:
        init_kwargs["id"] = wandb_run_id
        init_kwargs["resume"] = wandb_resume
    else:
        init_kwargs["name"] = wandb_run_name

    run = wandb_lib.init(**init_kwargs)
    try:
        resolved_run_id = str(run.id) if getattr(run, "id", None) else wandb_run_id
        prefixed_row = {f"{metric_prefix}/{key}": value for key, value in metric_row.items()}
        scalar_metrics, non_scalar_metrics = split_metrics(prefixed_row)
        config_updates: Dict[str, Any] = dict(non_scalar_metrics)
        if extra_config_updates:
            config_updates.update(extra_config_updates)
        run.config.update(config_updates, allow_val_change=True)
        run.log(scalar_metrics or {f"{metric_prefix}/_placeholder": total_results})
        return resolved_run_id, None
    except Exception as exc:
        return wandb_run_id, str(exc)
    finally:
        try:
            run.finish()
        except Exception:
            pass
