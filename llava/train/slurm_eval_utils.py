import json
import math
import os
import pathlib
import shlex
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional

import wandb

from llava.utils import rank0_print


class SlurmEvalManager:
    WANDB_EVAL_METRIC_KEYS = (
        "gaze_l2_normalized_mean",
        "inout_accuracy",
        "inout_recall",
        "inout_precision",
        "inout_total_samples",
    )

    def __init__(self, args: Any, is_world_process_zero: Callable[[], bool]):
        self.args = args
        self.is_world_process_zero = is_world_process_zero
        self.repo_root = pathlib.Path(__file__).resolve().parents[2]
        self._wandb_metrics_defined = False

    def marker_dir(self) -> pathlib.Path:
        return pathlib.Path(self.args.output_dir).resolve() / "slurm_eval_jobs"

    def output_root(self) -> pathlib.Path:
        output_root = pathlib.Path(str(getattr(self.args, "slurm_eval_output_root", "") or "./evaluation_results"))
        if output_root.is_absolute():
            return output_root
        return (self.repo_root / output_root).resolve()

    def log_completed_results(self) -> int:
        if not self._enabled_on_rank0():
            return 0
        if not (self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None):
            return 0
        self._define_wandb_metrics()

        output_root = self.output_root()
        if not output_root.is_dir():
            return 0

        logged = self._load_logged_metrics()
        logged_count = 0
        metrics_paths = sorted(output_root.glob("**/metrics.json"), key=lambda path: path.stat().st_mtime)
        for metrics_path in metrics_paths:
            resolved_path = str(metrics_path.resolve())
            if resolved_path in logged:
                continue

            metrics = self._load_metrics(metrics_path)
            if metrics is None:
                continue

            checkpoint_step = self._checkpoint_step_from_metrics(metrics, metrics_path)
            wandb_payload = {}
            if checkpoint_step is not None:
                wandb_payload["eval/checkpoint_step"] = checkpoint_step
            for key in self.WANDB_EVAL_METRIC_KEYS:
                value = metrics.get(key)
                if self._is_scalar(value):
                    wandb_payload[f"eval/{key}"] = value

            try:
                wandb.log(wandb_payload)
            except Exception as exc:  # noqa: BLE001
                rank0_print(f"Warning: failed to log SLURM eval metrics from {metrics_path} to wandb: {exc}")
                continue

            logged[resolved_path] = checkpoint_step
            logged_count += 1
            rank0_print(f"Logged SLURM eval metrics to wandb: {metrics_path}")

        if logged_count:
            self._save_logged_metrics(logged)
        return logged_count

    def _define_wandb_metrics(self) -> None:
        if self._wandb_metrics_defined or wandb.run is None:
            return
        try:
            wandb.define_metric("eval/checkpoint_step")
            wandb.define_metric("eval/*", step_metric="eval/checkpoint_step")
            self._wandb_metrics_defined = True
        except Exception as exc:  # noqa: BLE001
            rank0_print(f"Warning: failed to define SLURM eval wandb metrics: {exc}")

    def submit_for_checkpoint(self, checkpoint_dir: str, step: int) -> None:
        if not self._enabled_on_rank0():
            return

        eval_steps = int(getattr(self.args, "slurm_eval_steps", 0) or 0)
        if eval_steps <= 0 or step <= 0 or step % eval_steps != 0:
            return

        checkpoint_path = pathlib.Path(checkpoint_dir).resolve()
        if not checkpoint_path.is_dir():
            rank0_print(f"Skipping SLURM eval submission: checkpoint not found at {checkpoint_path}")
            return

        marker_dir = self.marker_dir()
        marker_dir.mkdir(parents=True, exist_ok=True)
        marker_path = marker_dir / f"checkpoint-{step}.json"
        if marker_path.exists():
            rank0_print(f"Skipping SLURM eval submission for checkpoint-{step}: marker already exists.")
            return

        script_path = pathlib.Path(getattr(self.args, "slurm_eval_script", ""))
        if not script_path.is_absolute():
            script_path = self.repo_root / script_path
        if not script_path.is_file():
            rank0_print(f"Skipping SLURM eval submission: script not found at {script_path}")
            return

        export_arg = self._build_export_arg(checkpoint_path, step)
        sbatch_args = shlex.split(str(getattr(self.args, "slurm_eval_sbatch_args", "") or ""))
        command = ["sbatch", "--parsable", f"--export={export_arg}", *sbatch_args, str(script_path)]

        rank0_print(f"Submitting SLURM eval for checkpoint-{step}: {' '.join(shlex.quote(part) for part in command)}")
        try:
            result = subprocess.run(command, cwd=str(self.repo_root), capture_output=True, text=True, check=False)
        except FileNotFoundError:
            rank0_print("SLURM eval submission failed: sbatch was not found.")
            return

        if result.returncode != 0:
            rank0_print(
                "SLURM eval submission failed "
                f"(exit={result.returncode}): {(result.stderr or result.stdout).strip()}"
            )
            return

        job_id = result.stdout.strip()
        marker_payload = {
            "checkpoint_step": step,
            "checkpoint_dir": str(checkpoint_path),
            "slurm_job_id": job_id,
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "script": str(script_path),
        }
        with marker_path.open("w", encoding="utf-8") as handle:
            json.dump(marker_payload, handle, indent=2)
        rank0_print(f"Submitted SLURM eval job {job_id} for checkpoint-{step}.")

    def wait_for_results(self, timeout_seconds: int = 3600, poll_interval_seconds: int = 60) -> None:
        if not self._enabled_on_rank0():
            return

        target_step = self._latest_submitted_step()
        if target_step is None:
            self.log_completed_results()
            return

        deadline = time.time() + max(0, int(timeout_seconds))
        poll_interval = max(1, int(poll_interval_seconds))
        rank0_print(f"Waiting up to {timeout_seconds}s for SLURM eval metrics for checkpoint-{target_step}.")
        while True:
            self.log_completed_results()
            logged = self._load_logged_metrics()
            if target_step in {step for step in logged.values() if step is not None}:
                rank0_print(f"Final SLURM eval metrics logged for checkpoint-{target_step}.")
                return
            if time.time() >= deadline:
                rank0_print(f"Timed out waiting for SLURM eval metrics for checkpoint-{target_step}.")
                return
            time.sleep(min(poll_interval, max(1, int(deadline - time.time()))))

    def _enabled_on_rank0(self) -> bool:
        return bool(getattr(self.args, "slurm_eval_enable", False)) and self.is_world_process_zero()

    def _build_export_arg(self, checkpoint_path: pathlib.Path, step: int) -> str:
        run_name = pathlib.Path(self.args.output_dir).resolve().name
        run_tag_base = getattr(self.args, "slurm_eval_run_tag", None) or run_name
        export_values = {
            "ADAPTER_PATH": str(checkpoint_path),
            "OUTPUT_ROOT": str(getattr(self.args, "slurm_eval_output_root", "") or "./evaluation_results"),
            "RUN_TAG": f"{run_tag_base}_ckpt{step}",
            "WANDB_TRAINING_STEP": str(step),
            "TRAINING_SLURM_JOB_ID": os.getenv("SLURM_JOB_ID", ""),
        }

        optional_exports = {
            "MODEL_PATH": getattr(self.args, "slurm_eval_model_path", None),
            "MODEL_BASE": getattr(self.args, "slurm_eval_model_base", None),
            "DATASET_JSON": getattr(self.args, "slurm_eval_dataset_json", None),
            "IMAGES_DIR": getattr(self.args, "slurm_eval_images_dir", None),
            "WANDB_RUN_ID": self._wandb_env_or_run_attr("WANDB_RUN_ID", "id"),
            "WANDB_RUN_NAME": self._wandb_env_or_run_attr("WANDB_RUN_NAME", "name"),
            "WANDB_PROJECT": self._wandb_env_or_run_attr("WANDB_PROJECT", "project"),
            "WANDB_ENTITY": self._wandb_env_or_run_attr("WANDB_ENTITY", "entity"),
        }
        for key, value in optional_exports.items():
            if value:
                export_values[key] = str(value)

        export_arg = "ALL," + ",".join(f"{key}={value}" for key, value in export_values.items())
        extra_export = str(getattr(self.args, "slurm_eval_extra_export", "") or "").strip()
        if extra_export:
            export_arg = f"{export_arg},{extra_export}"
        return export_arg

    def _load_logged_metrics(self) -> Dict[str, Optional[int]]:
        state_path = self.marker_dir() / "logged_eval_metrics.json"
        if not state_path.is_file():
            return {}
        try:
            with state_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            rank0_print(f"Warning: could not read SLURM eval log state {state_path}: {exc}")
            return {}
        if isinstance(payload, list):
            return {str(path): None for path in payload}
        if not isinstance(payload, dict):
            return {}
        logged = payload.get("logged_metrics", payload)
        if not isinstance(logged, dict):
            return {}
        return {
            str(path): entry.get("checkpoint_step") if isinstance(entry, dict) else None
            for path, entry in logged.items()
        }

    def _save_logged_metrics(self, logged: Dict[str, Optional[int]]) -> None:
        marker_dir = self.marker_dir()
        marker_dir.mkdir(parents=True, exist_ok=True)
        state_path = marker_dir / "logged_eval_metrics.json"
        payload = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "logged_metrics": {
                path: {"checkpoint_step": step}
                for path, step in sorted(logged.items())
            },
        }
        with state_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    @staticmethod
    def _load_metrics(metrics_path: pathlib.Path) -> Optional[Dict[str, Any]]:
        try:
            with metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            rank0_print(f"Skipping SLURM eval metrics at {metrics_path}: {exc}")
            return None
        return metrics if isinstance(metrics, dict) else None

    @staticmethod
    def _is_scalar(value: Any) -> bool:
        if isinstance(value, bool):
            return True
        if isinstance(value, (int, float)):
            return math.isfinite(float(value))
        return False

    @staticmethod
    def _checkpoint_step_from_metrics(metrics: Dict[str, Any], metrics_path: pathlib.Path) -> Optional[int]:
        for key in ("training_step", "checkpoint_step"):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str) and value.isdigit():
                return int(value)
        for part in metrics_path.parts:
            if "_ckpt" not in part:
                continue
            suffix = part.rsplit("_ckpt", 1)[-1].split("_", 1)[0]
            if suffix.isdigit():
                return int(suffix)
        return None

    def _latest_submitted_step(self) -> Optional[int]:
        marker_dir = self.marker_dir()
        if not marker_dir.is_dir():
            return None
        steps: List[int] = []
        for marker_path in marker_dir.glob("checkpoint-*.json"):
            payload = self._load_marker(marker_path)
            value = payload.get("checkpoint_step") if isinstance(payload, dict) else None
            if isinstance(value, (int, float)):
                steps.append(int(value))
                continue
            step_text = marker_path.stem.replace("checkpoint-", "", 1)
            if step_text.isdigit():
                steps.append(int(step_text))
        return max(steps) if steps else None

    @staticmethod
    def _load_marker(marker_path: pathlib.Path) -> Dict[str, Any]:
        try:
            with marker_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _wandb_env_or_run_attr(env_name: str, attr_name: str) -> str:
        env_value = os.getenv(env_name, "")
        if env_value:
            return env_value
        if wandb.run is None:
            return ""
        value = getattr(wandb.run, attr_name, "")
        if callable(value):
            value = value()
        return str(value) if value else ""
