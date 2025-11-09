from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, TypedDict

try:  # Allow execution both as a module and via direct script invocation.
    from .gaze_metrics import compute_gaze_errors, persist_ground_truth_updates
    from .data_proc.add_in_out_labels import iter_normalized_keys
except ImportError:  # pragma: no cover - fallback for CLI execution.
    from gaze_metrics import compute_gaze_errors, persist_ground_truth_updates
    from data_proc.add_in_out_labels import iter_normalized_keys


class FailureRecord(TypedDict):
    id: Any
    reason: str
    image: str


class DatasetUpdateRecord(TypedDict):
    id: Any
    image: str
    relative_path: str
    values: Dict[str, float]


class PersonLevelRecord(TypedDict, total=False):
    sample_id: Any
    person_id: str
    person_description: str
    gaze_target: Optional[str]
    gaze_score: Optional[float]
    person_score: Optional[float]
    gaze_l2_error: Optional[float]
    gaze_normalized_l2_error: Optional[float]
    gaze_angular_error: Optional[float]
    gaze_iou: Optional[float]
    gaze_modified_l2_error: Optional[float]


@dataclass
class EvaluationConfig:
    dataset_json: Path
    images_dir: Path
    output_dir: Path
    limit: Optional[int]
    gaze_model_id: str
    gaze_box_threshold: float
    gaze_text_threshold: float
    gaze_iou_radius_ratio: float
    gaze_device: Optional[str]
    max_new_tokens: int
    save_records: bool
    log_to_wandb: bool
    wandb_project: str
    wandb_entity: Optional[str]
    wandb_run_name: Optional[str]
    wandb_run_name_suffix: str
    use_gpt_gaze_targets: bool
    in_out_labels_csv: Optional[Path]

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "EvaluationConfig":
        if args.output_dir is None:
            dataset_stem = Path(args.dataset_json).parent.stem
            dataset_dir = Path(args.dataset_json).parent
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            args.output_dir = f"{dataset_dir}/qwen3vl_grounding_run_{dataset_stem}_{timestamp}"
        return cls(
            dataset_json=Path(args.dataset_json),
            images_dir=Path(args.images_dir),
            output_dir=Path(args.output_dir),
            limit=args.limit,
            gaze_model_id=args.gaze_model_id,
            gaze_box_threshold=args.gaze_box_threshold,
            gaze_text_threshold=args.gaze_text_threshold,
            gaze_iou_radius_ratio=args.gaze_iou_radius_ratio,
            gaze_device=args.gaze_device,
            max_new_tokens=args.max_new_tokens,
            save_records=args.save_records,
            log_to_wandb=args.log_to_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
            wandb_run_name_suffix=args.wandb_run_name_suffix,
            use_gpt_gaze_targets=args.use_gpt_gaze_targets,
            in_out_labels_csv=Path(args.in_out_labels_csv) if args.in_out_labels_csv else None,
        )

    def resolved_device_map(self) -> str:
        return self.gaze_device or "auto"


@dataclass
class EvaluationResults:
    metrics: Dict[str, Any]
    sample_records: List[Dict[str, Any]]
    person_records: List[PersonLevelRecord]
    failed_samples: List[FailureRecord]
    dataset_gt_updates: List[DatasetUpdateRecord]
    dataset_updated: bool


@dataclass
class PersistedPaths:
    metrics_path: Path
    records_path: Optional[Path]
    person_path: Optional[Path]
    failed_path: Optional[Path]


def build_qwen_query(
    gaze_target: str
) -> str:
    """Construct a prompt for Qwen3-VL grounding."""
    return (
        f"Locate the {gaze_target}. "
        "Return a JSON array with a single object following the schema: "
        "[{\"bbox1\": [x1, y1, x2, y2], \"score\": confidence}]. "
        # f"Reference: description={description}; gaze_target={gaze_json}; person_id={person_id_json}."
    )


def extract_bbox(detection: Dict[str, Any]) -> Optional[List[int]]:
    """Extract an integer bounding box from a generic detection mapping."""
    for key, value in detection.items():
        if not isinstance(key, str):
            continue
        if "bbox" not in key.lower():
            continue
        if isinstance(value, Sequence) and len(value) == 4:
            try:
                return [int(float(coord)) for coord in value]
            except (TypeError, ValueError):
                return None
    return None


_OUTSIDE_PATTERNS = (
    re.compile(r"\bout\s+of\s+(?:frame|image|view|screen)\b"),
    re.compile(r"\bout\s+of\s+(?:shot|picture|photo)\b"),
    re.compile(r"\bout-of-(?:frame|view)\b"),
    re.compile(r"\boff[-\s]?camera\b"),
    re.compile(r"\boff[-\s]?screen\b"),
    re.compile(r"\boffscreen\b"),
    re.compile(r"\boutside\b"),
    re.compile(r"\boutside\s+(?:of\s+)?the\s+(?:frame|image|photo|picture)\b"),
    re.compile(r"\bout\s+the\s+(?:frame|image)\b"),
    re.compile(r"\bnot\s+in\s+(?:frame|image|view)\b"),
)

_GENERIC_ONLY_PHRASES = {
    "someone",
    "somebody",
    "somewhere",
    "something",
    "unknown",
    "none",
    "nothing",
    "n/a",
    "n.a.",
    "unsure",
}


def infer_in_out_from_phrase(phrase: Optional[str]) -> Optional[int]:
    """
    Heuristically infer whether a gaze target phrase refers to an in-frame (1) or out-of-frame (0) target.

    Returns 0 if the phrase appears to reference an out-of-frame target, 1 if it appears in-frame,
    and None if no determination can be made (e.g., empty phrase).
    """
    if phrase is None:
        return None

    normalized = phrase.strip().lower()
    if not normalized:
        return None

    for pattern in _OUTSIDE_PATTERNS:
        if pattern.search(normalized):
            return 0

    cleaned = re.sub(r"[\s\W]+", " ", normalized).strip()
    if not cleaned:
        return None

    if cleaned in _GENERIC_ONLY_PHRASES or any(
        cleaned.startswith(prefix) for prefix in ("someone or something", "something or someone")
    ):
        return 0

    return 1


def extract_score(detection: Dict[str, Any]) -> Optional[float]:
    """Extract a numeric confidence score from a detection mapping."""
    for key in ("score", "confidence", "prob", "probability"):
        value = detection.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def coerce_in_out_value(value: Any) -> Optional[int]:
    """Normalize free-form in/out annotations to {0, 1}."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return 1 if value >= 1 else 0
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return 1 if value >= 0.5 else 0
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        lowered = stripped.lower()
        if lowered in {"1", "in", "inside", "inframe", "in-frame", "in_frame"}:
            return 1
        if lowered in {"0", "out", "outside", "outframe", "out-frame", "out_frame", "offscreen", "off-screen"}:
            return 0
        try:
            numeric = float(stripped)
        except ValueError:
            return None
        if math.isnan(numeric):
            return None
        return 1 if numeric >= 0.5 else 0
    return None


_IN_OUT_LOOKUP_FIELDS: Tuple[str, ...] = (
    "id",
    "image",
    "image_path",
    "relative_path",
    "image_relative_path",
)


def collect_in_out_lookup_keys(sample: Mapping[str, Any]) -> List[str]:
    """Generate normalized lookup keys for matching external in/out labels."""
    keys: List[str] = []
    seen: Set[str] = set()

    def _extend(value: Any) -> None:
        for key in iter_normalized_keys(value):
            if key not in seen:
                seen.add(key)
                keys.append(key)

    for field in _IN_OUT_LOOKUP_FIELDS:
        _extend(sample.get(field))

    metadata = sample.get("metadata")
    if isinstance(metadata, Mapping):
        for field in _IN_OUT_LOOKUP_FIELDS:
            _extend(metadata.get(field))

    return keys


def resolve_in_out_label(
    sample: Mapping[str, Any],
    lookup: Optional[Mapping[str, Any]],
    descriptions: Optional[Iterable[Any]],
) -> Optional[int]:
    """
    Resolve an in/out flag for a dataset sample using embedded annotations,
    an external lookup table, or gaze phrase heuristics.
    """
    value = coerce_in_out_value(sample.get("in_out"))

    if value is None and lookup:
        for key in collect_in_out_lookup_keys(sample):
            mapped = lookup.get(key)
            normalized = coerce_in_out_value(mapped)
            if normalized is not None:
                value = normalized
                break

    if value is None and descriptions:
        fallback: Optional[int] = None
        for entity in descriptions:
            inferred = infer_in_out_from_phrase(getattr(entity, "gaze_target", None))
            if inferred is not None:
                if inferred == 0:
                    return 0
                fallback = inferred
        if fallback is not None:
            value = fallback

    return value


def choose_best_detection_by_error(
    detections: List[Dict[str, Any]],
    ground_truth_point: Tuple[float, float],
    image_width: int,
    image_height: int,
    score_threshold: float,
    iou_radius_ratio: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Select the detection with the lowest modified L2 error."""
    best_detection: Optional[Dict[str, Any]] = None
    best_errors: Optional[Dict[str, Any]] = None
    best_modified_l2 = float("inf")

    for detection in detections:
        bbox = extract_bbox(detection)
        if not bbox:
            continue

        score = extract_score(detection)
        if score is not None and score < score_threshold:
            continue

        errors = compute_gaze_errors(
            predicted_box=bbox,
            person_box=None,
            ground_truth_point=ground_truth_point,
            image_width=image_width,
            image_height=image_height,
            iou_radius_ratio=iou_radius_ratio,
        )

        modified_l2 = errors.get("gaze_modified_l2_error")
        if modified_l2 is not None and modified_l2 < best_modified_l2:
            best_modified_l2 = modified_l2
            best_detection = detection
            best_errors = errors

    return best_detection, best_errors


def create_wandb_run(config: EvaluationConfig, metrics: Dict[str, Any]) -> Optional[Any]:
    """Initialize a wandb run if logging is enabled and wandb is available."""
    if not config.log_to_wandb:
        return None

    wandb_spec = importlib.util.find_spec("wandb")
    if wandb_spec is None:
        print(
            "Warning: wandb is not available. Disable logging with --no-log-to-wandb to silence this message."
        )
        return None

    wandb = importlib.import_module("wandb")
    dataset_stem = config.dataset_json.stem
    model_stem = config.gaze_model_id.split("/")[-1]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if config.wandb_run_name:
        run_name = config.wandb_run_name
    else:
        run_name = f"dataset-eval-{dataset_stem}-{model_stem}-{timestamp}"
        if config.wandb_run_name_suffix:
            run_name = f"{run_name}-{config.wandb_run_name_suffix}"

    config_payload = {
        "dataset_json": str(config.dataset_json),
        "images_dir": str(config.images_dir),
        "gaze_model_id": config.gaze_model_id,
        "gaze_box_threshold": config.gaze_box_threshold,
        "gaze_iou_radius_ratio": config.gaze_iou_radius_ratio,
        "max_new_tokens": config.max_new_tokens,
        "limit": config.limit,
        "device_map": config.resolved_device_map(),
        "use_gpt_gaze_targets": config.use_gpt_gaze_targets,
    }
    config_payload.update(
        {f"default_metrics/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
    )

    entity = config.wandb_entity
    project = config.wandb_project or "llava-dataset-eval"
    wandb_run = wandb.init(project=project, entity=entity, name=run_name, config=config_payload, reinit=True)
    print(f"Initialized wandb run: {wandb_run.name}")
    return wandb_run


def persist_evaluation_results(config: EvaluationConfig, results: EvaluationResults) -> PersistedPaths:
    """Write evaluation artifacts to disk and propagate ground-truth updates."""
    if results.dataset_updated and results.dataset_gt_updates:
        persisted = persist_ground_truth_updates(config.dataset_json, results.dataset_gt_updates)
        if persisted:
            print(f"Persisted gaze ground truth for {persisted} samples to {config.dataset_json}")

    metrics_path = config.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(results.metrics, metrics_file, indent=2, ensure_ascii=False)

    records_path: Optional[Path] = None
    person_path: Optional[Path] = None
    failed_path: Optional[Path] = None

    if config.save_records:
        records_path = config.output_dir / "dataset_localization_results.json"
        with records_path.open("w", encoding="utf-8") as results_file:
            json.dump(results.sample_records, results_file, indent=2, ensure_ascii=False)

        person_path = config.output_dir / "person_level_results.json"
        with person_path.open("w", encoding="utf-8") as person_file:
            json.dump(results.person_records, person_file, indent=2, ensure_ascii=False)

    if results.failed_samples:
        failed_path = config.output_dir / "failures.json"
        with failed_path.open("w", encoding="utf-8") as failed_file:
            json.dump(results.failed_samples, failed_file, indent=2, ensure_ascii=False)
    print(f"Persisted evaluation results to {config.output_dir}")
    return PersistedPaths(
        metrics_path=metrics_path,
        records_path=records_path,
        person_path=person_path,
        failed_path=failed_path,
    )


def log_results_to_wandb(
    config: EvaluationConfig,
    results: EvaluationResults,
    paths: PersistedPaths,
) -> None:
    """Upload metrics and artifacts to wandb."""
    wandb_run = create_wandb_run(config, results.metrics)
    if wandb_run is None:
        return

    wandb = importlib.import_module("wandb")
    wandb_metrics = {f"dataset_eval/{k}": v for k, v in results.metrics.items() if isinstance(v, (int, float))}
    wandb.log(wandb_metrics)

    if results.person_records:
        table = wandb.Table(
            columns=[
                "sample_id",
                "person_id",
                "person_description",
                "gaze_target",
                "gaze_score",
                "person_score",
                "gaze_l2_error",
                "gaze_normalized_l2_error",
                "gaze_angular_error",
                "gaze_iou",
                "gaze_modified_l2_error",
            ]
        )
        for record in results.person_records:
            table.add_data(
                record.get("sample_id"),
                record.get("person_id"),
                record.get("person_description"),
                record.get("gaze_target"),
                record.get("gaze_score"),
                record.get("person_score"),
                record.get("gaze_l2_error"),
                record.get("gaze_normalized_l2_error"),
                record.get("gaze_angular_error"),
                record.get("gaze_iou"),
                record.get("gaze_modified_l2_error"),
            )
        wandb.log({"dataset_eval/person_localization": table})

    artifact = wandb.Artifact(
        name=f"dataset_evaluation_{wandb_run.id}",
        type="dataset-evaluation",
        description="Ground-truth localization evaluation artifacts (Qwen3-VL).",
    )
    artifact.add_file(str(paths.metrics_path), name="metrics.json")
    if config.save_records and paths.records_path and paths.person_path:
        artifact.add_file(str(paths.records_path), name="dataset_localization_results.json")
        artifact.add_file(str(paths.person_path), name="person_level_results.json")
    if results.failed_samples and paths.failed_path:
        artifact.add_file(str(paths.failed_path), name="failures.json")
    wandb.log_artifact(artifact)
    wandb.finish()


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Pretty-print evaluation metrics to stdout."""
    print("\nEvaluation complete. Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
