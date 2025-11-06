#!/usr/bin/env python3
"""
Evaluate trained LLaVA models on custom JSON datasets using Qwen3-VL for gaze grounding.

This script mirrors evaluate_model.py but replaces the GroundingDINO-based gaze detector
with the Qwen3-VL vision-language model to localize gaze targets mentioned in model outputs.
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
from PIL import Image
from tqdm import tqdm

# Ensure project root is importable before local dependencies
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gazefollow.auto_phrase_grounding.detect_gaze_targets import (
    normalize_gaze_target_text,
    normalize_person_description,
    parse_person_descriptions,
)
from gazefollow.auto_phrase_grounding.qwen3vl_grounding import (
    load_qwen3vl_model,
    run_qwen3vl_grounding,
)
from gazefollow.data_proc.add_in_out_labels import load_in_out_lookup
from gazefollow.data_proc.normalize_outside_ground_truth import contains_outside_phrase
from gazefollow.gaze_metrics import (
    compute_gaze_errors,
    ensure_ground_truth_gaze,
    load_combined_description_cache,
    persist_ground_truth_updates,
)
from gazefollow.qwen3vl_utils import (
    build_qwen_query,
    extract_bbox,
    extract_score,
    resolve_in_out_label,
)

from generation_utils import (
    enable_inference_optimizations,
    load_model_and_setup,
    fix_wsl_paths,
    load_image,
)
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    IGNORE_INDEX,
)
from llava.conversation import conv_templates

from log_wandb_evaluations import (
    DEFAULT_PROJECT,
    GENERATION_TABLE_COLUMNS,
    build_run_name_from_adapter,
    format_generation_sample,
    split_metrics,
)


@dataclass
class EvaluationSampleOutput:
    index: int
    sample: Dict[str, Any]
    sample_id: str
    dataset_prompt: str
    prompt_used: str
    prompt_source: str
    ground_truth: str
    prediction: Optional[str]
    image_path: str
    image_size: Tuple[int, int]
    loss: Optional[float]
    predicted_in_out: Optional[int] = None


@dataclass
class GazeEvaluationState:
    predictions_output: List[Dict[str, Any]] = field(default_factory=list)
    model_generation_records: List[Dict[str, Any]] = field(default_factory=list)
    missing_in_out_samples: Set[str] = field(default_factory=set)
    dataset_gt_updates: List[Dict[str, Any]] = field(default_factory=list)
    dataset_updated: bool = False
    combined_cache: Optional[Dict[str, Dict[str, str]]] = None
    gaze_l2_errors: List[float] = field(default_factory=list)
    gaze_normalized_l2_errors: List[float] = field(default_factory=list)
    gaze_angular_errors: List[float] = field(default_factory=list)
    gaze_iou_scores: List[float] = field(default_factory=list)
    gaze_modified_l2_errors: List[float] = field(default_factory=list)
    generation_rows_buffer: List[Dict[str, Any]] = field(default_factory=list)


class JsonConversationDataset:
    """Minimal dataset wrapper to satisfy training-time evaluation helpers."""

    def __init__(self, samples: List[Dict[str, Any]], image_root: Path, data_path: Optional[Path] = None):
        self.list_data_dict = samples
        self.data_args = SimpleNamespace(image_folder=str(image_root))
        self.data_path = str(data_path) if data_path is not None else None

    def __len__(self) -> int:
        return len(self.list_data_dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained LLaVA model on custom JSON datasets using Qwen3-VL gaze grounding.")
    
    # Model arguments
    parser.add_argument("--model-path", default="lmms-lab/llava-onevision-qwen2-7b-ov-chat", help="Path to trained model or checkpoint to evaluate.")
    parser.add_argument("--model-base", default=None, help="Optional base model path when loading LoRA adapters.")
    parser.add_argument("--adapter-path", default=None, help="Optional LoRA adapter path to merge at inference time.")
    parser.add_argument("--attn-implementation", default="sdpa", help="Attention implementation (e.g. 'sdpa', 'flash_attention_2').")
    parser.add_argument("--load-4bit", action="store_true", help="Load model with 4-bit quantization.")
    parser.add_argument("--load-8bit", action="store_true", help="Load model with 8-bit quantization.")
    
    # Dataset arguments
    parser.add_argument("--dataset-json", required=True, help="Path to JSON dataset file.")
    parser.add_argument("--images-dir", required=True, help="Directory containing the images.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to evaluate (for testing).")
    
    # Generation arguments
    parser.add_argument("--conv-template", default=None, help="Conversation template key.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling value.")
    parser.add_argument("--num-beams", type=int, default=1, help="Number of beams for beam search.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling instead of greedy decoding.")
    
    # Output arguments
    parser.add_argument("--output-dir", default="./evaluation_results", help="Directory to save evaluation results.")
    parser.add_argument("--save-predictions", action="store_true", help="Save individual predictions to JSON.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation (currently only 1 supported).")
    parser.add_argument(
        "--table-log-interval",
        type=int,
        default=25,
        help="Number of samples between generation table streaming logs (<=0 disables periodic streaming).",
    )
    parser.add_argument(
        "--table-log-file",
        type=str,
        default=None,
        help="Optional JSONL file path for streaming generation table rows (defaults to <output_dir>/generation_progress.jsonl).",
    )
    
    # Other arguments
    parser.add_argument(
        "--generate-model-results",
        action="store_true",
        help="Generate full model outputs and gaze analysis per sample.",
    )
    parser.add_argument(
        "--prompt-override",
        type=str,
        default=None,
        help="Override the prompt used for model generation instead of the dataset-provided human conversation.",
    )
    parser.add_argument(
        "--gaze-model-id",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Qwen3-VL model identifier to use for gaze target detection.",
    )
    parser.add_argument(
        "--gaze-box-threshold",
        type=float,
        default=0.2,
        help="Minimum confidence score accepted for Qwen3-VL detections.",
    )
    parser.add_argument(
        "--gaze-text-threshold",
        type=float,
        default=0.20,
        help="Unused placeholder retained for CLI compatibility.",
    )
    parser.add_argument(
        "--gaze-device",
        type=str,
        default=None,
        help="Device map for the Qwen3-VL detector (defaults to 'auto').",
    )
    parser.add_argument(
        "--gaze-max-new-tokens",
        type=int,
        default=192,
        help="Maximum number of tokens generated by Qwen3-VL during grounding.",
    )
    parser.add_argument(
        "--gaze-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for Qwen3-VL grounding (0.0 for greedy decoding).",
    )
    parser.add_argument(
        "--gaze-iou-radius-ratio",
        type=float,
        default=0.05,
        help="Radius ratio used when computing IoU-based gaze metrics.",
    )
    parser.add_argument(
        "--in-out-labels-csv",
        type=str,
        default="gazefollow/data/combined_description_results.csv",
        help="Optional CSV file that provides precomputed in/out labels (via add_in_out_labels.py).",
    )
    parser.add_argument("--disable-optimizations", action="store_true", help="Skip enabling CUDA optimizations.")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print detailed progress information.")
    parser.add_argument("--safe-mode", action="store_true", help="Enable safe mode with more aggressive memory cleanup and smaller batches.")
    parser.add_argument("--no-loss", action="store_true", help="Disable loss calculation entirely to avoid CUDA errors.")
    parser.add_argument(
        "--log-to-wandb",
        action="store_true",
        default=True,
        help="Log evaluation results to Weights & Biases using the shared logging helper.",
    )
    parser.add_argument(
        "--wandb-project",
        default=DEFAULT_PROJECT,
        help="Weights & Biases project name to use when logging evaluations.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Optional Weights & Biases entity/organization name.",
    )
    parser.add_argument(
        "--focus-loss-after-looking",
        action="store_true",
        help="Compute loss only on tokens after the 'looking at' phrase and clamp missing cases to a threshold.",
    )
    parser.add_argument(
        "--focus-loss-threshold",
        type=float,
        default=5.0,
        help="Maximum loss value to apply when the focus phrase is not found.",
    )
    parser.add_argument(
        "--focus-loss-phrase",
        type=str,
        default="looking at",
        help="Target phrase used to locate the start of focused loss computation.",
    )
    parser.add_argument(
        "--use-iterative-generation",
        action="store_true",
        default=True,
        help="Use iterative token-by-token generation method instead of standard model.generate().",
    )

    return parser.parse_args()


def load_dataset(json_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load and validate the JSON dataset."""
    print(f"Loading dataset from: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must contain a list of samples.")
    
    if limit:
        data = data[:limit]
        print(f"Limited dataset to {limit} samples.")
    
    print(f"Loaded {len(data)} samples from dataset.")
    return data


def determine_template(model_name: str, override: Optional[str]) -> str:
    """Determine conversation template based on model name or override."""
    if override:
        if override not in conv_templates:
            available = ", ".join(sorted(conv_templates.keys()))
            raise ValueError(f"Conversation template '{override}' not found. Available: {available}")
        return override

    lowered = model_name.lower()
    if "qwen" in lowered:
        return "qwen_1_5"
    if "vicuna" in lowered:
        return "vicuna_v1"
    if "mpt" in lowered:
        return "mpt"
    return "qwen_1_5"  # Default fallback


def build_focus_phrase_token_ids(tokenizer, phrase: str) -> List[List[int]]:
    """Generate candidate token id sequences for the focus phrase."""
    if not phrase:
        return []

    stripped = phrase.strip()
    candidates = {phrase}
    if stripped:
        candidates.add(stripped)
        candidates.add(stripped.lower())
        candidates.add(stripped.capitalize())
        candidates.add(f" {stripped}")
        candidates.add(f" {stripped.lower()}")

    token_sequences: List[List[int]] = []
    seen: set = set()
    for candidate in candidates:
        if not candidate:
            continue
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if not ids:
            continue
        key = tuple(ids)
        if key not in seen:
            seen.add(key)
            token_sequences.append(ids)

    return token_sequences


def infer_predicted_in_out(prediction_text: Optional[str]) -> Optional[int]:
    """Infer an in/out flag from the model prediction using outside-phrase heuristics."""
    if prediction_text is None:
        return None
    stripped = prediction_text.strip()
    if not stripped:
        return None
    return 0 if contains_outside_phrase(stripped) else 1


def select_best_qwen_detection(
    detections: Iterable[Dict[str, Any]],
    score_threshold: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Return the highest-scoring detection that satisfies the score threshold."""
    detections_list = list(detections)
    best_detection: Optional[Dict[str, Any]] = None
    best_score = float("-inf")

    for detection in detections_list:
        score = extract_score(detection)
        if score is None:
            continue
        if score_threshold is not None and score < score_threshold:
            continue
        if score > best_score:
            best_detection = detection
            best_score = score

    if best_detection is not None:
        return best_detection

    return detections_list[0] if detections_list else None


def process_sample_with_qwen_grounding(
    output: EvaluationSampleOutput,
    *,
    args: argparse.Namespace,
    images_dir: Path,
    state: GazeEvaluationState,
    in_out_lookup: Optional[Dict[str, Any]],
    ensure_gaze_resources: Callable[[], Tuple[Any, Any]],
    generate_model_results: bool,
    gaze_device_map: str,
) -> None:
    """Process a single sample output, adding gaze grounding metadata and metrics."""

    sample = output.sample
    sample_id = output.sample_id
    full_image_path = Path(output.image_path)
    image_width_px, image_height_px = map(int, output.image_size)
    dataset_prompt = output.dataset_prompt
    prompt_used = output.prompt_used
    prediction_text = output.prediction or ""
    ground_truth = output.ground_truth
    predicted_in_out = output.predicted_in_out
    if predicted_in_out is None:
        predicted_in_out = infer_predicted_in_out(prediction_text)

    try:
        relative_image_path = str(full_image_path.relative_to(images_dir))
    except ValueError:
        relative_image_path = str(full_image_path)
    relative_image_path = relative_image_path.replace("\\", "/")

    ground_truth_descriptions = parse_person_descriptions(ground_truth)
    if not ground_truth_descriptions and ground_truth:
        ground_truth_descriptions = parse_person_descriptions(f"Person 1: {ground_truth}")

    in_out_value = resolve_in_out_label(sample, in_out_lookup, ground_truth_descriptions)
    if in_out_value is None and (args.in_out_labels_csv or sample.get("in_out") is not None):
        state.missing_in_out_samples.add(str(sample_id))

    mapping_ref = state.combined_cache if state.combined_cache is not None else {}
    ground_truth_gaze, gt_updated = ensure_ground_truth_gaze(
        sample,
        relative_image_path,
        image_width_px,
        image_height_px,
        mapping_ref,
    )
    if ground_truth_gaze is None and state.combined_cache is None:
        state.combined_cache = load_combined_description_cache()
        mapping_ref = state.combined_cache or {}
        ground_truth_gaze, gt_updated = ensure_ground_truth_gaze(
            sample,
            relative_image_path,
            image_width_px,
            image_height_px,
            mapping_ref,
        )
    if gt_updated and ground_truth_gaze:
        state.dataset_updated = True
        state.dataset_gt_updates.append(
            {
                "id": sample.get("id"),
                "image": sample.get("image"),
                "relative_path": relative_image_path,
                "values": {
                    "gaze_gt_x": ground_truth_gaze[0],
                    "gaze_gt_y": ground_truth_gaze[1],
                    "gaze_gt_width": ground_truth_gaze[2],
                    "gaze_gt_height": ground_truth_gaze[3],
                },
            }
        )

    ground_truth_point: Optional[Tuple[float, float]] = None
    if ground_truth_gaze:
        gt_x, gt_y, _, _ = ground_truth_gaze
        ground_truth_point = (gt_x, gt_y)

    prediction_entry: Dict[str, Any] = {
        "id": sample_id,
        "image_path": str(full_image_path),
        "prompt": prompt_used,
        "ground_truth": ground_truth,
        "prompt_source": output.prompt_source,
        "success": True,
    }
    if dataset_prompt and dataset_prompt != prompt_used:
        prediction_entry["dataset_prompt"] = dataset_prompt
    if prediction_text:
        prediction_entry["prediction"] = prediction_text
    if output.loss is not None:
        prediction_entry["loss"] = output.loss
    if predicted_in_out is not None:
        prediction_entry["predicted_in_out"] = predicted_in_out
    if ground_truth_point is not None:
        prediction_entry["gaze_ground_truth"] = {
            "x": ground_truth_point[0],
            "y": ground_truth_point[1],
            "image_width": image_width_px,
            "image_height": image_height_px,
        }
    if in_out_value is not None:
        prediction_entry["in_out"] = in_out_value
    state.predictions_output.append(prediction_entry)

    if generate_model_results:
        processed_people: Dict[str, Any] = {}
        person_descriptions = parse_person_descriptions(prediction_text)
        if not person_descriptions and prediction_text:
            person_descriptions = parse_person_descriptions(f"Person 1: {prediction_text}")

        processor = qwen_model = None
        if person_descriptions:
            processor, qwen_model = ensure_gaze_resources()

        for person in person_descriptions:
            sanitized_description = normalize_person_description(person.raw_description)
            description_for_query = sanitized_description or person.raw_description.strip()
            normalized_gaze_target = normalize_gaze_target_text(person.gaze_target)
            query = build_qwen_query(
                person.label,
                description_for_query,
                normalized_gaze_target or "",
                person.person_id,
            )

            detections: List[Dict[str, Any]] = []
            raw_response = ""
            if processor is not None and qwen_model is not None:
                try:
                    detections, raw_response = run_qwen3vl_grounding(
                        image_path=str(full_image_path),
                        query=query,
                        model_id=args.gaze_model_id,
                        max_new_tokens=args.gaze_max_new_tokens,
                        processor=processor,
                        model=qwen_model,
                        device_map=gaze_device_map,
                        temperature=args.gaze_temperature,
                    )
                except Exception as exc:  # noqa: BLE001
                    detections = []
                    raw_response = ""
                    if args.verbose:
                        print(f"[Qwen3-VL] Detection failed for {sample_id}/{person.person_id}: {exc}")

            best_detection = select_best_qwen_detection(detections, args.gaze_box_threshold)
            best_bbox = extract_bbox(best_detection) if best_detection else None
            best_score = extract_score(best_detection) if best_detection else None

            person_entry: Dict[str, Any] = {
                "label": person.label,
                "person_description": description_for_query,
                "gaze_target": normalized_gaze_target,
                "query": query,
                "detections": detections,
            }
            if raw_response:
                person_entry["raw_response"] = raw_response
            if best_detection:
                person_entry["best_detection"] = best_detection
            if best_bbox is not None:
                person_entry["gaze_coordinates"] = [float(coord) for coord in best_bbox]
            if best_score is not None:
                person_entry["gaze_score"] = float(best_score)

            if ground_truth_point is not None:
                person_entry["gaze_ground_truth"] = {
                    "x": ground_truth_point[0],
                    "y": ground_truth_point[1],
                }
                predicted_box = person_entry.get("gaze_coordinates")
                if predicted_box is not None:
                    errors = compute_gaze_errors(
                        predicted_box=predicted_box,
                        person_box=None,
                        ground_truth_point=ground_truth_point,
                        image_width=image_width_px,
                        image_height=image_height_px,
                        iou_radius_ratio=args.gaze_iou_radius_ratio,
                    )
                    if in_out_value == 0:
                        sanitized_errors = {key: None for key in errors.keys()}
                    else:
                        sanitized_errors = errors
                    person_entry.update(sanitized_errors)
                    if in_out_value != 0:
                        if sanitized_errors.get("gaze_l2_error") is not None:
                            state.gaze_l2_errors.append(sanitized_errors["gaze_l2_error"])
                        if sanitized_errors.get("gaze_normalized_l2_error") is not None:
                            state.gaze_normalized_l2_errors.append(sanitized_errors["gaze_normalized_l2_error"])
                        if sanitized_errors.get("gaze_angular_error") is not None:
                            state.gaze_angular_errors.append(sanitized_errors["gaze_angular_error"])
                        if sanitized_errors.get("gaze_iou") is not None:
                            state.gaze_iou_scores.append(sanitized_errors["gaze_iou"])
                        if sanitized_errors.get("gaze_modified_l2_error") is not None:
                            state.gaze_modified_l2_errors.append(sanitized_errors["gaze_modified_l2_error"])

            processed_people[person.person_id] = person_entry

        model_entry: Dict[str, Any] = {
            "id": sample_id,
            "image_path": str(full_image_path),
            "dataset_prompt": dataset_prompt,
            "prompt_used": prompt_used,
            "prompt_source": output.prompt_source,
            "ground_truth": ground_truth,
            "model_prediction": prediction_text,
            "gaze_detections": processed_people,
        }
        if output.loss is not None:
            model_entry["loss"] = output.loss
        if predicted_in_out is not None:
            model_entry["predicted_in_out"] = predicted_in_out
        if ground_truth_point is not None:
            model_entry["gaze_ground_truth"] = {
                "x": ground_truth_point[0],
                "y": ground_truth_point[1],
                "image_width": image_width_px,
                "image_height": image_height_px,
            }
        if in_out_value is not None:
            model_entry["in_out"] = in_out_value
        state.model_generation_records.append(model_entry)
        new_rows = format_generation_sample(model_entry)
        if new_rows:
            state.generation_rows_buffer.extend(new_rows)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_image_tensor(
    image_path: str, image_processor, model
) -> Optional[Tuple[torch.Tensor, Tuple[int, int], Image.Image]]:
    """Prepare image tensor, returning the tensor, original size, and PIL image object."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pil_image = load_image(image_path)
    processed = process_images([pil_image], image_processor, model.config)

    if isinstance(processed, tuple):
        image_tensor = processed[0]
    else:
        image_tensor = processed

    if isinstance(image_tensor, list):
        if not image_tensor:
            return None
        image_tensor = image_tensor[0]

    if isinstance(image_tensor, torch.Tensor) and image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    if not isinstance(image_tensor, torch.Tensor):
        return None

    image_tensor = image_tensor.to(model.device, dtype=model.dtype)
    return image_tensor, pil_image.size, pil_image


def compute_ground_truth_loss(
    prompt_text: str,
    ground_truth: str,
    tokenizer,
    model,
    conv_template: str,
    image_tensor: torch.Tensor,
    image_size: Tuple[int, int],
    focus_loss_after_phrase: bool = False,
    focus_loss_phrase_token_ids: Optional[List[List[int]]] = None,
    focus_loss_missing_value: Optional[float] = None,
) -> Optional[float]:
    """Teacher-force the ground truth response to compute language modeling loss."""
    conv = conv_templates[conv_template].copy()
    conv.tokenizer = tokenizer

    if DEFAULT_IMAGE_TOKEN not in prompt_text:
        user_content = f"{DEFAULT_IMAGE_TOKEN}\n{prompt_text}"
    else:
        user_content = prompt_text

    conv.append_message(conv.roles[0], user_content)
    conv.append_message(conv.roles[1], ground_truth)
    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        full_prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)

    labels = input_ids.clone()

    prompt_conv = conv_templates[conv_template].copy()
    prompt_conv.tokenizer = tokenizer
    prompt_conv.append_message(prompt_conv.roles[0], user_content)
    prompt_conv.append_message(prompt_conv.roles[1], None)
    prompt_only_ids = tokenizer_image_token(
        prompt_conv.get_prompt(),
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )

    prompt_length = prompt_only_ids.size(-1)
    labels = labels.to(model.device)
    invalid_label_mask = labels < 0
    if invalid_label_mask.any():
        labels = labels.masked_fill(invalid_label_mask, IGNORE_INDEX)

    if prompt_length > 0:
        labels[:, :prompt_length] = IGNORE_INDEX

    vocab_size = getattr(model.config, "vocab_size", None)
    if vocab_size is not None:
        overflow_mask = labels >= vocab_size
        if overflow_mask.any():
            labels = labels.masked_fill(overflow_mask, IGNORE_INDEX)
    model_kwargs = {
        "input_ids": input_ids,
        "labels": labels,
        "images": image_tensor,
        "image_sizes": [list(image_size)],
        "modalities": ["image"],
        "use_cache": False,
        "return_dict": True,
    }

    if focus_loss_after_phrase:
        if not focus_loss_phrase_token_ids:
            raise ValueError("Focus loss is enabled but no phrase token ids were provided.")
        model_kwargs.update(
            {
                "focus_loss_after_phrase": True,
                "focus_loss_phrase_token_ids": focus_loss_phrase_token_ids,
            }
        )
        if focus_loss_missing_value is not None:
            model_kwargs["focus_loss_missing_value"] = focus_loss_missing_value

    with torch.no_grad():
        outputs = model(**model_kwargs)

    loss_tensor = getattr(outputs, "loss", None)
    if loss_tensor is None:
        return None

    return loss_tensor.detach().to("cpu", dtype=torch.float32).item()


def generate_response(
    prompt_text: str,
    image_tensor: torch.Tensor,
    image_size: Tuple[int, int],
    tokenizer,
    model,
    conv_template: str,
    generation_kwargs: dict,
    return_loss_data: bool = False,
) -> Union[str, Tuple[str, torch.Tensor, torch.Tensor]]:
    """Generate response for a single prompt using iterative token generation similar to extract_attention script."""
    
    # Create conversation
    conv = conv_templates[conv_template].copy()
    conv.tokenizer = tokenizer
    
    # Add image token if not present
    if DEFAULT_IMAGE_TOKEN not in prompt_text:
        user_content = f"{DEFAULT_IMAGE_TOKEN}\n{prompt_text}"
    else:
        user_content = prompt_text
    
    conv.append_message(conv.roles[0], user_content)
    conv.append_message(conv.roles[1], None)
    prompt_for_tokenizer = conv.get_prompt()
    
    # Tokenize
    input_ids = tokenizer_image_token(
        prompt_for_tokenizer,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)
    
    # Initialize generation state
    max_new_tokens = generation_kwargs.get("max_new_tokens", 512)
    current_input_ids = input_ids
    past_key_values = None
    generated_tokens = []
    
    # Get EOS token ID
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]
    
    # Prepare image sizes as [width, height] for model
    image_sizes = [list(image_size)]
    
    # Store logits and labels for loss calculation if requested
    all_logits = [] if return_loss_data else None
    all_labels = [] if return_loss_data else None
    
    # Generation loop - similar to extract_attention script
    for i in range(max_new_tokens):
        with torch.inference_mode():
            # Prepare model inputs
            model_inputs = {
                "input_ids": current_input_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
                "output_hidden_states": generation_kwargs.get("output_hidden_states", False),
                "output_attentions": generation_kwargs.get("output_attentions", False),
            }
            
            # Only include images on first step
            if i == 0:
                model_inputs.update({
                    "images": image_tensor,
                    "image_sizes": image_sizes,
                    "modalities": ["image"]
                })
            
            # Generate next token
            outputs = model(**model_inputs)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Store logits for loss calculation if requested
            if return_loss_data:
                all_logits.append(outputs.logits.cpu())
            
            # Apply generation strategy (sampling vs greedy)
            if generation_kwargs.get("do_sample", False):
                # Apply top-k filtering if specified
                if generation_kwargs.get("top_k") is not None and generation_kwargs.get("top_k") > 0:
                    top_k = generation_kwargs["top_k"]
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                    filtered_logits = torch.full_like(next_token_logits, float('-inf'))
                    filtered_logits.scatter_(-1, top_k_indices, top_k_logits)
                    next_token_logits = filtered_logits
                
                # Apply temperature scaling
                if generation_kwargs.get("temperature", 1.0) != 1.0:
                    next_token_logits = next_token_logits / generation_kwargs["temperature"]
                
                # Apply top-p (nucleus) sampling if specified
                if generation_kwargs.get("top_p") is not None and generation_kwargs.get("top_p") < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > generation_kwargs["top_p"]
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Set logits to -inf for tokens to remove
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[0, indices_to_remove] = float('-inf')
                
                # Sample from the distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                if len(probs.shape) == 3:
                    next_token_id = torch.multinomial(probs[0], 1)
                else:
                    next_token_id = torch.multinomial(probs, 1)
            else:
                # Greedy decoding - use argmax without keepdim to match extract_attention script
                next_token_id = torch.argmax(next_token_logits, dim=-1)
            
            # Check for EOS token
            if next_token_id.item() == eos_token_id:
                break
            
            # Add token to generated sequence
            generated_tokens.append(next_token_id.item())
            
            # Store labels for loss calculation if requested
            if return_loss_data:
                all_labels.append(next_token_id.cpu())
            
            # Update for next iteration - ensure proper dimensions like in extract_attention script
            current_input_ids = next_token_id.view(1, -1)
                
            if hasattr(outputs, 'past_key_values'):
                past_key_values = outputs.past_key_values
    
    # Decode the generated tokens
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    if return_loss_data and all_logits and all_labels:
        # Concatenate logits and labels for loss calculation
        logits_tensor = torch.cat(all_logits, dim=1)  # Shape: [batch_size, seq_len, vocab_size]
        labels_tensor = torch.cat(all_labels, dim=0).unsqueeze(0)  # Shape: [batch_size, seq_len]
        return response, logits_tensor, labels_tensor
    
    return response


def extract_prompt_from_conversation(conversations: List[Dict[str, str]]) -> str:
    """Extract the human prompt from conversation format."""
    for conv in conversations:
        if conv.get("from") == "human":
            # Remove image token from prompt for processing
            prompt = conv["value"].replace("<image>", "").strip()
            return prompt
    return ""


def extract_ground_truth_from_conversation(conversations: List[Dict[str, str]]) -> str:
    """Extract the ground truth response from conversation format."""
    for conv in conversations:
        if conv.get("from") == "gpt":
            return conv["value"].strip()
    return ""


def calculate_basic_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """Calculate basic evaluation metrics."""
    metrics = {}
    
    # Exact match
    exact_matches = sum(1 for pred, gt in zip(predictions, ground_truths) if pred.strip() == gt.strip())
    metrics["exact_match"] = exact_matches / len(predictions) if predictions else 0.0
    
    # Average response length
    avg_pred_length = sum(len(pred.split()) for pred in predictions) / len(predictions) if predictions else 0.0
    avg_gt_length = sum(len(gt.split()) for gt in ground_truths) / len(ground_truths) if ground_truths else 0.0
    
    metrics["avg_prediction_length"] = avg_pred_length
    metrics["avg_ground_truth_length"] = avg_gt_length
    
    return metrics


def evaluate_dataset_for_training(
    model,
    tokenizer,
    image_processor,
    eval_dataset,
    conv_template: str = "qwen_1_5",
    max_new_tokens: int = 512,
    focus_loss_after_looking: bool = False,
    focus_loss_phrase: str = "looking at",
    focus_loss_threshold: float = 5.0,
    no_loss: bool = False,
    verbose: bool = False,
    limit: Optional[int] = None,
    prompt_override: Optional[str] = None,
    generation_params: Optional[Dict[str, Any]] = None,
    return_outputs: bool = False,
    qwen_grounding: Optional[Callable[[EvaluationSampleOutput], None]] = None,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], List[EvaluationSampleOutput], List[Dict[str, Any]]]]:
    """
    Custom evaluation function for training-time evaluation.

    This function is designed to be called during training to evaluate the model
    on a validation dataset using the same logic as evaluate_model.py.

    Args:
        model: The model to evaluate.
        tokenizer: Tokenizer.
        image_processor: Image processor.
        eval_dataset: Evaluation dataset (LazySupervisedDataset or compatible wrapper).
        conv_template: Conversation template to use.
        max_new_tokens: Maximum tokens to generate.
        focus_loss_after_looking: Whether to focus loss on tokens after phrase.
        focus_loss_phrase: Target phrase for focused loss.
        focus_loss_threshold: Maximum loss when focus phrase not found.
        no_loss: Disable loss calculation.
        verbose: Print detailed progress.
        limit: Optional limit on number of samples to evaluate (for testing).
        prompt_override: Optional prompt that replaces dataset prompts.
        generation_params: Optional overrides for generation keyword arguments.
        return_outputs: When True, return per-sample outputs and failure records.
        qwen_grounding: Optional callable invoked with each successful sample output as it is generated.

    Returns:
        metrics dict if return_outputs is False. Otherwise a tuple of:
        (metrics dict, successful sample outputs, failed sample metadata).
    """
    from tqdm import tqdm

    model.eval()
    predictions: List[str] = []
    ground_truths: List[str] = []
    losses: List[float] = []
    failed_samples: List[Dict[str, Any]] = []
    successful_outputs: List[EvaluationSampleOutput] = []

    # Build focus phrase token IDs if needed
    focus_phrase_token_ids: List[List[int]] = []
    if focus_loss_after_looking:
        focus_phrase_token_ids = build_focus_phrase_token_ids(tokenizer, focus_loss_phrase)

    # Generation kwargs
    generation_kwargs: Dict[str, Any] = dict(generation_params or {})
    generation_kwargs.setdefault("max_new_tokens", max_new_tokens)
    generation_kwargs.setdefault("do_sample", False)
    generation_kwargs.setdefault("use_cache", True)
    generation_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id or tokenizer.eos_token_id)
    if not generation_kwargs.get("do_sample", False):
        generation_kwargs.pop("temperature", None)
        generation_kwargs.pop("top_p", None)
        generation_kwargs.pop("top_k", None)

    # Dataset range
    total_samples = len(eval_dataset)
    if limit is not None and limit > 0:
        total_samples = min(total_samples, limit)
        if verbose:
            print(f"Running custom evaluation on {total_samples} samples (limited from {len(eval_dataset)})...")
    elif verbose:
        print(f"Running custom evaluation on {total_samples} samples...")

    if verbose:
        image_folder = getattr(getattr(eval_dataset, "data_args", None), "image_folder", None)
        if image_folder:
            print(f"Image folder: {image_folder}")
        else:
            print("Warning: No image_folder found in eval_dataset.data_args")

    with torch.no_grad():
        iterator = tqdm(range(total_samples), desc="Evaluating") if verbose else range(total_samples)
        for idx in iterator:
            sample = eval_dataset.list_data_dict[idx]
            sample_id = str(sample.get("id", idx))

            if verbose and hasattr(iterator, "set_description"):
                iterator.set_description(f"Evaluating {idx + 1}/{total_samples}")
            elif not verbose:
                print(f"[Status] Processing sample {idx + 1}/{total_samples} (id={sample_id})")
                sys.stdout.flush()

            conversations = sample.get("conversations", [])
            dataset_prompt = extract_prompt_from_conversation(conversations)
            ground_truth = extract_ground_truth_from_conversation(conversations)

            if not ground_truth:
                failed_samples.append({"index": idx, "id": sample_id, "reason": "Missing ground truth"})
                continue

            prompt_used = prompt_override if prompt_override is not None else dataset_prompt
            prompt_source = "override" if prompt_override is not None else "dataset"

            if not prompt_used:
                failed_samples.append(
                    {
                        "index": idx,
                        "id": sample_id,
                        "reason": "Missing prompt",
                        "dataset_prompt": dataset_prompt,
                        "prompt_override": prompt_override,
                    }
                )
                continue

            image_file = sample.get("image", "")
            if isinstance(image_file, list):
                image_file = image_file[0]

            if not os.path.isabs(image_file):
                image_folder = getattr(getattr(eval_dataset, "data_args", None), "image_folder", None)
                if image_folder:
                    image_file = os.path.join(image_folder, image_file)
                else:
                    data_path = getattr(eval_dataset, "data_path", None)
                    if data_path:
                        base_dir = Path(data_path).parent
                        image_file = str(base_dir / image_file)

            if not image_file.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")):
                image_file += ".jpg"

            if not Path(image_file).exists():
                if verbose:
                    print(f"Image not found: {image_file}")
                failed_samples.append({"index": idx, "id": sample_id, "reason": f"Image not found: {image_file}"})
                continue

            image_result = prepare_image_tensor(image_file, image_processor, model)

            if image_result is None:
                failed_samples.append({"index": idx, "id": sample_id, "reason": f"Failed to load image: {image_file}"})
                continue

            image_tensor, image_size, pil_image = image_result

            prediction = generate_response(
                prompt_text=prompt_used,
                image_tensor=image_tensor,
                image_size=image_size,
                tokenizer=tokenizer,
                model=model,
                conv_template=conv_template,
                generation_kwargs=generation_kwargs,
                return_loss_data=False,
            )

            predictions.append(prediction)
            ground_truths.append(ground_truth)
            predicted_in_out = infer_predicted_in_out(prediction)

            sample_loss: Optional[float] = None
            if not no_loss:
                loss = compute_ground_truth_loss(
                    prompt_text=prompt_used,
                    ground_truth=ground_truth,
                    tokenizer=tokenizer,
                    model=model,
                    conv_template=conv_template,
                    image_tensor=image_tensor,
                    image_size=image_size,
                    focus_loss_after_phrase=focus_loss_after_looking,
                    focus_loss_phrase_token_ids=focus_phrase_token_ids or None,
                    focus_loss_missing_value=focus_loss_threshold,
                )
                if loss is not None and math.isfinite(loss):
                    losses.append(loss)
                    sample_loss = loss

            if return_outputs or qwen_grounding is not None:
                sample_output = EvaluationSampleOutput(
                    index=idx,
                    sample=sample,
                    sample_id=sample_id,
                    dataset_prompt=dataset_prompt,
                    prompt_used=prompt_used,
                    prompt_source=prompt_source,
                    ground_truth=ground_truth,
                    prediction=prediction,
                    image_path=image_file,
                    image_size=image_size,
                    loss=sample_loss,
                    predicted_in_out=predicted_in_out,
                )
                if return_outputs:
                    successful_outputs.append(sample_output)
                if qwen_grounding is not None:
                    qwen_grounding(sample_output)

            if isinstance(pil_image, Image.Image):
                pil_image.close()

            if (idx + 1) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    metrics: Dict[str, Any] = {}

    if predictions:
        basic_metrics = calculate_basic_metrics(predictions, ground_truths)
        metrics.update(basic_metrics)

    if losses:
        metrics.update(
            {
                "eval_loss": sum(losses) / len(losses),
                "eval_min_loss": min(losses),
                "eval_max_loss": max(losses),
            }
        )

    metrics.update(
        {
            "eval_samples": total_samples,
            "eval_successful": len(predictions),
            "eval_failed": len(failed_samples),
            "eval_success_rate": len(predictions) / total_samples if total_samples > 0 else 0.0,
        }
    )

    model.train()

    if return_outputs:
        return metrics, successful_outputs, failed_samples

    return metrics


def build_generation_kwargs(args: argparse.Namespace, tokenizer) -> Dict[str, Any]:
    """Construct generation kwargs shared across evaluation utilities."""
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if args.do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
    if args.num_beams and args.num_beams > 1:
        generation_kwargs["num_beams"] = args.num_beams
    if hasattr(args, "top_k") and args.top_k is not None:
        generation_kwargs["top_k"] = args.top_k
    return generation_kwargs



def main():
    args = parse_args()

    if args.load_4bit and args.load_8bit:
        raise ValueError("Cannot enable both --load-4bit and --load-8bit.")

    if torch.cuda.is_available():
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        if args.verbose:
            print("Set CUDA_LAUNCH_BLOCKING=1 for better error reporting")

    if not args.disable_optimizations:
        enable_inference_optimizations()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table_log_interval = max(args.table_log_interval, 0)
    progress_log_path = Path(args.table_log_file) if args.table_log_file else output_dir / "generation_progress.jsonl"
    table_logging_streamed = False
    wandb_module = None
    wandb_run = None
    wandb_disabled_reason: Optional[str] = None

    print("=" * 60)
    print("LLaVA Model Evaluation")
    print("=" * 60)

    print("\n1. Loading model...")
    tokenizer, model, image_processor, _ = load_model_and_setup(
        model_path=args.model_path,
        attn_implementation=args.attn_implementation,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        model_base=args.model_base,
        adapter_path=fix_wsl_paths(args.adapter_path) if args.adapter_path else None,
    )

    model_name_source = getattr(model.config, "_name_or_path", args.model_path)
    model_name = get_model_name_from_path(model_name_source)
    conv_template = determine_template(model_name, args.conv_template)
    print(f"Using conversation template: {conv_template}")

    generation_kwargs = build_generation_kwargs(args, tokenizer)
    wandb_run_name = build_run_name_from_adapter(args.adapter_path or args.model_path)
    base_wandb_config: Dict[str, Any] = {
        "model_path": args.model_path,
        "model_base": args.model_base,
        "adapter_path": args.adapter_path,
        "dataset_json": args.dataset_json,
        "images_dir": args.images_dir,
        "conv_template": conv_template,
        "generation_kwargs": generation_kwargs,
        "limit": args.limit,
        "table_log_interval": table_log_interval,
        "table_log_file": str(progress_log_path),
    }
    if args.in_out_labels_csv:
        base_wandb_config["in_out_labels_csv"] = args.in_out_labels_csv

    if args.focus_loss_after_looking:
        focus_phrase_token_ids = build_focus_phrase_token_ids(tokenizer, args.focus_loss_phrase)
        if not focus_phrase_token_ids:
            print(
                "Warning: focus loss requested but tokenizer produced no token ids for the target phrase. "
                "Falling back to standard loss."
            )
        else:
            model.config.focus_loss_after_phrase = True
            model.config.focus_loss_phrase_token_ids = focus_phrase_token_ids
            model.config.focus_loss_missing_value = args.focus_loss_threshold

    print("\n2. Loading dataset...")
    dataset_samples = load_dataset(args.dataset_json, args.limit)
    images_dir = Path(args.images_dir)
    eval_dataset = JsonConversationDataset(dataset_samples, images_dir, Path(args.dataset_json))

    generate_model_results = args.generate_model_results or args.prompt_override is not None
    if generate_model_results:
        progress_log_path.parent.mkdir(parents=True, exist_ok=True)
        progress_log_path.unlink(missing_ok=True)

    in_out_lookup: Optional[Dict[str, Any]] = None
    if args.in_out_labels_csv:
        in_out_lookup = load_in_out_lookup(Path(args.in_out_labels_csv))
        if args.verbose:
            print(f"Loaded in/out labels from {args.in_out_labels_csv} (entries={len(in_out_lookup)})")


    gaze_device_map = args.gaze_device or "auto"
    gaze_processor = None
    gaze_model = None

    def ensure_gaze_resources() -> Tuple[Any, Any]:
        nonlocal gaze_processor, gaze_model
        if gaze_processor is None or gaze_model is None:
            if args.verbose:
                print(f"Loading Qwen3-VL model ({args.gaze_model_id}) with device map {gaze_device_map}...")
            try:
                gaze_processor, gaze_model = load_qwen3vl_model(
                    args.gaze_model_id,
                    device_map=gaze_device_map,
                )
            except RuntimeError as exc:
                print(f"\n[Qwen3-VL] {exc}")
                print(
                    "Set --gaze-model-id to an alternative (e.g. Qwen/Qwen2-VL-7B-Instruct) "
                    "after updating transformers, or upgrade the environment and retry."
                )
                raise
        return gaze_processor, gaze_model

    evaluation_state = GazeEvaluationState()

    def ensure_wandb_run() -> Optional[Any]:
        nonlocal wandb_module, wandb_run, wandb_disabled_reason
        if not args.log_to_wandb:
            if wandb_disabled_reason is None:
                wandb_disabled_reason = "wandb logging disabled"
            return None
        if wandb_disabled_reason is not None:
            return None
        if wandb_module is None:
            try:
                import wandb as wandb_lib
            except ImportError as exc:
                wandb_disabled_reason = f"wandb import failed: {exc}"
                print(f"\n⚠️  wandb logging skipped: {exc}")
                return None
            wandb_module = wandb_lib
        if wandb_run is None:
            init_config = dict(base_wandb_config)
            init_config["evaluation_dir"] = str(output_dir)
            init_kwargs: Dict[str, Any] = {
                "project": args.wandb_project,
                "name": wandb_run_name,
                "config": init_config,
            }
            if args.wandb_entity:
                init_kwargs["entity"] = args.wandb_entity
            wandb_run = wandb_module.init(**init_kwargs)
        return wandb_run

    def flush_generation_rows(force: bool = False) -> None:
        nonlocal table_logging_streamed
        buffer = evaluation_state.generation_rows_buffer
        if not buffer:
            return
        if not force and (table_log_interval <= 0 or len(buffer) < table_log_interval):
            return

        rows_to_log = list(buffer)
        buffer.clear()

        if generate_model_results:
            progress_log_path.parent.mkdir(parents=True, exist_ok=True)
            with progress_log_path.open("a", encoding="utf-8") as handle:
                for row in rows_to_log:
                    handle.write(json.dumps(row, ensure_ascii=False))
                    handle.write("\n")

        wandb_run_instance = ensure_wandb_run()
        if wandb_run_instance is not None:
            table = wandb_module.Table(columns=GENERATION_TABLE_COLUMNS)
            for row in rows_to_log:
                table.add_data(*(row.get(column) for column in GENERATION_TABLE_COLUMNS))
            wandb_run_instance.log({"generation_results": table}, commit=False)
            table_logging_streamed = True

    def handle_sample_output(sample_output: EvaluationSampleOutput) -> None:
        process_sample_with_qwen_grounding(
            sample_output,
            args=args,
            images_dir=images_dir,
            state=evaluation_state,
            in_out_lookup=in_out_lookup,
            ensure_gaze_resources=ensure_gaze_resources,
            generate_model_results=generate_model_results,
            gaze_device_map=gaze_device_map,
        )
        if generate_model_results:
            flush_generation_rows()

    print("\n3. Running evaluation...")
    start_time = time.time()
    metrics_result, sample_outputs, failed_samples = evaluate_dataset_for_training(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        eval_dataset=eval_dataset,
        conv_template=conv_template,
        max_new_tokens=args.max_new_tokens,
        focus_loss_after_looking=args.focus_loss_after_looking,
        focus_loss_phrase=args.focus_loss_phrase,
        focus_loss_threshold=args.focus_loss_threshold,
        no_loss=args.no_loss,
        verbose=args.verbose,
        limit=None,
        prompt_override=args.prompt_override,
        generation_params=generation_kwargs,
        return_outputs=True,
        qwen_grounding=handle_sample_output,
    )
    evaluation_time = time.time() - start_time

    predictions_output = evaluation_state.predictions_output
    model_generation_records = evaluation_state.model_generation_records
    missing_in_out_samples = evaluation_state.missing_in_out_samples
    dataset_gt_updates = evaluation_state.dataset_gt_updates
    dataset_updated = evaluation_state.dataset_updated
    gaze_l2_errors = evaluation_state.gaze_l2_errors
    gaze_normalized_l2_errors = evaluation_state.gaze_normalized_l2_errors
    gaze_angular_errors = evaluation_state.gaze_angular_errors
    gaze_iou_scores = evaluation_state.gaze_iou_scores
    gaze_modified_l2_errors = evaluation_state.gaze_modified_l2_errors

    total_samples = len(dataset_samples)
    processed_samples = len(sample_outputs)
    successful_predictions = sum(1 for output in sample_outputs if output.prediction is not None)

    final_metrics = dict(metrics_result)
    final_metrics.update(
        {
            "total_samples": total_samples,
            "successfully_processed_samples": processed_samples,
            "successful_predictions": successful_predictions,
            "failed_samples": len(failed_samples),
            "success_rate": successful_predictions / total_samples if total_samples else 0.0,
            "processing_rate": processed_samples / total_samples if total_samples else 0.0,
            "evaluation_time_seconds": evaluation_time,
            "average_time_per_sample": evaluation_time / processed_samples if processed_samples else 0.0,
        }
    )

    loss_values = [output.loss for output in sample_outputs if output.loss is not None]
    if loss_values:
        final_metrics.update(
            {
                "average_loss": sum(loss_values) / len(loss_values),
                "min_loss": min(loss_values),
                "max_loss": max(loss_values),
                "samples_with_loss": len(loss_values),
            }
        )

    if generate_model_results:
        final_metrics["model_generation_samples"] = len(model_generation_records)

    if gaze_l2_errors:
        final_metrics["gaze_l2_error_mean"] = sum(gaze_l2_errors) / len(gaze_l2_errors)
        final_metrics["gaze_l2_error_median"] = statistics.median(gaze_l2_errors)
        final_metrics["gaze_l2_error_count"] = len(gaze_l2_errors)
    if gaze_normalized_l2_errors:
        final_metrics["gaze_l2_normalized_mean"] = sum(gaze_normalized_l2_errors) / len(gaze_normalized_l2_errors)
        final_metrics["gaze_l2_normalized_median"] = statistics.median(gaze_normalized_l2_errors)
        final_metrics["gaze_l2_normalized_count"] = len(gaze_normalized_l2_errors)
    if gaze_angular_errors:
        final_metrics["gaze_angular_error_mean"] = sum(gaze_angular_errors) / len(gaze_angular_errors)
        final_metrics["gaze_angular_error_median"] = statistics.median(gaze_angular_errors)
        final_metrics["gaze_angular_error_count"] = len(gaze_angular_errors)
    if gaze_iou_scores:
        final_metrics["gaze_iou_mean"] = sum(gaze_iou_scores) / len(gaze_iou_scores)
        final_metrics["gaze_iou_median"] = statistics.median(gaze_iou_scores)
        final_metrics["gaze_iou_count"] = len(gaze_iou_scores)
    if gaze_modified_l2_errors:
        final_metrics["gaze_modified_l2_mean"] = sum(gaze_modified_l2_errors) / len(gaze_modified_l2_errors)
        final_metrics["gaze_modified_l2_median"] = statistics.median(gaze_modified_l2_errors)
        final_metrics["gaze_modified_l2_count"] = len(gaze_modified_l2_errors)

    print("\n4. Calculating metrics...")
    print("\nEvaluation summary:")
    print(f"  Total samples: {total_samples}")
    print(f"  Processed samples: {processed_samples}")
    print(f"  Successful predictions: {successful_predictions}")
    print(f"  Failed samples: {len(failed_samples)}")
    print(f"  Success rate: {final_metrics['success_rate']:.4f}")
    print(f"  Evaluation time (s): {final_metrics['evaluation_time_seconds']:.2f}")
    if loss_values:
        print(f"  Average loss: {final_metrics['average_loss']:.6f}")
        print(f"  Loss range: {min(loss_values):.6f} - {max(loss_values):.6f}")
    if gaze_l2_errors:
        print(f"  Gaze L2 mean: {final_metrics['gaze_l2_error_mean']:.4f}")

    if dataset_updated and dataset_gt_updates:
        try:
            persisted = persist_ground_truth_updates(Path(args.dataset_json), dataset_gt_updates)
            if persisted:
                print(f"Persisted gaze ground truth for {persisted} samples to {args.dataset_json}")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Failed to update dataset with gaze ground truth: {exc}")

    print(f"\n5. Saving results to {output_dir}...")

    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to: {metrics_file}")

    predictions_file = output_dir / "predictions.json"
    if args.save_predictions:
        with open(predictions_file, "w", encoding="utf-8") as f:
            json.dump(predictions_output, f, indent=2, ensure_ascii=False)
        print(f"Detailed predictions saved to: {predictions_file}")

    model_generation_file: Optional[Path] = None
    if generate_model_results:
        model_generation_file = output_dir / "model_generation_results.json"
        with open(model_generation_file, "w", encoding="utf-8") as f:
            json.dump(model_generation_records, f, indent=2, ensure_ascii=False)
        print(f"Model generation results saved to: {model_generation_file}")

    failed_file = output_dir / "failed_samples.json"
    if failed_samples:
        with open(failed_file, "w", encoding="utf-8") as f:
            json.dump(failed_samples, f, indent=2, ensure_ascii=False)
        print(f"Failed samples saved to: {failed_file}")

    config_file = output_dir / "evaluation_config.json"
    config = {
        "model_path": args.model_path,
        "model_base": args.model_base,
        "adapter_path": args.adapter_path,
        "dataset_json": args.dataset_json,
        "images_dir": args.images_dir,
        "conv_template": conv_template,
        "generation_kwargs": generation_kwargs,
        "limit": args.limit,
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if args.in_out_labels_csv:
        config["in_out_labels_csv"] = args.in_out_labels_csv
    config["table_log_interval"] = table_log_interval
    config["table_log_file"] = str(progress_log_path)
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Evaluation configuration saved to: {config_file}")

    flush_generation_rows(force=True)

    if missing_in_out_samples:
        print(f"\n⚠️  Missing in_out labels for {len(missing_in_out_samples)} samples.")

    wandb_logged = False
    if args.log_to_wandb:
        try:
            scalar_metrics, non_scalar_metrics = split_metrics(final_metrics)
            wandb_run_instance = ensure_wandb_run()
            if wandb_run_instance is None:
                if wandb_disabled_reason is None:
                    print("\n⚠️  Skipped wandb logging: unable to initialize wandb run.")
            else:
                config_updates: Dict[str, Any] = dict(config)
                config_updates.update(
                    {
                        "evaluation_dir": str(output_dir),
                        "metrics_path": str(metrics_file),
                        "config_path": str(config_file),
                    }
                )
                if args.save_predictions:
                    config_updates["predictions_path"] = str(predictions_file)
                if model_generation_file is not None:
                    config_updates["model_generation_path"] = str(model_generation_file)
                if failed_samples:
                    config_updates["failed_samples_path"] = str(failed_file)
                config_updates.update(non_scalar_metrics)

                try:
                    wandb_run_instance.config.update(config_updates, allow_val_change=True)
                    wandb_run_instance.log(scalar_metrics or {"_placeholder": final_metrics.get("total_samples", 0)})
                    if not table_logging_streamed and model_generation_records:
                        table_rows: List[Dict[str, Any]] = []
                        for record in model_generation_records:
                            table_rows.extend(format_generation_sample(record))
                        if table_rows:
                            table = wandb_module.Table(columns=GENERATION_TABLE_COLUMNS)
                            for row in table_rows:
                                table.add_data(*(row.get(column) for column in GENERATION_TABLE_COLUMNS))
                            wandb_run_instance.log({"generation_results": table}, commit=False)
                            table_logging_streamed = True
                    wandb_logged = True
                finally:
                    try:
                        wandb_run_instance.finish()
                    except Exception:
                        pass
                    wandb_run = None
                if wandb_logged:
                    print(f"\n✅ Logged evaluation results to wandb run: {wandb_run_name}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n⚠️  Failed to log evaluation results to wandb: {exc}")
        if not wandb_logged and wandb_disabled_reason is None and wandb_run is None:
            # Avoid duplicate messaging when wandb is unavailable; ensure users notice silent skips.
            print("\n⚠️  Skipped wandb logging: no data was sent.")

    print("\nEvaluation completed successfully!")
    if failed_samples:
        print(f"\n⚠️  Warning: {len(failed_samples)} samples failed to process. Check failed_samples.json for details.")
    if not loss_values and not args.no_loss:
        print("\n⚠️  No loss values were calculated. Consider using --verbose for debugging.")
    if args.no_loss:
        print("\n📊 Loss calculation was disabled via --no-loss flag.")

if __name__ == "__main__":
    main()
