#!/usr/bin/env python3
"""Batch inference on Vacation dataset frames using LLaVA.

This script processes frames from the Vacation dataset test_annotations.csv,
runs LLaVA inference with a configurable prompt, and stores results in a JSON file.

Features:
- Frame skipping for consecutive frames with the same event_attribute
- Periodic checkpointing to prevent data loss
- Resume capability from existing output JSON
- Includes bounding box annotations in output
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gazefollow.generate_vanilla_inference import (
    prepare_image_tensor,
    build_generation_kwargs,
    determine_template,
    ensure_image_config,
)
from gazefollow.generation_utils import (
    enable_inference_optimizations,
    load_model_and_setup,
    fix_wsl_paths,
)
from llava.mm_utils import get_model_name_from_path
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from gazefollow.evals.log_wandb_evaluations import (
    DEFAULT_PROJECT,
    build_run_name_from_adapter,
)
from vacation.gpt_extraction import extract_gaze_info_with_gpt
from vacation.recompute_vacation_metrics import (
    _extract_prompt_columns,
    _extract_run_config,
    compute_metrics,
)
from vacation.wandb_utils import (
    infer_wandb_run_id,
    log_prefixed_metrics_to_wandb,
)
from openai import OpenAI


DEFAULT_ANNOTATIONS = "datasets/Vacation/test_annotations.csv"
DEFAULT_FRAMES_DIR = "datasets/Vacation/frames"

# DEFAULT_PROMPT = "For each person in the image, describe who they are, what they are looking at, and then classify the interaction as non-communicative gaze, mutual gaze, or joint attention toward a shared object."
# DEFAULT_PROMPT = """For each person in the image:
# Briefly describe who they are (role/appearance) and what they are looking at (another person, an object, or off e.g. “off-screen left”).
# Then choose exactly one social interaction label using the rules below in this priority order:
# Priority order (apply top to bottom):
# Mutual gaze: at least two people are looking at each other (A→B and B→A).
# If this is true, the label must be Mutual gaze, even if you think they share attention.
# Single: one person looks at another (A→B) while the other looks elsewhere (B→≠A).
# Joint attention: two or more people are looking at the same external target (same object/location), and that target is not any person.
# Non-communicative gaze: none of the above apply; no clear gaze-based interaction.
# Important constraint:
# Joint attention never applies when the shared target is a person. If people are looking at each other, that is Mutual gaze, not Joint attention."""

DEFAULT_PROMPT = """Describe each person briefly and say what they are looking at (person, object, or off-screen).

Then choose exactly one social interaction label:
MutualGaze: at least two people are looking at each other (A looks at B and B looks at A).
SharedObjectAttention: at least two people are looking at the same external object or place (not a person), including one person following another person's reference to that external target.
OneSidedGaze: one person looks at another person but the other looks away or elsewhere (not reciprocated).
NonCommmunicative: no clear gaze interaction or gaze is unclear; use this when people are not engaging through gaze.

Rules:
Do not guess MutualGaze. If reciprocity is not obvious, it is not MutualGaze.
If only one person is described as looking at another, it is NonCommmunicative."""

PROMPT_A = """Describe the scene in free wording with focus on people and gaze.

Include:
1. How many people are visible in the image.
2. A brief description of each visible person.
3. For each person, where they are looking (another person, an object/place, off-screen, the camera, or unclear).

Keep the response concise but ensure all visible people are covered.
Do not assign any interaction label (no MutualGaze, SharedObjectAttention, OneSidedGaze, NonCommmunicative, or None)."""
# PROMPT_A = """Identify all visible people in the image.
# For each person:
# Assign a unique ID (e.g., Person_1, Person_2).
# Briefly describe their appearance (clothing, position, distinguishing features).
# Estimate their gaze direction.
# Identify what they are looking at: another person (use ID), an object (name it), or off-screen.
# If uncertain, state the uncertainty.
# Return your answer strictly in the following JSON format:
# {
#   "people": [
#     {
#       "id": "Person_1",
#       "description": "",
#       "gaze_direction": "",
#       "gaze_target_type": "person | object | off_screen | unclear",
#       "gaze_target": "",
#       "uncertainty": ""
#     }
#   ]
# }
# """
# gaze_target_type must be exactly one of the following values:
# "person" (if looking at another identified person — use their ID)
# "object" (if looking at a visible object — name it)
# "off_screen" (if looking outside the image frame)
# "unclear" (if gaze cannot be reliably determined)
# Do not list multiple options. Choose only one value.
# """
# PROMPT_A = """You are an expert vision assistant.
# Step 1 - Caption
# • Provide one concise sentence that broadly describes the entire scene.
# • Begin the line with: Caption:
# Step 2 - Foreground people & gaze
# 1. Detect every person whose height is at least 5% of the image (foreground).
# 2. List them from left to right and number sequentially starting at 1.
# 3. Write the total number of people detected.
# For each person output exactly one line in this format:
# Person {N}: {short description}, looking at {target | outside the frame | uncertain}
# Output format (no extra lines, no prose other than what is specified):
# -------------------------------------------------
# Caption: {your one-sentence scene description}
# Person 1: {short description}, looking at ...
# Person 2: {short description}, looking at ...
# ...
# Total people detected: {number}
# -------------------------------------------------
# Additional rules
# • Keep the phrase "looking at" unchanged.
# • {short description} must be 6 words or fewer (e.g., "man in red jacket").
# • If no foreground person is detected, write exactly: No foreground people detected.
# • If gaze cannot be determined, use "uncertain".
# • Do not output your reasoning or any extra text."""

PROMPT_B = """Given Step-1 gaze facts, choose exactly one label and briefly justify it from the listed person-to-target relations.

Labels:
MutualGaze: at least two people look at each other (A->B and B->A).
SharedObjectAttention: at least two people look at the same external object/place (not a person), including follow/reference toward that same external target.
OneSidedGaze: someone looks at another person, but reciprocity is absent.
NonCommmunicative: no clear interpersonal gaze interaction (off-screen/camera/object-only/unclear), or single-person scene.
None: no usable gaze-looking information is provided.

Decision rules:
1. Do not guess MutualGaze; require explicit reciprocity.
2. If only one person looks at another person and it is not reciprocated, use OneSidedGaze.
3. If all gaze targets are off-screen/camera/unclear, use NonCommmunicative.
4. If only one person is present, always use NonCommmunicative.
5. If no usable gaze info exists, use None.

Output format (strict):
Label: <MutualGaze|SharedObjectAttention|OneSidedGaze|NonCommmunicative|None>
Reason:
- <1 short bullet citing key gaze relation(s), e.g., P1->P2 and P2->P1>
- <optional 2nd short bullet for tie-break/rule applied>"""

# PROMPT_B = """Task: Assign exactly one Vacation gaze label:
# MutualGaze, SharedObjectAttention, OneSidedGaze, NonCommmunicative, None
# Use ONLY the Step 1 result (people + gaze targets). Ignore scene/story/context.
# Decision rules (apply in this exact order; first match wins):
# MutualGaze — two people look at each other (A looks at B AND B looks at A).
# SharedObjectAttention — at least two people look at the same visible object (same described object).
# OneSidedGaze — someone looks at another person, but it’s not reciprocated.
# NonCommmunicative — people are present but no mutual/shared/one-sided person-looking; gazes are toward objects, off-screen, or unclear.
# None — no people detected, OR gaze can’t be determined for everyone (all “unclear” with no usable targets).
# Output format (exactly):
# First line: Label: <ONE_LABEL>
# Then 2–4 bullet points explaining which rule triggered and the supporting gaze relations (use Person IDs).
# Don’ts:
# Do not output multiple labels.
# Do not hedge with “between X and Y”.
# If you’re unsure about “same object”, choose NonCommmunicative unless the match is very clear.
# Input (Step 1 JSON):
# """

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch LLaVA inference on Vacation dataset frames."
    )
    
    # Path arguments
    parser.add_argument(
        "--annotations-file",
        type=str,
        default=DEFAULT_ANNOTATIONS,
        help=f"Path to annotations CSV (default: {DEFAULT_ANNOTATIONS})",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default=DEFAULT_FRAMES_DIR,
        help=f"Path to frames directory (default: {DEFAULT_FRAMES_DIR})",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=(
            f"vacation_results/single_image/"
            f"vacation_general_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        ),
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Run inference on a single image path instead of annotation/frame batch mode.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Inference prompt to use for all frames",
    )
    parser.add_argument(
        "--two-step-inference",
        action="store_true",
        help="Enable two-step inference (prompt A then prompt B).",
    )
    parser.add_argument(
        "--prompt-a",
        dest="prompt_a",
        type=str,
        default=PROMPT_A,
        help="First prompt used for two-step inference.",
    )
    parser.add_argument(
        "--prompt-b",
        dest="prompt_b",
        type=str,
        default=PROMPT_B,
        help="Second prompt appended after the first response for two-step inference.",
    )
    parser.add_argument(
        "--step2-separator",
        "--second-separator",
        type=str,
        default=" \n",
        help="Separator placed between response A and prompt B in two-step inference.",
    )
    parser.add_argument(
        "--step2-keep-adapter",
        "--second-step-keep-adapter",
        action="store_true",
        help=(
            "In two-step inference, keep adapters enabled for step 2. "
            "Default behavior disables adapters in step 2 when --adapter-path is set."
        ),
    )
    parser.add_argument(
        "--step2-uses-image",
        "--second-step-uses-image",
        action="store_true",
        help=(
            "In two-step inference, include the image again in step 2. "
            "Default behavior uses text-only step 2."
        ),
    )
    parser.add_argument(
        "--step2-temperature",
        "--second-step-temperature",
        type=float,
        default=None,
        help=(
            "Optional temperature override for step 2 in two-step inference. "
            "If not set, step 2 uses --temperature."
        ),
    )
    
    # Frame selection arguments
    parser.add_argument(
        "--skip-step",
        type=int,
        default=10,
        help="Sample every N frames within consecutive same event_attribute runs (default: 1, no skipping)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the total number of frames to process",
    )
    
    # Checkpointing arguments
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoint every N frames (default: 100)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output JSON file",
    )
    
    # Queue randomization arguments
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize the processing queue order instead of running consecutively",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible queue randomization (only used with --randomize)",
    )
    
    # Model arguments (inherited from generate_vanilla_inference.py)
    parser.add_argument(
        "--model-path",
        default="lmms-lab/llava-onevision-qwen2-7b-ov-chat",
        help="Model or checkpoint path to load",
    )
    parser.add_argument(
        "--model-base",
        default=None,
        help="Optional base model path when loading LoRA adapters",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional LoRA adapter path to merge at inference time",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Attention implementation (e.g. 'sdpa', 'flash_attention_2')",
    )
    parser.add_argument(
        "--load-4bit",
        action="store_true",
        help="Load the model with 4-bit quantization",
    )
    parser.add_argument(
        "--load-8bit",
        action="store_true",
        help="Load the model with 8-bit quantization",
    )
    parser.add_argument(
        "--conv-template",
        default=None,
        help="Conversation template key",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--disable-optimizations",
        action="store_true",
        help="Skip enabling CUDA inference optimizations",
    )
    parser.add_argument(
        "--image-aspect-ratio",
        type=str,
        default="anyres_max_4",
        help="Override image aspect ratio (ANYRES)",
    )
    parser.add_argument(
        "--image-grid-pinpoints",
        type=str,
        default="(1x1),...,(2x2)",
        help="Image grid pinpoints",
    )
    
    # GPT extraction arguments
    parser.add_argument(
        "--enable-gpt-extraction",
        action="store_true",
        help="Enable GPT-based structured extraction of gaze information from responses",
    )
    parser.add_argument(
        "--gpt-model",
        type=str,
        default="gpt-5-nano",
        help="OpenAI model to use for GPT extraction (default: gpt-5-nano)",
    )

    # W&B arguments
    parser.add_argument(
        "--log-to-wandb",
        action="store_true",
        default=True,
        help="Log Vacation results to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-project",
        default=DEFAULT_PROJECT,
        help="Weights & Biases project name to use when logging.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Optional Weights & Biases entity/organization name.",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="Optional existing W&B run id to resume and update.",
    )
    parser.add_argument(
        "--wandb-resume",
        type=str,
        default="allow",
        choices=["allow", "must", "never", "auto"],
        help="W&B resume mode used when --wandb-run-id is provided.",
    )
    
    return parser.parse_args()


def build_frame_path(frames_dir: Path, video_id: int, frame_id: int) -> Path:
    """Build frame path: frames/{video_id}/{frame_id+1:06d}.png (1-indexed)."""
    return frames_dir / str(video_id) / f"{frame_id + 1:06d}.png"


def select_frames_with_skipping(df: pd.DataFrame, skip_step: int) -> pd.DataFrame:
    """
    Select frames to process, skipping every skip_step frames
    within consecutive runs of the same event_attribute per video.
    
    Args:
        df: DataFrame with columns video_id, frame_id, event_attribute
        skip_step: Sample every N frames within each event run
        
    Returns:
        DataFrame with selected (video_id, frame_id) pairs
    """
    if skip_step <= 1:
        # No skipping, return unique frames with their first event_attribute
        return (
            df.groupby(["video_id", "frame_id"])
            .agg({"event_attribute": "first"})
            .reset_index()
        )
    
    # Get unique frames with their first event_attribute
    unique_frames = (
        df.groupby(["video_id", "frame_id"])
        .agg({"event_attribute": "first"})
        .reset_index()
        .sort_values(["video_id", "frame_id"])
    )
    
    selected_frames = []
    
    for video_id in unique_frames["video_id"].unique():
        video_frames = unique_frames[unique_frames["video_id"] == video_id].copy()
        video_frames = video_frames.sort_values("frame_id").reset_index(drop=True)
        
        # Detect runs of consecutive frames with same event_attribute
        video_frames["event_changed"] = (
            video_frames["event_attribute"] != video_frames["event_attribute"].shift(1)
        ) | (video_frames["frame_id"] != video_frames["frame_id"].shift(1) + 1)
        video_frames["run_id"] = video_frames["event_changed"].cumsum()
        
        # Within each run, sample every skip_step frames
        for run_id in video_frames["run_id"].unique():
            run_frames = video_frames[video_frames["run_id"] == run_id]
            sampled = run_frames.iloc[::skip_step]
            selected_frames.append(sampled[["video_id", "frame_id", "event_attribute"]])
    
    return pd.concat(selected_frames, ignore_index=True)


def get_frame_annotations(df: pd.DataFrame, video_id: int, frame_id: int) -> List[dict]:
    """Get bbox, bbx_label, attention_focus, atomic_attribute for all persons in a frame."""
    frame_rows = df[(df["video_id"] == video_id) & (df["frame_id"] == frame_id)]
    
    annotations = []
    for _, row in frame_rows.iterrows():
        annotations.append({
            "bbx_id": int(row["bbx_id"]),
            "bbox": [int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])],
            "bbx_label": row["bbx_label"],
            "attention_focus": row["attention_focus"],
            "atomic_attribute": row["atomic_attribute"],
        })
    
    return annotations


def classify_atomic_attribute_combo(atomic_attributes: List[str]) -> Optional[str]:
    if not atomic_attributes:
        return None
    attrs = [str(a).strip().lower() for a in atomic_attributes if str(a).strip()]
    if not attrs:
        return None
    if all(a == "single" for a in attrs):
        return "NonCommmunicative"

    rules = [
        (("follow", "single"), "OneSidedGaze"),
        (("share", "share"), "SharedObjectAttention"),
        (("single", "avert"), "OneSidedGaze"),
        (("single", "refer"), "OneSidedGaze"),
        (("mutual", "mutual"), "MutualGaze"),
        (("avert", "avert"), "NonCommmunicative"),
        (("follow", "refer"), "SharedObjectAttention"),
    ]

    for (a, b), label in rules:
        if a == b:
            if sum(1 for v in attrs if v == a) >= 2:
                return label
        else:
            if a in attrs and b in attrs:
                return label
    return None


def _normalize_attention_focus(value: object) -> Optional[str]:
    if value is None:
        return None
    if pd.isna(value):
        return None
    cleaned = str(value).strip()
    if not cleaned or cleaned.lower() == "nan":
        return None
    return cleaned


def _person_id_from_label(label: object) -> Optional[str]:
    if label is None:
        return None
    cleaned = str(label).strip()
    if not cleaned:
        return None
    if cleaned.lower().startswith("person"):
        digits = "".join(ch for ch in cleaned if ch.isdigit())
        if digits:
            return f"P{digits}"
    if cleaned[:1].upper() == "P" and cleaned[1:].isdigit():
        return cleaned.upper()
    digits = "".join(ch for ch in cleaned if ch.isdigit())
    if digits:
        return f"P{digits}"
    return None


def _is_person_focus(value: str) -> bool:
    return value[:1].upper() == "P" and value[1:].isdigit()


def _is_object_focus(value: str) -> bool:
    return value[:1].upper() == "O" and value[1:].isdigit()


def classify_attention_focus_labels(
    annotations: List[dict],
    base_label: Optional[str],
) -> Tuple[Optional[str], List[str]]:
    person_ids = set()
    focus_pairs = []
    nan_count = 0

    for ann in annotations:
        pid = _person_id_from_label(ann.get("bbx_label"))
        if pid:
            person_ids.add(pid)
        focus = _normalize_attention_focus(ann.get("attention_focus"))
        if focus is None:
            nan_count += 1
            continue
        focus = focus.upper()
        if pid and _is_person_focus(focus):
            focus_pairs.append((pid, focus))
        elif _is_object_focus(focus):
            pass

    mutual = False
    one_sided = False
    focus_pairs_set = set(focus_pairs)
    for src, tgt in focus_pairs:
        if tgt in person_ids:
            one_sided = True
            if (tgt, src) in focus_pairs_set:
                mutual = True
                break

    primary_label = base_label
    labels = []
    if mutual:
        primary_label = "MutualGaze"
        labels.append("MutualGaze")
    elif one_sided:
        primary_label = "OneSidedGaze"
        labels.append("OneSidedGaze")
    elif base_label:
        labels.append(base_label)

    if nan_count >= 2:
        if primary_label is None:
            primary_label = "NonCommmunicative"
        labels.append("NonCommmunicative")

    seen = set()
    ordered = []
    for label in labels:
        if label and label not in seen:
            seen.add(label)
            ordered.append(label)

    return primary_label, ordered


def run_inference_on_frame(
    model,
    tokenizer,
    image_processor,
    frame_path: Path,
    prompt: str,
    args: argparse.Namespace,
    conv_name: str,
    include_image: bool = True,
    temperature_override: Optional[float] = None,
) -> str:
    """Run inference on a single frame, return the response."""
    image_tensor = None
    image_size = None
    if include_image:
        image_tensor, image_size = prepare_image_tensor(str(frame_path), image_processor, model)
    
    # Build conversation
    conv = conv_templates[conv_name].copy()
    conv.tokenizer = tokenizer
    
    # Add image token only when visual input is enabled
    user_content = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}" if include_image else prompt
    conv.append_message(conv.roles[0], user_content)
    conv.append_message(conv.roles[1], None)
    prompt_for_tokenizer = conv.get_prompt()
    
    if include_image:
        input_ids = tokenizer_image_token(
            prompt_for_tokenizer,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(model.device)
    else:
        input_ids = tokenizer(
            prompt_for_tokenizer,
            return_tensors="pt",
        ).input_ids.to(model.device)
    
    effective_temperature = (
        args.temperature if temperature_override is None else temperature_override
    )

    # Build generation kwargs
    gen_kwargs = {
        "inputs": input_ids,
        "do_sample": effective_temperature > 0,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if include_image:
        gen_kwargs["images"] = image_tensor
        gen_kwargs["image_sizes"] = [list(image_size)]
    
    if effective_temperature > 0:
        gen_kwargs["temperature"] = effective_temperature
    
    # Generate
    with torch.inference_mode():
        generation_output = model.generate(**gen_kwargs)
    
    if isinstance(generation_output, tuple):
        output_ids = generation_output[0]
    else:
        output_ids = generation_output

    # Decode response
    generated_tokens = output_ids[0, :]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    return response


def run_two_step_inference_on_frame(
    model,
    tokenizer,
    image_processor,
    frame_path: Path,
    prompt_a: str,
    prompt_b: str,
    args: argparse.Namespace,
    conv_name: str,
) -> Tuple[str, str, str]:
    response_a = run_inference_on_frame(
        model,
        tokenizer,
        image_processor,
        frame_path,
        prompt_a,
        args,
        conv_name,
    )
    if response_a.strip() and not response_a.strip().endswith("."):
        response_a = response_a.strip() + "."
    combined_prompt = f"{response_a}{args.step2_separator}{prompt_b}".strip()

    if (
        hasattr(model, "disable_adapter")
        and args.adapter_path
        and not args.step2_keep_adapter
    ):
        with model.disable_adapter():
            response_b = run_inference_on_frame(
                model,
                tokenizer,
                image_processor,
                frame_path,
                combined_prompt,
                args,
                conv_name,
                include_image=args.step2_uses_image,
                temperature_override=args.step2_temperature,
            )
    else:
        response_b = run_inference_on_frame(
            model,
            tokenizer,
            image_processor,
            frame_path,
            combined_prompt,
            args,
            conv_name,
            include_image=args.step2_uses_image,
            temperature_override=args.step2_temperature,
        )

    return response_a, response_b, combined_prompt


def _normalize_event_attribute(event_attribute: str | None) -> str | None:
    if event_attribute is None:
        return None
    cleaned = str(event_attribute).strip().lower()
    if not cleaned:
        return None
    if cleaned == "mutualgaze":
        return "mutual"
    if cleaned == "singlegaze":
        return "single"
    if cleaned == "jointatt":
        return "shared_object_attention"
    if cleaned == "avertgaze":
        return "single"
    if cleaned == "gazefollow":
        return "single"
    return None


def save_checkpoint(output_path: Path, config: dict, results: list, metrics: Optional[dict]):
    """Save current results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config,
        "results": results,
    }
    if metrics is not None:
        payload["metrics"] = metrics
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_wandb_metric_row(output_path: Path, config: dict, metrics: dict) -> Dict[str, Any]:
    payload = {"config": config}
    row: Dict[str, Any] = {
        "results_file": str(output_path),
        "adapter_name": output_path.stem,
    }
    row.update(_extract_run_config(payload))
    row.update(metrics)
    row.update(_extract_prompt_columns(payload))
    return row


def main():
    args = parse_args()
    
    # Resolve paths
    image_path = Path(fix_wsl_paths(args.image_path)) if args.image_path else None
    annotations_path = Path(fix_wsl_paths(args.annotations_file))
    frames_dir = Path(fix_wsl_paths(args.frames_dir))
    output_path = Path(fix_wsl_paths(args.output_json))
    adapter_path = fix_wsl_paths(args.adapter_path) if args.adapter_path else None
    wandb_run_name = build_run_name_from_adapter(adapter_path or args.model_path)
    wandb_run_id = args.wandb_run_id
    existing = {}
    log_to_wandb = bool(args.log_to_wandb and image_path is None)
    
    # Validate paths
    if image_path is not None:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
    else:
        if not annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    
    # Enable optimizations
    if not args.disable_optimizations:
        enable_inference_optimizations()
    
    # Load model
    print("Loading model...")
    tokenizer, model, image_processor, _ = load_model_and_setup(
        model_path=args.model_path,
        attn_implementation=args.attn_implementation,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        model_base=args.model_base,
        adapter_path=adapter_path,
    )
    aspect_ratio = getattr(getattr(model, "config", None), "image_aspect_ratio", None)
    print(f"INFO: image_aspect_ratio in checkpoint: {aspect_ratio}")
    ensure_image_config(
        model,
        args.image_aspect_ratio,
        args.image_grid_pinpoints,
    )
    
    # Determine conversation template
    model_name_source = model.config._name_or_path if hasattr(model.config, "_name_or_path") else args.model_path
    model_name = get_model_name_from_path(model_name_source)
    conv_name = determine_template(model_name, args.conv_template)
    
    # Load annotations
    print(f"Loading annotations from {annotations_path}...")
    all_annotations = pd.read_csv(annotations_path)
    
    # Select frames with skipping
    print(f"Selecting frames with skip_step={args.skip_step}...")
    selected_frames = select_frames_with_skipping(all_annotations, args.skip_step)
    print(f"Total unique frames to process: {len(selected_frames)}")
    
    # Initialize OpenAI client if GPT extraction is enabled
    openai_client = None
    if args.enable_gpt_extraction:
        print(f"GPT extraction enabled using model: {args.gpt_model}")
        openai_client = OpenAI()
    
    # Build config for output
    second_step_uses_adapter = None
    if args.two_step_inference:
        second_step_uses_adapter = bool(adapter_path) and (
            args.step2_keep_adapter or not hasattr(model, "disable_adapter")
        )

    config = {
        "prompt": args.prompt,
        "two_step_inference": args.two_step_inference,
        "prompt_a": args.prompt_a if args.two_step_inference else None,
        "prompt_b": args.prompt_b if args.two_step_inference else None,
        "second_separator": args.step2_separator if args.two_step_inference else None,
        "second_step_uses_image": args.step2_uses_image if args.two_step_inference else None,
        "second_step_keep_adapter": args.step2_keep_adapter if args.two_step_inference else None,
        "second_step_temperature": args.step2_temperature if args.two_step_inference else None,
        "second_step_uses_adapter": second_step_uses_adapter,
        "model_path": args.model_path,
        "adapter_path": adapter_path,
        "image_path": str(image_path) if image_path is not None else None,
        "frames_dir": str(frames_dir),
        "annotations_file": str(annotations_path),
        "skip_step": args.skip_step,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "gpt_extraction_enabled": args.enable_gpt_extraction,
        "gpt_model": args.gpt_model if args.enable_gpt_extraction else None,
        "wandb_project": args.wandb_project if log_to_wandb else None,
        "wandb_entity": args.wandb_entity if log_to_wandb else None,
        "wandb_run_name": wandb_run_name if log_to_wandb else None,
        "wandb_run_id": wandb_run_id if log_to_wandb else None,
        "wandb_resume": args.wandb_resume if log_to_wandb else None,
    }

    base_wandb_config: Dict[str, Any] = {
        "model_path": args.model_path,
        "model_base": args.model_base,
        "adapter_path": adapter_path,
        "annotations_file": str(annotations_path),
        "frames_dir": str(frames_dir),
        "output_json": str(output_path),
        "skip_step": args.skip_step,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "second_step_temperature": args.step2_temperature,
        "two_step_inference": args.two_step_inference,
        "gpt_extraction_enabled": args.enable_gpt_extraction,
    }
    if args.log_to_wandb and image_path is not None:
        print("W&B logging disabled for single-image inference mode.")
    if log_to_wandb:
        print(
            f"W&B logging enabled (project={args.wandb_project}, "
            f"entity={args.wandb_entity or 'default'}, run_name={wandb_run_name})"
        )
    else:
        print("W&B logging disabled.")

    def log_metrics_to_wandb(metrics: dict, total_results: int) -> None:
        nonlocal wandb_run_id
        if wandb_run_id is None:
            inferred_id, inferred_entity = infer_wandb_run_id(
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                inferred_run_name=wandb_run_name,
                model_path=adapter_path or args.model_path,
            )
            if inferred_id:
                wandb_run_id = inferred_id
                config["wandb_run_id"] = wandb_run_id
                print(f"Inferred wandb run id: {wandb_run_id}")
            if inferred_entity and args.wandb_entity is None:
                args.wandb_entity = inferred_entity
                config["wandb_entity"] = inferred_entity
        row = build_wandb_metric_row(output_path, config, metrics)
        updated_run_id, error = log_prefixed_metrics_to_wandb(
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=wandb_run_name,
            wandb_run_id=wandb_run_id,
            wandb_resume=args.wandb_resume,
            base_wandb_config=base_wandb_config,
            metric_row=row,
            metric_prefix="vacation",
            total_results=total_results,
            extra_config_updates={
            "vacation/output_json": str(output_path),
            "vacation/wandb_run_id": wandb_run_id,
            },
        )
        if updated_run_id:
            wandb_run_id = updated_run_id
            config["wandb_run_id"] = updated_run_id
        if error:
            print(f"\n⚠️  Failed to log Vacation metrics to wandb: {error}")
        else:
            print(
                f"\n✅ Logged Vacation metrics to wandb run id={wandb_run_id} "
                f"(project={args.wandb_project})"
            )

    if image_path is not None:
        if args.two_step_inference:
            print("Using two-step prompts:")
            print(f"Prompt A:\n{args.prompt_a}\n")
            print(f"Prompt B:\n{args.prompt_b}\n")
            step2_image_mode = "enabled" if args.step2_uses_image else "disabled"
            print(f"Step 2 image input: {step2_image_mode}\n")
            if args.step2_temperature is not None:
                print(f"Step 2 temperature override: {args.step2_temperature}\n")
            if args.adapter_path:
                step2_mode = "enabled" if args.step2_keep_adapter else "disabled"
                print(f"Step 2 adapters: {step2_mode}\n")
        else:
            print(f"Using prompt:\n{args.prompt}\n")

        if args.two_step_inference:
            response_a, response_b, combined_prompt = run_two_step_inference_on_frame(
                model,
                tokenizer,
                image_processor,
                image_path,
                args.prompt_a,
                args.prompt_b,
                args,
                conv_name,
            )
            response = response_b
            print("\nSingle-image Step 1 response:")
            print(response_a)
            print("\nSingle-image Step 2 response:")
            print(response_b)
        else:
            response = run_inference_on_frame(
                model,
                tokenizer,
                image_processor,
                image_path,
                args.prompt,
                args,
                conv_name,
            )
            response_a = None
            combined_prompt = None
            print("\nSingle-image response:")
            print(response)

        result_entry = {
            "video_id": None,
            "frame_id": None,
            "image_path": str(image_path),
            "atomic_attribute_combo_GT": None,
            "atomic_attribute_combo_GT_multi": [],
            "response": response,
            "response_step1": response_a if args.two_step_inference else None,
            "combined_prompt": combined_prompt if args.two_step_inference else None,
            "annotations": [],
            "atomic_attributes": [],
        }
        if openai_client is not None:
            extracted = extract_gaze_info_with_gpt(openai_client, response, args.gpt_model)
            result_entry["extracted_gaze_info"] = extracted

        metrics = compute_metrics([result_entry])
        if log_to_wandb:
            log_metrics_to_wandb(metrics, total_results=1)

        print(f"\nSaving final results to {output_path}...")
        save_checkpoint(output_path, config, [result_entry], metrics)
        print("Done! Processed 1 image.")
        return
    
    # Resume logic
    results = []
    processed_keys = set()
    
    if args.resume and output_path.exists():
        print(f"Resuming from {output_path}...")
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        results = existing.get("results", [])
        processed_keys = {(r["video_id"], r["frame_id"]) for r in results}
        print(f"Found {len(processed_keys)} already processed frames")
        existing_config = existing.get("config")
        if (
            isinstance(existing_config, dict)
            and not wandb_run_id
            and existing_config.get("wandb_run_id")
        ):
            wandb_run_id = str(existing_config["wandb_run_id"])
            config["wandb_run_id"] = wandb_run_id
            print(f"Reusing wandb run id from existing output: {wandb_run_id}")
    
    # Filter out already processed frames
    selected_frames = selected_frames[
        ~selected_frames.apply(lambda r: (r["video_id"], r["frame_id"]) in processed_keys, axis=1)
    ]
    
    # Randomize queue if requested
    if args.randomize:
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            print(f"Randomizing queue with seed={args.seed}...")
        else:
            print("Randomizing queue (no seed, non-reproducible)...")
        selected_frames = selected_frames.sample(
            frac=1,
            random_state=args.seed,
        ).reset_index(drop=True)
    
    # Apply limit if specified
    if args.limit:
        print(f"Limiting to {args.limit} frames...")
        selected_frames = selected_frames.head(args.limit)
        
    print(f"Remaining frames to process: {len(selected_frames)}")
    
    if len(selected_frames) == 0:
        print("All frames already processed!")
        if results:
            metrics = compute_metrics(results)
            if log_to_wandb:
                log_metrics_to_wandb(metrics, total_results=len(results))
            save_checkpoint(output_path, config, results, metrics)
            print(
                "Metrics: valid_social_interaction_label_percent="
                f"{metrics['valid_social_interaction_label_percent']:.4f}, "
                "accuracy_vs_atomic_attribute_combo="
                f"{metrics['accuracy_vs_atomic_attribute_combo']:.4f} "
                f"(evaluated={metrics['accuracy_evaluated_frames']})"
            )
        elif log_to_wandb:
            print("W&B logging skipped: no results available to log.")
        return
    
    # Process frames
    if args.two_step_inference:
        print("Using two-step prompts:")
        print(f"Prompt A:\n{args.prompt_a}\n")
        print(f"Prompt B:\n{args.prompt_b}\n")
        step2_image_mode = "enabled" if args.step2_uses_image else "disabled"
        print(f"Step 2 image input: {step2_image_mode}\n")
        if args.step2_temperature is not None:
            print(f"Step 2 temperature override: {args.step2_temperature}\n")
        if args.adapter_path:
            step2_mode = "enabled" if args.step2_keep_adapter else "disabled"
            print(f"Step 2 adapters: {step2_mode}\n")
    else:
        print(f"Using prompt:\n{args.prompt}\n")
    print("Starting inference...")
    for idx, (_, row) in enumerate(tqdm(selected_frames.iterrows(), total=len(selected_frames))):
        video_id = int(row["video_id"])
        frame_id = int(row["frame_id"])
        event_attribute = row["event_attribute"]
        
        frame_path = build_frame_path(frames_dir, video_id, frame_id)
        
        if not frame_path.exists():
            print(f"\nWarning: Frame not found: {frame_path}, skipping...")
            continue
        
        if args.two_step_inference:
            response_a, response_b, combined_prompt = run_two_step_inference_on_frame(
                model,
                tokenizer,
                image_processor,
                frame_path,
                args.prompt_a,
                args.prompt_b,
                args,
                conv_name,
            )
            response = response_b
        else:
            response = run_inference_on_frame(
                model,
                tokenizer,
                image_processor,
                frame_path,
                args.prompt,
                args,
                conv_name,
            )
        
        # Build result entry
        annotations = get_frame_annotations(all_annotations, video_id, frame_id)
        atomic_attributes = [
            ann.get("atomic_attribute")
            for ann in annotations
            if ann.get("atomic_attribute") not in (None, "")
        ]
        atomic_attribute_combo = classify_atomic_attribute_combo(atomic_attributes)
        gt_label, gt_labels_multi = classify_attention_focus_labels(
            annotations, atomic_attribute_combo
        )

        result_entry = {
            "video_id": video_id,
            "frame_id": frame_id,
            "image_path": str(frame_path),
            "atomic_attribute_combo_GT": gt_label,
            "atomic_attribute_combo_GT_multi": gt_labels_multi,
            "response": response,
            "response_step1": response_a if args.two_step_inference else None,
            "combined_prompt": combined_prompt if args.two_step_inference else None,
            "annotations": annotations,
            "atomic_attributes": atomic_attributes,
        }
        # Print response snippet
        print(f"\nProcessed video_id={video_id}, frame_id={frame_id}")
        print(f"Response: {response}")

        # Run GPT extraction if enabled
        if openai_client is not None:
            extracted = extract_gaze_info_with_gpt(openai_client, response, args.gpt_model)
            result_entry["extracted_gaze_info"] = extracted
        
        results.append(result_entry)
        
        # Checkpoint
        if (idx + 1) % args.checkpoint_interval == 0:
            print(f"\nSaving checkpoint at {idx + 1} frames...")
            metrics = compute_metrics(results)
            save_checkpoint(output_path, config, results, metrics)
    
    # Final save
    print(f"\nSaving final results to {output_path}...")
    metrics = compute_metrics(results)
    if log_to_wandb:
        log_metrics_to_wandb(metrics, total_results=len(results))
    save_checkpoint(output_path, config, results, metrics)
    print(
        "Metrics: valid_social_interaction_label_percent="
        f"{metrics['valid_social_interaction_label_percent']:.4f}, "
        "accuracy_vs_atomic_attribute_combo="
        f"{metrics['accuracy_vs_atomic_attribute_combo']:.4f} "
        f"(evaluated={metrics['accuracy_evaluated_frames']})"
    )
    print(f"Done! Processed {len(results)} frames total.")


if __name__ == "__main__":
    main()
