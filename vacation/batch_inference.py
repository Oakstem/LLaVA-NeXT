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
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from datetime import datetime

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate_vanilla_inference import (
    prepare_image_tensor,
    build_generation_kwargs,
    determine_template,
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
from vacation.gpt_extraction import extract_gaze_info_with_gpt
from openai import OpenAI


DEFAULT_ANNOTATIONS = "datasets/Vacation/test_annotations.csv"
DEFAULT_FRAMES_DIR = "datasets/Vacation/frames"

DEFAULT_PROMPT = "For each person in the image, describe who they are, what they are looking at, and then classify the interaction as non-communicative gaze, mutual gaze, or joint attention toward a shared object."


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
        default=f"vacation_general_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Inference prompt to use for all frames",
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
    """Get bbox, bbx_label, attention_focus for all persons in a frame."""
    frame_rows = df[(df["video_id"] == video_id) & (df["frame_id"] == frame_id)]
    
    annotations = []
    for _, row in frame_rows.iterrows():
        annotations.append({
            "bbx_id": int(row["bbx_id"]),
            "bbox": [int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])],
            "bbx_label": row["bbx_label"],
            "attention_focus": row["attention_focus"],
        })
    
    return annotations


def run_inference_on_frame(
    model,
    tokenizer,
    image_processor,
    frame_path: Path,
    prompt: str,
    args: argparse.Namespace,
    conv_name: str,
) -> str:
    """Run inference on a single frame, return the response."""
    # Prepare image
    image_tensor, image_size = prepare_image_tensor(str(frame_path), image_processor, model)
    
    # Build conversation
    conv = conv_templates[conv_name].copy()
    conv.tokenizer = tokenizer
    
    # Add image token to prompt
    user_content = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
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
    
    # Build generation kwargs
    gen_kwargs = {
        "inputs": input_ids,
        "images": image_tensor,
        "image_sizes": [list(image_size)],
        "do_sample": args.temperature > 0,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    
    if args.temperature > 0:
        gen_kwargs["temperature"] = args.temperature
    
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


def save_checkpoint(output_path: Path, config: dict, results: list):
    """Save current results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    
    # Resolve paths
    annotations_path = Path(fix_wsl_paths(args.annotations_file))
    frames_dir = Path(fix_wsl_paths(args.frames_dir))
    output_path = Path(fix_wsl_paths(args.output_json))
    
    # Validate paths
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
        adapter_path=fix_wsl_paths(args.adapter_path) if args.adapter_path else None,
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
    config = {
        "prompt": args.prompt,
        "model_path": args.model_path,
        "frames_dir": str(frames_dir),
        "annotations_file": str(annotations_path),
        "skip_step": args.skip_step,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "gpt_extraction_enabled": args.enable_gpt_extraction,
        "gpt_model": args.gpt_model if args.enable_gpt_extraction else None,
    }
    
    # Resume logic
    results = []
    processed_keys = set()
    
    if args.resume and output_path.exists():
        print(f"Resuming from {output_path}...")
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        results = existing.get("results", [])
        processed_keys = {(r["video_id"], r["frame_id"]) for r in results}
        print(f"Found {len(processed_keys)} already processed frames")
    
    # Filter out already processed frames
    selected_frames = selected_frames[
        ~selected_frames.apply(lambda r: (r["video_id"], r["frame_id"]) in processed_keys, axis=1)
    ]
    
    if args.limit:
        print(f"Limiting to {args.limit} frames...")
        selected_frames = selected_frames.head(args.limit)
        
    print(f"Remaining frames to process: {len(selected_frames)}")
    
    if len(selected_frames) == 0:
        print("All frames already processed!")
        return
    
    # Process frames
    print("Starting inference...")
    for idx, (_, row) in enumerate(tqdm(selected_frames.iterrows(), total=len(selected_frames))):
        video_id = int(row["video_id"])
        frame_id = int(row["frame_id"])
        event_attribute = row["event_attribute"]
        
        frame_path = build_frame_path(frames_dir, video_id, frame_id)
        
        if not frame_path.exists():
            print(f"\nWarning: Frame not found: {frame_path}, skipping...")
            continue
        
        response = run_inference_on_frame(
            model, tokenizer, image_processor,
            frame_path, args.prompt, args, conv_name
        )
        
        # Build result entry
        result_entry = {
            "video_id": video_id,
            "frame_id": frame_id,
            "event_attribute": event_attribute,
            "image_path": str(frame_path),
            "response": response,
            "annotations": get_frame_annotations(all_annotations, video_id, frame_id),
        }
        
        # Run GPT extraction if enabled
        if openai_client is not None:
            extracted = extract_gaze_info_with_gpt(openai_client, response, args.gpt_model)
            result_entry["extracted_gaze_info"] = extracted
        
        results.append(result_entry)
        
        # Checkpoint
        if (idx + 1) % args.checkpoint_interval == 0:
            print(f"\nSaving checkpoint at {idx + 1} frames...")
            save_checkpoint(output_path, config, results)
    
    # Final save
    print(f"\nSaving final results to {output_path}...")
    save_checkpoint(output_path, config, results)
    print(f"Done! Processed {len(results)} frames total.")


if __name__ == "__main__":
    main()
