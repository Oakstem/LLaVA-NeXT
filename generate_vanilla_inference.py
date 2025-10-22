#!/usr/bin/env python3
"""Minimal CLI for running vanilla LLaVA inference on an image and prompt.

This script reuses the shared model loading utilities from ``generation_utils``
so it supports optional LoRA adapters in addition to base checkpoints.
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Tuple
from copy import deepcopy

import re
import torch
import torch.nn.functional as F
# Patch PyTorch pytree for older torch versions used with newer Transformers
if hasattr(torch, 'utils') and hasattr(torch.utils, '_pytree'):
    _pytree = torch.utils._pytree
    if not hasattr(_pytree, 'register_pytree_node') and hasattr(_pytree, '_register_pytree_node'):
        def _compat_register_pytree_node(*args, **kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k not in {'serialized_type_name', 'serialized_context'}}
            return _pytree._register_pytree_node(*args, **kwargs)
        _pytree.register_pytree_node = _compat_register_pytree_node

from transformers import TextIteratorStreamer
from transformers import modeling_utils as _transformers_modeling_utils

if not hasattr(_transformers_modeling_utils, "apply_chunking_to_forward"):
    def _apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
        if chunk_size is None or chunk_size <= 0:
            return forward_fn(*input_tensors)
        if len(input_tensors) == 0:
            raise ValueError("apply_chunking_to_forward requires at least one input tensor")
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for tensor in input_tensors:
            if tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError("All input tensors must have the same shape in the chunk dimension")
        if tensor_shape % chunk_size != 0:
            chunk_size = tensor_shape
        num_chunks = max(tensor_shape // chunk_size, 1)
        chunked_inputs = [tensor.chunk(num_chunks, dim=chunk_dim) for tensor in input_tensors]
        output_chunks = []
        for chunk_idx in range(num_chunks):
            chunk_args = [chunk_input[chunk_idx] for chunk_input in chunked_inputs]
            output_chunks.append(forward_fn(*chunk_args))
        first_chunk = output_chunks[0]
        if isinstance(first_chunk, tuple):
            return tuple(torch.cat([chunk[i] for chunk in output_chunks], dim=chunk_dim) for i in range(len(first_chunk)))
        return torch.cat(output_chunks, dim=chunk_dim)

    _transformers_modeling_utils.apply_chunking_to_forward = _apply_chunking_to_forward

if not hasattr(_transformers_modeling_utils, 'find_pruneable_heads_and_indices'):
    def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        heads_to_prune = set(heads) - already_pruned_heads
        mask = torch.ones(n_heads, head_size, dtype=torch.bool)
        for head in heads_to_prune:
            mask[head % n_heads] = False
        index = torch.arange(n_heads * head_size, dtype=torch.long).view(n_heads, head_size)
        index = index[mask].view(1, -1)
        return heads_to_prune, index
    _transformers_modeling_utils.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

if not hasattr(_transformers_modeling_utils, 'prune_linear_layer'):
    def _prune_linear_layer(layer, index, dim=0):
        if not isinstance(index, torch.Tensor):
            index = torch.tensor(index, device=layer.weight.device, dtype=torch.long)
        else:
            index = index.to(layer.weight.device, dtype=torch.long)
        index = index.long()
        W = layer.weight.index_select(dim, index).clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = index.numel()
        new_layer = torch.nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
        new_layer.weight.requires_grad = layer.weight.requires_grad
        new_layer.weight.data.copy_(W.contiguous())
        if layer.bias is not None:
            if dim == 1:
                new_layer.bias.data.copy_(layer.bias.clone().detach())
            else:
                new_layer.bias.data.copy_(layer.bias[index].clone().detach())
        return new_layer
    _transformers_modeling_utils.prune_linear_layer = _prune_linear_layer

try:
    from gazefollow.auto_phrase_grounding.detect_gaze_targets import (
        parse_person_descriptions as gdino_parse_person_descriptions,
        detect_gaze_targets as gdino_detect_gaze_targets,
    )
except Exception:  # noqa: BLE001
    gdino_parse_person_descriptions = None
    gdino_detect_gaze_targets = None

# Ensure project root is importable (needed when running from subdirectories)
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generation_utils import (  # noqa: E402
    enable_inference_optimizations,
    load_model_and_setup,
    fix_wsl_paths,
    load_image,
)
from llava.mm_utils import (  # noqa: E402
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.constants import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates  # noqa: E402


DEFAULT_IMAGE_ASPECT_RATIO = "anyres_max_9"
DEFAULT_IMAGE_GRID_PINPOINTS_EXPR = "(1x1),...,(3x3)"


def _normalize_absolute_grid_pinpoints(pairs: List[Any]) -> List[List[int]]:
    normalized: List[List[int]] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("image_grid_pinpoints must contain [width, height] pairs.")
        try:
            width = int(pair[0])
            height = int(pair[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("image_grid_pinpoints pairs must be integers.") from exc
        normalized.append([width, height])
    return normalized


def determine_base_image_resolution(model) -> int:
    config = getattr(model, "config", None)
    if config is not None:
        vision_config = getattr(config, "vision_config", None)
        if vision_config is not None:
            size = getattr(vision_config, "image_size", None)
            if isinstance(size, (list, tuple)):
                size = size[0]
            if isinstance(size, int):
                return size
        size = getattr(config, "image_crop_resolution", None)
        if isinstance(size, int):
            return size
        size = getattr(config, "image_size", None)
        if isinstance(size, int):
            return size
    return 384


def resolve_image_grid_pinpoints(value: Any, base_resolution: int) -> Optional[List[List[int]]]:
    if value is None:
        return None

    effective_base = int(base_resolution) if base_resolution else 0
    if effective_base <= 0:
        effective_base = 384

    if isinstance(value, (list, tuple)):
        return _normalize_absolute_grid_pinpoints(list(value))

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None:
            return resolve_image_grid_pinpoints(parsed, effective_base)

        if "x" in raw and "(" in raw:
            matches = re.findall(r"\((\d+)x(\d+)\)", raw)
            if not matches:
                raise ValueError("image_grid_pinpoints string did not contain any '(WxH)' pairs.")
            multipliers = [[int(w), int(h)] for w, h in matches]
            if "..." in raw and len(multipliers) >= 2:
                start_w, start_h = multipliers[0]
                end_w, end_h = multipliers[-1]
                expanded = [[i, j] for i in range(start_w, end_w + 1) for j in range(start_h, end_h + 1)]
                multipliers = expanded
            return [[w * effective_base, h * effective_base] for w, h in multipliers]

        raise ValueError("unrecognized image_grid_pinpoints string format.")

    raise ValueError(f"Unsupported image_grid_pinpoints value type: {type(value)!r}")



def ensure_image_config(
    model,
    fallback_aspect_ratio: Optional[str],
    fallback_grid_pinpoints: Optional[Any],
    override_aspect_ratio: bool = False,
    override_grid_pinpoints: bool = False,
) -> None:
    config = getattr(model, "config", None)
    if config is None:
        return

    aspect_ratio = getattr(config, "image_aspect_ratio", None)
    print(f"INFO: Current image_aspect_ratio in checkpoint: {aspect_ratio}")
    if override_aspect_ratio and fallback_aspect_ratio:
        if aspect_ratio and aspect_ratio != fallback_aspect_ratio:
            print(f"Overriding image_aspect_ratio from '{aspect_ratio}' to '{fallback_aspect_ratio}'.")
        elif not aspect_ratio:
            print(f"Setting image_aspect_ratio to '{fallback_aspect_ratio}'.")
        config.image_aspect_ratio = fallback_aspect_ratio
        aspect_ratio = fallback_aspect_ratio
    elif not aspect_ratio and fallback_aspect_ratio:
        config.image_aspect_ratio = fallback_aspect_ratio
        print(f"image_aspect_ratio missing in checkpoint; using fallback '{fallback_aspect_ratio}'.")

    base_resolution = determine_base_image_resolution(model)

    current_grid = getattr(config, "image_grid_pinpoints", None)
    resolved_grid: Optional[List[List[int]]] = None
    if not override_grid_pinpoints and current_grid not in (None, "", []):
        try:
            resolved_grid = resolve_image_grid_pinpoints(current_grid, base_resolution)
            config.image_grid_pinpoints = deepcopy(resolved_grid)
        except ValueError:
            resolved_grid = None

    should_apply_fallback_grid = fallback_grid_pinpoints not in (None, "", [])
    if should_apply_fallback_grid and (override_grid_pinpoints or resolved_grid is None):
        try:
            resolved_grid = resolve_image_grid_pinpoints(fallback_grid_pinpoints, base_resolution)
        except ValueError as exc:
            raise ValueError(
                f"Failed to parse image_grid_pinpoints fallback '{fallback_grid_pinpoints}': {exc}"
            ) from exc
        config.image_grid_pinpoints = deepcopy(resolved_grid)
        if override_grid_pinpoints:
            print("image_grid_pinpoints overridden via arguments.")
        else:
            print("image_grid_pinpoints missing in checkpoint; using fallback from arguments.")

    vision_config = getattr(config, "vision_config", None)
    if vision_config is None:
        return

    if not getattr(vision_config, "image_aspect_ratio", None) and getattr(config, "image_aspect_ratio", None):
        vision_config.image_aspect_ratio = config.image_aspect_ratio

    if not getattr(vision_config, "image_grid_pinpoints", None) and resolved_grid:
        vision_config.image_grid_pinpoints = deepcopy(resolved_grid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vanilla LLaVA inference on a single image and prompt.")
    parser.add_argument("--model-path", default="lmms-lab/llava-onevision-qwen2-7b-ov-chat", help="Model or checkpoint path to load (defaults to LLaVA-OneVision 7B).")
    parser.add_argument("--model-base", default=None, help="Optional base model path when loading LoRA adapters.")
    parser.add_argument("--adapter-path", default=None, help="Optional LoRA adapter path to merge at inference time.")
    parser.add_argument("--select-adapter", action="store_true", help="Interactively select adapter from recent checkpoints.")
    parser.add_argument("--attn-implementation", default="sdpa", help="Attention implementation passed to the loader (e.g. 'sdpa', 'flash_attention_2').")
    parser.add_argument("--load-4bit", action="store_true", help="Load the model with 4-bit quantization.")
    parser.add_argument("--load-8bit", action="store_true", help="Load the model with 8-bit quantization.")
    parser.add_argument("--image-path", required=True, help="Path to the input image.")
    
    parser.add_argument("--image-aspect-ratio", default=DEFAULT_IMAGE_ASPECT_RATIO, help="Fallback image_aspect_ratio used when absent in the checkpoint.")
    parser.add_argument("--image-grid-pinpoints", default=DEFAULT_IMAGE_GRID_PINPOINTS_EXPR, help="Fallback image_grid_pinpoints used when absent in the checkpoint. Accepts formats like '(1x1),...,(2x2)' or a JSON array of [width, height] pairs.")
    
    # prompt_group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("--prompt", default="describe every person and where is he looking at?", help="User prompt to pair with the image.")
    # prompt_group.add_argument("--prompt-file", help="Path to a text file containing the prompt.")
    
    parser.add_argument("--conv-template", default=None, help="Conversation template key (defaults to qwen_1_5 for Qwen-style models).")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature (ignored if --do-sample is False).")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling value.")
    parser.add_argument("--num-beams", type=int, default=1, help="Number of beams for beam search decoding.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling instead of greedy decoding.")
    parser.add_argument("--log-topk-tokens", type=int, default=3, help="Log the top-K token probabilities for each generated step (0 to disable).")
    parser.add_argument("--log-topk-file", default=None, help="Optional path to write top-K token statistics as JSON Lines.")
    parser.add_argument("--disable-optimizations", action="store_true", help="Skip enabling CUDA inference optimizations.")
    parser.add_argument("--save-output", default=None, help="Optional path to save the generated text as JSON (with keys image, prompt, response).")
    parser.add_argument("--chat", action="store_true", help="Enter an interactive chat loop that reuses the loaded model and image.")
    parser.add_argument("--no-stream", action="store_true", help="Disable token streaming and only print responses after generation completes.")
    parser.add_argument("--run-gdino", action="store_true", help="Run GroundingDINO gaze detection on the generated response.")
    parser.add_argument("--gdino-model-id", default="IDEA-Research/grounding-dino-base", help="Hugging Face model identifier for GroundingDINO.")
    parser.add_argument("--gdino-box-threshold", type=float, default=0.3, help="Box confidence threshold for GroundingDINO detections.")
    parser.add_argument("--gdino-text-threshold", type=float, default=0.25, help="Text matching threshold for GroundingDINO detections.")
    parser.add_argument("--gdino-device", default="cuda", help="Device string for GroundingDINO (default: cuda).")
    return parser.parse_args()

def prompt_for_adapter_path(training_outputs_dir: str = "training_outputs") -> Optional[str]:
    """Interactively prompt user to select a checkpoint or provide a custom path.
    
    Returns:
        Selected adapter path or None if user chooses no adapter.
    """
    print("=" * 50)
    print("Available recent checkpoints:")
    
    checkpoints_pattern = Path(training_outputs_dir).glob("llava-*/checkpoint-*")
    checkpoints = sorted(checkpoints_pattern, key=lambda p: p.stat().st_mtime, reverse=True)[:10]
    
    if not checkpoints:
        print("No checkpoints found.")
        return None
    
    for i, ckpt in enumerate(checkpoints):
        print(f"  [{i}] {ckpt}")
    
    print("  [-1] No adapter (base model only)")
    print("=" * 50)
    print(f"Default: [0] {checkpoints[0]}")
    
    try:
        user_input = input("Enter checkpoint number, -1 for no adapter, full path, or press Enter for default: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nUsing default checkpoint.")
        return str(checkpoints[0])
    
    if not user_input:
        selected = str(checkpoints[0])
        print(f"Selected: {selected}")
        return selected
    
    if user_input == "-1":
        print("Selected: No adapter (base model only)")
        return None
    
    if user_input.isdigit():
        idx = int(user_input)
        if 0 <= idx < len(checkpoints):
            selected = str(checkpoints[idx])
            print(f"Selected: {selected}")
            return selected
        print(f"Invalid index {idx}. Using default.")
        return str(checkpoints[0])
    
    return user_input

def read_prompt(prompt: Optional[str]) -> str:
    if prompt is not None:
        return prompt.strip()


def determine_template(model_name: str, override: Optional[str]) -> str:
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
    # Default fallback works well for most modern checkpoints
    return "qwen_1_5"


def prepare_image_tensor(image_path: str, image_processor, model) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Prepare a single image tensor and return it with the original size."""
    pil_image = load_image(image_path)
    processed = process_images([pil_image], image_processor, model.config)

    if isinstance(processed, tuple):
        image_tensor = processed[0]
    else:
        image_tensor = processed

    if isinstance(image_tensor, list):
        if not image_tensor:
            raise ValueError("Image processor returned an empty list.")
        image_tensor = image_tensor[0]

    if isinstance(image_tensor, torch.Tensor) and image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError(f"Unexpected processed image type: {type(image_tensor)}")

    image_tensor = image_tensor.to(model.device, dtype=model.dtype)
    return image_tensor, pil_image.size


def maybe_run_gdino(args: argparse.Namespace, description_text: str, image_path: str) -> Optional[Dict[str, Any]]:
    """Run GroundingDINO on the generated description when requested."""
    if not args.run_gdino:
        return None

    if gdino_parse_person_descriptions is None or gdino_detect_gaze_targets is None:
        print("GroundingDINO utilities not available; skipping detection.", file=sys.stderr)
        return {"error": "detect_gaze_targets module not available"}

    try:
        persons = gdino_parse_person_descriptions(description_text)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to parse person descriptions for GroundingDINO: {exc}", file=sys.stderr)
        return {"error": f"parse error: {exc}"}

    payload_base: Dict[str, Any] = {
        "image_path": str(image_path),
        "model_id": args.gdino_model_id,
    }

    if not persons:
        payload_base["results"] = {}
        payload_base["warning"] = "No person descriptions detected in generated text."
        return payload_base

    try:
        pil_image = load_image(image_path)
        device_choice = args.gdino_device or "cpu"
        if device_choice.lower() == "auto":
            device_choice = "cuda" if torch.cuda.is_available() else "cpu"

        detections = gdino_detect_gaze_targets(
            image=pil_image,
            persons=persons,
            model_id=args.gdino_model_id,
            box_threshold=args.gdino_box_threshold,
            text_threshold=args.gdino_text_threshold,
            device=device_choice,
        )

        payload_base["results"] = detections
        return payload_base
    except Exception as exc:  # noqa: BLE001
        print(f"GroundingDINO detection failed: {exc}", file=sys.stderr)
        return {"error": str(exc)}


def build_generation_kwargs(args: argparse.Namespace, tokenizer, image_tensor: torch.Tensor, image_size: Tuple[int, int]):
    """Prepare generation kwargs shared across turns."""
    kwargs = {
        "images": image_tensor,
        "image_sizes": [list(image_size)],
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "sequence_bias": {(tokenizer.eos_token_id,): -5.0},
        # "eos_token_id": None,
    }

    if not args.do_sample:
        kwargs.pop("temperature", None)
        kwargs.pop("top_p", None)

    if getattr(args, "log_topk_tokens", 0) > 0:
        kwargs["return_dict_in_generate"] = True

    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return kwargs


def _extract_sequences_from_generate_output(output: Any) -> torch.Tensor:
    """Normalize different generate outputs to a sequences tensor."""
    if hasattr(output, "sequences"):
        return output.sequences[0]
    if isinstance(output, torch.Tensor):
        return output[0]
    if isinstance(output, (list, tuple)) and output:
        return output[0]
    raise TypeError(f"Unsupported generation output type: {type(output)!r}")


def _format_token_piece(tokenizer, token_id: int) -> str:
    piece = tokenizer.decode([token_id], skip_special_tokens=False)
    if not piece:
        converted = tokenizer.convert_ids_to_tokens([token_id])
        piece = converted[0] if converted else str(token_id)
    return piece.replace("\n", "\\n")


def collect_topk_token_probabilities(
    tokenizer,
    model,
    sequences: torch.Tensor,
    logits: Optional[torch.Tensor],
    prompt_token_length: int,
    topk: int,
    image_tensor: torch.Tensor,
    image_size: Tuple[int, int],
) -> Optional[Dict[str, Any]]:
    """Collect top-k token probabilities for the generated continuation."""
    if topk <= 0:
        return None

    if sequences.ndim == 1:
        sequences = sequences.unsqueeze(0)

    if sequences.size(0) != 1:
        print("INFO: Top-k logging currently supports batch size 1; skipping.")
        return None

    sequences_cpu = sequences.detach().cpu()
    generated_length = sequences_cpu.size(1) - 1
    if generated_length <= 0:
        print("INFO: No generated tokens to log.")
        return None

    if logits is None:
        print("INFO: Model output did not include logits; skipping top-k logging.")
        return None

    # if logits.size(1) < generated_length:
    #     print("INFO: Unable to compute top-k probabilities; received fewer logits than generated tokens.")
    #     return None

    log_probs = F.log_softmax(logits, dim=-1)

    generated_ids = sequences_cpu[0, :].tolist()

    effective_k = min(topk, log_probs.shape[-1])
    steps_payload: List[Dict[str, Any]] = []

    for step_idx, token_id in enumerate(generated_ids):
        step_log_probs = log_probs[0, step_idx]
        topk_log_probs, topk_indices = torch.topk(step_log_probs, k=effective_k)

        actual_log_prob = float(step_log_probs[token_id].item())
        actual_prob = math.exp(actual_log_prob)
        actual_piece = _format_token_piece(tokenizer, token_id)

        top_candidates: List[Dict[str, Any]] = []
        topk_log_probs_cpu = topk_log_probs.detach().cpu()
        topk_indices_cpu = topk_indices.detach().cpu()
        for rank in range(effective_k):
            candidate_id = int(topk_indices_cpu[rank].item())
            candidate_log_prob = float(topk_log_probs_cpu[rank].item())
            candidate_prob = math.exp(candidate_log_prob)
            candidate_piece = _format_token_piece(tokenizer, candidate_id)
            top_candidates.append(
                {
                    "rank": rank + 1,
                    "token_id": candidate_id,
                    "token": candidate_piece,
                    "prob": candidate_prob,
                    "log_prob": candidate_log_prob,
                    "selected": candidate_id == token_id,
                }
            )

        steps_payload.append(
            {
                "step": step_idx + 1,
                "token_id": int(token_id),
                "token": actual_piece,
                "prob": actual_prob,
                "log_prob": actual_log_prob,
                "top_candidates": top_candidates,
            }
        )

    return {
        "topk": effective_k,
        "prompt_token_length": prompt_token_length,
        "generated_length": generated_length,
        "vocab_size": log_probs.size(-1),
        "steps": steps_payload,
    }


def print_topk_summary(payload: Dict[str, Any], max_steps: int = 5, max_candidates: int = 5) -> None:
    """Emit a concise console summary of the recorded top-k statistics."""
    steps = payload.get("steps") or []
    if not steps:
        print("INFO: No top-k token data available.")
        return

    print("=== Top-k Token Probabilities ===")
    for step in steps[:max_steps]:
        candidates = step.get("top_candidates") or []
        pieces: List[str] = []
        for candidate in candidates[:max_candidates]:
            prob = candidate.get("prob")
            token = candidate.get("token", "")
            marker = " *" if candidate.get("selected") else ""
            if isinstance(prob, (int, float)):
                pieces.append(f"{token} ({prob:.3f}){marker}")
            else:
                pieces.append(f"{token}{marker}")
        summary = ", ".join(pieces) if pieces else "(no candidates)"
        print(f"Step {step.get('step')}: {summary}")

    remaining = len(steps) - min(len(steps), max_steps)
    if remaining > 0:
        print(f"... {remaining} additional step(s) omitted from console output.")

def append_topk_log(log_path: Path, payload: Dict[str, Any], is_first_turn: bool = False, checkpoint_name: str = "", full_response: str = "", user_prompt: str = "") -> None:
    """Append a JSON payload containing top-k statistics to the log file.
    
    If is_first_turn is True, creates a new file with metadata as the first entry.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    if is_first_turn:
        # Create new file with metadata as first entry
        metadata = {
            "metadata": {
                "checkpoint_name": checkpoint_name,
                "prompt": user_prompt,
                "full_response": full_response,
                "timestamp": datetime.now().isoformat(),
                "image_path": payload.get("image_path", ""),
            }
        }
        
        with log_path.open("w", encoding="utf-8") as log_file:
            json.dump([metadata, payload], log_file, ensure_ascii=False, indent=2)
        print(f"Created top-k token log at {log_path.resolve()}")
    else:
        # Append to existing file
        if log_path.exists():
            # Read existing content
            with log_path.open("r", encoding="utf-8") as log_file:
                try:
                    data = json.load(log_file)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        
        data.append(payload)
        
        with log_path.open("w", encoding="utf-8") as log_file:
            json.dump(data, log_file, ensure_ascii=False, indent=2)
        print(f"Appended top-k token log to {log_path.resolve()}")

def generate_turn(
    conv,
    prompt_text: str,
    tokenizer,
    model,
    image_tensor: torch.Tensor,
    image_size: Tuple[int, int],
    base_gen_kwargs: dict,
    include_image_token: bool,
    stream_printer: Optional[Callable[[str], None]] = None,
) -> Tuple[str, Optional[torch.Tensor], int]:
    """Generate a response for a single user turn and update the conversation.

    Returns:
        Tuple containing the decoded response text, the full generated sequences tensor
        (or None when streaming is used), and the length of the prompt tokens.
    """

    prompt_text = prompt_text.strip()
    if not prompt_text:
        raise ValueError("Prompt text must be non-empty for generation.")

    if include_image_token and DEFAULT_IMAGE_TOKEN not in prompt_text:
        user_content = f"{DEFAULT_IMAGE_TOKEN}\n{prompt_text}"
    else:
        user_content = prompt_text

    conv.append_message(conv.roles[0], user_content)
    conv.append_message(conv.roles[1], None)
    prompt_for_tokenizer = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_for_tokenizer,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)

    gen_kwargs = base_gen_kwargs.copy()
    gen_kwargs["inputs"] = input_ids
    gen_kwargs["output_scores"] = True

    prompt_token_length = input_ids.shape[1]

    if stream_printer:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

        def run_generate():
            with torch.inference_mode():
                model.generate(**gen_kwargs)

        thread = Thread(target=run_generate)
        thread.start()

        collected_chunks = []
        for chunk in streamer:
            stream_printer(chunk)
            collected_chunks.append(chunk)

        thread.join()
        response = "".join(collected_chunks).strip()
        sequences_tensor: Optional[torch.Tensor] = None
    else:
        with torch.inference_mode():
            generation_output = model.generate(**gen_kwargs)
        output_ids = _extract_sequences_from_generate_output(generation_output)
        # generated_tokens = output_ids.tolist()
        response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        sequences_tensor = output_ids
        if 'scores' in generation_output:
            # generation_output.scores is typically a tuple/list of per-step logits tensors
            # Stack them into a single tensor of shape (batch_size, seq_len, vocab_size)
            scores = getattr(generation_output, "scores", None)
            if isinstance(scores, (tuple, list)):
                logits = torch.stack(list(scores), dim=1)
            else:
                logits = scores
        else:
            logits = None

    conv.messages[-1][1] = response

    return response, sequences_tensor, prompt_token_length, logits


def main() -> None:
    args = parse_args()

    if args.load_4bit and args.load_8bit:
        raise ValueError("Cannot enable both --load-4bit and --load-8bit at the same time.")

    if not args.disable_optimizations:
        enable_inference_optimizations()

    adapter_path = args.adapter_path
    if args.select_adapter:
        adapter_path = prompt_for_adapter_path()

    tokenizer, model, image_processor, _ = load_model_and_setup(
        model_path=args.model_path,
        attn_implementation=args.attn_implementation,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        model_base=args.model_base,
        adapter_path=fix_wsl_paths(adapter_path) if adapter_path else None,
    )

    ensure_image_config(
        model,
        args.image_aspect_ratio,
        args.image_grid_pinpoints,
        override_aspect_ratio=args.image_aspect_ratio != DEFAULT_IMAGE_ASPECT_RATIO,
        override_grid_pinpoints=args.image_grid_pinpoints != DEFAULT_IMAGE_GRID_PINPOINTS_EXPR,
    )

    prompt_text = read_prompt(args.prompt)
    image_path = fix_wsl_paths(args.image_path)

    image_tensor, image_size = prepare_image_tensor(image_path, image_processor, model)

    model_name_source = model.config._name_or_path if hasattr(model.config, "_name_or_path") else args.model_path
    model_name = get_model_name_from_path(model_name_source)
    conv_name = determine_template(model_name, args.conv_template)

    conv = conv_templates[conv_name].copy()
    conv.tokenizer = tokenizer

    if args.log_topk_tokens < 0:
        raise ValueError("--log-topk-tokens must be non-negative.")

    topk_log_path: Optional[Path] = None
    if args.log_topk_tokens > 0:
        if args.log_topk_file:
            topk_log_path = Path(args.log_topk_file)
        else:
            # Create dedicated topk_logs folder
            topk_dir = Path("topk_logs")
            topk_dir.mkdir(parents=True, exist_ok=True)
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            topk_log_path = topk_dir / f"topk_log_{timestamp}.json"
        print(f"INFO: Top-k token statistics will be written to {topk_log_path}.")

    base_gen_kwargs = build_generation_kwargs(args, tokenizer, image_tensor, image_size)
    stream_output = not args.no_stream
    if args.log_topk_tokens > 0 and stream_output:
        print("INFO: Disabling streaming to enable top-k token logging.")
        stream_output = False

    if args.chat and not prompt_text:
        try:
            prompt_text = input("Enter initial user prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting without running inference.")
            return

    if not prompt_text:
        raise ValueError("A non-empty prompt is required to start the conversation.")

    def stream_printer(chunk: str) -> None:
        print(chunk, end="", flush=True)

    def maybe_record_topk(
        user_prompt: str,
        response_text: str,
        sequences_tensor: Optional[torch.Tensor],
        logits: Optional[torch.Tensor],
        prompt_token_length: int,
        turn_id: int,
    ) -> bool:
        if args.log_topk_tokens <= 0 or sequences_tensor is None or topk_log_path is None:
            print(f"DEBUG: Skipping top-k token logging for this turn due to:{' no logging requested' if args.log_topk_tokens <= 0 else ''}{' missing sequences tensor' if sequences_tensor is None else ''}{' no log path' if topk_log_path is None else ''}")
            return False

        payload = collect_topk_token_probabilities(
            tokenizer,
            model,
            sequences_tensor,
            logits,
            prompt_token_length,
            args.log_topk_tokens,
            image_tensor,
            image_size,
        )

        if payload is None:
            return False

        payload.update(
            {
                "turn_index": turn_id,
                "mode": "chat" if args.chat else "single",
                "prompt": user_prompt,
                "response": response_text,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "image_path": image_path,
                "model_path": args.model_path,
            }
        )

        print_topk_summary(payload)
        # Pass additional metadata for first turn
        is_first_turn = (turn_id == 0)
        checkpoint_name = args.model_path if args.adapter_path is None else args.adapter_path
        append_topk_log(
            topk_log_path, 
            payload, 
            is_first_turn=is_first_turn,
            checkpoint_name=checkpoint_name,
            full_response=response_text,
            user_prompt=user_prompt
        )
        return True

    turn_index = 0

    print("Generating response...")
    print("\n=== Model Response ===")
    if stream_output:
        response, sequences, prompt_token_length, logits = generate_turn(
            conv,
            prompt_text,
            tokenizer,
            model,
            image_tensor,
            image_size,
            base_gen_kwargs,
            include_image_token=True,
            stream_printer=stream_printer,
        )
        print()
    else:
        response, sequences, prompt_token_length, logits = generate_turn(
            conv,
            prompt_text,
            tokenizer,
            model,
            image_tensor,
            image_size,
            base_gen_kwargs,
            include_image_token=True,
        )
        print(response)

    if maybe_record_topk(prompt_text, response, sequences, logits, prompt_token_length, turn_index):
        turn_index += 1

    if args.chat:
        while True:
            try:
                next_prompt = input("\nUser> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break

            if not next_prompt:
                continue

            if next_prompt.lower() in {"exit", "quit", "q"}:
                print("Exiting chat.")
                break

            if stream_output:
                print("\nAssistant> ", end="", flush=True)
                response, sequences, prompt_token_length, logits = generate_turn(
                    conv,
                    next_prompt,
                    tokenizer,
                    model,
                    image_tensor,
                    image_size,
                    base_gen_kwargs,
                    include_image_token=False,
                    stream_printer=stream_printer,
                )
                print()
            else:
                response, sequences, prompt_token_length, logits = generate_turn(
                    conv,
                    next_prompt,
                    tokenizer,
                    model,
                    image_tensor,
                    image_size,
                    base_gen_kwargs,
                    include_image_token=False,
                )
                print("\nAssistant>")
                print(response)

            if maybe_record_topk(next_prompt, response, sequences, prompt_token_length, turn_index):
                turn_index += 1

    gdino_payload = maybe_run_gdino(args, response, image_path)
    if gdino_payload is not None:
        print("\n=== GroundingDINO Results ===")
        print(json.dumps(gdino_payload, indent=2))

    if args.save_output:
        output_path = Path(args.save_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.chat:
            conversation_payload = [
                {"role": role, "message": message}
                for role, message in conv.messages
            ]
            payload = {
                "image": image_path,
                "conversation": conversation_payload,
            }
        else:
            payload = {
                "image": image_path,
                "prompt": prompt_text,
                "response": response,
            }

        if gdino_payload is not None:
            payload["gdino"] = gdino_payload

        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved response to {output_path}")


if __name__ == "__main__":
    main()
