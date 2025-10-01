#!/usr/bin/env python3
"""
Evaluation script for trained LLaVA models using custom JSON datasets.

This script loads a trained model and evaluates it on a given JSON dataset
with images, calculating various metrics and saving detailed results.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from tqdm import tqdm
import torch
import time
import os
import math

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

# Import metrics calculation if available
try:
    from generation_metrics import calculate_metrics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Warning: generation_metrics module not available. Only basic evaluation will be performed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained LLaVA model on custom JSON dataset.")
    
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
    
    # Other arguments
    parser.add_argument("--disable-optimizations", action="store_true", help="Skip enabling CUDA optimizations.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress information.")
    parser.add_argument("--safe-mode", action="store_true", help="Enable safe mode with more aggressive memory cleanup and smaller batches.")
    parser.add_argument("--no-loss", action="store_true", help="Disable loss calculation entirely to avoid CUDA errors.")
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


def prepare_image_tensor(image_path: str, image_processor, model) -> Optional[Tuple[torch.Tensor, Tuple[int, int]]]:
    """Prepare image tensor, return None if image cannot be loaded."""
    try:
        # Clear CUDA cache before processing (with error handling)
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass  # Ignore CUDA errors during cache cleanup
            
        pil_image = load_image(image_path)
        
        # Process image with error handling
        try:
            processed = process_images([pil_image], image_processor, model.config)
        except Exception as e:
            print(f"Error in process_images for {image_path}: {e}")
            return None

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

        # Move to device with error handling
        try:
            image_tensor = image_tensor.to(model.device, dtype=model.dtype)
        except Exception as e:
            print(f"CUDA error when moving tensor to device for {image_path}: {e}")
            return None
            
        return image_tensor, pil_image.size
    
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


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
    try:
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
        # Mask out image placeholder tokens so they do not participate in the loss
        invalid_label_mask = labels < 0
        if invalid_label_mask.any():
            labels = labels.masked_fill(invalid_label_mask, IGNORE_INDEX)

        # ensure we ignore sequence regions beyond the language tokens
        if prompt_length > 0:
            labels[:, :prompt_length] = IGNORE_INDEX

        vocab_size = getattr(model.config, "vocab_size", None)
        # tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
        # vocab_size = min(vocab_size, tokenizer_vocab_size) if vocab_size and tokenizer_vocab_size else vocab_size or tokenizer_vocab_size
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
    except Exception as exc:  # pragma: no cover - best effort safeguard
        print(f"Loss calculation failed: {exc}")
        return None


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
) -> Dict[str, Any]:
    """
    Custom evaluation function for training-time evaluation.
    
    This function is designed to be called during training to evaluate the model
    on a validation dataset using the same logic as evaluate_model.py.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer
        image_processor: Image processor
        eval_dataset: Evaluation dataset (LazySupervisedDataset)
        conv_template: Conversation template to use
        max_new_tokens: Maximum tokens to generate
        focus_loss_after_looking: Whether to focus loss on tokens after phrase
        focus_loss_phrase: Target phrase for focused loss
        focus_loss_threshold: Maximum loss when focus phrase not found
        no_loss: Disable loss calculation
        verbose: Print detailed progress
        limit: Optional limit on number of samples to evaluate (for testing)
        
    Returns:
        Dictionary of evaluation metrics
    """
    from tqdm import tqdm
    
    model.eval()
    predictions = []
    ground_truths = []
    losses = []
    failed_samples = []
    
    # Build focus phrase token IDs if needed
    focus_phrase_token_ids = []
    if focus_loss_after_looking:
        focus_phrase_token_ids = build_focus_phrase_token_ids(tokenizer, focus_loss_phrase)
    
    # Get dataset samples
    total_samples = len(eval_dataset)
    if limit is not None and limit > 0:
        total_samples = min(total_samples, limit)
        if verbose:
            print(f"Running custom evaluation on {total_samples} samples (limited from {len(eval_dataset)})...")
    elif verbose:
        print(f"Running custom evaluation on {total_samples} samples...")
    
    with torch.no_grad():
        iterator = tqdm(range(total_samples), desc="Evaluating") if verbose else range(total_samples)
        for idx in iterator:
            try:
                # Get sample from dataset
                sample = eval_dataset.list_data_dict[idx]
                
                # Extract conversations
                conversations = sample.get("conversations", [])
                prompt = extract_prompt_from_conversation(conversations)
                ground_truth = extract_ground_truth_from_conversation(conversations)
                
                if not prompt or not ground_truth:
                    failed_samples.append({"index": idx, "reason": "Missing prompt or ground truth"})
                    continue
                
                # Get image path
                image_file = sample.get("image", "")
                if isinstance(image_file, list):
                    image_file = image_file[0]
                
                # Handle relative paths from dataset
                if not os.path.isabs(image_file):
                    # Try to resolve relative to dataset's data_path if available
                    data_path = getattr(eval_dataset, 'data_path', None)
                    if data_path:
                        base_dir = Path(data_path).parent
                        image_file = str(base_dir / image_file)
                
                # Process image
                image_result = prepare_image_tensor(
                    image_file,
                    image_processor,
                    model
                )
                
                if image_result is None:
                    failed_samples.append({"index": idx, "reason": f"Failed to load image: {image_file}"})
                    continue
                
                image_tensor, image_size = image_result
                
                # Generate response
                generation_kwargs = {
                    "do_sample": False,
                    "max_new_tokens": max_new_tokens,
                    "use_cache": True,
                    "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                }
                
                prediction = generate_response(
                    prompt_text=prompt,
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
                
                # Compute loss if enabled
                if not no_loss:
                    loss = compute_ground_truth_loss(
                        prompt_text=prompt,
                        ground_truth=ground_truth,
                        tokenizer=tokenizer,
                        model=model,
                        conv_template=conv_template,
                        image_tensor=image_tensor,
                        image_size=image_size,
                        focus_loss_after_phrase=focus_loss_after_looking,
                        focus_loss_phrase_token_ids=focus_phrase_token_ids if focus_phrase_token_ids else None,
                        focus_loss_missing_value=focus_loss_threshold,
                    )
                    
                    if loss is not None and math.isfinite(loss):
                        losses.append(loss)
                
            except Exception as e:
                if verbose:
                    print(f"Error evaluating sample {idx}: {e}")
                failed_samples.append({"index": idx, "reason": str(e)})
                continue
            
            # Clear cache periodically
            if (idx + 1) % 10 == 0 and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass
    
    # Calculate metrics
    metrics = {}
    
    if predictions:
        basic_metrics = calculate_basic_metrics(predictions, ground_truths)
        metrics.update(basic_metrics)
        
        # Try advanced metrics if available
        if METRICS_AVAILABLE:
            try:
                advanced_metrics = calculate_metrics(predictions, ground_truths)
                metrics.update(advanced_metrics)
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not calculate advanced metrics: {e}")
    
    if losses:
        metrics.update({
            "eval_loss": sum(losses) / len(losses),
            "eval_min_loss": min(losses),
            "eval_max_loss": max(losses),
        })
    
    metrics.update({
        "eval_samples": total_samples,
        "eval_successful": len(predictions),
        "eval_failed": len(failed_samples),
        "eval_success_rate": len(predictions) / total_samples if total_samples > 0 else 0.0,
    })
    
    model.train()
    return metrics


def main():
    args = parse_args()
    
    if args.load_4bit and args.load_8bit:
        raise ValueError("Cannot enable both --load-4bit and --load-8bit.")
    
    # Set CUDA launch blocking for better error reporting
    import os
    if torch.cuda.is_available():
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        print("Set CUDA_LAUNCH_BLOCKING=1 for better error reporting")
    
    # Enable optimizations
    if not args.disable_optimizations:
        enable_inference_optimizations()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("LLaVA Model Evaluation")
    print("=" * 60)
    
    # Load model
    print("\n1. Loading model...")
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
    conv_template = determine_template(model_name, args.conv_template)
    print(f"Using conversation template: {conv_template}")

    # Prepare generation kwargs
    generation_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }

    if not args.do_sample:
        generation_kwargs.pop("temperature", None)
        generation_kwargs.pop("top_p", None)

    focus_loss_kwargs = {}
    focus_phrase_token_ids: List[List[int]] = []
    if args.focus_loss_after_looking:
        focus_phrase_token_ids = build_focus_phrase_token_ids(tokenizer, args.focus_loss_phrase)
        if not focus_phrase_token_ids:
            print(
                "Warning: focus loss requested but tokenizer produced no token ids for the target phrase. "
                "Falling back to standard loss."
            )
        else:
            focus_loss_kwargs = {
                "focus_loss_after_phrase": True,
                "focus_loss_phrase_token_ids": focus_phrase_token_ids,
                "focus_loss_missing_value": args.focus_loss_threshold,
            }
            # Persist configuration for downstream utilities that may use the model directly.
            model.config.focus_loss_after_phrase = True
            model.config.focus_loss_phrase_token_ids = focus_phrase_token_ids
            model.config.focus_loss_missing_value = args.focus_loss_threshold

    # Load dataset
    print("\n2. Loading dataset...")
    dataset = load_dataset(args.dataset_json, args.limit)
    images_dir = Path(args.images_dir)
    
    # Evaluation loop
    print("\n3. Running evaluation...")
    predictions = []
    ground_truths = []
    failed_samples = []
    evaluation_results = []
    losses = []
    
    start_time = time.time()
    
    # Process in smaller batches to prevent memory issues
    batch_size = 1 if args.safe_mode else args.batch_size  # Process 1 sample at a time in safe mode, 5 otherwise
    
    for batch_start in range(0, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch_samples = dataset[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(dataset) + batch_size - 1)//batch_size} (samples {batch_start+1}-{batch_end})")
        
        for i, sample in enumerate(tqdm(batch_samples, desc=f"Batch {batch_start//batch_size + 1}")):
            sample_idx = batch_start + i
            sample_id = sample.get("id", f"sample_{sample_idx}")
        image_path = sample.get("image", "")
        conversations = sample.get("conversations", [])
        
        if args.verbose:
            print(f"\nProcessing sample {sample_id}...")
        
        # Extract prompt and ground truth
        prompt = extract_prompt_from_conversation(conversations)
        ground_truth = extract_ground_truth_from_conversation(conversations)
        
        if not prompt or not ground_truth:
            failed_samples.append({
                "id": sample_id,
                "reason": "Missing prompt or ground truth",
                "prompt": prompt,
                "ground_truth": ground_truth
            })
            continue
        
        # Construct full image path
        if image_path.endswith('.jpg') or image_path.endswith('.png') or image_path.endswith('.jpeg'):
            full_image_path = images_dir / image_path
        else:
            # Try common extensions
            for ext in ['.jpg', '.png', '.jpeg']:
                potential_path = images_dir / f"{image_path}{ext}"
                if potential_path.exists():
                    full_image_path = potential_path
                    break
            else:
                failed_samples.append({
                    "id": sample_id,
                    "reason": f"Image not found: {image_path}",
                    "prompt": prompt,
                    "ground_truth": ground_truth
                })
                continue
        
        # Prepare image tensor
        image_result = prepare_image_tensor(str(full_image_path), image_processor, model)
        if image_result is None:
            failed_samples.append({
                "id": sample_id,
                "reason": f"Failed to process image: {full_image_path}",
                "prompt": prompt,
                "ground_truth": ground_truth
            })
            continue
        
        image_tensor, image_size = image_result
        
        # Generate prediction and optionally compute loss against ground truth
        try:
            prediction = None
            # Generate prediction only if iterative generation is enabled
            if args.use_iterative_generation:
                prediction = generate_response(
                    prompt,
                    image_tensor,
                    image_size,
                    tokenizer,
                    model,
                    conv_template,
                    generation_kwargs,
                )

            sample_loss = None
            if not args.no_loss:
                loss_value = compute_ground_truth_loss(
                    prompt,
                    ground_truth,
                    tokenizer,
                    model,
                    conv_template,
                    image_tensor,
                    image_size,
                    **focus_loss_kwargs,
                )
                if loss_value is not None and math.isfinite(loss_value):
                    sample_loss = loss_value
                    losses.append(sample_loss)
                    if args.verbose:
                        print(f"Sample {sample_id} loss: {sample_loss:.6f}")
                else:
                    if args.verbose:
                        reason = "None" if loss_value is None else "non-finite"
                        print(f"Loss calculation skipped for {sample_id}: returned {reason} value")

            # Only add to predictions list if we actually generated a prediction
            if prediction is not None:
                predictions.append(prediction)
            ground_truths.append(ground_truth)
            
            # Store detailed result
            result = {
                "id": sample_id,
                "image_path": str(full_image_path),
                "prompt": prompt,
                "ground_truth": ground_truth,
                "success": True
            }
            if prediction is not None:
                result["prediction"] = prediction
            if sample_loss is not None:
                result["loss"] = sample_loss
            evaluation_results.append(result)
            
            if args.verbose:
                print(f"Prompt: {prompt}")
                print(f"Ground truth: {ground_truth}")
                if prediction is not None:
                    print(f"Prediction: {prediction}")
                else:
                    print("Prediction: [skipped - iterative generation not enabled]")
                if sample_loss is not None:
                    print(f"Loss: {sample_loss:.6f}")
                print("-" * 40)
        
        except Exception as e:
            failed_samples.append({
                "id": sample_id,
                "reason": f"Generation error: {str(e)}",
                "prompt": prompt,
                "ground_truth": ground_truth
            })
            continue
        
        finally:
            # Clean up GPU memory after each sample to prevent CUDA errors
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass  # Ignore CUDA errors during cleanup
        
        # Clean up GPU memory after each batch
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                if args.verbose:
                    print(f"Cleaned GPU memory after batch {batch_start//batch_size + 1}")
            except RuntimeError:
                pass  # Ignore CUDA errors during cleanup
    
    end_time = time.time()
    evaluation_time = end_time - start_time
    
    # Calculate metrics
    print("\n4. Calculating metrics...")
    # Calculate loss metrics if available
    loss_metrics = {}
    if losses:
        loss_metrics.update({
            "average_loss": sum(losses) / len(losses),
            "min_loss": min(losses),
            "max_loss": max(losses),
            "samples_with_loss": len(losses),
        })

    if predictions:
        basic_metrics = calculate_basic_metrics(predictions, ground_truths)
        
        # Try to calculate advanced metrics if available
        advanced_metrics = {}
        if METRICS_AVAILABLE:
            try:
                advanced_metrics = calculate_metrics(predictions, ground_truths)
            except Exception as e:
                print(f"Warning: Could not calculate advanced metrics: {e}")
        
        
        # Combine all metrics
        final_metrics = {**basic_metrics, **advanced_metrics, **loss_metrics}
        
        # Add evaluation statistics
        total_processed = len(evaluation_results)  # Total samples that were successfully processed (regardless of prediction generation)
        final_metrics.update({
            "total_samples": len(dataset),
            "successful_predictions": len(predictions),
            "failed_samples": len(failed_samples),
            "successfully_processed_samples": total_processed,
            "success_rate": len(predictions) / len(dataset) if len(dataset) > 0 else 0,
            "processing_rate": total_processed / len(dataset) if len(dataset) > 0 else 0,
            "evaluation_time_seconds": evaluation_time,
            "average_time_per_sample": evaluation_time / total_processed if total_processed > 0 else 0,
        })
        
    else:
        total_processed = len(evaluation_results)
        if total_processed > 0:
            print(f"Processed {total_processed} samples but no predictions were generated (--use-iterative-generation not enabled).")
        else:
            print("No successful predictions to evaluate!")
        final_metrics = {
            "total_samples": len(dataset),
            "successful_predictions": 0,
            "failed_samples": len(failed_samples),
            "successfully_processed_samples": total_processed,
            "success_rate": 0.0,
            "processing_rate": total_processed / len(dataset) if len(dataset) > 0 else 0,
            "evaluation_time_seconds": evaluation_time,
            "average_time_per_sample": evaluation_time / total_processed if total_processed > 0 else 0,
            "average_loss": sum(losses) / len(losses) if losses else None,
            "samples_with_loss": len(losses),
        }

        final_metrics.update(loss_metrics)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # Print basic metrics
    if predictions:
        print("\n📊 BASIC METRICS:")
        for metric_name, value in basic_metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")
        
        # Print advanced metrics if available
        if advanced_metrics:
            print("\n🔍 ADVANCED METRICS:")
            for metric_name, value in advanced_metrics.items():
                if isinstance(value, float):
                    print(f"  {metric_name}: {value:.4f}")
                else:
                    print(f"  {metric_name}: {value}")
    else:
        print("\n📊 PREDICTION METRICS: No predictions generated (--use-iterative-generation not enabled)")
    
    # Print loss metrics prominently (regardless of whether predictions were generated)
    if loss_metrics:
        print("\n📉 LOSS METRICS:")
        for metric_name, value in loss_metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.6f}")
            else:
                print(f"  {metric_name}: {value}")
    elif losses:  # Handle case where we have losses but no loss_metrics dict
        print("\n📉 LOSS METRICS:")
        print(f"  average_loss: {sum(losses)/len(losses):.6f}")
        print(f"  min_loss: {min(losses):.6f}")
        print(f"  max_loss: {max(losses):.6f}")
        print(f"  samples_with_loss: {len(losses)}")
    else:
        print("\n⚠️  LOSS METRICS: No loss values calculated")
        
    # Print evaluation statistics
    print("\n📈 EVALUATION STATISTICS:")
    stats_metrics = {
        "total_samples": final_metrics["total_samples"],
        "successfully_processed_samples": final_metrics["successfully_processed_samples"],
        "successful_predictions": final_metrics["successful_predictions"],
        "failed_samples": final_metrics["failed_samples"],
        "processing_rate": final_metrics["processing_rate"],
        "success_rate": final_metrics["success_rate"],
        "evaluation_time_seconds": final_metrics["evaluation_time_seconds"],
        "average_time_per_sample": final_metrics["average_time_per_sample"],
    }
    for metric_name, value in stats_metrics.items():
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.4f}")
        else:
            print(f"  {metric_name}: {value}")
    
    # Save results
    print(f"\n5. Saving results to {output_dir}...")
    
    # Save metrics
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to: {metrics_file}")
    
    # Save detailed predictions if requested
    if args.save_predictions:
        predictions_file = output_dir / "predictions.json"
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        print(f"Detailed predictions saved to: {predictions_file}")
    
    # Save failed samples
    if failed_samples:
        failed_file = output_dir / "failed_samples.json"
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_samples, f, indent=2, ensure_ascii=False)
        print(f"Failed samples saved to: {failed_file}")
    
    # Save configuration
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
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Evaluation configuration saved to: {config_file}")
    
    print("\nEvaluation completed successfully!")
    
    if failed_samples:
        print(f"\n⚠️  Warning: {len(failed_samples)} samples failed to process. Check failed_samples.json for details.")
    
    # Print loss summary
    if losses:
        print(f"\n📉 Loss Summary: Calculated loss for {len(losses)}/{total_processed} successful predictions")
        print(f"   Average Loss: {sum(losses)/len(losses):.6f}")
        print(f"   Loss Range: {min(losses):.6f} - {max(losses):.6f}")
    else:
        if args.no_loss:
            print(f"\n📊 Loss calculation was disabled via --no-loss flag.")
        else:
            print(f"\n⚠️  No loss values were calculated. Consider using --verbose for debugging.")


if __name__ == "__main__":
    main()
