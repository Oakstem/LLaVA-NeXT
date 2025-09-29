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
import torch.nn.functional as F
import time
import os
from collections import defaultdict

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
    parser.add_argument("--model-path", required=True, help="Path to trained model or checkpoint to evaluate.")
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


def calculate_loss(
    labels: torch.Tensor,
    logits: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> float:
    """
    Calculate cross-entropy loss between predicted logits and target labels.
    This matches the loss calculation used during training.
    """
    # Simple cross-entropy loss calculation
    loss_fct = F.cross_entropy
    # Flatten the tokens
    shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = labels[..., 1:].contiguous().view(-1)
    # Move labels to same device as logits
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels, ignore_index=ignore_index)
    return loss.item()


def compute_ground_truth_loss(
    prompt_text: str,
    ground_truth: str,
    tokenizer,
    model,
    conv_template: str,
    image_tensor: torch.Tensor,
    image_size: Tuple[int, int],
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
        labels[:, :prompt_length] = IGNORE_INDEX

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                images=image_tensor,
                image_sizes=[list(image_size)],
                modalities=["image"],
                use_cache=False,
            )

        return calculate_loss(labels, outputs.logits, ignore_index=IGNORE_INDEX)
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
    batch_size = 1 if args.safe_mode else 5  # Process 1 sample at a time in safe mode, 5 otherwise
    
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
        
        # Generate prediction (loss calculation temporarily disabled due to CUDA assertion errors)
        try:
            # Generate prediction normally
            prediction = generate_response(
                prompt,
                image_tensor,
                image_size,
                tokenizer,
                model,
                conv_template,
                generation_kwargs,
            )
            
            # Temporarily disable loss calculation to avoid CUDA assertion errors
            # TODO: Fix loss calculation with proper token ID validation
            sample_loss = None
            predictions.append(prediction)
            ground_truths.append(ground_truth)
            if sample_loss is not None:
                losses.append(sample_loss)
                if args.verbose:
                    print(f"Sample {sample_id} loss: {sample_loss:.4f}")
            
            # Store detailed result
            result = {
                "id": sample_id,
                "image_path": str(full_image_path),
                "prompt": prompt,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "success": True
            }
            if sample_loss is not None:
                result["loss"] = sample_loss
            evaluation_results.append(result)
            
            if args.verbose:
                print(f"Prompt: {prompt}")
                print(f"Ground truth: {ground_truth}")
                print(f"Prediction: {prediction}")
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
    
    if predictions:
        basic_metrics = calculate_basic_metrics(predictions, ground_truths)
        
        # Try to calculate advanced metrics if available
        advanced_metrics = {}
        if METRICS_AVAILABLE:
            try:
                advanced_metrics = calculate_metrics(predictions, ground_truths)
            except Exception as e:
                print(f"Warning: Could not calculate advanced metrics: {e}")
        
        # Calculate loss metrics if available
        loss_metrics = {}
        if losses:
            loss_metrics.update({
                "average_loss": sum(losses) / len(losses),
                "min_loss": min(losses),
                "max_loss": max(losses),
                "samples_with_loss": len(losses),
            })
        
        # Combine all metrics
        final_metrics = {**basic_metrics, **advanced_metrics, **loss_metrics}
        
        # Add evaluation statistics
        final_metrics.update({
            "total_samples": len(dataset),
            "successful_predictions": len(predictions),
            "failed_samples": len(failed_samples),
            "success_rate": len(predictions) / len(dataset),
            "evaluation_time_seconds": evaluation_time,
            "average_time_per_sample": evaluation_time / len(predictions) if predictions else 0,
        })
        
    else:
        print("No successful predictions to evaluate!")
        final_metrics = {
            "total_samples": len(dataset),
            "successful_predictions": 0,
            "failed_samples": len(failed_samples),
            "success_rate": 0.0,
            "evaluation_time_seconds": evaluation_time,
            "average_loss": None,
            "samples_with_loss": 0,
        }
    
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
        
        # Print loss metrics prominently
        if loss_metrics:
            print("\n📉 LOSS METRICS:")
            for metric_name, value in loss_metrics.items():
                if isinstance(value, float):
                    print(f"  {metric_name}: {value:.6f}")
                else:
                    print(f"  {metric_name}: {value}")
        else:
            print("\n⚠️  LOSS METRICS: No loss values calculated")
        
        # Print evaluation statistics
        print("\n📈 EVALUATION STATISTICS:")
        stats_metrics = {
            "total_samples": final_metrics["total_samples"],
            "successful_predictions": final_metrics["successful_predictions"],
            "failed_samples": final_metrics["failed_samples"],
            "success_rate": final_metrics["success_rate"],
            "evaluation_time_seconds": final_metrics["evaluation_time_seconds"],
            "average_time_per_sample": final_metrics["average_time_per_sample"],
        }
        for metric_name, value in stats_metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")
    
    else:
        print("\n❌ No successful predictions to evaluate!")
        for metric_name, value in final_metrics.items():
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
        print(f"\n� Loss Summary: Calculated loss for {len(losses)}/{len(predictions)} successful predictions")
        print(f"   Average Loss: {sum(losses)/len(losses):.6f}")
        print(f"   Loss Range: {min(losses):.6f} - {max(losses):.6f}")
    else:
        if args.no_loss:
            print(f"\n📊 Loss calculation was disabled via --no-loss flag.")
        else:
            print(f"\n⚠️  No loss values were calculated. Consider using --verbose for debugging.")


if __name__ == "__main__":
    main()