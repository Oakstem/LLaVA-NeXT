import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path
import copy
from typing import Dict, List, Optional, Any, Tuple, Union

import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from io import BytesIO
import requests
import pandas as pd

from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import modeling_utils as _transformers_modeling_utils

if not hasattr(_transformers_modeling_utils, 'apply_chunking_to_forward'):
    def _apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
        if chunk_size is None or chunk_size <= 0:
            return forward_fn(*input_tensors)
        if len(input_tensors) == 0:
            raise ValueError('apply_chunking_to_forward requires at least one input tensor')
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for tensor in input_tensors:
            if tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError('All input tensors must have the same shape in the chunk dimension')
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

from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN, IGNORE_INDEX
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from gazefollow.gazefollow_utils import _pixel_to_token_indices_helper_anyres
import traceback

from llava.model.llava_arch import unpad_image

# Import our enhanced generation metrics
from generation_metrics import (
    ConfidenceMetrics, RepetitivityMetrics, TopKCandidateEvaluator,
    generate_next_token_with_evaluation, create_generation_summary,
    analyze_generation_quality, calculate_attention_correlation_from_similarity
)

def enable_inference_optimizations() -> None:
    """Enable tf32 and other CUDA optimizations for faster inference"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping inference optimizations")
        return
    
    # Enable tf32 for faster inference on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("tf32 enabled for faster inference")
    
    # Additional inference optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    print("CUDNN optimizations enabled for inference")

def normalize_embedding(embedding: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize an embedding vector to unit length (L2 norm).
    
    Args:
        embedding: Tensor of shape [dim] or [batch, dim]
        eps: Small value to avoid division by zero
    
    Returns:
        Normalized embedding tensor of the same shape
    """
    norm = embedding.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)
    return embedding / norm

def cross_cosine_similarity(vectors: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between all vectors in a [N, D] tensor.
    Returns a [N, N] similarity matrix.
    """
    # Normalize each vector to unit length
    vectors_norm = torch.nn.functional.normalize(vectors, p=2, dim=1)
    # Compute cosine similarity matrix
    similarity_matrix = torch.matmul(vectors_norm, vectors_norm.T)
    return similarity_matrix

# Gaze-Guided Token Selection Functions
def select_token_by_gaze_correlation(
    logits: torch.Tensor,
    text_embedding: torch.Tensor,
    image_embeddings: torch.Tensor,
    mask: Optional[np.ndarray],
    tokenizer: "PreTrainedTokenizer",
    model: "PreTrainedModel",
    top_k: int = 10,
    similarity_weight: float = 0.7,
    probability_weight: float = 0.3
) -> Tuple[torch.Tensor, str, Dict[str, Any]]:
    """
    Select next token based on correlation with target gaze area rather than just probability.
    
    Args:
        logits: Model output logits for next token prediction
        text_embedding: Current text token embedding (not used in updated version)
        image_embeddings: Image patch embeddings
        mask: Binary mask indicating target or source gaze area
        tokenizer: Tokenizer for decoding candidate tokens
        model: The language model to extract token embeddings from
        top_k: Number of top probability candidates to consider
        similarity_weight: Weight for similarity score (0-1)
        probability_weight: Weight for probability score (0-1)
        
    Returns:
        Tuple of (selected_token_id, token_text, selection_metrics)
    """
    # Get top-k candidates by probability
    probs = torch.softmax(logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, top_k)
    
    # Calculate similarity scores for each candidate
    candidate_scores = []
    selection_metrics = {
        "candidates": [],
        "similarity_scores": [],
        "probability_scores": [],
        "combined_scores": [],
        "selected_index": 0
    }
    
    # If no target mask provided, fall back to probability-based selection
    if mask is None or image_embeddings is None:
        selected_idx = top_k_indices[0]
        selected_text = tokenizer.decode([selected_idx], skip_special_tokens=True)
        selection_metrics["fallback_reason"] = "no_mask_or_embeddings"
        return selected_idx.unsqueeze(0), selected_text, selection_metrics
    
    # Get target area embeddings
    target_indices = np.where(mask.flatten() > 0)[0]
    if len(target_indices) == 0:
        # No target area found, fall back to probability
        selected_idx = top_k_indices[0]
        selected_text = tokenizer.decode([selected_idx], skip_special_tokens=True)
        selection_metrics["fallback_reason"] = "empty_mask"
        return selected_idx.unsqueeze(0), selected_text, selection_metrics
    
    target_embeddings = image_embeddings[target_indices]  # Shape: [n_target_patches, embed_dim]
    target_center_embedding = target_embeddings.mean(dim=0)
    
    token_embedding_layer = model.get_model().embed_tokens

    # Lets convert target embeddings to the same embeddings space as the given model output
    target_token_ids = torch.argmax(model.get_output_embeddings()(target_embeddings), dim=-1)
    target_texts = [tokenizer.decode([idx], skip_special_tokens=True) for idx in target_token_ids]
    target_converted_embeddings = token_embedding_layer(target_token_ids)  # Shape: [n_target_patches, embed_dim]

    # Optional: Analyze semantic content of target area (for debugging/insights)
    target_semantics = None
    
    for i, (prob, token_idx) in enumerate(zip(top_k_probs, top_k_indices)):
        token_text = tokenizer.decode([token_idx], skip_special_tokens=True)
        
        # Get the actual embedding for this specific token
        if token_embedding_layer is not None:
            candidate_token_embedding = token_embedding_layer(token_idx.unsqueeze(0)).squeeze(0)
        else:
            # Fallback to using the provided text_embedding
            candidate_token_embedding = text_embedding
        
        # Calculate cosine similarity between token embedding and target area
        similarity = torch.cosine_similarity(
            candidate_token_embedding.unsqueeze(0), 
            target_converted_embeddings, 
            dim=1
        ).mean().item()
        
        # Combine probability and similarity scores
        prob_score = prob.item()
        combined_score = (similarity_weight * similarity) + (probability_weight * prob_score)
        
        candidate_scores.append({
            "token_idx": token_idx.item(),
            "token_text": token_text,
            "probability": prob_score,
            "similarity": similarity,
            "combined_score": combined_score,
            "rank": i
        })
        
        # Store metrics
        selection_metrics["candidates"].append(token_text)
        selection_metrics["similarity_scores"].append(similarity)
        selection_metrics["probability_scores"].append(prob_score)
        selection_metrics["combined_scores"].append(combined_score)
    
    # Select token with highest combined score
    best_candidate = max(candidate_scores, key=lambda x: x["combined_score"])
    selected_token_idx = torch.tensor([best_candidate["token_idx"]], device=logits.device)
    selected_text = best_candidate["token_text"]
    
    # Find the index of selected candidate in the original top-k list
    selection_metrics["selected_index"] = next(
        i for i, cand in enumerate(candidate_scores) 
        if cand["token_idx"] == best_candidate["token_idx"]
    )
    selection_metrics["selected_candidate"] = best_candidate
    selection_metrics["target_embeddings_count"] = len(target_indices)
    selection_metrics["target_area_semantics"] = target_semantics
    
    return selected_token_idx, selected_text, selection_metrics


def decode_embeddings_to_text(
    embeddings: torch.Tensor,
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    top_k: int = 5,
    temperature: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Decode embeddings to the most semantically similar text tokens using output embeddings.
    
    This function takes image patch embeddings or any other embeddings and finds
    the vocabulary tokens that are most semantically similar to them, helping to
    understand what concepts the embeddings represent.
    
    Args:
        embeddings: Tensor of shape [embed_dim] or [n_embeddings, embed_dim]
        model: The language model to extract output embeddings from
        tokenizer: Tokenizer for decoding tokens
        top_k: Number of top similar tokens to return
        temperature: Temperature for similarity scoring (higher = more diverse)
        
    Returns:
        List of dictionaries containing token information and similarity scores
        
    Example:
        >>> target_embeddings = image_embeddings[target_indices]
        >>> semantic_info = decode_embeddings_to_text(target_embeddings, model, tokenizer)
        >>> print(f"Target area represents: {semantic_info[0]['top_tokens'][0]['token_text']}")
    """
    # Ensure embeddings is 2D
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)  # Shape: [1, embed_dim]
    
    # Get the output embedding layer (lm_head) from the model
    if hasattr(model, 'get_output_embeddings'):
        output_embedding_layer = model.get_output_embeddings()
    elif hasattr(model, 'lm_head'):
        output_embedding_layer = model.lm_head
    else:
        raise ValueError("Could not find output embedding layer in the model")
    
    # Get output embedding weights
    # Shape: [vocab_size, embed_dim]
    output_weights = output_embedding_layer.weight
    
    # Calculate similarities for each input embedding
    results = []
    for i, embedding in enumerate(embeddings):
        # Calculate cosine similarity between the embedding and all output token embeddings
        # embedding: [embed_dim], output_weights: [vocab_size, embed_dim]
        similarities = torch.nn.functional.cosine_similarity(
            embedding.unsqueeze(0),  # Shape: [1, embed_dim]
            output_weights,          # Shape: [vocab_size, embed_dim]
            dim=1
        )  # Shape: [vocab_size]
        
        # Apply temperature scaling
        if temperature != 1.0:
            similarities = similarities / temperature
        
        # Handle NaNs and get top-k most similar tokens
        similarities = torch.nan_to_num(similarities, nan=-float('inf'))
        top_k_similarities, top_k_indices = torch.topk(similarities, top_k)
        
        # Decode tokens and create result
        embedding_result = {
            "embedding_index": i,
            "top_tokens": []
        }
        
        for j, (sim_score, token_idx) in enumerate(zip(top_k_similarities, top_k_indices)):
            try:
                token_text = tokenizer.decode([token_idx.item()], skip_special_tokens=True)
                # Clean up the token text
                token_text = token_text.strip()
                if not token_text:
                    token_text = f"<token_{token_idx.item()}>"
            except Exception:
                token_text = f"<token_{token_idx.item()}>"
            
            embedding_result["top_tokens"].append({
                "rank": j + 1,
                "token_id": token_idx.item(),
                "token_text": token_text,
                "similarity_score": sim_score.item(),
                "raw_token": tokenizer.convert_ids_to_tokens([token_idx.item()])[0] if hasattr(tokenizer, 'convert_ids_to_tokens') else token_text
            })
        
        results.append(embedding_result)
    
    return results


def analyze_target_area_semantics(
    target_embeddings: torch.Tensor,
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    top_k: int = 10,
    temperature: float = 1.0
) -> Dict[str, Any]:
    """
    Analyze the semantic content of target area embeddings.
    
    Args:
        target_embeddings: Tensor of shape [n_target_patches, embed_dim]
        model: The language model
        tokenizer: Tokenizer for decoding
        top_k: Number of top tokens to analyze
        temperature: Temperature for similarity scoring
        
    Returns:
        Dictionary containing semantic analysis results
    """
    # Calculate the mean embedding of the target area
    target_center_embedding = target_embeddings.mean(dim=0)  # Shape: [embed_dim]
    
    # Decode the center embedding to text
    center_results = decode_embeddings_to_text(
        target_center_embedding, model, tokenizer, top_k, temperature
    )
    
    # Also decode a few individual patch embeddings for diversity analysis
    num_patches_to_analyze = min(3, target_embeddings.shape[0])
    individual_patch_results = []
    
    if num_patches_to_analyze > 1:
        # Select patches: first, middle, and last
        patch_indices = [0, target_embeddings.shape[0] // 2, target_embeddings.shape[0] - 1]
        patch_indices = list(set(patch_indices))  # Remove duplicates
        patch_indices = patch_indices[:num_patches_to_analyze]
        
        for idx in patch_indices:
            patch_embedding = target_embeddings[idx]
            patch_results = decode_embeddings_to_text(
                patch_embedding, model, tokenizer, top_k // 2, temperature
            )
            individual_patch_results.append({
                "patch_index": idx,
                "results": patch_results[0]  # Only one embedding per patch
            })
    
    # Extract common themes from the top tokens
    all_tokens = [token["token_text"] for token in center_results[0]["top_tokens"]]
    for patch_result in individual_patch_results:
        all_tokens.extend([token["token_text"] for token in patch_result["results"]["top_tokens"]])
    
    # Simple analysis of token types
    semantic_categories = {
        "objects": [],
        "actions": [],
        "descriptors": [],
        "spatial": [],
        "other": []
    }
    
    # Basic categorization (this could be enhanced with more sophisticated NLP)
    for token in set(all_tokens):  # Remove duplicates
        token_lower = token.lower().strip()
        if not token_lower or len(token_lower) < 2:
            continue
            
        # Simple heuristic categorization
        if any(word in token_lower for word in ['look', 'see', 'watch', 'gaze', 'stare', 'glance']):
            semantic_categories["actions"].append(token)
        elif any(word in token_lower for word in ['left', 'right', 'up', 'down', 'above', 'below', 'center']):
            semantic_categories["spatial"].append(token)
        elif any(word in token_lower for word in ['big', 'small', 'red', 'blue', 'bright', 'dark', 'large']):
            semantic_categories["descriptors"].append(token)
        elif len(token_lower) > 2 and token_lower.isalpha():
            semantic_categories["objects"].append(token)
        else:
            semantic_categories["other"].append(token)
    
    return {
        "target_area_center": center_results[0],
        "individual_patches": individual_patch_results,
        "semantic_analysis": {
            "categories": semantic_categories,
            "most_likely_concept": center_results[0]["top_tokens"][0]["token_text"] if center_results[0]["top_tokens"] else "unknown",
            "confidence_score": center_results[0]["top_tokens"][0]["similarity_score"] if center_results[0]["top_tokens"] else 0.0,
            "diversity_score": len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0
        },
        "summary": {
            "num_target_patches": target_embeddings.shape[0],
            "embedding_dim": target_embeddings.shape[1],
            "top_concepts": [token["token_text"] for token in center_results[0]["top_tokens"][:5]]
        }
    }


def generate_next_token_with_gaze_guidance(
    model_inputs: Dict[str, Any],
    model: "PreTrainedModel", 
    tokenizer: "PreTrainedTokenizer",
    gen_config: Dict[str, Any],
    confidence_tracker: "ConfidenceMetrics",
    repetitivity_tracker: "RepetitivityMetrics", 
    candidate_evaluator: "TopKCandidateEvaluator",
    step_num: int,
    image_embeddings: Optional[torch.Tensor] = None,
    target_mask: Optional[np.ndarray] = None,
    source_mask: Optional[np.ndarray] = None,
    apply_only_target_mask: bool = True,
    guidance_config: Optional[Dict[str, Any]] = None
) -> Tuple[torch.Tensor, str, Any, Dict[str, Any]]:
    """
    Enhanced token generation with gaze-based guidance.
    
    Args:
        model_inputs: Dictionary of inputs to the model
        model: The language model
        tokenizer: The tokenizer
        gen_config: Generation configuration
        confidence_tracker: Metrics tracker for confidence
        repetitivity_tracker: Metrics tracker for repetitivity
        candidate_evaluator: Top-k candidate evaluator
        step_num: Current generation step number
        image_embeddings: Image patch embeddings for similarity calculation
        target_mask: Binary mask indicating target gaze area
        apply_only_target_mask: Whether to apply the target mask only
        guidance_config: Configuration for guidance behavior
        
    Returns:
        Tuple of (next_token_id, token_text, model_outputs, evaluation_metrics)
    """
    # Default guidance configuration
    default_guidance_config = {
        "top_k": 10,
        "similarity_weight": 0.7,
        "probability_weight": 0.3,
        "enable_after_step": 0,  # Start guidance immediately
        "enable_after_keyword": "looking"  # Enable guidance after seeing "looking"
    }
    guidance_config = {**default_guidance_config, **(guidance_config or {})}
    
    # Generate model outputs
    outputs = model(**model_inputs)
    logits = outputs.logits[0, -1, :]  # Get logits for the last token
    
    # Extract current text embedding if available
    text_embedding = None
    if outputs.hidden_states and len(outputs.hidden_states) > 0:
        text_embedding = outputs.hidden_states[-1].squeeze(0)[-1]  # Last token of last layer
    
    # Determine if we should use gaze guidance
    should_use_guidance = (
        apply_only_target_mask and 
        step_num >= guidance_config["enable_after_step"] and
        image_embeddings is not None and
        text_embedding is not None and
        target_mask is not None
    )
    
    # Check if we've seen the trigger keyword
    if guidance_config.get("enable_after_keyword"):
        # This would need to be tracked externally, for now assume it's always enabled
        pass
    
    selection_metrics = {}
    should_use_guidance = True
    if should_use_guidance:
        # Use gaze-guided selection
        selected_mask = target_mask if apply_only_target_mask else source_mask
        next_token_id, token_text, selection_metrics = select_token_by_gaze_correlation(
            logits=logits,
            text_embedding=text_embedding,
            image_embeddings=image_embeddings,
            mask=selected_mask,
            tokenizer=tokenizer,
            model=model,
            top_k=guidance_config["top_k"],
            similarity_weight=guidance_config["similarity_weight"],
            probability_weight=guidance_config["probability_weight"]
        )
        selection_metrics["guidance_used"] = True
    else:
        # Fall back to standard probability-based selection
        probs = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, 1) if gen_config.get("do_sample", False) else logits.argmax(dim=-1, keepdim=True)
        token_text = tokenizer.decode(next_token_id, skip_special_tokens=True)
        selection_metrics = {
            "guidance_used": False,
            "fallback_reason": "guidance_disabled_or_missing_data"
        }
    
    # Update evaluation metrics
    evaluation_metrics = {}
    if outputs.logits is not None:
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        
        # Compute confidence metrics first
        confidence_metrics = confidence_tracker.compute_confidence_metrics(logits)
        
        # Evaluate top-k candidates
        candidate_eval = candidate_evaluator.evaluate_candidates(logits, step_num)
        
        # Update trackers with proper method calls
        confidence_tracker.update_history(step_num, confidence_metrics)
        repetitivity_tracker.add_token(next_token_id.item(), token_text)
        
        # Compute repetitivity metrics
        repetitivity_metrics = repetitivity_tracker.compute_diversity_metrics()
        recent_repetitions = repetitivity_tracker.get_recent_repetitions()
        
        evaluation_metrics = {
            "step": step_num,
            "selected_token": {
                "id": next_token_id.item(),
                "text": token_text
            },
            "confidence": confidence_metrics,
            "repetitivity": repetitivity_metrics,
            "recent_repetitions": recent_repetitions,
            "top_k_analysis": candidate_eval,
            "selection_metrics": selection_metrics
        }
    
    return next_token_id, token_text, outputs, evaluation_metrics

# Model Loading and Setup
def load_model_and_setup(
    model_path: str = "lmms-lab/llava-onevision-qwen2-7b-ov-chat",
    attn_implementation: str = "sdpa",
    load_4bit: bool = False,
    load_8bit: bool = False,
    attn_layer_ind: int = -1,
    model_base: Optional[str] = None,
    adapter_path: Optional[str] = None
) -> Tuple[PreTrainedTokenizer, PreTrainedModel, SigLipImageProcessor, int]:
    """
    Load and initialize the LLaVA model with specified configurations.
    This function should be called once at the beginning of your session.
    """
    print("Loading model and components...")

    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_map = "auto" if target_device.type == "cuda" else None
    llava_model_args = {"multimodal": True}
    custom_config = {'attn_layer_ind': attn_layer_ind}

    use_adapter = adapter_path is not None
    base_model_path = model_base or model_path
    target_model_path = adapter_path if use_adapter else model_path

    model_name_source = base_model_path if base_model_path else model_path
    model_name = get_model_name_from_path(model_name_source) or "llava_qwen"

    if use_adapter:
        print(f"Base model path: {base_model_path}")
        print(f"LoRA adapter path: {adapter_path}")
    else:
        print(f"Model path: {model_path}")
        if model_base:
            print(f"Using auxiliary base path: {model_base}")

    print(f"Derived model name: {model_name}")
    print(f"Attention implementation: {attn_implementation}")
    print(f"Custom config: {custom_config}")

    if load_4bit and load_8bit:
        raise ValueError("Cannot load in both 4-bit and 8-bit mode.")

    if use_adapter:
        if not base_model_path:
            raise ValueError("LoRA adapter loading requires a base model path. Provide --model_base or use --model_path to point to the base checkpoint.")

        # Suppress warnings during model loading
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module")
            tokenizer, model, image_processor, max_length = load_pretrained_model(
                base_model_path,
                None,
                model_name,
                load_8bit=load_8bit,
                load_4bit=load_4bit,
                device_map=device_map,
                attn_implementation=attn_implementation,
                overwrite_config=custom_config,
                **llava_model_args
            )

        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("peft must be installed to load LoRA adapters") from exc

        print("Loading LoRA adapter weights...")
        # Suppress warnings during adapter loading
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module")
            peft_model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        # if hasattr(peft_model, "merge_and_unload"):
        #     print("Merging LoRA weights into the base model...")
        #     model = peft_model.merge_and_unload()
        # else:
        model = peft_model
        print("LoRA adapter loaded successfully.")
    else:
        # Suppress warnings during model loading
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module")
            tokenizer, model, image_processor, max_length = load_pretrained_model(
                target_model_path,
                model_base,
                model_name,
                load_8bit=load_8bit,
                load_4bit=load_4bit,
                device_map=device_map,
                attn_implementation=attn_implementation,
                overwrite_config=custom_config,
                **llava_model_args
            )

    if device_map is None:
        model = model.to(target_device)
        if target_device.type != "cuda":
            model = model.float()

    model.eval()
    print("✅ Model loaded successfully!")
    if hasattr(model, "device"):
        print(f"Model device: {model.device}")
    else:
        print(f"Model running on: {target_device}")
    print(f"Max context length: {max_length}")

    return tokenizer, model, image_processor, max_length

# Helper Functions
def fix_wsl_paths(path: str) -> str:
    """Convert Windows paths to WSL paths if necessary."""
    if not isinstance(path, str):
        path = str(path)
        
    if path.startswith('/mnt/'):
        return path
    path = path.replace("\\", os.sep)
    drive_parts = path.split(os.sep)
    if len(drive_parts) > 0 and len(drive_parts[0]) > 1 and drive_parts[0][1] == ':':
        drive_letter = drive_parts[0][0].lower()
        wsl_path = f'/mnt/{drive_letter}/' + os.sep.join(drive_parts[1:])
        return wsl_path
    else:
        return path

def load_image(image_path: Union[str, Path]) -> Image.Image:
    """Load an image from path or URL."""
    image_path = str(image_path)
    if image_path.startswith("http"):
        response = requests.get(image_path, stream=True)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found at {image_path}")
        image = Image.open(image_path).convert("RGB")
    return image

def load_mask_from_file(mask_path: Union[str, Path]) -> np.ndarray:
    """Load mask from numpy file."""
    mask_path = fix_wsl_paths(str(mask_path))
    if not Path(mask_path).exists():
        raise FileNotFoundError(f"Mask file not found at {mask_path}")

    mask_data = np.load(mask_path, allow_pickle=True)
    if isinstance(mask_data, np.ndarray) and mask_data.dtype == object:
        # Handle pickled dict format
        mask_dict = mask_data.item()
        if 'static_masks' in mask_dict:
            if mask_dict['static_masks'].shape[1] == 2:
                # new format with direct [x,y] indices
                mask = mask_dict['static_masks']
            else:
                mask = mask_dict['static_masks'][0]  # Take first mask
        elif 'masks' in mask_dict:
            mask = mask_dict['masks'][0]
        else:
            mask = mask_dict
    else:
        mask = mask_data

    return mask


def calculate_coordinate_mapping(original_img_shape, patch_boxes, patched_final_dim, patched_resized_before_pad_dim, vision_tower):
    """
    Calculate coordinate mapping from tokens to pixels.

    Args:
        original_img_shape: Original PIL image shape or object with .width and .height
        patch_boxes: List of patch boxes
        patched_final_dim: Final dimension after patching (width, height)
        patched_resized_before_pad_dim: Image size before padding (width, height)
        vision_tower: Vision tower for getting patch size

    AnyRes Processing:
    1. The image is encoded with ViT to 14-sized patches resulting in a 27x27 grid for every patch
    2. All patches are concatenated first on every axis, x and y resulting in (N*27)x(M*27) grid for an NxM patch grid
    3. For every line, an additional newline patch is added to the end of the line resulting with (width*27)x((height*27)+1) grid
    4. Flatten the grid to a single dimension, resulting in (width*height*27*27 + width*27) tokens
    5. The base image patches are added to the start of the sequence, resulting in 729+tokens in total
    """
    patch_size = vision_tower.config.patch_size if hasattr(vision_tower.config, 'patch_size') else 14
    nb_height_patches = patched_final_dim[1] // 384
    nb_width_patches = patched_final_dim[0] // 384
    nb_height_tokens = nb_height_patches * (384 // patch_size)
    nb_width_tokens = nb_width_patches * (384 // patch_size) + 1 # +1 for the newline patch
    
    patched_img_scale_x = original_img_shape[0] / patched_resized_before_pad_dim[0]
    patched_img_scale_y = original_img_shape[1] / patched_resized_before_pad_dim[1]

    dummy_img = np.zeros((nb_height_tokens, nb_width_tokens, 3), dtype=np.uint8)
    unpadded = unpad_image(np.transpose(dummy_img, (2, 0, 1)), original_img_shape)
    nb_height_tokens, nb_width_tokens = unpadded.shape[1], unpadded.shape[2]
    # Calculate the number of patches per side
    token_patch_boxes = []
    for box in patch_boxes:
        token_box = [val // patch_size for val in box]
        # since every anyres patch is 384x384 which is 27x27 tokens, we need to round to 27 multiples
        token_box[0] = (token_box[0] // 27) * 27
        token_box[1] = (token_box[1] // 27) * 27
        token_box[2] = (token_box[2] // 27) * 27
        token_box[3] = (token_box[3] // 27) * 27
        token_patch_boxes.append(token_box)

    # now lets build a matrix with the all the patches in token size, where every cell is the coordinate of the center pixel before tokenization
    coordinate_mat = np.ones((nb_height_tokens, nb_width_tokens, 2), dtype=int)*(-1)
    for token_box, patch_box in zip(token_patch_boxes, patch_boxes):
        token2pixel_ratio = patch_box[-1] / token_box[-1]
        token_inds_in_box_x = np.arange(token_box[0], token_box[2])
        token_inds_in_box_y = np.arange(token_box[1], token_box[3])
        for token_idx_y in token_inds_in_box_y:
            for token_idx_x in token_inds_in_box_x:
                # Calculate the center pixel of the token box
                pixel_x_min = int(token_idx_x * token2pixel_ratio)
                pixel_y_min = int(token_idx_y * token2pixel_ratio)

                # Map to the coordinate matrix
                coordinate_mat[token_idx_y, token_idx_x] = (pixel_x_min, pixel_y_min)

    # Flatten to make it the same shape as the token sequence
    coordinate_mat_flat = coordinate_mat.reshape(-1, 2)
    # scale back to origina image size before resizing
    coordinate_mat_flat[:, 0] = np.round(coordinate_mat_flat[:, 0] * patched_img_scale_x).astype(int)
    coordinate_mat_flat[:, 1] = np.round(coordinate_mat_flat[:, 1] * patched_img_scale_y).astype(int)
    # Clip the coordinate values since the image might be padded, -1 means no image coordinate (since these tokens added as placeholders for every end of line)
    coordinate_mat_flat[:, 0] = np.clip(coordinate_mat_flat[:, 0], -1, original_img_shape[0] - 1)
    coordinate_mat_flat[:, 1] = np.clip(coordinate_mat_flat[:, 1], -1, original_img_shape[1] - 1)

    # Insert additional patch at the start of the sequence as placeholder for the base image
    base_image_patch = np.ones([(384 // patch_size) * (384 // patch_size), 2], dtype=int)*(-1)
    coordinate_mat_flat = np.insert(coordinate_mat_flat, 0, base_image_patch, axis=0)

    # lets also do an inverse mapping to get the token index from pixel coordinates
    pixel_to_token_indices = {}
    for token_idx, (x, y) in enumerate(coordinate_mat_flat):
        # skip if indices are -1
        if x == -1 or y == -1:
            continue
        # if (x, y) not in pixel_to_token_indices:
        #     pixel_to_token_indices[(x, y)] = token_idx

        # Calculate all covered pixel coordinates for this token
        patch_size_scaled = np.round(patch_size*patched_img_scale_x).astype(int)
        # todo: instead of patch_radius, use the actual (scaled) patch size in pixels
        covered_coords = []
        covered_x_indices = np.arange(x, x + patch_size_scaled)
        covered_y_indices = np.arange(y, y + patch_size_scaled)
        # remove indices that are out of bounds
        covered_x_indices = covered_x_indices[(covered_x_indices >= 0) & (covered_x_indices < original_img_shape[0])]
        covered_y_indices = covered_y_indices[(covered_y_indices >= 0) & (covered_y_indices < original_img_shape[1])]
        # create all combinations of covered coordinates without a for loop
        covered_coords = np.array(np.meshgrid(covered_x_indices, covered_y_indices)).T
        covered_coords = covered_coords.reshape(-1, 2)

        # Insert all valid coordinates into the dict
        for coord in covered_coords:
            coord = tuple(coord)
            # Append the token index to the list for this coordinate
            pixel_to_token_indices[coord] = token_idx


    # Calculate coverage statistics
    total_image_pixels = original_img_shape[0] * original_img_shape[1]
    covered_mask = np.zeros((original_img_shape[1], original_img_shape[0]), dtype=bool)
    for coord in pixel_to_token_indices.keys():
        covered_mask[coord[1], coord[0]] = True
    covered_pixels = np.sum(covered_mask)
    coverage_percentage = (covered_pixels / total_image_pixels) * 100

    print(f"Image coverage statistics:")
    print(f"  Total image pixels: {total_image_pixels:,}")
    print(f"  Covered pixels: {covered_pixels:,}")
    print(f"  Coverage percentage: {coverage_percentage:.2f}%")
    

    return coordinate_mat_flat, pixel_to_token_indices

def get_attention_indices_from_mask(
    mask: np.ndarray, 
    image_size: Tuple[int, int], 
    model_config: Any,
    original_img_shape: Tuple[int, int],
    patched_final_dim: Tuple[int, int],
    patch_boxes: List[List[int]],
    patched_resized_before_pad_dim: Tuple[int, int],
    vision_tower: Any,
    apply_for_anyres_patches: bool = True
) -> Tuple[List[int], np.ndarray]:
    """Convert mask pixels to token indices."""
    if len(mask.shape)>1 and mask.shape[1] == 2:
        # new mask format, we get the coordinates directly, the <x, y> pairs
        mask_coords = mask.reshape(-1, 2)
        # flip x and y to row and column
        mask_coords = mask_coords[:, [1, 0]]
    else:
        mask_coords = np.argwhere(mask)
    print(f"Mask coordinates shape: {mask_coords.shape}")

    # Get attention indices from pixel coordinates
    atten_indices, resized_mask = _pixel_to_token_indices_helper_anyres(
        mask_coords, image_size, possible_resolutions=model_config.image_grid_pinpoints,
        add_system_prompt_tokens= False,
        add_user_prompt_tokens= False,
        user_prompt_range=[1849, 1860]
    )
    if apply_for_anyres_patches:
        anyres_token2pixel_map = calculate_coordinate_mapping(
            original_img_shape,
            patch_boxes,
            patched_final_dim,
            patched_resized_before_pad_dim,
            vision_tower
        )

        # add additional patch tokens form anyres structure to attention indices
        add_tokens = []
        for coord in mask_coords:
            pixel_coord = tuple(coord)
            if pixel_coord in anyres_token2pixel_map[1]:
                token_idx = anyres_token2pixel_map[1][pixel_coord]
                if token_idx not in atten_indices:
                    add_tokens.append(token_idx)

        # Add additional tokens to the attention indices
        atten_indices.extend(add_tokens)
        atten_indices = sorted(set(atten_indices))

    return atten_indices, resized_mask

def save_raw_attention_tensor(attention_map: np.ndarray, full_attention, output_path: Path,
                             token_id: int, token_text: str, step_idx: int) -> None:
    """Save raw attention tensor to file."""
    output_path.mkdir(parents=True, exist_ok=True)

    safe_token_text = "".join(c if c.isalnum() else "_" for c in token_text)
    if not safe_token_text:
        safe_token_text = f"tokenid_{token_id}"

    filename = f"attn_tensor_{step_idx:03d}_{safe_token_text}.pt"
    file_path = output_path / filename

    data = {
        "attention_map": torch.from_numpy(attention_map) if isinstance(attention_map, np.ndarray) else attention_map,
        "full_attention": full_attention,
        "token_id": token_id,
        "token_text": token_text,
        "step_idx": step_idx,
        "timestamp": str(datetime.now())
    }

    torch.save(data, file_path)
    # print(f"Saved raw attention tensor to {file_path}")

def visualize_processed_attention(
    attention_map: np.ndarray,
    original_image: Image.Image,
    output_path: Union[str, Path],
    threshold_value: float = 0.4,
    opening_kernel_size: int = 5,
    min_blob_area: int = 20,
    min_avg_attention: float = 0.2,
    show_highest_attn_blob: bool = False,
    dilate_kernel_size: int = 0
) -> Image.Image:
    """Process and visualize attention map with filtering."""

    if not isinstance(attention_map, np.ndarray) or attention_map.ndim != 2:
        print(f"Error: Invalid attention map for {output_path}")
        return None

    # Resize attention map to match image size
    map_img = Image.fromarray(attention_map.astype(np.float32))
    resized_map_img = map_img.resize(original_image.size, Image.Resampling.LANCZOS)
    resized_map = np.array(resized_map_img)

    # Normalize the raw map
    if np.max(resized_map) > np.min(resized_map):
        norm_raw_map = (resized_map - np.min(resized_map)) / (np.max(resized_map) - np.min(resized_map))
    else:
        norm_raw_map = np.zeros_like(resized_map)

    # Binary thresholding
    binary_map = np.where(norm_raw_map >= threshold_value, 255, 0).astype(np.uint8)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel_size, opening_kernel_size))
    opened_map = cv2.morphologyEx(binary_map, cv2.MORPH_DILATE, kernel)

    # Blob filtering
    contours, _ = cv2.findContours(opened_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_blobs = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_blob_area:
            contour_mask = np.zeros_like(opened_map)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            masked_attention = norm_raw_map[contour_mask == 255]
            if masked_attention.size > 0:
                avg_attention = np.mean(masked_attention)
                if avg_attention >= min_avg_attention:
                    valid_blobs.append((avg_attention, contour))

    # Select blobs
    final_mask = np.zeros_like(opened_map)
    if valid_blobs:
        if show_highest_attn_blob:
            valid_blobs.sort(key=lambda x: x[0], reverse=True)
            selected_contours = [valid_blobs[0][1]]
        else:
            selected_contours = [blob[1] for blob in valid_blobs]

        cv2.drawContours(final_mask, selected_contours, -1, 255, thickness=cv2.FILLED)

        # Optional dilation
        if show_highest_attn_blob and dilate_kernel_size > 1:
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
            final_mask = cv2.dilate(final_mask, dilate_kernel, iterations=1)

    # Create visualization
    masked_norm_map = np.where(final_mask > 0, norm_raw_map, 0)
    heatmap_colors = cm.viridis(masked_norm_map)[:, :, :3]
    heatmap_uint8 = (heatmap_colors * 255).astype(np.uint8)

    original_np = np.array(original_image.convert('RGB'))
    blended_np = original_np.copy()

    mask_indices = final_mask > 0
    if np.any(mask_indices):
        alpha = 0.5
        blended_np[mask_indices] = cv2.addWeighted(
            original_np[mask_indices], 1 - alpha,
            heatmap_uint8[mask_indices], alpha, 0.0
        )

    overlay_img = Image.fromarray(blended_np)

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_img.save(output_path)
        # print(f"Saved processed attention to: {output_path}")

    return overlay_img

def visualize_attention_collage(
    attention_maps: List[Tuple[Image.Image, str]],
    output_path: Union[str, Path],
    grid_size: Tuple[int, int] = (3, 4)
) -> None:
    """Create a collage of attention maps."""
    rows, cols = grid_size
    total_maps = min(len(attention_maps), rows * cols)

    if total_maps == 0:
        print("No attention maps to create collage")
        return

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    if total_maps == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(total_maps):
        attention_map, token_text = attention_maps[i]
        ax = axes[i]
        ax.imshow(attention_map)
        ax.set_title(token_text, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty subplots
    for i in range(total_maps, rows * cols):
        axes[i].axis('off')

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved attention collage to: {output_path}")


def visualize_embedding_similarity(
    text_token_embedding: torch.Tensor,
    image_token_embeddings: torch.Tensor,
    original_image: Image.Image,
    grid_size: int,
    output_path: Union[str, Path],
    region_radius: int = 3,
    top_k: int = 2,
    normalize: bool = True,
    save_file: bool = False
) -> None:
    """
    Calculates and visualizes the semantic similarity between a text token's embedding
    and all image patch embeddings. Highlights the top-k similarity regions with area around them.

    Args:
        text_token_embedding: The feature embedding of the generated text token. Shape: [hidden_dim].
        image_token_embeddings: The feature embeddings of all image patches. Shape: [num_patches, hidden_dim].
        original_image: The original PIL image for the background.
        grid_size: The dimension of the square patch grid (e.g., 27 for a 27x27 grid).
        output_path: The path to save the visualization.
        region_radius: The radius around each top similarity value to include in visualization.
        top_k: The number of top similarity values to highlight (default: 3).
        save_file: Whether to save the visualization to file (default: False).
    """
    # 1. Calculate dot product similarity
    # text_token_embedding: [D], image_token_embeddings: [N, D] -> similarity_scores: [N]
    similarity_scores = torch.nn.functional.cosine_similarity(
        image_token_embeddings, text_token_embedding.unsqueeze(0), dim=1
    )

    # 2. Reshape into a 2D grid
    num_patches = image_token_embeddings.shape[0]
    expected_elements = grid_size * grid_size
    if expected_elements > num_patches:
        padding_size = expected_elements - num_patches
        similarity_scores = torch.cat([
            similarity_scores,
            torch.zeros(padding_size, device=similarity_scores.device)
        ])
    
    similarity_map = similarity_scores.reshape(grid_size, grid_size).cpu().float().numpy()
    
    # Find top-k similarity values and their positions
    flat_map = similarity_map.flatten()
    # Get indices of top-k values
    topk_indices = np.argpartition(flat_map, -top_k)[-top_k:]

    # Also add all indices with similarity above a threshold
    sim_threshold = 0.1  # Hardcoded threshold
    high_sim_indices = np.where(flat_map > sim_threshold)[0]
    # Combine and deduplicate
    all_indices = np.unique(np.concatenate([topk_indices, high_sim_indices]))
    topk_positions = [(idx // grid_size, idx % grid_size) for idx in all_indices]
    
    # Create a mask for top-k regions with area around them
    topk_mask = np.zeros_like(similarity_map)
    
    for row, col in topk_positions:
        # Create circular region around each top value
        for r in range(max(0, row - region_radius), min(grid_size, row + region_radius + 1)):
            for c in range(max(0, col - region_radius), min(grid_size, col + region_radius + 1)):
                # Check if within circular radius
                distance = np.sqrt((r - row)**2 + (c - col)**2)
                if distance <= region_radius:
                    topk_mask[r, c] = 1
    
    # Apply mask to similarity map
    masked_similarity_map = similarity_map * topk_mask

    if normalize:
        # Normalize the masked similarity map
        min_val, max_val = np.min(masked_similarity_map), np.max(masked_similarity_map)
        if max_val > min_val:
            masked_similarity_map = (masked_similarity_map - min_val) / (max_val - min_val)
        else:
            masked_similarity_map = np.zeros_like(masked_similarity_map)

    # Resize heatmap to image size for overlay
    map_img = Image.fromarray(masked_similarity_map.astype(np.float32))
    resized_map_img = map_img.resize(original_image.size, Image.Resampling.LANCZOS)
    
    if normalize:
        # Normalize the resized map
        min_val, max_val = np.min(resized_map_img), np.max(resized_map_img)
        if max_val > min_val:
            resized_map_img = (resized_map_img - min_val) / (max_val - min_val)
        else:
            resized_map_img = np.zeros_like(resized_map_img)
    
    heatmap_colors = cm.inferno(np.array(resized_map_img))[:, :, :3]
    heatmap_uint8 = (heatmap_colors * 255).astype(np.uint8)

    # 4. Blend with original image and conditionally save
    overlay_img = Image.blend(original_image, Image.fromarray(heatmap_uint8), alpha=0.6)
    
    if save_file:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_img.save(output_path)

    return masked_similarity_map


# JSON Data Loading and Prompt Building Functions

def load_train_results_json(json_path: Union[str, Path]) -> Dict[str, str]:
    """Load the train_results.json file."""
    json_path = fix_wsl_paths(str(json_path))
    if not Path(json_path).exists():
        raise FileNotFoundError(f"JSON file not found at {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded train_results.json with {len(data)} entries")
    return data

def build_prompt_with_subject_description(subject_description: str, base_prompt_template: str = "Complete the sentence. The {} is looking at") -> str:
    """Build a prompt using the subject description."""
    if not subject_description:
        # Fallback to generic prompt
        return "Complete the sentence. The person is looking at"

    # Clean the subject description
    subject_description = subject_description.strip()

    # Build the prompt
    prompt = base_prompt_template.format(subject_description)
    return prompt

def create_image_key_from_path(image_path: Union[str, Path]) -> str:
    """Create the JSON key from image path (e.g., '00000000/00000021.jpg' -> '00000000_00000021')."""
    image_path = Path(image_path)

    # Extract the folder and filename
    folder_name = image_path.parent.name
    filename = image_path.stem  # filename without extension

    # Create the key
    key = f"{folder_name}_{filename}"
    return key

def save_results_to_json(results: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Save the generation results to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        # Convert all objects to JSON-serializable types
        serializable_results = _make_json_serializable(results)
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {output_path}")


def append_results_to_json(results: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Append results to an existing JSON file or create a new one if it doesn't exist.
    
    The file will contain a JSON array where each element is a result entry.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    serializable_results = _make_json_serializable(results)
    
    # Read existing data if file exists
    existing_data = []
    if output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    existing_data = json.loads(content)
                    if not isinstance(existing_data, list):
                        # If existing file is not a list, wrap it in a list
                        existing_data = [existing_data]
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read existing file {output_path}: {e}")
            print("Creating new file...")
            existing_data = []
    
    # Append new results
    existing_data.append(serializable_results)
    
    # Write back to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
    
    print(f"Appended results to {output_path} (entry #{len(existing_data)})")

# Helper Functions for Generation Process

def _prepare_configs(generation_config: Optional[Dict], attention_config: Optional[Dict]) -> Tuple[Dict, Dict]:
    """Prepare and merge generation and attention configurations with defaults."""
    default_generation_config = {
        "max_new_tokens": 50,
        "temperature": 0.1,
        "do_sample": True,
        "top_k": None  # None = disabled, int > 0 = enabled with specified value
    }

    default_attention_config = {
        "attn_threshold": 0.4,
        "opening_kernel_size": 5,
        "min_blob_area": 50,
        "min_avg_attention": 0.2,
        "show_highest_attn_blob": False,
        "dilate_kernel_size": 0,
        "create_collage": True,
        "collage_grid_rows": 3,
        "collage_grid_cols": 4,
        "visualize_attn_overlays": False,
        "save_tensors": False
    }

    gen_config = {**default_generation_config, **(generation_config or {})}
    attn_config = {**default_attention_config, **(attention_config or {})}

    return gen_config, attn_config

def _setup_output_directories(output_dir: Union[str, Path]) -> Tuple[Path, Path, Path, Path, Path, Path]:
    """Create and return all output directory paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    vis_output_dir_raw = output_dir / "attention_maps_raw"
    vis_output_dir_processed = output_dir / "attention_maps_processed"
    tensor_output_dir = output_dir / "attention_tensors"
    collage_output_dir = output_dir / "attention_collages"
    similarity_output_dir = output_dir / "embedding_similarity_overlays" # New directory

    for dir_path in [vis_output_dir_raw, vis_output_dir_processed, tensor_output_dir, collage_output_dir, similarity_output_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)

    return output_dir, vis_output_dir_raw, vis_output_dir_processed, tensor_output_dir, collage_output_dir, similarity_output_dir

def _prepare_inputs(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    prompt: str,
    image_processor: SigLipImageProcessor,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel
) -> Tuple[Any, Any, torch.Tensor, List, List[int], Any, torch.Tensor]:
    """Load and prepare image, mask, and input tensors."""
    person_mask = None
    person_mask_indices = None
    # Load image and mask
    image = load_image(image_path)
    mask = load_mask_from_file(mask_path)
    person_mask_path = str(mask_path).replace("gaze__", "person__")
    if Path(person_mask_path).exists():
        person_mask = load_mask_from_file(person_mask_path)


    print(f"Image size: {image.size}")
    print(f"Mask shape: {mask.shape}")


    # Process image: get tensor of shape [1, C, H, W]
    processed, patched_resized_before_pad_dim, patched_final_dim, patch_boxes = process_images([image], image_processor, model.config)
    if isinstance(processed, list):
        # Take first element if list returned
        image_tensor = processed[0]
    else:
        image_tensor = processed
    # Ensure batch dimension
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    # Move to device and cast to model's dtype
    image_tensor = image_tensor.to(model.device, dtype=model.dtype)

    # Get attention indices from mask
    atten_indices, target_mask = get_attention_indices_from_mask(
        mask=mask,
        image_size=image.size,
        model_config=model.config,
        original_img_shape=image.size,
        patched_final_dim=patched_final_dim,
        patch_boxes=patch_boxes,
        patched_resized_before_pad_dim=patched_resized_before_pad_dim,
        vision_tower=model.get_vision_tower() if hasattr(model, 'get_vision_tower') else None,
        apply_for_anyres_patches=False
    )
    if person_mask is not None:
        person_mask_indices, person_mask = get_attention_indices_from_mask(
            mask=person_mask,
            image_size=image.size,
            model_config=model.config,
            original_img_shape=image.size,
            patched_final_dim=patched_final_dim,
            patch_boxes=patch_boxes,
            patched_resized_before_pad_dim=patched_resized_before_pad_dim,
            vision_tower=model.get_vision_tower() if hasattr(model, 'get_vision_tower') else None,
            apply_for_anyres_patches=False
        )
    print(f"Initial attention indices: {len(atten_indices)} tokens")
    masks = {'target_mask': target_mask, 'person_mask': person_mask}        # masks in [model's] input image resolution 
    # Prepare conversation
    conv_template = "qwen_1_5"  # Default for the model

    if DEFAULT_IMAGE_TOKEN not in prompt:
        full_prompt = f"{DEFAULT_IMAGE_TOKEN}\\n{prompt}"
    else:
        full_prompt = prompt

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], full_prompt)
    conv.append_message(conv.roles[1], None)        # 5 tokens are always added to the end of the user prompt
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_question, tokenizer,
        IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    # Prepare image sizes as [width, height] for model
    image_sizes = [[image.size[0], image.size[1]]]
    # tokenizer.decode(input_ids.cpu().numpy()[0][-11:-9])
    return image, masks, image_tensor, image_sizes, atten_indices, person_mask_indices, input_ids

def _determine_image_patch_info(model: PreTrainedModel, input_ids: torch.Tensor) -> Tuple[int, int, int, int]:
    """Determine image patch information and token indices."""
    # Determine image patch information
    num_patches = 729  # Default fallback
    vision_tower = None
    if hasattr(model, 'get_vision_tower'):
        vision_tower = model.get_vision_tower()

    if vision_tower is not None:
        if hasattr(vision_tower, 'num_patches'):
            num_patches = vision_tower.num_patches
        elif hasattr(vision_tower, 'patch_embed') and hasattr(vision_tower.patch_embed, 'num_patches'):
            num_patches = vision_tower.patch_embed.num_patches
        elif hasattr(vision_tower, 'embeddings') and hasattr(vision_tower.embeddings, 'num_patches'):
            num_patches = vision_tower.embeddings.num_patches

    print(f"Using number of image patches: {num_patches}")

    # Find image token indices
    image_token_indices_in_original = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0]
    if len(image_token_indices_in_original) == 0:
        image_token_start_index_in_llm = 1
    else:
        image_token_start_index_in_llm = image_token_indices_in_original[0].item()

    image_token_end_index_in_llm = image_token_start_index_in_llm + num_patches
    grid_size = int(np.sqrt(num_patches))

    if grid_size * grid_size != num_patches:
        grid_size = int(np.ceil(np.sqrt(num_patches)))

    print(f"Image token range: {image_token_start_index_in_llm} to {image_token_end_index_in_llm}")
    print(f"Grid size: {grid_size}x{grid_size}")

    return num_patches, grid_size, image_token_start_index_in_llm, image_token_end_index_in_llm

def _generate_next_token(
    model_inputs: Dict,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    gen_config: Dict
) -> Tuple[torch.Tensor, str, Any]:
    """Generate next token and return token info and model outputs."""
    outputs = model(**model_inputs)

    # Get next token
    next_token_logits = outputs.logits[:, -1, :]
    
    if gen_config.get("do_sample", False):
        # 1. Apply top-k filtering first (if enabled)
        if gen_config.get("top_k") is not None and gen_config.get("top_k") > 0:
            top_k = gen_config["top_k"]
            # Get top-k indices and values
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            # Create filtered logits tensor (set non-top-k to -inf)
            filtered_logits = torch.full_like(next_token_logits, float('-inf'))
            filtered_logits.scatter_(-1, top_k_indices, top_k_logits)
            next_token_logits = filtered_logits
        
        # 2. Apply temperature scaling (after top-k filtering)

        if gen_config.get("temperature", 1.) != 1.0:
            next_token_logits = next_token_logits / gen_config["temperature"]
        
        # 3. Sample from the (potentially filtered) distribution
        probs = torch.softmax(next_token_logits, dim=-1)
        if len(probs.shape) == 3:
            next_token_id = torch.multinomial(probs[0], 1)
        else:
            next_token_id = torch.multinomial(probs, 1)
    else:
        # Greedy decoding (no sampling)
        next_token_id = torch.argmax(next_token_logits, dim=-1)

    token_text = tokenizer.decode([next_token_id.item()]).strip()

    return next_token_id, token_text, outputs

def _extract_and_process_attention(
    outputs: Any,
    next_token_id: torch.Tensor,
    token_text: str,
    step_idx: int,
    num_patches: int,
    grid_size: int,
    image_token_start_index_in_llm: int,
    image: Any,
    attn_config: Dict,
    vis_output_dir_raw: Path,
    vis_output_dir_processed: Path,
    tensor_output_dir: Path
) -> Optional[Any]:
    """Extract attention map and save visualizations. Returns processed image for collage and the raw attention map."""
    if outputs.attentions is None:
        return None, None

    attentions = outputs.attentions
    selected_attentions = attentions[0][0].squeeze(0)  # Remove batch dimension
    avg_attentions = selected_attentions.mean(dim=0)  # Average across heads

    current_token_index = outputs.logits.shape[1] - 1
    token_attention_to_image = avg_attentions[current_token_index, image_token_start_index_in_llm:image_token_start_index_in_llm + num_patches]

    if token_attention_to_image.shape[0] != num_patches:
        return None, None

    # Handle padding for square grid
    expected_elements = grid_size * grid_size
    if expected_elements > num_patches:
        padding_size = expected_elements - num_patches
        token_attention_to_image = torch.cat([
            token_attention_to_image,
            torch.zeros(padding_size, device=token_attention_to_image.device)
        ])

    attention_map = token_attention_to_image.reshape(grid_size, grid_size).cpu().float().numpy()
    processed_img = None

    # Save visualizations
    if attn_config["visualize_attn_overlays"]:
        # Create safe filename
        safe_token_text = "".join(c if c.isalnum() else "_" for c in token_text)
        if not safe_token_text:
            safe_token_text = f"tokenid_{next_token_id.item()}"

        # Save raw attention
        raw_path = vis_output_dir_raw / f"token_{step_idx:03d}_{safe_token_text}.png"
        map_img = Image.fromarray(attention_map.astype(np.float32))
        resized_map = np.array(map_img.resize(image.size, Image.Resampling.LANCZOS))

        min_val, max_val = np.min(resized_map), np.max(resized_map)
        norm_map = np.zeros_like(resized_map) if max_val <= min_val else (resized_map - min_val) / (max_val - min_val)

        heatmap = cm.viridis(norm_map)[:, :, :3]
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        overlay_img = Image.blend(image, Image.fromarray(heatmap_uint8), alpha=0.5)
        overlay_img.save(raw_path)

        # Processed visualization
        processed_path = vis_output_dir_processed / f"token_{step_idx:03d}_{safe_token_text}.png"
        processed_img = visualize_processed_attention(
            attention_map=attention_map,
            original_image=image,
            output_path=processed_path,
            **{k: v for k, v in attn_config.items() if k in [
                'threshold_value', 'opening_kernel_size', 'min_blob_area',
                'min_avg_attention', 'show_highest_attn_blob', 'dilate_kernel_size'
            ]}
        )

    # Save tensor
    if attn_config["save_tensors"]:
        save_raw_attention_tensor(
            attention_map, attentions, tensor_output_dir,
            next_token_id.item(), token_text, step_idx
        )

    return processed_img, attention_map

def _create_collages(
    collected_maps: List,
    attn_config: Dict,
    collage_output_dir: Path
) -> None:
    """Create attention collages from collected maps."""
    if not (attn_config["create_collage"] and collected_maps):
        return

    maps_per_collage = attn_config["collage_grid_rows"] * attn_config["collage_grid_cols"]
    num_collages = (len(collected_maps) + maps_per_collage - 1) // maps_per_collage

    for collage_idx in range(num_collages):
        start_idx = collage_idx * maps_per_collage
        end_idx = min(start_idx + maps_per_collage, len(collected_maps))
        collage_maps = collected_maps[start_idx:end_idx]

        collage_path = collage_output_dir / f"attention_collage_{collage_idx + 1}.png"
        visualize_attention_collage(
            collage_maps, collage_path,
            (attn_config["collage_grid_rows"], attn_config["collage_grid_cols"])
        )

def print_summary(generation_summary, quality_analysis, final_text: str, generated_ids: List[int], output_dir: Path) -> None:
    print("" + "="*50)
    print("GENERATION COMPLETE")
    print("="*50)
    print(f"Generated text: {final_text}")
    print(f"Generated {len(generated_ids)} tokens")
    print(f"Output directory: {output_dir}")

    # Print enhanced evaluation summary
    if generation_summary and quality_analysis:
        print("" + "="*50)
        print("ENHANCED EVALUATION SUMMARY")
        print("="*50)
        
        # Overall quality score
        overall_score = quality_analysis.get("overall_quality_score", 0)
        interpretation = quality_analysis.get("interpretation", {})
        quality_level = "Unknown"
        for level, is_level in interpretation.items():
            if is_level:
                quality_level = level.title()
                break
        
        print(f"Overall Quality Score: {overall_score:.2f}/10 ({quality_level})")
        
        # Key metrics
        gen_metrics = generation_summary.get("average_confidence", {})
        print(f"Average Confidence: {gen_metrics.get('confidence_score', 0):.3f}")
        print(f"Average Entropy: {gen_metrics.get('entropy', 0):.3f}")
        print(f"Confidence Trend: {generation_summary.get('confidence_trajectory', {}).get('trend', 'unknown')}")
        
        # Diversity metrics
        diversity = generation_summary.get("diversity_metrics", {})
        print(f"Type-Token Ratio: {diversity.get('type_token_ratio', 0):.3f}")
        print(f"Bigram Repetition: {diversity.get('repetition_penalty_2gram', 0):.3f}")
        
        # Decision points
        decision_points = generation_summary.get("decision_points", [])
        print(f"Decision Points (low confidence): {len(decision_points)}")
        
        if decision_points:
            print("Key Decision Points:")
            for dp in decision_points[:3]:  # Show first 3
                step = dp["step"]
                prob = dp["top1_probability"]
                alts = dp.get("alternatives", [])[:2]  # Show top 2 alternatives
                conf = dp["confidence"]
                print(f"  Step {step}: Confidence ({conf:.3f})")
                for alt in alts:
                    print(f"    {alt}")
        
        # Quality factors breakdown
        factors = quality_analysis.get("quality_factors", {})
        print(f"Quality Breakdown:")
        print(f"  Confidence Factor: {factors.get('confidence', 0):.2f}")
        print(f"  Diversity Factor: {factors.get('diversity', 0):.2f}")
        print(f"  Repetition Penalty: {factors.get('repetition', 0):.2f}")
        print(f"  Stability Factor: {factors.get('stability', 0):.2f}")
        print(f"  Attention Correlation with Mask Factor: {factors.get('attention_correlation', 0):.2f}")

    # Print Attention Correlation
    attention_correlation = generation_summary.get('average_attention_correlation', 0)
    person_correlation = generation_summary.get('person_mask_correlation', 0)
    if attention_correlation:
        print("" + "="*50)
        print("ATTENTION CORRELATION METRICS")
        print("="*50)
        print(f"Normalized Correlation Score: {attention_correlation.get('normalized_correlation_score', 0):.3f}")
        print(f"  - On-Target Attention Mean: {attention_correlation.get('target_attention_mean', 0):.4f}")
        print(f"  - Off-Target Attention Mean: {attention_correlation.get('off_target_attention_mean', 0):.4f}")
        print(f"  - Focus Ratio: {attention_correlation.get('attention_focus_ratio', 0):.2f}x")
        print(f"  - Person Mask Attention Mean: {person_correlation.get('target_attention_mean', 0):.4f}")
        print(f"  - Person Focus Ratio: {person_correlation.get('attention_focus_ratio', 0):.2f}x")

# Summary and analysis of all bias sweep results

def _make_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    # Handle PyTorch tensors
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    else:
        return obj

def analyze_bias_sweep_results(all_results: Dict[float, Dict[str, Any]], base_output_dir: Union[str, Path], save_summary: bool = True) -> Dict[float, Dict[str, Any]]:
    """
    Analyze and summarize bias sweep experiment results.
    
    Args:
        all_results: Dictionary mapping bias strength to experiment results
        base_output_dir: Base output directory for saving summary
        save_summary: Whether to save the summary to file and print detailed results
        
    Returns:
        Dictionary of performance summary data
    """
    print(f"\n{'='*80}")
    print("BIAS SWEEP EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    # Analyze and sort results by performance
    performance_summary = []
    for bias_val, result in all_results.items():
        evaluation_summary = result.get("evaluation_summary", {})
        quality_analysis = result.get("quality_analysis", {})
        attention_correlation = result.get("attention_correlation", {})
        
        # Extract key performance metrics
        avg_confidence = evaluation_summary.get("average_confidence", {}).get("confidence_score", 0.0)
        avg_entropy = evaluation_summary.get("average_confidence", {}).get("entropy", float('inf'))
        correlation_score = attention_correlation.get("normalized_correlation_score", 0.0)
        generated_text = result.get("generated_text", "")
        num_tokens = result.get("num_tokens", 0)
        
        # Use existing overall quality score from quality_analysis
        overall_quality_score = quality_analysis.get("overall_quality_score", 0.0)
        if correlation_score == 0.0:
            continue  # Skip if correlation score is zero (no correlation with target mask)
        if 'looking' not in generated_text.lower():
            continue    # Skip if generated text does not contain 'looking'
        performance_summary.append({
            'bias_strength': bias_val,
            'overall_quality_score': overall_quality_score,
            'avg_confidence': avg_confidence,
            'avg_entropy': avg_entropy,
            'correlation_score': correlation_score,
            'generated_text': generated_text,
            'num_tokens': num_tokens,
            'quality_analysis': quality_analysis
        })
    
    # Sort by overall quality score (descending - higher is better)
    performance_summary.sort(key=lambda x: x['overall_quality_score'], reverse=True)
    
    # Display top performers
    print(f"\nTOP 5 PERFORMING BIAS STRENGTHS:")
    print(f"{'Rank':<4} {'Bias':<6} {'Quality':<8} {'Confidence':<11} {'Entropy':<8} {'Correlation':<11} {'Tokens':<7} {'Generated Text':<30}")
    print("-" * 100)
    
    for i, result in enumerate(performance_summary[:5], 1):
        print(f"{i:<4} {result['bias_strength']:<6.2f} {result['overall_quality_score']:<8.2f} "
                f"{result['avg_confidence']:<11.3f} {result['avg_entropy']:<8.3f} "
                f"{result['correlation_score']:<11.3f} {result['num_tokens']:<7} "
                f"{result['generated_text'][:30]:<30}")
    
    # Display worst performers for comparison
    print(f"\nWORST 3 PERFORMING BIAS STRENGTHS:")
    print(f"{'Rank':<4} {'Bias':<6} {'Quality':<8} {'Confidence':<11} {'Entropy':<8} {'Correlation':<11} {'Tokens':<7} {'Generated Text':<30}")
    print("-" * 100)
    
    for i, result in enumerate(performance_summary[-3:], len(performance_summary)-2):
        print(f"{i:<4} {result['bias_strength']:<6.2f} {result['overall_quality_score']:<8.2f} "
                f"{result['avg_confidence']:<11.3f} {result['avg_entropy']:<8.3f} "
                f"{result['correlation_score']:<11.3f} {result['num_tokens']:<7} "
                f"{result['generated_text'][:30]:<30}")
    
    if len(performance_summary) == 0:
        print("No valid results found with non-zero correlation scores.")
        return {}
    # Best bias strength recommendation
    best_result = performance_summary[0]
    print(f"\n🏆 RECOMMENDED BIAS STRENGTH: {best_result['bias_strength']:.2f}")
    print(f"   • Overall Quality Score: {best_result['overall_quality_score']:.2f}/10")
    print(f"   • Average Confidence: {best_result['avg_confidence']:.3f}")
    print(f"   • Average Entropy: {best_result['avg_entropy']:.3f}")
    print(f"   • Attention Correlation: {best_result['correlation_score']:.3f}")
    print(f"   • Generated Text: '{best_result['generated_text']}'")
    
    # Save detailed summary to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = Path(base_output_dir) / f"bias_sweep_summary_{timestamp}.json"
    summary_data = {
        "experiment_timestamp": timestamp,
        "bias_range": list(all_results.keys()),
        "performance_ranking": performance_summary,
        "best_bias_strength": best_result['bias_strength'],
        "summary_metrics": {
            "total_experiments": len(all_results),
            "best_quality_score": best_result['overall_quality_score'],
            "quality_score_range": [performance_summary[-1]['overall_quality_score'], performance_summary[0]['overall_quality_score']]
        }
    }
        
    if save_summary:
        # Convert to JSON-serializable format
        summary_data = _make_json_serializable(summary_data)
        
        with open(summary_path, 'w') as f:
            import json
            json.dump(summary_data, f, indent=2)
        
        print(f"\n📊 Detailed summary saved to: {summary_path}")
        print(f"{'='*80}")
    
    return performance_summary

# Batch processing helper functions
def prepare_person_desc_data(
    json_path: Union[str, Path],
    filter_keys: Optional[List[str]] = None,
    limit_items: Optional[int] = None
) -> Dict[str, str]:
    """Load and optionally filter and limit person description data."""
    json_path = fix_wsl_paths(str(json_path))
    data = load_train_results_json(json_path)
    if filter_keys:
        data = {k: v for k, v in data.items() if k in filter_keys}
    if limit_items:
        data = dict(list(data.items())[:limit_items])
    return data

def default_bias_range(
    min_bias: float = 1.0,
    max_bias: float = 4.0,
    steps: int = 4
) -> np.ndarray:
    """Return default bias sweep range."""
    return np.linspace(min_bias, max_bias, steps)

def prepare_batch_paths(
    image_key: str,
    base_image_dir: Union[str, Path],
    base_mask_dir: Union[str, Path],
    mask_filename_template: str = "gaze__{}_masks.npy",
    mask_filename_template2: str = "gaze__{}_results.npy"
) -> Tuple[Path, Path]:
    """Derive image and mask paths from image_key and validate existence."""
    base_image_dir = Path(fix_wsl_paths(str(base_image_dir)))
    base_mask_dir = Path(fix_wsl_paths(str(base_mask_dir)))
    parts = image_key.strip("/").split("/")
    folder, filename = parts[-2], parts[-1]
    image_path = base_image_dir / folder / filename
    mask_path = base_mask_dir / mask_filename_template.format(Path(filename).stem)
    if not image_path.exists() or not mask_path.exists():
        mask_path = base_mask_dir / mask_filename_template2.format(Path(filename).stem)
        if not mask_path.exists():
            print(f"Image or mask not found for {image_key}")
            return None, None
    return image_path, mask_path

def create_experiment_config(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    prompt: str,
    output_dir: Union[str, Path],
    generation_config: Optional[Dict[str, Any]] = None,
    attention_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build base experiment config dict for bias sweep."""
    return {
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "prompt": prompt,
        "output_dir": str(output_dir),
        "generation_config": generation_config or {},
        "attention_config": attention_config or {},
    }

def save_image_results(
    results: Dict[str, Any],
    output_dir: Union[str, Path],
    prefix: str = "bias_sweep_results",
    append_mode: bool = False,
    run_id: Optional[str] = None
) -> Path:
    """Save individual image results to a JSON file. 
    
    Args:
        results: Results to save
        output_dir: Directory to save results to
        prefix: Prefix for the filename
        append_mode: If True, append to existing file instead of creating new timestamped file
        run_id: Optional run identifier. If not provided and append_mode is True, uses current timestamp
    """
    output_dir = Path(output_dir)
    
    if append_mode:
        # Use single file per run for appending
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"{prefix}_{run_id}.json"
        
        # Append to existing JSON array or create new one
        append_results_to_json(results, path)
    else:
        # Original behavior: create timestamped file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"{prefix}_{timestamp}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_results_to_json(results, path)
    
    return path

    
def log_generation_step(step: int,
                        token_text: str,
                        evaluation_metrics: Dict[str, Any],
                        next_token_id: torch.Tensor,
                        eos_token_id: int,
                        conf_threshold: float = 0.8,
                        top_n: int = 3) -> bool:
    """
    Logs confidence, entropy and top-k alternatives.
    Returns True if the generated token is EOS (so the caller should break).
    """
    conf_score = evaluation_metrics["confidence"].get("confidence_score", 0)
    entropy = evaluation_metrics["confidence"].get("entropy", 0)
    print(f"Step {step}: '{token_text}' | Confidence: {conf_score:.3f} | Entropy: {entropy:.3f}")

    candidates = evaluation_metrics.get("top_k_analysis", {}).get("candidates", [])
    if conf_score < conf_threshold and len(candidates) >= top_n:
        print("  Top alternatives:")
        for idx, cand in enumerate(candidates[:top_n], start=1):
            print(f"    {idx}. '{cand['token_text']}' (p={cand['probability']:.3f})")

    if next_token_id.item() == eos_token_id:
        print("EOS token generated. Stopping.")
        return True
    if token_text.lower() == ".":
        print("Empty token generated. Stopping.")
        return True

    return False


def summarize_batch_results(
    all_image_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a summary of batch results containing only the prompt used,
    the generated prompt at the best bias strength, the best bias value,
    and the processing timestamp.
    """
    summary_results: Dict[str, Any] = {}
    for key, entry in all_image_results.items():
        best_bias = entry.get("best_bias_strength")
        gen_text = None
        if best_bias is not None and best_bias in entry.get("bias_sweep_results", {}):
            gen_text = entry["bias_sweep_results"][best_bias].get("generated_text")
        summary_results[key] = {
            "prompt_used": entry.get("prompt_used"),
            "generated_prompt": gen_text,
            "best_bias_strength": best_bias,
            "processing_timestamp": entry.get("processing_timestamp"),
        }
    return summary_results


# Dataset handling helper functions
def find_annotations_path(base_image_dir: Path) -> Optional[Path]:
    """Return the expected annotations file path if it exists, else None."""
    is_train = "train" in str(base_image_dir).lower()
    ann_name = "train_annotations_release.txt" if is_train else "test_annotations_release.txt"
    ann_path = base_image_dir.parent / ann_name
    return ann_path if ann_path.exists() else None


def load_image_files_from_annotations(ann_path: Path, dataset_root: Path) -> List[Path]:
    """Load image file paths from the first column of the annotations file.
    Builds paths by simple string concatenation: str(dataset_root) + '/' + rel_path.
    Does not resolve or check file existence.
    """
    df = pd.read_csv(str(ann_path), sep="\t", header=None, engine="python")
    # Split the single text column by comma into columns; we will use the first column
    df = df[0].astype(str).str.split(",", expand=True)
    rel_paths = df.iloc[:, 0].astype(str)

    files = (str(dataset_root) + '/' + rel_paths).tolist()

    # Keep stable ordering (no existence checks, no resolve)
    return sorted(files)


def filter_and_limit_files(
    image_files: List[Path],
    filter_keys: Optional[List[str]],
    limit_items: Optional[int],
) -> List[Path]:
    """Filter by provided keys and limit the number of files."""
    if filter_keys:
        image_files = [
            p for p in image_files
            if any(k in p.stem or k in str(p) for k in filter_keys)
        ]
    if limit_items is not None:
        image_files = image_files[: max(0, int(limit_items))]
    return image_files


def build_stem_index(image_files: List[Path]) -> Dict[str, Path]:
    """Index files by their stem. Last one wins for duplicates."""
    idx: Dict[str, Path] = {}
    for p in sorted(image_files):
        idx[p.stem] = p
    return idx


def map_person_desc_to_paths(
    person_desc_data: Dict[str, str],
    base_image_dir: Path,
) -> Dict[str, Path]:
    """Resolve image keys to file paths using annotations only (no directory scanning).
    Matches keys to stems first, then substring match over annotation-listed files.
    """
    ann_path = find_annotations_path(base_image_dir)
    if not ann_path:
        print(f"WARNING: No annotations file found near {base_image_dir}. Cannot resolve JSON keys to images.")
        return {}

    files = load_image_files_from_annotations(ann_path, base_image_dir.parent)
    if not files:
        print("WARNING: No images resolved from annotations (first column). Cannot map JSON keys to images.")
        return {}

    # Build stem index from annotation-listed files
    stem_index: Dict[str, Path] = {}
    for p in sorted(files):
        stem_index[p.stem] = p

    image_paths_map: Dict[str, Path] = {}
    for image_key in person_desc_data.keys():
        if image_key in stem_index:
            image_paths_map[image_key] = stem_index[image_key]
        else:
            # Restrict substring search to annotation-provided files
            candidates = [p for p in files if image_key in str(p)]
            if candidates:
                image_paths_map[image_key] = candidates[0]
            else:
                print(f"WARNING: Could not locate image for key '{image_key}' using annotations in {base_image_dir}")
    return image_paths_map


def get_image_files(
    base_image_dir: Path,
    filter_keys: Optional[List[str]],
    limit_items: Optional[int],
) -> List[Path]:
    """Load images from annotations only. Does not scan directories.
    Always returns a list of Path objects (possibly empty).
    """
    ann_path = find_annotations_path(base_image_dir)
    if not ann_path:
        print(f"WARNING: No annotations file found near {base_image_dir}. Not scanning directories.")
        return []

    print(f"Using annotations file: {ann_path}")
    image_files = load_image_files_from_annotations(ann_path, base_image_dir.parent)
    if not image_files:
        print("WARNING: No images resolved from annotations (first column). Not scanning directories.")
        return []

    return filter_and_limit_files(image_files, filter_keys, limit_items)


# ============================================================================
# Refactored utility functions for run_generation_with_attention
# ============================================================================

def initialize_generation_state(
    gen_config: Dict[str, Any],
    tokenizer: "PreTrainedTokenizer",
    input_ids: torch.Tensor
) -> Dict[str, Any]:
    """
    Initialize the generation state variables.
    
    Args:
        gen_config: Generation configuration dictionary
        tokenizer: Model tokenizer
        input_ids: Initial input token IDs
        
    Returns:
        Dictionary containing initialized state variables
    """
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]
    
    return {
        "max_new_tokens": gen_config["max_new_tokens"],
        "generated_ids": [],
        "past_key_values": None,
        "current_input_ids": input_ids,
        "confidence_tracker": ConfidenceMetrics(),
        "repetitivity_tracker": RepetitivityMetrics(window_size=10),
        "candidate_evaluator": TopKCandidateEvaluator(k=5, tokenizer=tokenizer),
        "all_step_metrics": [],
        "all_attention_maps": [],
        "eos_token_id": eos_token_id,
        "target_tokens": 0,
        "image_embeddings": None,
        "first_step_hidden_state": None,
        "set_layer_image_embeddings": None,  # Store embeddings from first step for reuse
        "all_correlation_metrics": [],
        "person_mask_correlation_metrics": [],
        "apply_only_target_mask": False
    }


def process_hidden_states_and_embeddings(
    outputs: Any,
    attn_config: Dict[str, Any],
    image_token_start_index_in_llm: int,
    num_patches: int,
    model: "PreTrainedModel",
    state: Dict[str, Any]
) -> Optional[torch.Tensor]:
    """
    Process hidden states and extract image embeddings on first step only.
    
    Args:
        outputs: Model outputs containing hidden states
        attn_config: Attention configuration
        image_token_start_index_in_llm: Start index of image tokens
        num_patches: Number of image patches
        model: The language model
        state: Generation state dictionary (modified in place)
        
    Returns:
        Image embeddings for similarity computation (only on first step, None afterwards)
    """
    if not outputs.hidden_states:
        return None
        
    last_hidden_state = outputs.hidden_states[-1].squeeze(0)
    
    if state["image_embeddings"] is None:  # First step only
        # Store the first step hidden state for later output
        state["first_step_hidden_state"] = last_hidden_state
        state["image_embeddings"] = last_hidden_state[
            image_token_start_index_in_llm : image_token_start_index_in_llm + num_patches
        ]
        
        # Get embeddings from specified layer for similarity computation
        # This will be reused for all subsequent steps
        set_layer_hidden_state = outputs.hidden_states[-1].squeeze(0)
        set_layer_image_embeddings = set_layer_hidden_state[
            image_token_start_index_in_llm : image_token_start_index_in_llm + num_patches
        ]
        
        # Store these embeddings for reuse in all steps
        state["set_layer_image_embeddings"] = set_layer_image_embeddings
        
        return set_layer_image_embeddings
    
    # For subsequent steps, return the stored embeddings from first step
    return state.get("set_layer_image_embeddings", None)


def create_similarity_visualization(
    set_layer_image_embeddings: Optional[torch.Tensor],
    next_token_id: torch.Tensor,
    model: "PreTrainedModel",
    image: Image.Image,
    grid_size: Tuple[int, int],
    similarity_output_dir: Path,
    step: int,
    token_text: str
) -> Optional[np.ndarray]:
    """
    Create and save embedding similarity visualization.
    
    Args:
        set_layer_image_embeddings: Image embeddings from specific layer
        next_token_id: Generated token ID
        model: The language model
        image: Original image
        grid_size: Grid dimensions for visualization
        similarity_output_dir: Output directory for similarity maps
        step: Current generation step
        token_text: Generated token text
        
    Returns:
        Similarity map array or None if no embeddings provided
    """
    if set_layer_image_embeddings is None:
        return None
        
    try:
        safe_token_text = "".join(c if c.isalnum() else "_" for c in token_text) or f"tokenid_{next_token_id.item()}"
        sim_path = similarity_output_dir / f"similarity_{step:03d}_{safe_token_text}.png"
        
        # Convert embeddings to token space for similarity computation
        tmp_image_embeddings_converted = model.get_model().embed_tokens(
            torch.argmax(model.get_output_embeddings()(set_layer_image_embeddings), dim=-1)
        )
        tmp_text_embedding_converted = model.get_model().embed_tokens(next_token_id).squeeze(0)
        
        similarity_map = visualize_embedding_similarity(
            text_token_embedding=tmp_text_embedding_converted,
            image_token_embeddings=tmp_image_embeddings_converted,
            original_image=image,
            grid_size=grid_size,
            output_path=sim_path,
        )
        
        return similarity_map
    except Exception as e:
        print(f"Warning: Failed to create similarity visualization: {e}")
        return None


def calculate_correlation_metrics(
    state: Dict[str, Any],
    input_masks: Dict[str, Any],
    similarity_map: Optional[np.ndarray],
    step: int,
    token_text: str,
    person_mask_indices: List[int],
    atten_indices: List[int]
) -> None:
    """
    Calculate and store correlation metrics for person and target masks.
    
    Args:
        state: Generation state dictionary (modified in place)
        input_masks: Dictionary containing mask data
        similarity_map: Text-to-image similarity matrix
        step: Current generation step
        token_text: Generated token text
        person_mask_indices: Indices for person mask
        atten_indices: Indices for attention target mask
    """
    # Calculate person mask correlation before switching to target mask
    if (not state["apply_only_target_mask"] and 
        state["all_attention_maps"] and 
        len(person_mask_indices) > 0 and
        similarity_map is not None):
        
        person_attention_correlation = calculate_attention_correlation_from_similarity(
            text_to_image_similarity_matrix=similarity_map,
            attention_mask=input_masks.get('person_mask', None),
            attention_map=state["all_attention_maps"][-1] if state["all_attention_maps"] else None
        )
        state["person_mask_correlation_metrics"].append({
            'step': step,
            'token': token_text,
            'mask_type': 'person_source',
            **person_attention_correlation
        })

    # Calculate target mask correlation after switching to target mode
    if (state["all_attention_maps"] and 
        len(atten_indices) > 0 and 
        state["apply_only_target_mask"] and 
        similarity_map is not None and
        token_text.lower().strip() not in ['at', 'looking', 'is', 'a', 'an']):
        
        attention_correlation = calculate_attention_correlation_from_similarity(
            text_to_image_similarity_matrix=similarity_map,
            attention_mask=input_masks.get('target_mask', None),
        )
        state["all_correlation_metrics"].append({
            'step': step,
            'token': token_text,
            'mask_type': 'target',
            **attention_correlation
        })


def update_generation_state(
    state: Dict[str, Any],
    next_token_id: torch.Tensor,
    token_text: str,
    outputs: Any,
    attention_map: Optional[np.ndarray]
) -> None:
    """
    Update generation state with new token and attention information.
    
    Args:
        state: Generation state dictionary (modified in place)
        next_token_id: Generated token ID
        token_text: Generated token text
        outputs: Model outputs
        attention_map: Processed attention map
    """
    # Update token tracking
    if state["apply_only_target_mask"]:
        state["target_tokens"] += 1

    # Check for transition to target mask mode
    if 'looking' in token_text.lower():
        state["apply_only_target_mask"] = True

    # Add token to generated sequence
    state["generated_ids"].append(next_token_id.item())

    # Store attention map if available
    if attention_map is not None:
        state["all_attention_maps"].append(attention_map)

    # Update model state for next iteration
    state["current_input_ids"] = next_token_id.view(1, -1)
    state["past_key_values"] = outputs.past_key_values


def create_generation_results(
    state: Dict[str, Any],
    tokenizer: "PreTrainedTokenizer",
    output_directories: Dict[str, str],
    gen_config: Dict[str, Any],
    attn_config: Dict[str, Any],
    guidance_config: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Create the final results dictionary from generation state.
    
    Args:
        state: Generation state dictionary
        tokenizer: Model tokenizer
        output_directories: Dictionary of output directory paths
        gen_config: Generation configuration
        attn_config: Attention configuration
        guidance_config: Guidance configuration
        
    Returns:
        Complete results dictionary
    """
    final_text = tokenizer.decode(state["generated_ids"], skip_special_tokens=True).strip()
    
    generation_summary, quality_analysis = None, None
    if state["all_step_metrics"]:
        generation_summary = create_generation_summary(
            state["confidence_tracker"], 
            state["repetitivity_tracker"], 
            state["candidate_evaluator"],
            final_text, 
            state["all_step_metrics"], 
            state["all_correlation_metrics"],
            person_mask_correlation_metrics=state["person_mask_correlation_metrics"]
        )
        quality_analysis = analyze_generation_quality(generation_summary)

    return {
        "generated_text": final_text,
        "generated_tokens": state["generated_ids"],
        "num_tokens": len(state["generated_ids"]),
        "output_directories": output_directories,
        "config_used": {
            "generation": gen_config,
            "attention": attn_config,
            "guidance": guidance_config,
        },
        "evaluation_summary": generation_summary,
        "quality_analysis": quality_analysis,
        "attention_correlation": generation_summary.get("average_attention_correlation", {}) if generation_summary else {},
        "person_mask_correlation": state["person_mask_correlation_metrics"],
        "target_mask_correlation": state["all_correlation_metrics"],
        "step_metrics": state["all_step_metrics"],
        "first_step_hidden_state": state["first_step_hidden_state"],
    }
