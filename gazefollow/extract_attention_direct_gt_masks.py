import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
import copy
import math
from typing import Dict, List, Optional, Any, Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
if not hasattr(torch.utils._pytree, "register_pytree_node") and hasattr(
    torch.utils._pytree, "_register_pytree_node"
):
    def _compat_register_pytree_node(*args, **kwargs):
        kwargs.pop("serialized_type_name", None)
        return torch.utils._pytree._register_pytree_node(*args, **kwargs)

    torch.utils._pytree.register_pytree_node = _compat_register_pytree_node
from transformers import PreTrainedModel, PreTrainedTokenizer
# Add project root to sys.path
try:
    project_root = Path(__file__).resolve().parent.parent
except NameError:
    project_root = Path.cwd().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
from gazefollow.extract_cls_token_image_similarity import draw_topk_similarity_overlay
from gazefollow.generation_utils import (
    enable_inference_optimizations,
    load_model_and_setup,
    fix_wsl_paths,
    _prepare_configs,
    _setup_output_directories,
    _prepare_inputs,
    _determine_image_patch_info,
    _generate_next_token,
    _extract_and_process_attention,
    _create_collages,
    visualize_embedding_similarity,
    load_train_results_json,
    build_prompt_with_subject_description,
    save_results_to_json,
    print_summary,
    analyze_bias_sweep_results,
    prepare_person_desc_data,
    default_bias_range,
    prepare_batch_paths,
    create_experiment_config,
    save_image_results,
    summarize_batch_results,
    log_generation_step,
    DEFAULT_GT_GAZE_CSV,
    save_mask_overlay_image,
    _mask_array_to_binary,
    # New gaze guidance functions
    select_token_by_gaze_correlation,
    generate_next_token_with_gaze_guidance,
    # Dataset annotation functions
    find_annotations_path,
    load_image_files_from_annotations,
    filter_and_limit_files,
    build_stem_index,
    map_person_desc_to_paths,
    get_image_files,
    # Refactored utility functions
    initialize_generation_state,
    process_hidden_states_and_embeddings,
    create_similarity_visualization,
    calculate_correlation_metrics,
    update_generation_state,
    create_generation_results,
    set_gt_annotation_lookup_use_body_bbox,
)
from llava.model.language_model.attention_mask_visualizer import (
    AttentionMaskFrame,
    capture_attention_mask_frame,
    save_attention_mask_sequence,
)
from gazefollow.repr_layer_token_injection_experiment import run_repr_layer_image_token_injection_experiment
from generation_metrics import (
    ConfidenceMetrics, RepetitivityMetrics, TopKCandidateEvaluator,
    generate_next_token_with_evaluation, create_generation_summary,
    analyze_generation_quality, calculate_attention_correlation_from_similarity,
    print_detailed_step_analysis, compare_generation_configs, analyze_top_k_impact
)
DEFAULT_PROMPT = """Complete the sentence in the following format, for example:
a tired guy in a hoodie → a guy in a gray hoodie and ripped jeans sitting on a worn wooden bench.  
a chic woman in a beige coat → a woman in a beige coat and ankle boots holding a phone.  
a techy man in a leather jacket → a man in a black leather jacket and glasses.  
a relaxed woman in a green sweater → a woman in a dark green sweater and black jeans carrying a tan shoulder bag.

The sentence: a _ → """
DEFAULT_TARGET_PROMPT = """Complete the sentence in the following format, for example:
a scruffy guy in a tee → a guy with messy hair wearing a faded graphic t-shirt and loose jeans.
a stylish woman in red → a woman with sleek hair wearing a bright red blazer and matching heels.
a bulky backpack → a large black backpack with thick straps and a padded mesh back.
a glossy metal bottle → a tall stainless-steel bottle with a smooth reflective finish.
a nerdy man with glasses → a man with round glasses, a plaid button-up, and tucked-in chinos.
a worn-out notebook → a small notebook with frayed edges and a cracked leather cover.

The sentence: a _ → """
DEFAULT_TARGET_PROMPT = """Complete the sentence in the following format, for example:
a bulky backpack → a large black backpack with thick straps and a padded mesh back.
a glossy metal bottle → a tall stainless-steel bottle with a smooth reflective finish.
a warm croissant → a golden, flaky croissant with crisp layers and a soft buttery center.
a vintage camera → a compact silver-and-black camera with a textured grip and a chunky lens.
a loaded hotdog → a toasted bun filled with a browned sausage, topped with bright mustard and diced onions.
a cozy wool blanket → a thick cream-colored blanket with a soft woven pattern.

The sentence: a _ →"""
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def create_person_mask_similarity_overlay(
    image_embeddings: Optional[torch.Tensor],
    person_mask_repr: Optional[torch.Tensor],
    image: Image.Image,
    grid_size: int,
    output_dir: Union[str, Path],
    step: int,
    top_k: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    Generate a top-k similarity overlay between cached image embeddings and
    the person mask representation produced by the model.
    """
    if image_embeddings is None or person_mask_repr is None:
        return None

    flattened = image_embeddings
    if flattened.dim() == 3 and flattened.shape[0] == 1:
        flattened = flattened.squeeze(0)
    if flattened.dim() != 2:
        flattened = flattened.reshape(-1, flattened.shape[-1])

    if flattened.shape[0] == 0 or flattened.shape[-1] == 0:
        return None

    current_grid = int(grid_size) if grid_size and grid_size > 0 else int(math.ceil(math.sqrt(flattened.shape[0])))
    working_embeddings = flattened.detach().to(device=person_mask_repr.device, dtype=torch.float32)
    normalized_image_embeddings = F.normalize(working_embeddings, p=2, dim=-1)
    normalized_person_repr = F.normalize(
        person_mask_repr.detach().to(device=working_embeddings.device, dtype=torch.float32),
        p=2,
        dim=-1,
    )

    usable_k = min(max(int(top_k), 1), normalized_image_embeddings.shape[0])
    similarities = torch.matmul(normalized_image_embeddings, normalized_person_repr)
    top_scores, top_indices = torch.topk(similarities, k=usable_k, dim=0)

    top_entries: List[Dict[str, Any]] = []
    for rank, (score_tensor, idx_tensor) in enumerate(zip(top_scores, top_indices), start=1):
        token_idx = int(idx_tensor.item())
        grid_row = token_idx // current_grid
        grid_col = token_idx % current_grid
        top_entries.append(
            {
                "rank": rank,
                "similarity": float(score_tensor.item()),
                "token_index": token_idx,
                "grid_row": grid_row,
                "grid_col": grid_col,
            }
        )

    if not top_entries:
        return None

    overlay_dir = Path(output_dir) / "person_mask_similarity"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = overlay_dir / f"person_mask_similarity_step{step:03d}.png"
    draw_topk_similarity_overlay(image, current_grid, top_entries, overlay_path)

    return {
        "path": str(overlay_path),
        "grid_size": current_grid,
        "top_entries": top_entries,
    }


# Main Generation Function with Attention Extraction
def run_generation_with_attention(
    image_path: Union[str, Path],
    mask_path: Optional[Union[str, Path]],
    prompt: str,
    output_dir: Union[str, Path],
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    image_processor: "SigLipImageProcessor",
    generation_config: Optional[Dict] = None,
    attention_config: Optional[Dict] = None,
    bias_strength: float = 0.0,
    prev_run_last_hidden_state: Optional[torch.Tensor] = None,
    break_after_first_step: bool = False,
    use_gaze_guidance: bool = True,  # New parameter
    guidance_config: Optional[Dict[str, Any]] = None,  # New parameter
    save_debug_files: bool = False,
    use_gt_gaze_csv: bool = False,
    gt_gaze_csv_path: Optional[Union[str, Path]] = None,
    gt_gaze_mask_radius: Optional[int] = None,
    gt_gaze_mask_radius_ratio: float = 0.02,
    save_mask_overlays: bool = False,
    mask_overlay_alpha: float = 0.4,
    include_image_inputs: bool = True,
    filter_image_tokens_to_person_mask: bool = False,
    same_mask_for_person: bool = False,
    use_target_insert_for_source: bool = False,
    attention_mask_viz_dir: Optional[Union[str, Path]] = None,
    use_body_bbox: bool = False,
    person_bbox_scale: float = 1.0,
) -> Dict[str, Any]:
    """
    Run generation with attention extraction and optional gaze guidance.

    Args:
        image_path: Path to input image
        mask_path: Path to attention mask file
        prompt: Input text prompt
        output_dir: Directory for saving outputs
        model: Pre-trained language model
        tokenizer: Model tokenizer
        image_processor: Image preprocessing component
        generation_config: Generation parameters
        attention_config: Attention extraction parameters
        bias_strength: Strength of attention bias
        prev_run_last_hidden_state: Hidden state from previous run
        break_after_first_step: Stop generation after first token
        use_gaze_guidance: Enable gaze-guided token selection
        guidance_config: Configuration for gaze guidance behavior
        beam_search_config: Configuration for beam search (unused)
        use_gt_gaze_csv: Replace gaze target mask with CSV-derived coordinates when True
        gt_gaze_csv_path: Optional override path for the GT CSV
        gt_gaze_mask_radius: Optional fixed radius (pixels) for generated mask
        gt_gaze_mask_radius_ratio: Relative radius fallback (fraction of min dimension)
        save_mask_overlays: Generate and save person/target overlay visualization
        mask_overlay_alpha: Alpha blend to use for overlay visualization
        include_image_inputs: Whether to send image tensors to the model on the first decoding step
        filter_image_tokens_to_person_mask: Restrict image tokens to the person_mask selection when True
        use_target_insert_for_source: When True, reuse gaze target insert indices/representations for the source slots
        attention_mask_viz_dir: Directory to store compressed custom attention mask visualizations.
            When unset or False, HTML visualization generation is disabled.

    Returns:
        Dictionary containing generation results and analysis
    """
    # Prepare configurations and paths
    gen_config, attn_config = _prepare_configs(generation_config, attention_config)
    mask_path = fix_wsl_paths(str(mask_path)) if mask_path is not None else None
    image_path = fix_wsl_paths(str(image_path))

    print(f"Processing image: {image_path}")
    # print(f"Using mask: {mask_path}")
    print(f"Prompt: {prompt}")
    print(f"Gaze guidance enabled: {use_gaze_guidance}")

    # Setup output directories
    output_directories = _setup_output_directories(output_dir)
    (output_dir, vis_output_dir_raw, vis_output_dir_processed, 
     tensor_output_dir, collage_output_dir, similarity_output_dir) = output_directories
    mask_overlay_dir = Path(output_dir) / "mask_overlays"
    mask_viz_dir: Optional[Path] = None
    if attention_mask_viz_dir:
        mask_viz_dir = Path(fix_wsl_paths(str(attention_mask_viz_dir))).expanduser().resolve()
    if mask_viz_dir is not None:
        mask_viz_dir.mkdir(parents=True, exist_ok=True)

    # Prepare inputs
    if use_target_insert_for_source and not same_mask_for_person:
        same_mask_for_person = True

    (image, input_masks, image_tensor, image_sizes, atten_indices,
     person_mask_indices, input_ids) = _prepare_inputs(
        image_path,
        mask_path,
        prompt,
        image_processor,
        tokenizer,
        model,
        use_gt_gaze_csv=use_gt_gaze_csv,
        gt_gaze_csv_path=gt_gaze_csv_path,
        gt_gaze_mask_radius=gt_gaze_mask_radius,
        gt_gaze_mask_radius_ratio=gt_gaze_mask_radius_ratio,
        insert_image_token=include_image_inputs,
        same_mask_for_person=same_mask_for_person,
        use_body_bbox=use_body_bbox,
        person_bbox_scale=person_bbox_scale,
    )

    target_mask_raw = input_masks.get('target_mask_raw')
    person_mask_raw = input_masks.get('person_mask_raw')

    if use_target_insert_for_source:
        print("Target insert override enabled: using gaze target mask for both source and target injections.")
        if target_mask_raw is not None:
            person_mask_raw = np.copy(target_mask_raw)
            input_masks['person_mask_raw'] = person_mask_raw
        if input_masks.get('target_mask') is not None:
            input_masks['person_mask'] = input_masks.get('target_mask')
        if atten_indices:
            person_mask_indices = list(atten_indices)

    person_mask_bbox: Optional[List[float]] = None
    if person_mask_raw is not None:
        binary_person_mask = _mask_array_to_binary(person_mask_raw, image.size)
        if binary_person_mask is not None:
            coords = np.argwhere(binary_person_mask > 0)
            if coords.size > 0:
                y_min = int(coords[:, 0].min())
                x_min = int(coords[:, 1].min())
                y_max = int(coords[:, 0].max())
                x_max = int(coords[:, 1].max())
                person_mask_bbox = [
                    float(x_min),
                    float(y_min),
                    float(x_max + 1),
                    float(y_max + 1),
                ]

    overlay_path: Optional[Path] = None
    if save_mask_overlays:
        overlay_path = save_mask_overlay_image(
            image=image,
            target_mask_raw=target_mask_raw,
            person_mask_raw=person_mask_raw,
            output_dir=mask_overlay_dir,
            filename_prefix=Path(image_path).stem,
            alpha=mask_overlay_alpha
        )
        if overlay_path:
            print(f"Saved mask overlay visualization to {overlay_path}")
        else:
            print("⚠️ Warning: Unable to create mask overlay (missing mask data).")

    # Raw masks are only needed for visualization; remove them before passing to the model
    input_masks.pop('target_mask_raw', None)
    input_masks.pop('person_mask_raw', None)

    # Setup model configuration
    boost_source_indices = atten_indices if use_target_insert_for_source else person_mask_indices
    boost_positions = {'gaze_source': boost_source_indices, 'gaze_target': atten_indices}
    image_token_filter_indices: Optional[List[int]] = None
    if filter_image_tokens_to_person_mask and person_mask_indices:
        image_token_filter_indices = list(person_mask_indices)
    num_patches, grid_size, image_token_start_index_in_llm, image_token_end_index_in_llm = _determine_image_patch_info(model, input_ids)

    # Initialize generation state
    print("Starting generation with attention extraction and advanced evaluation...")
    state = initialize_generation_state(gen_config, tokenizer, input_ids)
    attention_mask_frames: List[AttentionMaskFrame] = []

    def _pop_mask_snapshot(model_obj: Any) -> Optional[Dict[str, Any]]:
        getter = getattr(model_obj, "pop_latest_attention_mask_snapshot", None)
        if getter is None and hasattr(model_obj, "module"):
            getter = getattr(model_obj.module, "pop_latest_attention_mask_snapshot", None)
        if callable(getter):
            return getter()
        return None

    repr_layer_map = None
    repr_capture_layer_idx = None
    repr_inject_layer_idx = None
    if attn_config:
        repr_capture_layer_idx = attn_config.get("repr_capture_layer_idx")
        repr_inject_layer_idx = attn_config.get("repr_inject_layer_idx")
        repr_layer_map = attn_config.get("repr_layer_idx")
        # multi_layer_requested = (
        #     repr_capture_layer_idx is not None or repr_inject_layer_idx is not None
        # )
        # if multi_layer_requested:
        fallback_layer = repr_layer_map
        if fallback_layer is None:
            fallback_layer = attn_config.get("layer_idx")
        if fallback_layer is None:
            fallback_layer = -1
        repr_layer_map = {
            "capture": repr_capture_layer_idx if repr_capture_layer_idx is not None else fallback_layer,
            "inject": repr_inject_layer_idx if repr_inject_layer_idx is not None else fallback_layer,
        }

    # Main generation loop
    for i in range(state["max_new_tokens"]):
        with torch.inference_mode():
            # Prepare model inputs
            model_inputs = {
                "input_ids": state["current_input_ids"],
                "past_key_values": state["past_key_values"],
                "use_cache": True,
                "output_attentions": True,
                "output_hidden_states": True,
                "atten_ids": None,
                "boost_positions": boost_positions,     #if include_image_inputs else None,
                "include_image_inputs": include_image_inputs,
                "bias_strength": bias_strength,
                "query_indices": attn_config.get("query_indices", None),
                "repr_layer_idx": repr_layer_map,
                "target_mask_embedding": prev_run_last_hidden_state,
                "base_image_token_inds": [image_token_start_index_in_llm, image_token_start_index_in_llm + num_patches],
                "input_masks": input_masks,
                "apply_only_target_mask": state["apply_only_target_mask"],
                "target_tokens": state["target_tokens"],
                "target_tokens": state["target_tokens"],
                "image_token_filter_indices": image_token_filter_indices if include_image_inputs else None,
            }
            if use_target_insert_for_source:
                model_inputs["use_target_insert_indices_for_source"] = True
            if mask_viz_dir is not None:
                model_inputs["attention_mask_viz"] = {"capture_only": True}
            if i == 0 and include_image_inputs:
                model_inputs.update({"images": image_tensor, "image_sizes": image_sizes, "modalities": ["image"]})

            ## Todo: remove after testing
            # decoded = []
            # for ind, val in enumerate(model_inputs["input_ids"][0]):
            #     if val == tokenizer.convert_tokens_to_ids("<image>"):
            #         print(f"Image token at position {ind}")
            #     if val < tokenizer.vocab_size and val >= 0:
            #         decoded.append(tokenizer.convert_ids_to_tokens([val]))
            #         print(f"Text token '{tokenizer.convert_ids_to_tokens([val])}' at position {ind}")
            target_mask_embedding = model_inputs.get("target_mask_embedding", None)
            if isinstance(target_mask_embedding, dict):
                for key, tensor in target_mask_embedding.items():
                    if tensor is None:
                        continue
            # elif target_mask_embedding is not None:
            #     print(
            #         f"Using target mask embedding from previous run for guidance with mean: {target_mask_embedding.mean().item():.4f}, "
            #         f"min: {target_mask_embedding.min().item():.4f}, max: {target_mask_embedding.max().item():.4f}"
            #     )


            # Generate next token with gaze guidance
            next_token_id, token_text, outputs, evaluation_metrics = generate_next_token_with_gaze_guidance(
                model_inputs, model, tokenizer, gen_config,
                state["confidence_tracker"], state["repetitivity_tracker"], 
                state["candidate_evaluator"], i,
                image_embeddings=state["image_embeddings"],
                target_mask=input_masks.get('target_mask', None) if state["apply_only_target_mask"] else None,
                source_mask=input_masks.get('person_mask', None) if not state["apply_only_target_mask"] else None,
                apply_only_target_mask=state["apply_only_target_mask"],
                guidance_config=guidance_config
            )
            state["all_step_metrics"].append(evaluation_metrics)

            if mask_viz_dir is not None:
                snapshot = _pop_mask_snapshot(model)
                if snapshot and snapshot.get("mask") is not None:
                    try:
                        top_candidates = (
                            (evaluation_metrics.get("top_k_analysis") or {}).get("candidates")
                            if isinstance(evaluation_metrics, dict)
                            else None
                        )
                        frame = capture_attention_mask_frame(
                            attention_mask=snapshot["mask"],
                            tokens_indexing=snapshot.get("tokens_indexing"),
                            step_idx=i,
                            token_text=token_text or f"id_{next_token_id.item()}",
                            top_candidates=top_candidates,
                        )
                        attention_mask_frames.append(frame)
                    except Exception as exc:
                        print(f"[ATTN_VIZ] Failed to capture visualization frame for step {i}: {exc}")
            
            # print(f"Step {i+1}: Hidden state -1: mean {outputs.hidden_states[-1].mean().item():.4f}, min {outputs.hidden_states[-1].min().item():.4f}, max {outputs.hidden_states[-1].max().item():.4f}")
            # ## Todo: remove after testing
            # decoded = []
            # all_probs = torch.softmax(outputs.logits[0, :, :], dim=-1)
            # predall_probs = all_probs.argmax(dim=-1)
            # for ind, val in enumerate(predall_probs):
            #     if val == tokenizer.convert_tokens_to_ids("<image>"):
            #         print(f"Image token at position {ind}")
            #     if val < tokenizer.vocab_size and val >= 0:
            #         decoded.append(tokenizer.convert_ids_to_tokens([val]))
            #         print(f"Text token '{tokenizer.convert_ids_to_tokens([val])}' at position {ind}")


            # Process hidden states and extract embeddings
            set_layer_image_embeddings = process_hidden_states_and_embeddings(
                outputs, attn_config, image_token_start_index_in_llm, 
                num_patches, model, state
            )

            # Create similarity visualization
            similarity_map = create_similarity_visualization(
                set_layer_image_embeddings, next_token_id, model, image,
                grid_size, similarity_output_dir, i, token_text
            )

            person_mask_repr = getattr(model, "latest_person_mask_repr", None)
            if set_layer_image_embeddings is not None and person_mask_repr is not None:
                overlay_info = create_person_mask_similarity_overlay(
                    image_embeddings=set_layer_image_embeddings,
                    person_mask_repr=person_mask_repr,
                    image=image,
                    grid_size=grid_size,
                    output_dir=similarity_output_dir,
                    step=i,
                )
                if overlay_info:
                    overlay_info["step"] = i
                    state["person_mask_similarity_overlays"].append(overlay_info)
                    print(f"Saved person mask similarity overlay to {overlay_info['path']}")

            # Check for early termination
            if log_generation_step(i, token_text, evaluation_metrics, next_token_id, state["eos_token_id"]):
                break

            # Extract and process attention
            processed_img, attention_map = _extract_and_process_attention(
                outputs, next_token_id, token_text, i, num_patches, grid_size, 
                image_token_start_index_in_llm, image, attn_config, 
                vis_output_dir_raw, vis_output_dir_processed, tensor_output_dir
            )

            # Calculate correlation metrics
            calculate_correlation_metrics(
                state, input_masks, similarity_map, i, token_text,
                person_mask_indices, atten_indices
            )

            # Update generation state
            update_generation_state(state, next_token_id, token_text, outputs, attention_map)

            if break_after_first_step:
                print("Breaking after the first step as requested.")
                break

    # Create final results
    output_directories_dict = {
        "main": str(output_dir),
        "raw_attention": str(vis_output_dir_raw),
        "processed_attention": str(vis_output_dir_processed),
        "embedding_similarity": str(similarity_output_dir),
        "tensors": str(tensor_output_dir),
        "collages": str(collage_output_dir),
    }
    if save_mask_overlays:
        output_directories_dict["mask_overlays"] = str(mask_overlay_dir)
    
    results = create_generation_results(
        state, tokenizer, output_directories_dict, 
        gen_config, attn_config, guidance_config
    )

    if overlay_path:
        results["mask_overlay_path"] = str(overlay_path)
    if person_mask_bbox is not None:
        results["person_mask_bbox"] = person_mask_bbox
    if mask_viz_dir is not None and attention_mask_frames:
        combined_path = save_attention_mask_sequence(
            attention_mask_frames,
            Path(mask_viz_dir) / "attention_mask_sequence.html",
        )
        if combined_path:
            results["attention_mask_visualization"] = str(combined_path)

    # Print summary
    print_summary(
        results["evaluation_summary"], 
        results["quality_analysis"], 
        results["generated_text"], 
        results["generated_tokens"], 
        output_dir
    )

    return results


def load_previous_batch_results(results_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Load the most recent batch summary file to determine completed images.

    Args:
        results_dir: Path to the directory containing previous batch results

    Returns:
        Dictionary of completed image results from the batch summary file
    """
    results_dir = Path(results_dir)

    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist.")
        return {}

    # Look for batch summary results file
    batch_files = list(results_dir.glob("batch_bias_sweep_results_*.json"))
    if not batch_files:
        print(f"No batch summary files found in {results_dir}")
        return {}

    # Use the most recent batch file
    latest_batch_file = max(batch_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading completed results from: {latest_batch_file}")

    try:
        with open(latest_batch_file, 'r') as f:
            completed_results = json.load(f)
            print(f"Found {len(completed_results)} completed images")
            return completed_results
    except Exception as e:
        print(f"Error loading batch summary: {e}")
        return {}


def validate_resume_directory(resume_dir: Union[str, Path]) -> bool:
    """
    Validate that the resume directory exists and contains valid batch results.

    Args:
        resume_dir: Path to the directory to validate

    Returns:
        True if the directory is valid for resuming, False otherwise
    """
    resume_dir = Path(resume_dir)

    if not resume_dir.exists():
        print(f"ERROR: Resume directory does not exist: {resume_dir}")
        return False

    if not resume_dir.is_dir():
        print(f"ERROR: Resume path is not a directory: {resume_dir}")
        return False

    # Check for any subdirectories (image results) or batch summary files
    has_subdirs = any(item.is_dir() and not item.name.startswith('.') for item in resume_dir.iterdir())
    has_batch_files = any(resume_dir.glob("batch_bias_sweep_results_*.json"))

    if not has_subdirs and not has_batch_files:
        print(f"ERROR: Resume directory appears empty or invalid: {resume_dir}")
        return False

    print(f"Resume directory validation passed: {resume_dir}")
    return True


def load_sgl_conversation_data(sgl_path: Union[str, Path]) -> set:
    """
    Load SGL conversation data and extract processed image IDs.

    Args:
        sgl_path: Path to the SGL conversation data JSON file

    Returns:
        Set of processed image IDs
    """
    sgl_path = Path(fix_wsl_paths(str(sgl_path)))
    
    if not sgl_path.exists():
        print(f"Warning: SGL conversation file not found at {sgl_path}")
        return set()
    
    try:
        with open(sgl_path, 'r') as f:
            sgl_data = json.load(f)
        
        processed_ids = set()
        for entry in sgl_data:
            if 'id' in entry:
                processed_ids.add(entry['id'])
        
        print(f"Loaded {len(processed_ids)} processed image IDs from SGL conversation data")
        return processed_ids
    
    except Exception as e:
        print(f"Error loading SGL conversation data: {e}")
        return set()


def filter_images_by_sgl_data(
    person_desc_data: Dict[str, str],
    sgl_processed_ids: set
) -> Dict[str, str]:
    """
    Filter out images that are already processed in SGL conversation data.

    Args:
        person_desc_data: Dictionary of all images to process
        sgl_processed_ids: Set of processed image IDs from SGL data

    Returns:
        Dictionary of remaining images to process
    """
    remaining = {}
    original_count = len(person_desc_data)
    
    for image_key, subject_description in person_desc_data.items():
        # Extract image ID from the key
        if isinstance(image_key, Path):
            image_id = image_key.stem
        else:
            image_id = Path(image_key).stem
        
        # Check if this image ID is already processed
        if image_id not in sgl_processed_ids:
            remaining[image_key] = subject_description
        else:
            print(f"Skipping already processed image: {image_id}")

    filtered_count = original_count - len(remaining)
    print(f"Filtered out {filtered_count} already processed images")
    print(f"Found {len(remaining)} images remaining to process out of {original_count} total")
    return remaining


def get_remaining_images(
    person_desc_data: Dict[str, str],
    completed_results: Dict[str, Any]
) -> Dict[str, str]:
    """
    Determine which images still need to be processed.

    Args:
        person_desc_data: Dictionary of all images to process
        completed_results: Dictionary of already completed results

    Returns:
        Dictionary of remaining images to process
    """
    remaining = {}
    for image_key, subject_description in person_desc_data.items():
        if image_key not in completed_results:
            remaining[image_key] = subject_description
        # else:
        #     # Check if the results are complete (has bias sweep results)
        #     result_entry = completed_results[image_key]
        #     if not result_entry.get('bias_sweep_results') or not result_entry.get('performance_summary'):
        #         print(f"Results for {image_key} appear incomplete, will reprocess")
        #         remaining[image_key] = subject_description

    print(f"Found {len(remaining)} images remaining to process out of {len(person_desc_data)} total")
    return remaining


def process_batch_from_json(
    base_image_dir: Union[str, Path],
    base_mask_dir: Union[str, Path],
    base_output_dir: Union[str, Path],
    model,
    tokenizer,
    image_processor,
    mask_filename_template: str = "gaze__{}_masks.npy",
    prompt_template: str = "Complete the sentence. The {} is looking at",
    generation_config: Optional[Dict] = None,
    attention_config: Optional[Dict] = None,
    limit_items: Optional[int] = None,
    filter_keys: Optional[List[str]] = None,
    bias_range: Optional[np.ndarray] = None,
    resume_from_dir: Optional[Union[str, Path]] = None,
    use_person_descriptions: bool = False,
    json_path: Optional[Union[str, Path]] = None,
    start_from: Optional[str] = None,
    skip_first: int = 0,
    mask_filename_template2: str = "gaze__{}_results.npy",
    sgl_conversation_path: Optional[Union[str, Path]] = None,
    use_gt_gaze_csv: bool = False,
    gt_gaze_csv_path: Optional[Union[str, Path]] = None,
    gt_gaze_mask_radius: Optional[int] = None,
    gt_gaze_mask_radius_ratio: float = 0.02,
    save_mask_overlays: bool = False,
    mask_overlay_alpha: float = 0.4,
    include_image_inputs: Optional[bool] = None,
    use_target_insert_for_source: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Process multiple images, performing bias sweeps for each image.

    Args:
        start_from: Optional image ID (stem name without extension) to start processing from,
                   skipping all images that come before it in sorted order.

    Notes:
        - If use_person_descriptions is True and json_path is provided, descriptions are
          loaded via prepare_person_desc_data(json_path, ...).
        - Otherwise, images are discovered by scanning base_image_dir (recursively) and
          descriptions are left empty.
        - json_path is optional and only used when use_person_descriptions is True.
        - By default, person descriptions are not used.
        - start_from filtering is applied after resume logic but before processing begins.
    """
    # Generate unique run ID for this batch processing session
    from datetime import datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Handle resume functionality
    all_image_results: Dict[str, Any] = {}
    if resume_from_dir:
        resume_from_dir = fix_wsl_paths(str(resume_from_dir))
        if not validate_resume_directory(resume_from_dir):
            print("Resume directory validation failed. Starting fresh batch processing.")
            resume_from_dir = None
        else:
            print(f"Attempting to resume from previous run: {resume_from_dir}")
            all_image_results = load_previous_batch_results(resume_from_dir)

    # Prepare entries (either from JSON or by scanning image directory)
    person_desc_data: Dict[str, str] = {}
    image_paths_map: Dict[str, Path] = {}

    base_image_dir = Path(fix_wsl_paths(str(base_image_dir)))
    base_mask_dir = Path(fix_wsl_paths(str(base_mask_dir)))
    base_output_dir = Path(fix_wsl_paths(str(base_output_dir)))
    bias_vals = bias_range if bias_range is not None else default_bias_range()

    if use_person_descriptions and json_path:
        # Use JSON-provided mapping (image_key -> description)
        person_desc_data = prepare_person_desc_data(json_path, filter_keys, limit_items)
        # Resolve image paths by matching stems or substrings from annotations
        image_paths_map = map_person_desc_to_paths(person_desc_data, base_image_dir)
    else:
        # Load images using annotations (no directory scanning)
        image_files: List[Path] = get_image_files(base_image_dir, filter_keys, limit_items)
        # Map key -> empty description, and track path
        # Create dataframe from image files and empty descriptions
        files_df = pd.DataFrame({
            'image_file': image_files,
            'description': [""] * len(image_files)
        })

        # set the index as the first column (image_file)
        files_df.set_index('image_file', inplace=True)
        person_desc_data = files_df['description'].to_dict()

    # Apply SGL conversation filtering if path is provided
    if sgl_conversation_path:
        print(f"SGL conversation path provided: {sgl_conversation_path}")
        sgl_processed_ids = load_sgl_conversation_data(sgl_conversation_path)
        print(f"Original dataset: {len(person_desc_data)} images")
        person_desc_data = filter_images_by_sgl_data(person_desc_data, sgl_processed_ids)
        print(f"After SGL filtering: {len(person_desc_data)} images")

    # Determine which images still need processing (if resuming)
    full_entries_before_resume = dict(person_desc_data)  # for total count
    if resume_from_dir and all_image_results:
        person_desc_data = get_remaining_images(person_desc_data, all_image_results)
        if not person_desc_data:
            print("All images already processed! Nothing to resume.")
            # Don't save summary again if resuming and all processed, 
            # as individual results were already appended during original processing
            print("All processing already complete.")
            return all_image_results

    # Filter images based on start_from argument
    original_image_count = len(person_desc_data)
    if start_from:
        # Convert to list to maintain order and find start index
        image_keys = list(person_desc_data.keys())
        start_index = None
        
        # Find the index of the image to start from
        for i, image_key in enumerate(image_keys):
            # Extract stem from the image key (handle both Path objects and strings)
            if isinstance(image_key, Path):
                image_stem = image_key.stem
            else:
                image_stem = Path(image_key).stem
            
            if image_stem == start_from:
                start_index = i
                break
        
        if start_index is not None:
            # Keep only images from start_index onwards
            filtered_keys = image_keys[start_index:]
            person_desc_data = {key: person_desc_data[key] for key in filtered_keys}
            print(f"Starting from image '{start_from}' - skipping {start_index} image(s)")
        else:
            print(f"Warning: Image with ID '{start_from}' not found. Processing all images.")

    # Apply skip_first filtering if specified
    if skip_first > 0:
        image_keys = list(person_desc_data.keys())
        if skip_first < len(image_keys):
            skipped_keys = image_keys[skip_first:]
            person_desc_data = {key: person_desc_data[key] for key in skipped_keys}
            print(f"Skipping first {skip_first} image(s) - processing {len(person_desc_data)} remaining images")
        else:
            print(f"Warning: skip_first ({skip_first}) is greater than or equal to available images ({len(image_keys)}). No images to process.")
            return {}

    # Report resume status
    total_images = len(full_entries_before_resume)
    completed_count = len(all_image_results)
    remaining_count = len(person_desc_data)
    
    if start_from:
        print(f"START-FROM STATUS: Found {original_image_count} total image(s), processing {remaining_count} image(s) starting from '{start_from}'")
    
    if skip_first > 0:
        print(f"SKIP-FIRST STATUS: Skipped first {skip_first} image(s), processing {remaining_count} remaining image(s)")
    
    if resume_from_dir:
        print(f"RESUME STATUS: {completed_count}/{total_images} images completed, {remaining_count} remaining")
    else:
        print(f"BATCH PROCESSING: {remaining_count} images to process")

    final_path = None

    # Iterate through each remaining image entry
    for idx, (image_key, subject_description) in enumerate(person_desc_data.items(), start=1):
        if use_person_descriptions and json_path:
            subject_description = full_entries_before_resume.get(image_key, "")
            prompt = build_prompt_with_subject_description(subject_description, prompt_template)
            img_path = image_paths_map.get(image_key)
        else:
            prompt = prompt_template
            subject_description = ""
            img_path = Path(image_key)
            image_key = img_path.stem  # Use stem as key
        current_index = completed_count + idx
        print(f"{'='*80}\nProcessing {current_index}/{total_images}: {image_key} (remaining: {idx}/{remaining_count})\n{'='*80}")
        
        if img_path is None or not img_path.exists():
            print(f"Skipping {image_key}: image not found in scanned paths.")
            continue
        mask_path = base_mask_dir / mask_filename_template.format(img_path.stem)

        if not mask_path.exists():
            mask_path = base_mask_dir / mask_filename_template2.format(img_path.stem)
            if not mask_path.exists():
                print(f"Skipping {image_key}: mask not found at {mask_path}")
                continue

        output_dir = base_output_dir / image_key

        # Run bias sweep experiment
        print(f"Running bias sweep for {image_key} with range {bias_vals}")
        base_config = create_experiment_config(
            str(img_path), str(mask_path), prompt, output_dir,
            generation_config, attention_config,
            use_gt_gaze_csv=use_gt_gaze_csv,
            gt_gaze_csv_path=gt_gaze_csv_path,
            gt_gaze_mask_radius=gt_gaze_mask_radius,
            gt_gaze_mask_radius_ratio=gt_gaze_mask_radius_ratio,
            save_mask_overlays=save_mask_overlays,
            mask_overlay_alpha=mask_overlay_alpha,
        )
        base_config["use_target_insert_for_source"] = use_target_insert_for_source
        if include_image_inputs is not None:
            base_config["include_image_inputs"] = include_image_inputs
        bias_sweep_results = run_bias_sweep_experiment(
            base_experiment_config=base_config,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            bias_range=bias_vals,
            save_summary=False
        )

        # Analyze and store results
        perf_summary = analyze_bias_sweep_results(
            bias_sweep_results, output_dir, save_summary=False
        )
        if not perf_summary:
            print(f"No valid performance summary for {image_key}. Skipping.")
            continue

        result_entry = {
            "subject_description": subject_description,
            "bias_sweep_results": bias_sweep_results,
            "performance_summary": perf_summary,
            "best_bias_strength": perf_summary[0]['bias_strength'] if perf_summary else None,
            "prompt_used": prompt,
            "processing_timestamp": str(datetime.now()),
        }
        all_image_results[image_key] = result_entry

        # Save individual image results
        saved_path = save_image_results(result_entry, output_dir, append_mode=True, run_id=run_id)
        print(f"Results for {image_key} saved to: {saved_path}")

        # Append only the current image's summary to the batch file
        current_image_summary = {
            image_key: {
                "prompt_used": result_entry.get("prompt_used"),
                "generated_prompt": (result_entry["bias_sweep_results"][result_entry["best_bias_strength"]].get("generated_text") 
                                   if result_entry["best_bias_strength"] and result_entry["best_bias_strength"] in result_entry.get("bias_sweep_results", {}) 
                                   else None),
                "best_bias_strength": result_entry.get("best_bias_strength"),
                "processing_timestamp": result_entry.get("processing_timestamp"),
            }
        }
        final_path = save_image_results(current_image_summary, base_output_dir, prefix="batch_bias_sweep_results", append_mode=True, run_id=run_id)

    if final_path is not None:
        print(f"\nBATCH BIAS SWEEP PROCESSING COMPLETE. Final results saved to: {final_path}")
    else:
        print("\nBATCH BIAS SWEEP PROCESSING COMPLETE. No images were processed.")
    return all_image_results

def run_bias_sweep_experiment(
    base_experiment_config: Dict[str, Any],
    model,
    tokenizer,
    image_processor,
    bias_range: np.ndarray = np.linspace(0., 5., 5),
    save_summary: bool = True,
    use_gaze_guidance: bool = True,
    guidance_config: Optional[Dict[str, Any]] = None,
) -> Dict[float, Dict[str, Any]]:
    """
    Runs the generation experiment across a range of bias strengths.
    """
    all_results = {}
    base_output_dir = base_experiment_config.get("output_dir", "attention_output/bias_sweep")

    # initial run to get hidden state embeddings
    print(f"Running initial generation to get hidden state embeddings for bias sweep...")
    init_bias = 0.0
    experiment_config = copy.deepcopy(base_experiment_config)
    experiment_config["output_dir"] = f"{base_output_dir}/init_run_bias_{init_bias:.2f}"
    init_run_results = run_generation_with_attention(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_path=experiment_config["image_path"],
            mask_path=experiment_config["mask_path"],
            prompt="",       #experiment_config["prompt"],
            output_dir=experiment_config["output_dir"],
            generation_config=experiment_config["generation_config"],
            attention_config=experiment_config.get("attention_config"),
            bias_strength=init_bias,  # Initial run with no bias to get embeddings
            break_after_first_step=True,  # Only run the first step to get embeddings
            use_gaze_guidance=use_gaze_guidance,
            guidance_config=guidance_config,
            save_debug_files=base_experiment_config.get("save_debug_files", False),
            use_gt_gaze_csv=experiment_config.get("use_gt_gaze_csv", False),
            gt_gaze_csv_path=experiment_config.get("gt_gaze_csv_path"),
            gt_gaze_mask_radius=experiment_config.get("gt_gaze_mask_radius"),
            gt_gaze_mask_radius_ratio=experiment_config.get("gt_gaze_mask_radius_ratio", 0.02),
            save_mask_overlays=experiment_config.get("save_mask_overlays", False),
            mask_overlay_alpha=experiment_config.get("mask_overlay_alpha", 0.4),
            include_image_inputs=experiment_config.get("include_image_inputs", True),
            use_target_insert_for_source=experiment_config.get("use_target_insert_for_source", False),
        )

    for bias_i in bias_range:
        print(f"\n{'='*60}")
        print(f"Running bias sweep experiment with bias strength: {bias_i:.2f}")
        experiment_config = copy.deepcopy(base_experiment_config)
        experiment_config["output_dir"] = f"{base_output_dir}/bias_{bias_i:.2f}"
        experiment_config.setdefault("generation_config", {})["bias_strength"] = bias_i
        print(f"Starting attention extraction experiment for bias {bias_i:.2f}...")
        results = run_generation_with_attention(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_path=experiment_config["image_path"],
            mask_path=experiment_config["mask_path"],
            prompt=experiment_config["prompt"],
            output_dir=experiment_config["output_dir"],
            generation_config=experiment_config["generation_config"],
            attention_config=experiment_config.get("attention_config"),
            bias_strength=bias_i,
            prev_run_last_hidden_state=init_run_results['first_step_hidden_state'],  # Use initial run results for embeddings
            use_gaze_guidance=use_gaze_guidance,
            guidance_config=guidance_config,
            save_debug_files=base_experiment_config.get(
                "save_debug_files",
                base_experiment_config.get("generation_config", {}).get("save_debug_files", False)
            ),
            use_gt_gaze_csv=experiment_config.get("use_gt_gaze_csv", False),
            gt_gaze_csv_path=experiment_config.get("gt_gaze_csv_path"),
            gt_gaze_mask_radius=experiment_config.get("gt_gaze_mask_radius"),
            gt_gaze_mask_radius_ratio=experiment_config.get("gt_gaze_mask_radius_ratio", 0.02),
            save_mask_overlays=experiment_config.get("save_mask_overlays", False),
            mask_overlay_alpha=experiment_config.get("mask_overlay_alpha", 0.4),
            include_image_inputs=experiment_config.get("include_image_inputs", True),
            use_target_insert_for_source=experiment_config.get("use_target_insert_for_source", False),
        )
        results.pop('first_step_hidden_state', None)
        all_results[bias_i] = results
        print(f"Finished experiment for bias {bias_i:.2f}. Results saved to: {results['output_directories']['main']}")

    # Summary and analysis of all bias sweep results
    performance_summary = analyze_bias_sweep_results(all_results, base_output_dir, save_summary=save_summary)

    return all_results


def run_repr_layer_sweep_experiment(
    base_experiment_config: Dict[str, Any],
    model,
    tokenizer,
    image_processor,
    repr_layer_indices: List[int],
    bias_strength: Optional[float] = None,
    use_gaze_guidance: bool = True,
    guidance_config: Optional[Dict[str, Any]] = None,
    repr_inject_layer_indices: Optional[List[int]] = None,
    repr_capture_layer_indices: Optional[List[int]] = None,
    enable_pairwise_sweep: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Runs the generation experiment across different representation layer indices.
    """
    if not repr_layer_indices:
        raise ValueError("repr_layer_indices must contain at least one layer index to sweep.")
    repr_layer_indices = [int(idx) for idx in repr_layer_indices]
    if repr_inject_layer_indices is None:
        repr_inject_layer_indices = repr_layer_indices
    inject_indices = [int(idx) for idx in repr_inject_layer_indices] if repr_inject_layer_indices else repr_layer_indices
    if repr_capture_layer_indices is None:
        repr_capture_layer_indices = repr_layer_indices
    capture_indices = [int(idx) for idx in repr_capture_layer_indices] if repr_capture_layer_indices else repr_layer_indices

    combos: List[Tuple[int, int]]
    combos = [(src, tgt) for src in capture_indices for tgt in inject_indices]

    if not combos:
        raise ValueError("No representation layer combinations computed for sweep.")

    all_results: Dict[str, Dict[str, Any]] = {}
    base_output_dir = base_experiment_config.get("output_dir", "attention_output/repr_sweep")
    base_output_dir_path = Path(base_output_dir)
    base_output_dir_path.mkdir(parents=True, exist_ok=True)

    print("Running generation passes to cache hidden-state embeddings for repr layer sweep...")
    init_bias = bias_strength if bias_strength is not None else base_experiment_config.get(
        "generation_config", {}
    ).get("bias_strength", 0.0)

    effective_bias = bias_strength if bias_strength is not None else base_experiment_config.get(
        "generation_config", {}
    ).get("bias_strength", 0.0)

    required_cache_layers = sorted({idx for combo in combos for idx in combo})
    layer_hidden_state_cache: Dict[int, torch.Tensor] = {}
    if required_cache_layers:
        cache_layers_label = ", ".join(str(idx) for idx in required_cache_layers)
        print(f" - Caching hidden states for layers [{cache_layers_label}] in a single initial run")
        cache_config = copy.deepcopy(base_experiment_config)
        cache_config["output_dir"] = f"{base_output_dir}/repr_cache_all_layers"
        cache_config.setdefault("generation_config", {})["bias_strength"] = effective_bias
        attn_cfg = copy.deepcopy(cache_config.get("attention_config", {}))
        # attn_cfg.pop("repr_capture_layer_idx", None)
        # attn_cfg.pop("repr_inject_layer_idx", None)
        # attn_cfg["repr_layer_idx"] = required_cache_layers[0]
        attn_cfg["return_full_hidden_states"] = True
        cache_config["attention_config"] = attn_cfg
        init_run_results = run_generation_with_attention(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_path=cache_config["image_path"],
            mask_path=cache_config["mask_path"],
            prompt="",
            output_dir=cache_config["output_dir"],
            generation_config=cache_config["generation_config"],
            attention_config=cache_config.get("attention_config"),
            bias_strength=init_bias,
            break_after_first_step=True,
            use_gaze_guidance=use_gaze_guidance,
            guidance_config=guidance_config,
            save_debug_files=base_experiment_config.get("save_debug_files", False),
            use_gt_gaze_csv=cache_config.get("use_gt_gaze_csv", False),
            gt_gaze_csv_path=cache_config.get("gt_gaze_csv_path"),
            gt_gaze_mask_radius=cache_config.get("gt_gaze_mask_radius"),
            gt_gaze_mask_radius_ratio=cache_config.get("gt_gaze_mask_radius_ratio", 0.02),
            save_mask_overlays=cache_config.get("save_mask_overlays", False),
            mask_overlay_alpha=cache_config.get("mask_overlay_alpha", 0.4),
            include_image_inputs=cache_config.get("include_image_inputs", True),
            use_target_insert_for_source=cache_config.get("use_target_insert_for_source", False),
        )
        all_hidden_states = init_run_results.get("first_step_all_hidden_states")
        if not all_hidden_states:
            raise RuntimeError("Failed to retrieve full hidden state cache from initial run.")
        total_layers = len(all_hidden_states)
        for layer_idx in required_cache_layers:
            normalized_idx = layer_idx if layer_idx >= 0 else total_layers + layer_idx
            if normalized_idx < 0 or normalized_idx >= total_layers:
                raise ValueError(
                    f"Requested hidden-state layer {layer_idx} is out of range for {total_layers} total layers."
                )
            layer_hidden_state_cache[layer_idx] = all_hidden_states[normalized_idx]

    for source_idx, target_idx in combos:
        print(f"\n{'='*60}")
        if enable_pairwise_sweep:
            print(
                f"Running repr-layer sweep experiment with capture layer {source_idx} -> inject layer {target_idx}"
            )
        else:
            print(f"Running repr-layer sweep experiment with repr_layer_idx: {target_idx}")

        experiment_config = copy.deepcopy(base_experiment_config)
        combo_label = f"src{source_idx}_tgt{target_idx}" #if enable_pairwise_sweep else f"layer_{target_idx}"
        experiment_config["output_dir"] = f"{base_output_dir}/{combo_label}"
        experiment_config.setdefault("generation_config", {})["bias_strength"] = effective_bias
        attn_cfg = copy.deepcopy(experiment_config.get("attention_config", {}))
        # attn_cfg["repr_layer_idx"] = target_idx
        # if enable_pairwise_sweep:
        attn_cfg["repr_capture_layer_idx"] = source_idx
        attn_cfg["repr_inject_layer_idx"] = target_idx
        # else:
        #     attn_cfg.pop("repr_capture_layer_idx", None)
        #     attn_cfg.pop("repr_inject_layer_idx", None)
        experiment_config["attention_config"] = attn_cfg

        # if enable_pairwise_sweep:
        #     prev_hidden_state: Union[torch.Tensor, Dict[str, torch.Tensor]] = {
        #         "source": layer_hidden_state_cache[source_idx],
        #         "target": layer_hidden_state_cache[target_idx],
        #     }
        # else:
        prev_hidden_state = layer_hidden_state_cache[source_idx]

        results = run_generation_with_attention(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_path=experiment_config["image_path"],
            mask_path=experiment_config["mask_path"],
            prompt=experiment_config["prompt"],
            output_dir=experiment_config["output_dir"],
            generation_config=experiment_config["generation_config"],
            attention_config=experiment_config.get("attention_config"),
            bias_strength=experiment_config["generation_config"].get("bias_strength", 0.0),
            prev_run_last_hidden_state=prev_hidden_state,
            use_gaze_guidance=use_gaze_guidance,
            guidance_config=guidance_config,
            save_debug_files=base_experiment_config.get(
                "save_debug_files",
                base_experiment_config.get("generation_config", {}).get("save_debug_files", False)
            ),
            use_gt_gaze_csv=experiment_config.get("use_gt_gaze_csv", False),
            gt_gaze_csv_path=experiment_config.get("gt_gaze_csv_path"),
            gt_gaze_mask_radius=experiment_config.get("gt_gaze_mask_radius"),
            gt_gaze_mask_radius_ratio=experiment_config.get("gt_gaze_mask_radius_ratio", 0.02),
            save_mask_overlays=experiment_config.get("save_mask_overlays", False),
            mask_overlay_alpha=experiment_config.get("mask_overlay_alpha", 0.4),
            include_image_inputs=experiment_config.get("include_image_inputs", True),
            use_target_insert_for_source=experiment_config.get("use_target_insert_for_source", False),
        )
        results.pop("first_step_hidden_state", None)
        results["repr_capture_layer_idx"] = source_idx if enable_pairwise_sweep else target_idx
        results["repr_inject_layer_idx"] = target_idx
        all_results[combo_label] = results
        print(f"Finished experiment for {combo_label}. Results saved to: {results['output_directories']['main']}")

    if not all_results:
        print("No representation layer sweep results were generated.")
        return {}

    print(f"\n{'='*80}")
    print("REPRESENTATION LAYER SWEEP SUMMARY")
    print(f"{'='*80}")

    performance_summary: List[Dict[str, Any]] = []
    for _, result in all_results.items():
        evaluation_summary = result.get("evaluation_summary") or {}
        quality_analysis = result.get("quality_analysis") or {}
        attention_correlation = result.get("attention_correlation") or {}

        avg_confidence = (evaluation_summary.get("average_confidence") or {}).get("confidence_score", 0.0)
        avg_entropy = (evaluation_summary.get("average_confidence") or {}).get("entropy", float("inf"))
        correlation_score = attention_correlation.get("normalized_correlation_score", 0.0)
        generated_text = result.get("generated_text", "") or ""
        num_tokens = result.get("num_tokens", 0)
        overall_quality_score = quality_analysis.get("overall_quality_score", 0.0) if quality_analysis else 0.0
        source_idx = result.get("repr_capture_layer_idx")
        if source_idx is None:
            source_idx = result.get("repr_inject_layer_idx")
        target_idx = result.get("repr_inject_layer_idx", source_idx)

        if correlation_score == 0.0:
            continue
        if "looking" not in generated_text.lower():
            continue

        performance_summary.append({
            "repr_capture_layer_idx": source_idx,
            "repr_inject_layer_idx": target_idx,
            "overall_quality_score": overall_quality_score,
            "avg_confidence": avg_confidence,
            "avg_entropy": avg_entropy,
            "correlation_score": correlation_score,
            "generated_text": generated_text,
            "num_tokens": num_tokens,
            "quality_analysis": quality_analysis
        })

    performance_summary.sort(key=lambda x: x["overall_quality_score"], reverse=True)

    if performance_summary:
        heading = "TOP 5 REPRESENTATION LAYERS"
        print(f"\n{heading}:")
        if enable_pairwise_sweep:
            print(
                f"{'Rank':<4} {'Cap':<5} {'Inj':<5} {'Quality':<8} {'Confidence':<11} "
                f"{'Entropy':<8} {'Correlation':<11} {'Tokens':<7} {'Generated Text':<30}"
            )
        else:
            print(
                f"{'Rank':<4} {'Layer':<7} {'Quality':<8} {'Confidence':<11} "
                f"{'Entropy':<8} {'Correlation':<11} {'Tokens':<7} {'Generated Text':<30}"
            )
        print("-" * 105)
        for i, result in enumerate(performance_summary[:5], 1):
            display_source = result.get("repr_capture_layer_idx")
            if display_source is None:
                display_source = result.get("repr_inject_layer_idx")
            display_target = result.get("repr_inject_layer_idx")
            if enable_pairwise_sweep:
                print(
                    f"{i:<4} {display_source:<5} {display_target:<5} {result['overall_quality_score']:<8.2f} "
                    f"{result['avg_confidence']:<11.3f} {result['avg_entropy']:<8.3f} "
                    f"{result['correlation_score']:<11.3f} {result['num_tokens']:<7} "
                    f"{result['generated_text'][:30]:<30}"
                )
            else:
                print(
                    f"{i:<4} {display_target:<7} {result['overall_quality_score']:<8.2f} "
                    f"{result['avg_confidence']:<11.3f} {result['avg_entropy']:<8.3f} "
                    f"{result['correlation_score']:<11.3f} {result['num_tokens']:<7} "
                    f"{result['generated_text'][:30]:<30}"
                )

        print(f"\nWORST 3 REPRESENTATION LAYERS:")
        if enable_pairwise_sweep:
            print(
                f"{'Rank':<4} {'Cap':<5} {'Inj':<5} {'Quality':<8} {'Confidence':<11} "
                f"{'Entropy':<8} {'Correlation':<11} {'Tokens':<7} {'Generated Text':<30}"
            )
        else:
            print(
                f"{'Rank':<4} {'Layer':<7} {'Quality':<8} {'Confidence':<11} "
                f"{'Entropy':<8} {'Correlation':<11} {'Tokens':<7} {'Generated Text':<30}"
            )
        print("-" * 105)
        worst_start = max(len(performance_summary) - 3, 0)
        for i, result in enumerate(performance_summary[worst_start:], worst_start + 1):
            display_source = result.get("repr_capture_layer_idx")
            if display_source is None:
                display_source = result.get("repr_inject_layer_idx")
            display_target = result.get("repr_inject_layer_idx")
            if enable_pairwise_sweep:
                print(
                    f"{i:<4} {display_source:<5} {display_target:<5} {result['overall_quality_score']:<8.2f} "
                    f"{result['avg_confidence']:<11.3f} {result['avg_entropy']:<8.3f} "
                    f"{result['correlation_score']:<11.3f} {result['num_tokens']:<7} "
                    f"{result['generated_text'][:30]:<30}"
                )
            else:
                print(
                    f"{i:<4} {display_target:<7} {result['overall_quality_score']:<8.2f} "
                    f"{result['avg_confidence']:<11.3f} {result['avg_entropy']:<8.3f} "
                    f"{result['correlation_score']:<11.3f} {result['num_tokens']:<7} "
                    f"{result['generated_text'][:30]:<30}"
                )

        best_result = performance_summary[0]
        best_capture_idx = best_result.get("repr_capture_layer_idx")
        if best_capture_idx is None:
            best_capture_idx = best_result.get("repr_inject_layer_idx")
        best_inject_idx = best_result.get("repr_inject_layer_idx")
        if enable_pairwise_sweep:
            print(
                f"\n🏆 RECOMMENDED REPR LAYERS: "
                f"capture={best_capture_idx} -> inject={best_inject_idx}"
            )
        else:
            print(f"\n🏆 RECOMMENDED REPR LAYER: {best_inject_idx}")
        print(f"   • Overall Quality Score: {best_result['overall_quality_score']:.2f}/10")
        print(f"   • Average Confidence: {best_result['avg_confidence']:.3f}")
        print(f"   • Average Entropy: {best_result['avg_entropy']:.3f}")
        print(f"   • Attention Correlation: {best_result['correlation_score']:.3f}")
        print(f"   • Generated Text: '{best_result['generated_text']}'")
    else:
        best_result = None
        best_capture_idx = None
        best_inject_idx = None
        print("No representation layers met the filtering criteria (non-zero correlation and containing 'looking').")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generated_text_by_combo = {
        combo_label: (result.get("generated_text", "") or "")
        for combo_label, result in all_results.items()
    }
    sweep_payload = {
        "timestamp": timestamp,
        "image_path": str(base_experiment_config.get("image_path") or ""),
        "mask_path": str(base_experiment_config.get("mask_path") or ""),
        "output_dir": str(base_output_dir_path),
        "generated_text_by_layer_combination": generated_text_by_combo,
    }
    saved_results_path = save_image_results(
        sweep_payload,
        base_output_dir_path,
        prefix="repr_layer_sweep_results",
        append_mode=False
    )
    print(f"\n📁 Representation layer sweep results saved to: {saved_results_path}")
    print(f"{'='*80}")

def print_resume_usage_examples():
    """Print usage examples for the resume functionality."""
    print("\n" + "="*80)
    print("RESUME AND START-FROM FUNCTIONALITY USAGE EXAMPLES:")
    print("="*80)
    print("1. Resume a previous batch run:")
    print("   python extract_attention_interactive_refactored.py --mode batch \\")
    print("          --resume_from_dir /path/to/previous/run/output \\")
    print("          --json_path /path/to/data.json \\")
    print("          --base_image_dir /path/to/images \\")
    print("          --base_mask_dir /path/to/masks")
    print()
    print("2. Start processing from a specific image (useful for partial reprocessing):")
    print("   python extract_attention_interactive_refactored.py --mode batch \\")
    print("          --start_from image_12345 \\")
    print("          --json_path /path/to/data.json \\")
    print("          --base_image_dir /path/to/images \\")
    print("          --base_mask_dir /path/to/masks")
    print()
    print("3. Combine resume and start-from (resume first, then apply start-from filter):")
    print("   python extract_attention_interactive_refactored.py --mode batch \\")
    print("          --resume_from_dir /path/to/previous/run/output \\")
    print("          --start_from image_12345 \\")
    print("          --json_path /path/to/data.json \\")
    print("          --base_image_dir /path/to/images \\")
    print("          --base_mask_dir /path/to/masks")
    print()
    print("4. Use SGL conversation filtering with regular batch mode:")
    print("   python extract_attention_interactive_refactored.py --mode batch \\")
    print("          --sgl_conversation_path sgl_conversation_data.json \\")
    print("          --base_image_dir /path/to/images \\")
    print("          --base_mask_dir /path/to/masks")
    print()
    print("5. Combine SGL filtering with resume and start-from:")
    print("   python extract_attention_interactive_refactored.py --mode batch \\")
    print("          --sgl_conversation_path sgl_conversation_data.json \\")
    print("          --resume_from_dir /path/to/previous/run/output \\")
    print("          --start_from image_12345 \\")
    print("          --base_image_dir /path/to/images \\")
    print("          --base_mask_dir /path/to/masks")
    print()
    print("6. Skip the first X images in batch processing:")
    print("   python extract_attention_interactive_refactored.py --mode batch \\")
    print("          --skip_first 100 \\")
    print("          --base_image_dir /path/to/images \\")
    print("          --base_mask_dir /path/to/masks")
    print()
    print("7. Combine all filtering options:")
    print("   python extract_attention_interactive_refactored.py --mode batch \\")
    print("          --sgl_conversation_path sgl_conversation_data.json \\")
    print("          --skip_first 50 \\")
    print("          --start_from image_12345 \\")
    print("          --resume_from_dir /path/to/previous/run/output \\")
    print("          --base_image_dir /path/to/images \\")
    print("          --base_mask_dir /path/to/masks")
    print()
    print("8. Resume functionality will automatically:")
    print("   - Load the most recent batch_bias_sweep_results_*.json file")
    print("   - Determine which images were already processed")
    print("   - Continue processing only the remaining images")
    print()
    print("9. Start-from functionality will:")
    print("   - Skip all images that come before the specified image ID in sorted order")
    print("   - Use the image stem (filename without extension) for matching")
    print("   - Work with both resume and fresh processing")
    print()
    print("10. Skip-first functionality will:")
    print("   - Skip the first X images in the dataset after all other filtering")
    print("   - Applied after SGL filtering, resume filtering, and start-from filtering")
    print("   - Useful for distributed processing or continuing from a specific point")
    print()
    print("11. SGL filtering (--sgl_conversation_path) will:")
    print("   - Load processed image IDs from sgl_conversation_data.json")
    print("   - Skip images that already have conversation entries")
    print("   - Apply before resume and start-from filtering")
    print()
    print("12. Example directory structure for resuming:")
    print("   /previous/run/output/")
    print("   ├── batch_bias_sweep_results_20250714_123456.json  <- Resume from here")
    print("   ├── image1/")
    print("   │   └── bias_sweep_results_*.json")
    print("   └── image2/")
    print("       └── bias_sweep_results_*.json")
    print("="*80 + "\n")


if __name__ == '__main__':
    enable_inference_optimizations()
    
    parser = argparse.ArgumentParser(description="Run LLaVA-NeXT generation with attention extraction.")
    parser.add_argument('--mode', type=str, default='repr_sweep', choices=['single', 'batch', 'sweep', 'repr_sweep'],
                        help="Execution mode: 'single' (one image), 'batch' (JSON list), 'sweep' (bias sweep), 'repr_sweep' (representation layer sweep).")

    # --- Model Loading Arguments ---
    parser.add_argument('--model_path', type=str, default="lmms-lab/llava-onevision-qwen2-7b-ov-chat", help="Path to the base model or fully merged checkpoint.")
    parser.add_argument('--model_base', type=str, default=None, help="Optional explicit base model path when loading adapters.")
    parser.add_argument('--adapter_path', type=str, default=None, help="Optional LoRA adapter checkpoint to load on top of the base model.")
    parser.add_argument('--attn_implementation', type=str, default="sdpa", help="Attention implementation ('sdpa' or 'eager').")
    parser.add_argument('--load_4bit', action='store_true', help="Load model in 4-bit.")
    parser.add_argument('--load_8bit', action='store_true', help="Load model in 8-bit.")
    parser.add_argument('--attn_layer_ind', type=int, default=23, help="Attention layer index for attention maps and default similarity.")
    parser.add_argument(
        '--repr_layer_idx', type=int, default=None,
        help="Hidden-state layer index used for similarity/representation computations (defaults to attn_layer_ind)."
    )

    # --- Single Experiment Arguments ---
    parser.add_argument('--image_path', type=str, default=r"D:\Projects\data\gazefollow\train\00000000\00000691.jpg", help="Path to the input image.")
    parser.add_argument('--mask_path', type=str, default=r"D:\Projects\data\gazefollow\train_gaze_segmentations\small_masks\gaze__00000691_results.npy", help="Path to the attention mask.")
    # parser.add_argument('--image_path', type=str, default=r"D:\Projects\Annotators\data\llava_results\our_llava_results\109166.png", help="Path to the input image.")
    # parser.add_argument('--mask_path', type=str, default=r"D:\Projects\data\gazefollow\train_gaze_segmentations\manual_masks\gaze__109166_masks.npy", help="Path to the attention mask.")
    # parser.add_argument('--prompt', type=str, default="The _ is looking at _ . Where is the _ person looking?", help="Input prompt.")
    # parser.add_argument('--prompt', type=str, default="Describe the person _ which is looking at _", help="Input prompt.")
#     parser.add_argument('--prompt', type=str, default="""Complete the sentence in the following format, examples:
# a woman looking at a red mug → a woman in a cream sweater with straight dark hair, looking at a small red ceramic mug
# a guy looking at a laptop → a guy in a gray hoodie and jeans, looking at an open silver laptop
# a girl looking at a book → a girl with a ponytail and a denim jacket, looking at a thick hardcover book
# a boy looking at another boy → a boy in a blue t-shirt with curly hair, looking at a shorter boy in a yellow hoodie
# a man looking at a woman → a man in a black jacket and glasses, looking at a woman in a long beige coat
# a woman looking at a child → a woman with wavy brown hair and a green coat, looking at a small child in a red jacket
# a person looking at a dog → a person in a puffer vest and beanie, looking at a small brown dog
# The sentence: a _ looking at _ → """, help="Input prompt.")
#                         a guy → a guy in a gray hoodie and ripped jeans sitting on a worn wooden bench. 
#                         a woman → a woman in a beige coat and ankle boots holding a phone. 
#                         a man → a man in a black leather jacket and glasses. 
#                         a woman → a woman in a dark green sweater and black jeans carrying a tan shoulder bag
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT, help="Input prompt.")
    parser.add_argument('--use-gt-gaze-csv', action=argparse.BooleanOptionalAction, default=False,
                        dest='use_gt_gaze_csv', help="Use ground-truth gaze CSV to override gaze masks (default: enabled).")
    parser.add_argument('--gt_gaze_csv_path', type=str, default=str(DEFAULT_GT_GAZE_CSV),
                        help="Path to the ground-truth gaze CSV file.")
    parser.add_argument('--gt_gaze_mask_radius', type=int, default=None,
                        help="Optional fixed radius (pixels) for GT gaze mask blobs.")
    parser.add_argument('--gt_gaze_mask_radius_ratio', type=float, default=0.05,
                        help="Relative radius used when no fixed radius is provided for GT gaze mask blobs.")
    parser.add_argument('--use_body_bbox', action='store_true', default=False,
                        help="Use normalized body bounding boxes from the GT CSV instead of head boxes when creating person masks.")
    parser.add_argument('--use_target_insert_for_source', action='store_true', default=False,
                        help="Use gaze target insert indices and representations for both source and target slots.")
    # parser.add_argument('--prompt', type=str, default="You are provided with embeddings representing people or objects in an image." \
    # " Your task is to describe each embedding and where it is looking clearly and succinctly in the following exact format: 'The _ [description of the person] is looking at  _ [description of the object or person]. Repeat the sentence.' " \
    # "Make sure to include 'looking at' in each sentence and that each description accurately captures key visual attributes (e.g., age, gender, clothing, appearance for objects or people; type, color, state for objects) in no more than one short phrase.", help="Input prompt.")
    parser.add_argument('--output_dir', type=str, default=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}", help="Directory to save outputs.")

    # --- Batch Processing Arguments ---
    parser.add_argument('--json_path', type=str, default=None, help="Path to the JSON file with image descriptions for batch processing.")
    parser.add_argument('--base_image_dir', type=str, default=r"D:\Projects\data\gazefollow\train", help="Base directory for images in batch mode.")
    parser.add_argument('--base_mask_dir', type=str, default=r"D:\Projects\data\gazefollow\train_gaze_segmentations\small_masks", help="Base directory for masks in batch mode.")
    parser.add_argument('--limit_items', type=int, default=None, help="Limit the number of items to process in batch mode.")
    # parser.add_argument('--resume_from_dir', type=str, default=r"D:\Projects\LLaVA-NeXT\attention_output\refactored_experiment_20250714_000947", help="Path to previous run directory to resume batch processing from.")
    parser.add_argument('--resume_from_dir', default=False, help="Path to previous run directory to resume batch processing from.")
    parser.add_argument('--use_person_descriptions', action='store_true', help="Use person descriptions from JSON file in prompts.")
    parser.add_argument('--start_from', type=str, default=None, help="Image ID (stem name without extension) to start processing from, skipping all images that come before it in sorted order.")
    parser.add_argument('--skip_first', type=int, default=0, help="Skip the first X images in the dataset (applied after filtering and sorting).")
    parser.add_argument('--sgl_conversation_path', type=str, default='sgl_conversation_data.json', help="Path to SGL conversation data JSON file for filtering already processed images.")
    parser.add_argument('--save_debug_files', action='store_true', default=True, help="Save debug files during generation.")
    parser.add_argument('--save-mask-overlays', action=argparse.BooleanOptionalAction, default=False,
                        help="Save visualization overlays that show person and gaze target masks.")
    parser.add_argument('--mask-overlay-alpha', type=float, default=0.6,
                        help="Alpha blending factor (0-1) for mask overlay visualization.")
    parser.add_argument(
        '--exclude-image-inputs',
        action='store_true',
        dest='exclude_image_inputs',
        default=False,
        help="Exclude image tensors from the model during decoding (default includes them).",
    )

    # --- Bias Sweep Arguments ---
    parser.add_argument('--bias_min', type=float, default=1., help="Minimum bias strength for the sweep.")
    parser.add_argument('--bias_max', type=float, default=3.5, help="Maximum bias strength for the sweep.")
    parser.add_argument('--bias_steps', type=int, default=5, help="Number of steps in the bias sweep.")
    parser.add_argument('--repr_sweep_layers', type=int, nargs='*', default=None,
                        help="List of repr_layer_idx values to sweep (defaults to all decoder layers, e.g., 0-27).")
    parser.add_argument('--repr_sweep_start_idx', type=int, default=None,
                        help="Optional starting layer index; when set and no explicit repr_sweep_layers are provided, sweeps all layers from this index onward.")
    parser.add_argument('--repr_sweep_bias', type=float, default=2.5,
                        help="Bias strength to use during repr layer sweep (defaults to generation bias).")
    parser.add_argument('--repr_combo_sweep', action='store_true',
                        help="When provided, performs a pairwise sweep over capture/inject repr layers instead of a single shared index.")
    parser.add_argument('--repr_capture_layers', type=int, nargs='*', default=None,
                        dest='repr_capture_layers',
                        help="Optional override for capture/person representation layers used during combo sweeps.")
    parser.add_argument('--repr_inject_layers', type=int, nargs='*', default=None,
                        dest='repr_inject_layers',
                        help="Optional override for inject/gaze representation layers used during combo sweeps.")
    parser.add_argument('--token_injection_repr_sweep', action=argparse.BooleanOptionalAction, default=False,
                        help="Run the representation sweep with alternating runs that filter image tokens to the person mask.")
    parser.add_argument('--token_injection_runs_per_combo', type=int, default=2,
                        help="Number of runs per layer combination when image token filtering sweep is enabled (>=1).")

    # --- Gaze Guidance Arguments ---
    parser.add_argument('--use_gaze_guidance', action='store_true', help="Enable gaze-guided token selection.")
    parser.add_argument('--guidance_top_k', type=int, default=10, help="Number of top-k candidates to consider for guidance.")
    parser.add_argument('--similarity_weight', type=float, default=0.7, help="Weight for similarity score in guided selection.")
    parser.add_argument('--probability_weight', type=float, default=0.3, help="Weight for probability score in guided selection.")

    # --- Help and Usage ---
    parser.add_argument('--show_resume_examples', action='store_true', help="Show usage examples for resume functionality and exit.")

    args = parser.parse_args()
    set_gt_annotation_lookup_use_body_bbox(args.use_body_bbox)

    model_path = fix_wsl_paths(args.model_path)
    adapter_path = fix_wsl_paths(args.adapter_path) if args.adapter_path else None
    model_base_path = fix_wsl_paths(args.model_base) if args.model_base else None
    args.gt_gaze_csv_path = fix_wsl_paths(args.gt_gaze_csv_path) if args.gt_gaze_csv_path else None

    if adapter_path and model_base_path is None:
        model_base_path = model_path

    user_specified_output = '--output_dir' in sys.argv
    if user_specified_output:
        args.output_dir = Path(fix_wsl_paths(args.output_dir))
    else:
        args.output_dir = Path(fix_wsl_paths(args.base_image_dir)).parent / 'results' / 'steered_generation' / Path(args.output_dir).name
    # Show resume examples if requested
    if args.show_resume_examples:
        print_resume_usage_examples()
        sys.exit(0)

    # --- Model Loading ---
    MODEL_CONFIG = {
        "model_path": model_path,
        "attn_implementation": args.attn_implementation,
        "load_4bit": args.load_4bit,
        "load_8bit": args.load_8bit,
        "attn_layer_ind": args.attn_layer_ind,
        "model_base": model_base_path,
        "adapter_path": adapter_path
    }
    tokenizer, model, image_processor, max_length = load_model_and_setup(**MODEL_CONFIG)
    total_decoder_layers = getattr(model.config, "num_hidden_layers", None)
    if total_decoder_layers is None or total_decoder_layers <= 0:
        decoder = getattr(model, "model", None)
        total_decoder_layers = len(getattr(decoder, "layers", [])) if decoder is not None else 28
    repr_start_idx = args.repr_sweep_start_idx if args.repr_sweep_start_idx is not None else 0
    if repr_start_idx < 0:
        repr_start_idx = 0
    if repr_start_idx >= total_decoder_layers:
        raise ValueError(
            f"repr_sweep_start_idx ({repr_start_idx}) must be less than total decoder layers ({total_decoder_layers})."
        )
    default_repr_layers = list(range(repr_start_idx, total_decoder_layers))
    if args.repr_sweep_layers is None:
        args.repr_sweep_layers = default_repr_layers

    repr_capture_layers = args.repr_capture_layers if args.repr_capture_layers is not None else args.repr_sweep_layers
    repr_inject_layers = (
        args.repr_inject_layers if args.repr_inject_layers is not None else
        (args.repr_sweep_layers if args.repr_combo_sweep else None)
    )

    # --- Common Generation & Attention Configs ---
    # These can be further customized or exposed as arguments if needed
    generation_config = {
        "bias_strength": 2.5, "max_new_tokens": 128, "temperature": 0.1,
        "do_sample": False, "top_k": 50, "output_hidden_states": True,
        "save_debug_files": args.save_debug_files
    }
    attention_config = {
        "attn_threshold": 0.4, "opening_kernel_size": 5, "min_blob_area": 50,
        "min_avg_attention": 0.2, "show_highest_attn_blob": False, "dilate_kernel_size": 0,
        "create_collage": False,
        "query_indices": {"gaze_source": [-6, -3], # taking the person indices from the end of the prompt, using range format
                          "gaze_target": [-3, 0]  # taking the attention indices from the end of the prompt
                         },
        "layer_idx": args.attn_layer_ind,
        "save_tensors": False
    }
    # if args.repr_layer_idx is not None:
    #     attention_config["repr_layer_idx"] = args.repr_layer_idx

    # New guidance configuration
    guidance_config = {
        "top_k": args.guidance_top_k,
        "similarity_weight": args.similarity_weight,
        "probability_weight": args.probability_weight,
        "enable_after_step": 0,
        "enable_after_keyword": "looking"
    }

    if args.use_target_insert_for_source:
        args.prompt = DEFAULT_TARGET_PROMPT

    if args.mode == 'single':
        print("--- Running Single Experiment ---")
        run_generation_with_attention(
             image_path=args.image_path,
             mask_path=args.mask_path,
             prompt=args.prompt,
             output_dir=args.output_dir,
             model=model,
             tokenizer=tokenizer,
             image_processor=image_processor,
             generation_config=generation_config,
             attention_config=attention_config,
             bias_strength=generation_config.get("bias_strength", 0.0),
             use_gaze_guidance=args.use_gaze_guidance,
             guidance_config=guidance_config,
             save_debug_files=args.save_debug_files,
             use_gt_gaze_csv=args.use_gt_gaze_csv,
             gt_gaze_csv_path=args.gt_gaze_csv_path,
             gt_gaze_mask_radius=args.gt_gaze_mask_radius,
             gt_gaze_mask_radius_ratio=args.gt_gaze_mask_radius_ratio,
             save_mask_overlays=args.save_mask_overlays,
             mask_overlay_alpha=args.mask_overlay_alpha,
             include_image_inputs=not args.exclude_image_inputs,
             use_target_insert_for_source=args.use_target_insert_for_source,
        )

    elif args.mode == 'batch':
        print("--- Running Batch Processing with Bias Sweeps ---")
        if args.resume_from_dir:
            print(f"Resuming from previous run: {args.resume_from_dir}")
        if args.sgl_conversation_path:
            print(f"Using SGL conversation filtering with: {args.sgl_conversation_path}")
        bias_range = np.linspace(args.bias_min, args.bias_max, args.bias_steps)
        process_batch_from_json(
            json_path=args.json_path,
            base_image_dir=args.base_image_dir,
            base_mask_dir=args.base_mask_dir,
            base_output_dir=args.output_dir,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            generation_config=generation_config,
            attention_config=attention_config,
            limit_items=args.limit_items,
            bias_range=bias_range,
            resume_from_dir=args.resume_from_dir,
            prompt_template=args.prompt,
            use_person_descriptions=args.use_person_descriptions,
            start_from=args.start_from,
            skip_first=args.skip_first,
            sgl_conversation_path=args.sgl_conversation_path,
            use_gt_gaze_csv=args.use_gt_gaze_csv,
            gt_gaze_csv_path=args.gt_gaze_csv_path,
            gt_gaze_mask_radius=args.gt_gaze_mask_radius,
            gt_gaze_mask_radius_ratio=args.gt_gaze_mask_radius_ratio,
            save_mask_overlays=args.save_mask_overlays,
            mask_overlay_alpha=args.mask_overlay_alpha,
            include_image_inputs=not args.exclude_image_inputs,
            use_target_insert_for_source=args.use_target_insert_for_source,
        )

    elif args.mode == 'sweep':
        print("--- Running Bias Sweep Experiment ---")
        base_experiment_config = create_experiment_config(
            args.image_path,
            args.mask_path,
            args.prompt,
            args.output_dir,
            generation_config,
            attention_config,
            use_gt_gaze_csv=args.use_gt_gaze_csv,
            gt_gaze_csv_path=args.gt_gaze_csv_path,
            gt_gaze_mask_radius=args.gt_gaze_mask_radius,
            gt_gaze_mask_radius_ratio=args.gt_gaze_mask_radius_ratio,
            save_mask_overlays=args.save_mask_overlays,
            mask_overlay_alpha=args.mask_overlay_alpha,
        )
        base_experiment_config["use_target_insert_for_source"] = args.use_target_insert_for_source
        base_experiment_config["save_debug_files"] = args.save_debug_files
        if args.exclude_image_inputs:
            base_experiment_config["include_image_inputs"] = False
        bias_range = np.linspace(args.bias_min, args.bias_max, args.bias_steps)
        run_bias_sweep_experiment(
            base_experiment_config=base_experiment_config,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            bias_range=bias_range,
            save_summary=True,
            use_gaze_guidance=args.use_gaze_guidance,
            guidance_config=guidance_config,
        )
    elif args.mode == 'repr_sweep':
        if not args.repr_sweep_layers:
            raise ValueError("Mode 'repr_sweep' requires --repr_sweep_layers to specify layer indices.")
        print("--- Running Representation Layer Sweep Experiment ---")
        base_experiment_config = create_experiment_config(
            args.image_path,
            args.mask_path,
            args.prompt,
            args.output_dir,
            generation_config,
            attention_config,
            use_gt_gaze_csv=args.use_gt_gaze_csv,
            gt_gaze_csv_path=args.gt_gaze_csv_path,
            gt_gaze_mask_radius=args.gt_gaze_mask_radius,
            gt_gaze_mask_radius_ratio=args.gt_gaze_mask_radius_ratio,
            save_mask_overlays=args.save_mask_overlays,
            mask_overlay_alpha=args.mask_overlay_alpha,
        )
        base_experiment_config["use_target_insert_for_source"] = args.use_target_insert_for_source
        base_experiment_config["save_debug_files"] = args.save_debug_files
        if args.exclude_image_inputs:
            base_experiment_config["include_image_inputs"] = False
        if args.token_injection_repr_sweep:
            runs_per_combo = max(1, int(args.token_injection_runs_per_combo))
            print(f"Image token filtering sweep enabled (runs per combo = {runs_per_combo}).")
            run_repr_layer_image_token_injection_experiment(
                base_experiment_config=base_experiment_config,
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                repr_layer_indices=args.repr_sweep_layers,
                bias_strength=args.repr_sweep_bias,
                use_gaze_guidance=args.use_gaze_guidance,
                guidance_config=guidance_config,
                repr_target_layer_indices=repr_inject_layers if args.repr_combo_sweep else None,
                enable_pairwise_sweep=args.repr_combo_sweep,
                runs_per_combo=runs_per_combo,
            )
        else:
            run_repr_layer_sweep_experiment(
                base_experiment_config=base_experiment_config,
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                repr_layer_indices=args.repr_sweep_layers,
                bias_strength=args.repr_sweep_bias,
                use_gaze_guidance=args.use_gaze_guidance,
                guidance_config=guidance_config,
                repr_inject_layer_indices=args.repr_inject_layers,
                repr_capture_layer_indices=args.repr_capture_layers,
                enable_pairwise_sweep=args.repr_combo_sweep,
            )
