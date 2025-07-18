import os
import sys
import json
from datetime import datetime
from pathlib import Path
import copy
from typing import Dict, List, Optional, Any, Tuple, Union
...  # removed traceback import

import torch
import numpy as np
from PIL import Image
from transformers import PreTrainedModel, PreTrainedTokenizer
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor

# Add project root to sys.path
try:
    project_root = Path(__file__).resolve().parent.parent
except NameError:
    project_root = Path.cwd().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from generation_utils import (
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
)
from generation_metrics import (
    ConfidenceMetrics, RepetitivityMetrics, TopKCandidateEvaluator,
    generate_next_token_with_evaluation, create_generation_summary,
    analyze_generation_quality, calculate_attention_correlation_from_similarity,
    print_detailed_step_analysis, compare_generation_configs, analyze_top_k_impact
)

"""
todo: it seems that the more we increase the bias strength, the more the model confidence increases in the right direction, for example:
Bias 1.67::
Step 6: 'ceiling' | Confidence: 0.035 | Entropy: 3.536
  Top alternatives:
    1. 'ceiling' (p=0.295)
    2. 'camera' (p=0.260)
    3. 'man' (p=0.245)

Bias 3.89::
Step 6: 'man' | Confidence: 0.232 | Entropy: 2.832
  Top alternatives:
    1. 'man' (p=0.558)
    2. 'camera' (p=0.289)
    3. 'photographer' (p=0.065)

This only works for hidden state-based embedding (and not the raw image embeddings) ->
Maybe, we can try using the image embeddings after the projections

todo: during the end of sweep, we need to find the best bias strength with the highest score
"""

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Main Generation Function with Attention Extraction
def run_generation_with_attention(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    prompt: str,
    output_dir: Union[str, Path],
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    image_processor: "SigLipImageProcessor",
    generation_config: Optional[Dict] = None,
    attention_config: Optional[Dict] = None,
    bias_strength: float = 0.0,
    prev_run_last_hidden_state: Optional[torch.Tensor] = None,
    break_after_first_step: bool = False
) -> Dict[str, Any]:
    """
    Run generation with attention extraction for a given image, mask, and prompt.
    """
    gen_config, attn_config = _prepare_configs(generation_config, attention_config)
    mask_path = fix_wsl_paths(str(mask_path))
    image_path = fix_wsl_paths(str(image_path))
    mask_embedding_path = mask_path.replace("masks.npy", "_target_embeddings.pt")

    # target_mask_embedding = torch.load(mask_embedding_path) if Path(mask_embedding_path).exists() else None
    # if target_mask_embedding is not None:
    #     print(f"Using existing mask embedding from {mask_embedding_path}")

    print(f"Processing image: {image_path}")
    print(f"Using mask: {mask_path}")
    print(f"Prompt: {prompt}")

    output_dir, vis_output_dir_raw, vis_output_dir_processed, tensor_output_dir, collage_output_dir, similarity_output_dir = _setup_output_directories(output_dir)

    image, input_masks, image_tensor, image_sizes, atten_indices, person_mask_indices, input_ids = \
        _prepare_inputs(image_path, mask_path, prompt, image_processor, tokenizer, model)
    
    boost_positions = {'gaze_source': person_mask_indices, 'gaze_target': atten_indices}
    num_patches, grid_size, image_token_start_index_in_llm, image_token_end_index_in_llm = _determine_image_patch_info(model, input_ids)

    print("Starting generation with attention extraction and advanced evaluation...")
    max_new_tokens = gen_config["max_new_tokens"]
    generated_ids = []
    past_key_values = None
    current_input_ids = input_ids
    collected_maps = []
    confidence_tracker = ConfidenceMetrics()
    repetitivity_tracker = RepetitivityMetrics(window_size=10)
    candidate_evaluator = TopKCandidateEvaluator(k=5, tokenizer=tokenizer)
    all_step_metrics = []
    all_attention_maps = []
    eos_token_id = tokenizer.eos_token_id

    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]
    image_embeddings = None
    first_step_hidden_state = None # To store the first step's hidden state for later output
    all_correlation_metrics = []

    for i in range(max_new_tokens):
        with torch.inference_mode():
            model_inputs = {
                "input_ids": current_input_ids, "past_key_values": past_key_values, "use_cache": True,
                "output_attentions": True, "output_hidden_states": True, "atten_ids": None,
                "boost_positions": boost_positions, "bias_strength": bias_strength,
                "query_indices": attn_config.get("query_indices", None),
                "target_mask_embedding": prev_run_last_hidden_state,        # target_mask_embedding
                "base_image_token_inds": [image_token_start_index_in_llm, image_token_start_index_in_llm + num_patches],
                "input_masks": input_masks,
            }
            if i == 0:
                model_inputs.update({"images": image_tensor, "image_sizes": image_sizes, "modalities": ["image"]})

            # Direct enhanced generation without exception handling
            next_token_id, token_text, outputs, evaluation_metrics = generate_next_token_with_evaluation(
                model_inputs, model, tokenizer, gen_config,
                confidence_tracker, repetitivity_tracker, candidate_evaluator, i
            )
            all_step_metrics.append(evaluation_metrics)

            # Call the logging function and break if EOS
            if log_generation_step(i, token_text, evaluation_metrics, next_token_id, eos_token_id):
                break

            generated_ids.append(next_token_id.item())

            if outputs.hidden_states:
                last_hidden_state = outputs.hidden_states[-1].squeeze(0)
                text_embedding = last_hidden_state[-1]
                text_embedding = outputs.hidden_states[attn_config["layer_idx"]].squeeze(0)[-1]
                if image_embeddings is None:    # happens only on the first step
                    # Store the first step (last) hidden state for later output
                    first_step_hidden_state = last_hidden_state
                    first_step_all_layers = outputs.hidden_states
                    image_embeddings = last_hidden_state[image_token_start_index_in_llm : image_token_start_index_in_llm + num_patches]
                    set_layer_hidden_state = outputs.hidden_states[26].squeeze(0)
                    set_layer_image_embeddings = set_layer_hidden_state[image_token_start_index_in_llm : image_token_start_index_in_llm + num_patches]
                    # set_layer_last_token_embedding = set_layer_hidden_state[-1]

                safe_token_text = "".join(c if c.isalnum() else "_" for c in token_text) or f"tokenid_{next_token_id.item()}"
                # if bias_strength > 2.2:
                #     # lets do a grid search for the best localization layer for both text and image embeddings
                #     for layer_idx in range(len(outputs.hidden_states)):
                #         set_layer_hidden_state = first_step_all_layers[layer_idx].squeeze(0)
                #         set_layer_image_embeddings = set_layer_hidden_state[image_token_start_index_in_llm : image_token_start_index_in_llm + num_patches]
                #         set_layer_text_embedding = outputs.hidden_states[attn_config["layer_idx"]].squeeze(0)[-1]
                #         set_layer_text_embedding = outputs.hidden_states[-1].squeeze(0)[-1]
                #         sim_path = similarity_output_dir / f"layer_{layer_idx}" / f"similarity_{i:03d}_{safe_token_text}_layer_{layer_idx}.png"
                #         sim_path.parent.mkdir(parents=True, exist_ok=True)
                #         # Direct embedding similarity visualization
                #         similarity_map = visualize_embedding_similarity(
                #             text_token_embedding=set_layer_text_embedding,
                #             image_token_embeddings=set_layer_image_embeddings,
                #             original_image=image,
                #             grid_size=grid_size,
                #             output_path=sim_path,
                #             # threshold_value=0.3
                #         )
                
                # Direct embedding similarity visualization
                # else:
                sim_path = similarity_output_dir / f"similarity_{i:03d}_{safe_token_text}.png"
                similarity_map = visualize_embedding_similarity(
                    text_token_embedding=text_embedding,
                    image_token_embeddings=set_layer_image_embeddings,
                    original_image=image,
                    grid_size=grid_size,
                    output_path=sim_path,
                    # threshold_value=0.3
                )
                # Compute mask indices only for valid numpy arrays^
                # if isinstance(resized_mask, np.ndarray):
                #     mask_indices = np.where(resized_mask.flatten() > 0)[0]
                # else:
                #     mask_indices = ([], [])
                # if i == 0 and len(mask_indices) > 0:
                #     mask_embeddings = image_embeddings[mask_indices, :].mean(dim=0)
                #     torch.save(mask_embeddings, mask_embedding_path)
                #     print(f"Saved target mask embeddings to {mask_embedding_path}")

            processed_img, attention_map = _extract_and_process_attention(
                outputs, next_token_id, token_text, i, num_patches, grid_size, image_token_start_index_in_llm,
                image, attn_config, vis_output_dir_raw, vis_output_dir_processed, tensor_output_dir
            )
            if attention_map is not None:
                all_attention_maps.append(attention_map)
            # if attn_config["create_collage"] and processed_img is not None:
            #     collected_maps.append((processed_img, token_text))

            current_input_ids = next_token_id.view(1, -1)
            past_key_values = outputs.past_key_values

            if all_attention_maps and len(atten_indices) > 0:
                attention_correlation = calculate_attention_correlation_from_similarity(
                    text_to_image_similarity_matrix=similarity_map, attention_mask=input_masks.get('target_mask', None),
                )
                all_correlation_metrics.append(attention_correlation)
            
            if break_after_first_step:
                print("Breaking after the first step as requested.")
                break

    final_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    generation_summary, quality_analysis = None, None
    if all_step_metrics:
        generation_summary = create_generation_summary(
            confidence_tracker, repetitivity_tracker, candidate_evaluator,
            final_text, all_step_metrics, all_correlation_metrics,
        )
        quality_analysis = analyze_generation_quality(generation_summary)

    # Final summary printing would go here
    print_summary(generation_summary, quality_analysis, final_text, generated_ids, output_dir)

    return {
        "generated_text": final_text,
        "generated_tokens": generated_ids,
        "num_tokens": len(generated_ids),
        "output_directories": {
            "main": str(output_dir),
            "raw_attention": str(vis_output_dir_raw),
            "processed_attention": str(vis_output_dir_processed),
            "embedding_similarity": str(similarity_output_dir),
            "tensors": str(tensor_output_dir),
            "collages": str(collage_output_dir),
        },
        "config_used": {
            "generation": gen_config,
            "attention": attn_config,
        },
        "evaluation_summary": generation_summary,
        "quality_analysis": quality_analysis,
        "attention_correlation": generation_summary.get(
            "average_attention_correlation", {}
        ),
        "step_metrics": all_step_metrics,
        "first_step_hidden_state": first_step_hidden_state,
    }


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
    json_path: Union[str, Path], base_image_dir: Union[str, Path], base_mask_dir: Union[str, Path],
    base_output_dir: Union[str, Path], model, tokenizer, image_processor,
    mask_filename_template: str = "gaze__{}_masks.npy",
    prompt_template: str = "Complete the sentence. The {} is looking at",
    generation_config: Optional[Dict] = None, attention_config: Optional[Dict] = None,
    limit_items: Optional[int] = None, filter_keys: Optional[List[str]] = None,
    bias_range: Optional[np.ndarray] = None,
    resume_from_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Process multiple images from JSON file, performing bias sweeps for each image.
    
    Args:
        bias_range: Array of bias strengths to sweep. Defaults to np.linspace(1.0, 4.0, 4)
        resume_from_dir: Optional path to previous run directory to resume from
    """
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
    
    # Prepare description data and directories
    person_desc_data = prepare_person_desc_data(json_path, filter_keys, limit_items)
    
    # Determine which images still need processing
    if resume_from_dir and all_image_results:
        person_desc_data = get_remaining_images(person_desc_data, all_image_results)
        if not person_desc_data:
            print("All images already processed! Nothing to resume.")
            summary_results = summarize_batch_results(all_image_results)
            final_path = save_image_results(summary_results, base_output_dir, prefix="batch_bias_sweep_results")
            print(f"Final results saved to: {final_path}")
            return all_image_results
    bias_vals = bias_range if bias_range is not None else default_bias_range()
    base_image_dir = fix_wsl_paths(str(base_image_dir))
    base_mask_dir = fix_wsl_paths(str(base_mask_dir))
    base_output_dir = Path(fix_wsl_paths(str(base_output_dir)))

    # Report resume status
    total_images = len(prepare_person_desc_data(json_path, filter_keys, limit_items))
    completed_count = len(all_image_results)
    remaining_count = len(person_desc_data)
    
    if resume_from_dir:
        print(f"RESUME STATUS: {completed_count}/{total_images} images completed, {remaining_count} remaining")
    else:
        print(f"BATCH PROCESSING: {remaining_count} images to process")

    # Iterate through each remaining image entry
    for idx, (image_key, subject_description) in enumerate(person_desc_data.items(), start=1):
        subject_description = False  # todo: remove this to use person (gaze source) description in the prompt
        current_index = completed_count + idx
        print(f"{'='*80}\nProcessing {current_index}/{total_images}: {image_key} (remaining: {idx}/{remaining_count})\n{'='*80}")
        prompt = build_prompt_with_subject_description(subject_description, prompt_template)
        image_path, mask_path = prepare_batch_paths(
            image_key, base_image_dir, base_mask_dir, mask_filename_template
        )
        if image_path is None or mask_path is None:
            print(f"Skipping {image_key} due to missing image or mask.")
            continue
        output_dir = base_output_dir / image_key

        # Run bias sweep experiment
        print(f"Running bias sweep for {image_key} with range {bias_vals}")
        base_config = create_experiment_config(
            image_path, mask_path, prompt, output_dir,
            generation_config, attention_config
        )
        bias_sweep_results = run_bias_sweep_experiment(
            base_experiment_config=base_config,
            model=model, tokenizer=tokenizer,
            image_processor=image_processor, bias_range=bias_vals,
            save_summary=False
        )

        # Analyze and store results
        perf_summary = analyze_bias_sweep_results(
            bias_sweep_results, output_dir, save_summary=False
        )
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
        saved_path = save_image_results(result_entry, output_dir)
        print(f"Results for {image_key} saved to: {saved_path}")

        # Final summary and saving of all results
        summary_results = summarize_batch_results(all_image_results)
        final_path = save_image_results(summary_results, base_output_dir, prefix="batch_bias_sweep_results")
    print(f"\nBATCH BIAS SWEEP PROCESSING COMPLETE. Final results saved to: {final_path}")
    return all_image_results

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

    return False


def run_bias_sweep_experiment(
    base_experiment_config: Dict[str, Any],
    model,
    tokenizer,
    image_processor,
    bias_range: np.ndarray = np.linspace(0., 5., 5),
    save_summary: bool = True
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
            break_after_first_step=True  # Only run the first step to get embeddings
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
            prev_run_last_hidden_state=init_run_results['first_step_hidden_state']  # Use initial run results for embeddings

        )
        results.pop('first_step_hidden_state', None)
        all_results[bias_i] = results
        print(f"Finished experiment for bias {bias_i:.2f}. Results saved to: {results['output_directories']['main']}")

    # Summary and analysis of all bias sweep results
    performance_summary = analyze_bias_sweep_results(all_results, base_output_dir, save_summary=save_summary)

    return all_results

def print_resume_usage_examples():
    """Print usage examples for the resume functionality."""
    print("\n" + "="*80)
    print("RESUME FUNCTIONALITY USAGE EXAMPLES:")
    print("="*80)
    print("1. Resume a previous batch run:")
    print("   python extract_attention_interactive_refactored.py --mode batch \\")
    print("          --resume_from_dir /path/to/previous/run/output \\")
    print("          --json_path /path/to/data.json \\")
    print("          --base_image_dir /path/to/images \\")
    print("          --base_mask_dir /path/to/masks")
    print()
    print("2. The script will automatically:")
    print("   - Load the most recent batch_bias_sweep_results_*.json file")
    print("   - Determine which images were already processed")
    print("   - Continue processing only the remaining images")
    print()
    print("3. Example directory structure for resuming:")
    print("   /previous/run/output/")
    print("   ├── batch_bias_sweep_results_20250714_123456.json  <- Resume from here")
    print("   ├── image1/")
    print("   │   └── bias_sweep_results_*.json")
    print("   └── image2/")
    print("       └── bias_sweep_results_*.json")
    print("="*80 + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run LLaVA-NeXT generation with attention extraction.")
    parser.add_argument('--mode', type=str, default='sweep', choices=['single', 'batch', 'sweep'],
                        help="Execution mode: 'single' for one image, 'batch' for multiple images from a JSON file, 'sweep' for a bias strength sweep.")

    # --- Model Loading Arguments ---
    parser.add_argument('--model_path', type=str, default="lmms-lab/llava-onevision-qwen2-7b-ov-chat", help="Path to the model.")
    parser.add_argument('--attn_implementation', type=str, default="sdpa", help="Attention implementation ('sdpa' or 'eager').")
    parser.add_argument('--load_4bit', action='store_true', help="Load model in 4-bit.")
    parser.add_argument('--load_8bit', action='store_true', help="Load model in 8-bit.")
    parser.add_argument('--attn_layer_ind', type=int, default=23, help="Attention layer index to extract from.")

    # --- Single Experiment Arguments ---
    parser.add_argument('--image_path', type=str, default=r"D:\Projects\data\gazefollow\train\00000000\00000004.jpg", help="Path to the input image.")
    parser.add_argument('--mask_path', type=str, default=r"D:\Projects\data\gazefollow\train_gaze_segmentations\masks\gaze__00000004_masks.npy", help="Path to the attention mask.")
    # parser.add_argument('--prompt', type=str, default="Repeat the sentence and make sure to include the words 'looking at'. The _ is looking at _", help="Input prompt.")\
    parser.add_argument('--prompt', type=str, default="The _ is looking at _ . Repeat the sentence.", help="Input prompt.")
    parser.add_argument('--output_dir', type=str, default=f"attention_output/refactored_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}", help="Directory to save outputs.")

    # --- Batch Processing Arguments ---
    parser.add_argument('--json_path', type=str, default=r"D:\Projects\data\gazefollow\train_results.json", help="Path to the JSON file with image descriptions for batch processing.")
    parser.add_argument('--base_image_dir', type=str, default=r"D:\Projects\data\gazefollow\train", help="Base directory for images in batch mode.")
    parser.add_argument('--base_mask_dir', type=str, default=r"D:\Projects\data\gazefollow\train_gaze_segmentations\masks", help="Base directory for masks in batch mode.")
    parser.add_argument('--limit_items', type=int, default=None, help="Limit the number of items to process in batch mode.")
    # parser.add_argument('--resume_from_dir', type=str, default=r"D:\Projects\LLaVA-NeXT\attention_output\refactored_experiment_20250714_000947", help="Path to previous run directory to resume batch processing from.")
    parser.add_argument('--resume_from_dir', default=False, help="Path to previous run directory to resume batch processing from.")

    # --- Bias Sweep Arguments ---
    parser.add_argument('--bias_min', type=float, default=1.0, help="Minimum bias strength for the sweep.")
    parser.add_argument('--bias_max', type=float, default=5., help="Maximum bias strength for the sweep.")
    parser.add_argument('--bias_steps', type=int, default=4, help="Number of steps in the bias sweep.")
    
    # --- Help and Usage ---
    parser.add_argument('--show_resume_examples', action='store_true', help="Show usage examples for resume functionality and exit.")

    args = parser.parse_args()
    
    # Show resume examples if requested
    if args.show_resume_examples:
        print_resume_usage_examples()
        sys.exit(0)

    # --- Model Loading ---
    MODEL_CONFIG = {
        "model_path": args.model_path,
        "attn_implementation": args.attn_implementation,
        "load_4bit": args.load_4bit,
        "load_8bit": args.load_8bit,
        "attn_layer_ind": args.attn_layer_ind
    }
    tokenizer, model, image_processor, max_length = load_model_and_setup(**MODEL_CONFIG)

    # --- Common Generation & Attention Configs ---
    # These can be further customized or exposed as arguments if needed
    generation_config = {
        "bias_strength": 2.5, "max_new_tokens": 30, "temperature": 0.1,
        "do_sample": False, "top_k": 50, "output_hidden_states": True,
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
            bias_strength=generation_config.get("bias_strength", 0.0)
        )

    elif args.mode == 'batch':
        print("--- Running Batch Processing with Bias Sweeps ---")
        if args.resume_from_dir:
            print(f"Resuming from previous run: {args.resume_from_dir}")
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
        )

    elif args.mode == 'sweep':
        print("--- Running Bias Sweep Experiment ---")
        base_experiment_config = {
            "image_path": args.image_path,
            "mask_path": args.mask_path,
            "prompt": args.prompt,
            "output_dir": args.output_dir,
            "generation_config": generation_config,
            "attention_config": attention_config,
        }
        bias_range = np.linspace(args.bias_min, args.bias_max, args.bias_steps)
        run_bias_sweep_experiment(
            base_experiment_config=base_experiment_config,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            bias_range=bias_range,
            save_summary=True
        )
