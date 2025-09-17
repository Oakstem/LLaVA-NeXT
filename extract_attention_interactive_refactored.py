import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
import copy
from typing import Dict, List, Optional, Any, Tuple, Union

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
    break_after_first_step: bool = False,
    use_gaze_guidance: bool = True,
    guidance_config: Optional[Dict[str, Any]] = None,
    beam_search_config: Optional[Dict[str, Any]] = None
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

    Returns:
        Dictionary containing generation results and analysis
    """
    # Prepare configurations and paths
    gen_config, attn_config = _prepare_configs(generation_config, attention_config)
    mask_path = fix_wsl_paths(str(mask_path))
    image_path = fix_wsl_paths(str(image_path))

    print(f"Processing image: {image_path}")
    print(f"Using mask: {mask_path}")
    print(f"Prompt: {prompt}")
    print(f"Gaze guidance enabled: {use_gaze_guidance}")

    # Setup output directories
    output_directories = _setup_output_directories(output_dir)
    (output_dir, vis_output_dir_raw, vis_output_dir_processed, 
     tensor_output_dir, collage_output_dir, similarity_output_dir) = output_directories

    # Prepare inputs
    (image, input_masks, image_tensor, image_sizes, atten_indices,
     person_mask_indices, input_ids) = _prepare_inputs(
        image_path, mask_path, prompt, image_processor, tokenizer, model
    )

    # Setup model configuration
    boost_positions = {'gaze_source': person_mask_indices, 'gaze_target': atten_indices}
    num_patches, grid_size, image_token_start_index_in_llm, image_token_end_index_in_llm = _determine_image_patch_info(model, input_ids)

    # Initialize generation state
    print("Starting generation with attention extraction and advanced evaluation...")
    state = initialize_generation_state(gen_config, tokenizer, input_ids)

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
                "boost_positions": boost_positions,
                "bias_strength": bias_strength,
                "query_indices": attn_config.get("query_indices", None),
                "target_mask_embedding": prev_run_last_hidden_state,
                "base_image_token_inds": [image_token_start_index_in_llm, image_token_start_index_in_llm + num_patches],
                "input_masks": input_masks,
                "apply_only_target_mask": state["apply_only_target_mask"],
                "target_tokens": state["target_tokens"],
            }
            if i == 0:
                model_inputs.update({"images": image_tensor, "image_sizes": image_sizes, "modalities": ["image"]})

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
    
    results = create_generation_results(
        state, tokenizer, output_directories_dict, 
        gen_config, attn_config, guidance_config
    )

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
    start_from: Optional[str] = None
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

    # Report resume status
    total_images = len(full_entries_before_resume)
    completed_count = len(all_image_results)
    remaining_count = len(person_desc_data)
    
    if start_from:
        print(f"START-FROM STATUS: Found {original_image_count} total image(s), processing {remaining_count} image(s) starting from '{start_from}'")
    
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
            print(f"Skipping {image_key}: mask not found at {mask_path}")
            continue

        output_dir = base_output_dir / image_key

        # Run bias sweep experiment
        print(f"Running bias sweep for {image_key} with range {bias_vals}")
        base_config = create_experiment_config(
            str(img_path), str(mask_path), prompt, output_dir,
            generation_config, attention_config
        )
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
    beam_search_config: Optional[Dict[str, Any]] = None
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
            beam_search_config=beam_search_config
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
            guidance_config=guidance_config
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
    print("4. Resume functionality will automatically:")
    print("   - Load the most recent batch_bias_sweep_results_*.json file")
    print("   - Determine which images were already processed")
    print("   - Continue processing only the remaining images")
    print()
    print("5. Start-from functionality will:")
    print("   - Skip all images that come before the specified image ID in sorted order")
    print("   - Use the image stem (filename without extension) for matching")
    print("   - Work with both resume and fresh processing")
    print()
    print("6. Example directory structure for resuming:")
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
    parser.add_argument('--mode', type=str, default='sweep', choices=['single', 'batch', 'sweep'],
                        help="Execution mode: 'single' for one image, 'batch' for multiple images from a JSON file, 'sweep' for a bias strength sweep.")

    # --- Model Loading Arguments ---
    parser.add_argument('--model_path', type=str, default="lmms-lab/llava-onevision-qwen2-7b-ov-chat", help="Path to the model.")
    parser.add_argument('--attn_implementation', type=str, default="sdpa", help="Attention implementation ('sdpa' or 'eager').")
    parser.add_argument('--load_4bit', action='store_true', help="Load model in 4-bit.")
    parser.add_argument('--load_8bit', action='store_true', help="Load model in 8-bit.")
    parser.add_argument('--attn_layer_ind', type=int, default=23, help="Attention layer index to extract from.")

    # --- Single Experiment Arguments ---
    parser.add_argument('--image_path', type=str, default=r"D:\Projects\data\gazefollow\train\00000000\00000032.jpg", help="Path to the input image.")
    parser.add_argument('--mask_path', type=str, default=r"D:\Projects\data\gazefollow\train_gaze_segmentations\small_masks\gaze__00000032_masks.npy", help="Path to the attention mask.")
    # parser.add_argument('--image_path', type=str, default=r"D:\Projects\Annotators\data\llava_results\our_llava_results\109166.png", help="Path to the input image.")
    # parser.add_argument('--mask_path', type=str, default=r"D:\Projects\data\gazefollow\train_gaze_segmentations\manual_masks\gaze__109166_masks.npy", help="Path to the attention mask.")
    parser.add_argument('--prompt', type=str, default="The _ is looking at _ . Where is the _ person looking?", help="Input prompt.")
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

    # --- Bias Sweep Arguments ---
    parser.add_argument('--bias_min', type=float, default=1., help="Minimum bias strength for the sweep.")
    parser.add_argument('--bias_max', type=float, default=6., help="Maximum bias strength for the sweep.")
    parser.add_argument('--bias_steps', type=int, default=3, help="Number of steps in the bias sweep.")

    # --- Gaze Guidance Arguments ---
    parser.add_argument('--use_gaze_guidance', action='store_true', help="Enable gaze-guided token selection.")
    parser.add_argument('--guidance_top_k', type=int, default=10, help="Number of top-k candidates to consider for guidance.")
    parser.add_argument('--similarity_weight', type=float, default=0.7, help="Weight for similarity score in guided selection.")
    parser.add_argument('--probability_weight', type=float, default=0.3, help="Weight for probability score in guided selection.")

    # --- Help and Usage ---
    parser.add_argument('--show_resume_examples', action='store_true', help="Show usage examples for resume functionality and exit.")
    # --- Beam Search Arguments ---
    parser.add_argument('--use_beam_search', action='store_true', help="Enable beam search for generation.")
    parser.add_argument('--num_beams', type=int, default=1, help="Number of beams for beam search.")
    parser.add_argument('--length_penalty', type=float, default=1.0, help="Length penalty for beam search.")
    parser.add_argument('--early_stopping', action='store_true', help="Enable early stopping in beam search.")

    args = parser.parse_args()
    args.output_dir = Path(fix_wsl_paths(args.base_image_dir)).parent / 'results' / 'steered_generation' / Path(args.output_dir).name
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

    # New guidance configuration
    guidance_config = {
        "top_k": args.guidance_top_k,
        "similarity_weight": args.similarity_weight,
        "probability_weight": args.probability_weight,
        "enable_after_step": 0,
        "enable_after_keyword": "looking"
    }

    beam_search_config = {
        'use_beam_search': args.use_beam_search,
        'length_penalty': args.length_penalty,
        'early_stopping': args.early_stopping,
        'num_beams': args.num_beams
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
             bias_strength=generation_config.get("bias_strength", 0.0),
             use_gaze_guidance=args.use_gaze_guidance,
             guidance_config=guidance_config,
             use_beam_search=args.use_beam_search,
             beam_search_config=beam_search_config
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
            use_person_descriptions=args.use_person_descriptions,
            start_from=args.start_from,
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
            save_summary=True,
            use_gaze_guidance=args.use_gaze_guidance,
            guidance_config=guidance_config,
            beam_search_config=beam_search_config
        )
