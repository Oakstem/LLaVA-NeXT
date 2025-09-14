#!/usr/bin/env python3
"""
Simplified parallel inference script for single GPU with large memory.
Uses multiprocessing to spawn workers that process image batches in parallel.
"""

import argparse
import json
import multiprocessing as mp
import numpy as np
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from generation_utils import (
    prepare_person_desc_data,
    get_image_files,
    fix_wsl_paths
)


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def worker_function(
    worker_id: int,
    image_batch: List[Tuple[str, str]],  # (image_key, description) pairs
    base_args: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """Worker function that processes a batch of images."""
    
    print(f"Worker {worker_id}: Starting with {len(image_batch)} images")
    print(f"Worker {worker_id}: PID = {os.getpid()}")
    
    try:
        # Import inside worker to avoid CUDA context issues
        print(f"Worker {worker_id}: Importing required modules...")
        from extract_attention_interactive_refactored import (
            load_model_and_setup, 
            run_bias_sweep_experiment, 
            enable_inference_optimizations,
            create_experiment_config
        )
        print(f"Worker {worker_id}: Modules imported successfully")
    except Exception as e:
        print(f"Worker {worker_id}: ERROR importing modules: {e}")
        import traceback
        traceback.print_exc()
        return {
            "worker_id": worker_id,
            "error": f"Import error: {e}",
            "results": {},
            "processed_count": 0,
            "failed_images": []
        }
    
    try:
        # Enable optimizations and set GPU
        print(f"Worker {worker_id}: Setting up optimizations...")
        enable_inference_optimizations()
        
        # File-based synchronization to prevent GPU conflicts
        sync_dir = output_dir / "worker_sync"
        sync_dir.mkdir(parents=True, exist_ok=True)
        worker_ready_file = sync_dir / f"worker_{worker_id}_ready.txt"
        
        # Add a longer delay to stagger worker GPU access more aggressively
        delay = worker_id * 30  # 30 seconds between workers
        print(f"Worker {worker_id}: Waiting {delay} seconds to stagger GPU access...")
        time.sleep(delay)
        
        # Wait for previous worker to finish loading (if any)
        if worker_id > 0:
            prev_worker_file = sync_dir / f"worker_{worker_id - 1}_ready.txt"
            wait_time = 0
            while not prev_worker_file.exists() and wait_time < 300:  # Wait max 5 minutes
                print(f"Worker {worker_id}: Waiting for worker {worker_id - 1} to finish loading... ({wait_time}s)")
                time.sleep(10)
                wait_time += 10
            
            if wait_time >= 300:
                print(f"Worker {worker_id}: WARNING - Previous worker didn't signal ready, proceeding anyway")
        
        # Force garbage collection before GPU access
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        # Load model
        print(f"Worker {worker_id}: Loading model...")
        model_config = {
            "model_path": base_args["model_path"],
            "attn_implementation": base_args["attn_implementation"],
            "load_4bit": base_args["load_4bit"],
            "load_8bit": base_args["load_8bit"],
            "attn_layer_ind": base_args["attn_layer_ind"]
        }
        
        tokenizer, model, image_processor, max_length = load_model_and_setup(**model_config)
        print(f"Worker {worker_id}: Model loaded successfully")
        
        # Additional cleanup after model loading
        torch.cuda.empty_cache()
        print(f"Worker {worker_id}: GPU cache cleared after model loading")
        
        # Quick test to ensure model is working
        print(f"Worker {worker_id}: Testing model readiness...")
        print(f"Worker {worker_id}: Model device: {next(model.parameters()).device}")
        print(f"Worker {worker_id}: Model is ready for inference")
        
        # Signal that this worker is ready
        with open(worker_ready_file, 'w') as f:
            f.write(f"Worker {worker_id} ready at {time.time()}")
        print(f"Worker {worker_id}: Signaled ready for processing")
        
    except Exception as e:
        print(f"Worker {worker_id}: ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return {
            "worker_id": worker_id,
            "error": f"Model loading error: {e}",
            "results": {},
            "processed_count": 0,
            "failed_images": []
        }
    
    # Create worker output directory
    try:
        worker_output_dir = output_dir / f"worker_{worker_id}"
        worker_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Worker {worker_id}: Output directory created at {worker_output_dir}")
    except Exception as e:
        print(f"Worker {worker_id}: ERROR creating output directory: {e}")
        import traceback
        traceback.print_exc()
        return {
            "worker_id": worker_id,
            "error": f"Directory creation error: {e}",
            "results": {},
            "processed_count": 0,
            "failed_images": []
        }
    
    # Process each image in the batch
    worker_results = {}
    failed_images = []
    
    print(f"Worker {worker_id}: Starting image processing loop with {len(image_batch)} images")
    
    for idx, (image_key, subject_description) in enumerate(image_batch):
        print(f"Worker {worker_id}: Processing image {idx+1}/{len(image_batch)} - {image_key}")
        
        # Add a heartbeat every 10 images to confirm worker is alive
        if idx > 0 and idx % 10 == 0:
            print(f"Worker {worker_id}: HEARTBEAT - Processed {idx} images so far")
        
        try:
            # Find image and mask paths
            image_path = find_image_path(image_key, base_args["base_image_dir"])
            mask_path = find_mask_path(image_key, base_args["base_mask_dir"])
            
            if not image_path or not mask_path:
                print(f"Worker {worker_id}: Could not find files for {image_key}")
                print(f"Worker {worker_id}: Image path: {image_path}")
                print(f"Worker {worker_id}: Mask path: {mask_path}")
                failed_images.append(image_key)
                continue
            
            # Build prompt
            if base_args["use_person_descriptions"] and subject_description:
                prompt = f"The {subject_description} is looking at"
            else:
                prompt = base_args["prompt"]
            
            # Create experiment config and run bias sweep
            experiment_config = create_experiment_config(
                image_path=str(image_path),
                mask_path=str(mask_path),
                prompt=prompt,
                output_dir=str(worker_output_dir / image_key),
                generation_config=base_args["generation_config"],
                attention_config=base_args["attention_config"]
            )
            
            image_results = run_bias_sweep_experiment(
                base_experiment_config=experiment_config,
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                bias_range=base_args["bias_range"],
                save_summary=True,
                use_gaze_guidance=base_args.get("use_gaze_guidance", True),
                guidance_config=base_args.get("guidance_config"),
                beam_search_config=base_args.get("beam_search_config")
            )
            
            worker_results[image_key] = {
                "bias_sweep_results": image_results,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "prompt": prompt,
                "subject_description": subject_description
            }
            print(f"Worker {worker_id}: Successfully processed {image_key}")
            
        except Exception as e:
            print(f"Worker {worker_id}: ERROR processing {image_key}: {e}")
            import traceback
            traceback.print_exc()
            failed_images.append(image_key)
    
    # Save worker results
    try:
        results_file = worker_output_dir / "worker_results.json"
        with open(results_file, 'w') as f:
            json.dump(convert_numpy_types(worker_results), f, indent=2)
        print(f"Worker {worker_id}: Results saved to {results_file}")
    except Exception as e:
        print(f"Worker {worker_id}: ERROR saving results: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"Worker {worker_id}: Completed! Processed {len(worker_results)} images, {len(failed_images)} failed")
    
    return {
        "worker_id": worker_id,
        "results": worker_results,
        "processed_count": len(worker_results),
        "failed_images": failed_images,
        "results_file": str(results_file) if 'results_file' in locals() else None,
        "error": None
    }


def find_image_path(image_key: str, base_image_dir: str) -> Path:
    """Find the full path to an image file."""
    base_dir = Path(base_image_dir)
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Try direct path first
    for ext in extensions:
        path = base_dir / f"{image_key}{ext}"
        if path.exists():
            return path
    
    # Search in subdirectories
    for img_file in base_dir.rglob(f"{image_key}.*"):
        if img_file.suffix.lower() in extensions:
            return img_file
    
    return None


def find_mask_path(image_key: str, base_mask_dir: str) -> Path:
    """Find the full path to a mask file."""
    base_dir = Path(base_mask_dir)
    
    # Try standard naming
    mask_path = base_dir / f"gaze__{image_key}_masks.npy"
    if mask_path.exists():
        return mask_path
    
    # Search for alternative naming
    for mask_file in base_dir.rglob(f"*{image_key}*masks.npy"):
        return mask_file
    
    return None


def split_image_list(image_list: List[Tuple[str, str]], num_workers: int) -> List[List[Tuple[str, str]]]:
    """Split image list into roughly equal batches for workers."""
    batch_size = len(image_list) // num_workers
    remainder = len(image_list) % num_workers
    
    batches = []
    start_idx = 0
    
    for i in range(num_workers):
        # Give extra images to first few workers if there's a remainder
        current_batch_size = batch_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_batch_size
        
        batch = image_list[start_idx:end_idx]
        batches.append(batch)
        start_idx = end_idx
    
    return batches


def run_parallel_processing(
    num_workers: int,
    image_data: Dict[str, str],  # image_key -> description
    base_args: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """Run parallel processing using multiple workers on a single GPU."""
    
    print(f"Starting parallel processing with {num_workers} workers")
    print(f"Total images to process: {len(image_data)}")
    
    # Split work among workers
    image_list = list(image_data.items())
    worker_batches = split_image_list(image_list, num_workers)
    
    # Print worker distribution
    for i, batch in enumerate(worker_batches):
        print(f"Worker {i}: {len(batch)} images")
    
    # Start worker processes
    print(f"\nStarting {num_workers} worker processes...")
    start_time = time.time()
    
    try:
        with mp.Pool(processes=num_workers) as pool:
            worker_args = [
                (i, worker_batches[i], base_args, output_dir)
                for i in range(num_workers)
            ]
            
            print(f"Worker arguments prepared:")
            for i, (worker_id, batch, _, _) in enumerate(worker_args):
                print(f"  Worker {worker_id}: {len(batch)} images")
            
            worker_results = pool.starmap(worker_function, worker_args)
    except Exception as e:
        print(f"ERROR in multiprocessing pool: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Multiprocessing error: {e}"}
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nAll workers completed in {total_time:.1f} seconds!")
    
    # Check for worker errors
    for result in worker_results:
        if result.get("error"):
            print(f"Worker {result['worker_id']} had an error: {result['error']}")
    
    return merge_results(worker_results, output_dir, total_time)


def merge_results(worker_results: List[Dict], output_dir: Path, total_time: float) -> Dict[str, Any]:
    """Merge results from all workers."""
    
    print("\nMerging results from all workers...")
    
    merged_results = {}
    total_processed = 0
    all_failed_images = []
    worker_errors = []
    
    for result in worker_results:
        if result.get("error"):
            worker_errors.append({
                "worker_id": result["worker_id"],
                "error": result["error"]
            })
            print(f"Worker {result['worker_id']} had error: {result['error']}")
        else:
            merged_results.update(result.get("results", {}))
            total_processed += result.get("processed_count", 0)
            all_failed_images.extend(result.get("failed_images", []))
    
    # Save merged results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    merged_file = output_dir / f"merged_results_{timestamp}.json"
    
    with open(merged_file, 'w') as f:
        json.dump(convert_numpy_types(merged_results), f, indent=2)
    
    # Create summary
    total_images = total_processed + len(all_failed_images)
    summary = {
        "timestamp": timestamp,
        "total_processed": total_processed,
        "total_failed": len(all_failed_images),
        "total_images": total_images,
        "success_rate": (total_processed / total_images) * 100 if total_images > 0 else 0,
        "total_time_seconds": total_time,
        "avg_time_per_image": total_time / total_processed if total_processed > 0 else 0,
        "failed_images": all_failed_images,
        "worker_errors": worker_errors,
        "worker_summary": [
            {
                "worker_id": r["worker_id"],
                "processed": r.get("processed_count", 0),
                "failed": len(r.get("failed_images", [])),
                "error": r.get("error")
            }
            for r in worker_results
        ]
    }
    
    summary_file = output_dir / f"parallel_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(convert_numpy_types(summary), f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PARALLEL PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Images processed: {total_processed}")
    print(f"Images failed: {len(all_failed_images)}")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Average time per image: {summary['avg_time_per_image']:.1f} seconds")
    print(f"\nResults saved to: {merged_file}")
    print(f"Summary saved to: {summary_file}")
    
    for worker_summary in summary["worker_summary"]:
        status = f" (ERROR: {worker_summary['error']})" if worker_summary['error'] else ""
        print(f"Worker {worker_summary['worker_id']}: {worker_summary['processed']} processed, {worker_summary['failed']} failed{status}")
    
    if worker_errors:
        print(f"\nWorker errors ({len(worker_errors)}):")
        for error in worker_errors:
            print(f"  Worker {error['worker_id']}: {error['error']}")
    
    if all_failed_images:
        print(f"\nFailed images ({len(all_failed_images)}): {all_failed_images[:10]}{'...' if len(all_failed_images) > 10 else ''}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Simplified parallel inference for single GPU")
    
    # Basic arguments
    parser.add_argument('--num_workers', type=int, default=4, 
                       help="Number of worker processes (default: 4)")
    # parser.add_argument('--base_image_dir', type=str, required=True,
                    #    help="Base directory containing images")
    # parser.add_argument('--base_mask_dir', type=str, required=True,
    #                    help="Base directory containing masks")
    parser.add_argument('--output_dir', type=str, 
                       default=f"parallel_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       help="Output directory for results")
    parser.add_argument('--base_image_dir', type=str, default="/mnt/d/Projects/data/gazefollow/train", help="Base directory for images in batch mode.")
    parser.add_argument('--base_mask_dir', type=str, default="/mnt/d/Projects/data/gazefollow/train_gaze_segmentations/small_masks", help="Base directory for masks in batch mode.")

    # Data source arguments
    parser.add_argument('--json_path', type=str, default=None,
                       help="Path to JSON file with image descriptions")
    parser.add_argument('--use_person_descriptions', action='store_true',
                       help="Use person descriptions from JSON in prompts")
    parser.add_argument('--limit_items', type=int, default=None,
                       help="Limit number of images to process (for testing)")
    
    # Model arguments
    parser.add_argument('--model_path', type=str, 
                       default="lmms-lab/llava-onevision-qwen2-7b-ov-chat",
                       help="Path to the model")
    parser.add_argument('--load_4bit', action='store_true',
                       help="Load model in 4-bit")
    parser.add_argument('--attn_layer_ind', type=int, default=23,
                       help="Attention layer index")
    
    # Generation arguments
    parser.add_argument('--prompt', type=str, 
                       default="You are provided with embeddings representing people or objects in an image. Your task is to describe each embedding and where it is looking clearly and succinctly in the following exact format: 'The _ [description of the person] is looking at _ [description of the object or person]. Repeat the sentence.' Make sure to include 'looking at' in each sentence and that each description accurately captures key visual attributes (e.g., age, gender, clothing, appearance for objects or people; type, color, state for objects) in no more than one short phrase.",
                       help="Default prompt template")
    parser.add_argument('--bias_min', type=float, default=1.0,
                       help="Minimum bias strength")
    parser.add_argument('--bias_max', type=float, default=6.0,
                       help="Maximum bias strength")
    parser.add_argument('--bias_steps', type=int, default=3,
                       help="Number of bias steps")
    
    args = parser.parse_args()
    args.output_dir = Path(fix_wsl_paths(args.base_image_dir)).parent / 'results' / 'steered_generation' / Path(args.output_dir).name
    
    # Setup paths
    output_dir = Path(fix_wsl_paths(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image data
    print("Loading image data...")
    if args.use_person_descriptions and args.json_path:
        image_data = prepare_person_desc_data(
            args.json_path, 
            filter_keys=None, 
            limit_items=args.limit_items
        )
        print(f"Loaded {len(image_data)} images with descriptions from JSON")
    else:
        # Get images from directory
        base_image_dir = Path(fix_wsl_paths(args.base_image_dir))
        image_files = get_image_files(
            base_image_dir, 
            filter_keys=None, 
            limit_items=args.limit_items
        )
        image_data = {Path(img_path).stem: "" for img_path in image_files}
        print(f"Found {len(image_data)} images in directory")
    
    if not image_data:
        print("ERROR: No images found to process!")
        return
    
    # Prepare base arguments for workers
    base_args = {
        "model_path": args.model_path,
        "attn_implementation": "sdpa",
        "load_4bit": args.load_4bit,
        "load_8bit": False,
        "attn_layer_ind": args.attn_layer_ind,
        "base_image_dir": args.base_image_dir,
        "base_mask_dir": args.base_mask_dir,
        "prompt": args.prompt,
        "use_person_descriptions": args.use_person_descriptions,
        "bias_range": np.linspace(args.bias_min, args.bias_max, args.bias_steps),
        "generation_config": {
            "bias_strength": 2.5,
            "max_new_tokens": 30,
            "temperature": 0.1,
            "do_sample": False,
            "top_k": 50,
            "output_hidden_states": True,
        },
        "attention_config": {
            "attn_threshold": 0.4,
            "opening_kernel_size": 5,
            "min_blob_area": 50,
            "min_avg_attention": 0.2,
            "show_highest_attn_blob": False,
            "dilate_kernel_size": 0,
            "create_collage": False,
            "query_indices": {
                "gaze_source": [-6, -3],
                "gaze_target": [-3, 0]
            },
            "layer_idx": args.attn_layer_ind,
            "save_tensors": False
        },
        "use_gaze_guidance": True,
        "guidance_config": {
            "top_k": 10,
            "similarity_weight": 0.7,
            "probability_weight": 0.3,
            "enable_after_step": 0,
            "enable_after_keyword": "looking"
        },
        "beam_search_config": {
            'use_beam_search': False,
            'length_penalty': 1.0,
            'early_stopping': False,
            'num_beams': 1
        }
    }
    
    # Run parallel processing
    print(f"Starting parallel processing with {args.num_workers} workers...")
    summary = run_parallel_processing(
        num_workers=args.num_workers,
        image_data=image_data,
        base_args=base_args,
        output_dir=output_dir
    )
    
    print(f"\nParallel processing completed successfully!")
    return summary


if __name__ == "__main__":
    # Set multiprocessing start method for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    main()