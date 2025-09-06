#!/home/alonz/llava/bin/python
"""
Parallel processing version of temporal attention focus analysis.

This script optimizes the processing of attention maps by utilizing multiple
workers to run tasks in parallel, significantly reducing processing time for
large datasets.
"""

import os
import sys
import json
import logging
import argparse
import traceback
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from dataclasses import dataclass
from tqdm import tqdm

# Import the core processing functions from the original script
from temporal_attn_focus_refactored import (
    process_single_image,
    gather_caption_and_descriptions,
    save_person_desc_data,
    visualize_individual_attention_maps
)
from attn_utils import fix_wsl_paths


@dataclass
class ProcessingTask:
    """Container for a single processing task."""
    attn_dir: Path
    image_path: Path
    layer: int
    person_threshold: float
    gaze_threshold: float
    kernel_size: int
    sigma: float
    visualize_attention_maps: bool
    
    def to_args_namespace(self) -> argparse.Namespace:
        """Convert to args namespace for compatibility with existing function."""
        return argparse.Namespace(
            result_dir=str(self.attn_dir),
            image_path=str(self.image_path),
            layer=self.layer,
            person_threshold=self.person_threshold,
            gaze_threshold=self.gaze_threshold,
            kernel_size=self.kernel_size,
            sigma=self.sigma,
            visualize_attention_maps=self.visualize_attention_maps
        )


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('temporal_attn_parallel.log')
        ]
    )
    return logging.getLogger(__name__)


def process_task_wrapper(task: ProcessingTask) -> Tuple[bool, str, Optional[str]]:
    """
    Wrapper function to process a single task.
    
    Returns:
        Tuple of (success, task_id, error_message)
    """
    task_id = f"{task.attn_dir.name}"
    
    try:
        # # Check if already processed
        # each_person_attn_maps_dir = task.attn_dir / f'layer_{task.layer}' / 'each_person_attn_maps'
        # if each_person_attn_maps_dir.exists():
        #     return True, task_id, None
            
        # Convert to namespace and process
        args = task.to_args_namespace()
        process_single_image(args)
        return True, task_id, None
        
    except Exception as e:
        error_msg = f"Error processing {task_id}: {str(e)}"
        return False, task_id, error_msg


def discover_processing_tasks(
    input_dir: Path,
    image_path_base: Path,
    pattern: str,
    layer: int,
    person_threshold: float,
    gaze_threshold: float,
    kernel_size: int,
    sigma: float,
    visualize_attention_maps: bool
) -> List[ProcessingTask]:
    """
    Discover all processing tasks from the input directory.
    
    Args:
        input_dir: Directory containing attention map directories
        image_path_base: Base directory for finding original images
        pattern: Glob pattern to find attention map directories
        Other parameters: Processing configuration
        
    Returns:
        List of ProcessingTask objects
    """
    logger = logging.getLogger(__name__)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Find all attention map directories
    attention_dirs = list(input_dir.glob(pattern))
    
    if not attention_dirs:
        raise FileNotFoundError(f"No attention map directories found in {input_dir}")
    
    logger.info(f"Found {len(attention_dirs)} attention map directories")
    
    tasks = []
    missing_images = []
    
    for attn_dir in attention_dirs:
        # Extract image ID and build image path
        image_id = f"{attn_dir.name.split('_')[0]}.jpg"
        image_path = image_path_base / str(int(image_id.split('.')[0]) // 1000).zfill(8) / image_id
        
        if not image_path.exists():
            missing_images.append((attn_dir.name, str(image_path)))
            continue
            
        task = ProcessingTask(
            attn_dir=attn_dir,
            image_path=image_path,
            layer=layer,
            person_threshold=person_threshold,
            gaze_threshold=gaze_threshold,
            kernel_size=kernel_size,
            sigma=sigma,
            visualize_attention_maps=visualize_attention_maps
        )
        tasks.append(task)
    
    if missing_images:
        logger.warning(f"Found {len(missing_images)} directories with missing images")
        if len(missing_images) <= 10:  # Log first few missing images
            for attn_dir, img_path in missing_images[:10]:
                logger.warning(f"Missing image for {attn_dir}: {img_path}")
    
    logger.info(f"Created {len(tasks)} processing tasks")
    return tasks


def process_tasks_parallel(
    tasks: List[ProcessingTask],
    num_workers: Optional[int] = None,
    chunk_size: int = 1
) -> Dict[str, Any]:
    """
    Process tasks in parallel using multiprocessing.
    
    Args:
        tasks: List of processing tasks
        num_workers: Number of worker processes (defaults to CPU count)
        chunk_size: Number of tasks per chunk for load balancing
        
    Returns:
        Dictionary with processing results and statistics
    """
    logger = logging.getLogger(__name__)
    
    if num_workers is None:
        num_workers = min(cpu_count(), len(tasks))
    
    logger.info(f"Processing {len(tasks)} tasks with {num_workers} workers")
    
    results = {
        'successful': [],
        'failed': [],
        'errors': [],
        'total_tasks': len(tasks),
        'num_workers': num_workers
    }
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_task_wrapper, task): task 
            for task in tasks
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(tasks), desc="Processing tasks") as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                success, task_id, error_msg = future.result()
                
                if success:
                    results['successful'].append(task_id)
                else:
                    results['failed'].append(task_id)
                    if error_msg:
                        results['errors'].append(error_msg)
                        logger.error(error_msg)
                
                pbar.update(1)
                pbar.set_postfix({
                    'Success': len(results['successful']),
                    'Failed': len(results['failed'])
                })
    
    # Log final statistics
    success_rate = len(results['successful']) / len(tasks) * 100
    logger.info(f"Processing completed: {len(results['successful'])}/{len(tasks)} "
                f"tasks successful ({success_rate:.1f}%)")
    
    if results['failed']:
        logger.warning(f"Failed tasks: {len(results['failed'])}")
        for task_id in results['failed'][:5]:  # Log first 5 failed tasks
            logger.warning(f"Failed: {task_id}")
    
    return results


def save_processing_results(results: Dict[str, Any], output_path: Path) -> None:
    """Save processing results to JSON file."""
    results_copy = results.copy()
    # Convert any Path objects to strings for JSON serialization
    for key, value in results_copy.items():
        if isinstance(value, list):
            results_copy[key] = [str(item) if isinstance(item, Path) else item for item in value]
    
    with open(output_path, 'w') as f:
        json.dump(results_copy, f, indent=2)


def filter_completed_tasks(tasks: List[ProcessingTask]) -> List[ProcessingTask]:
    """Filter out tasks that have already been completed."""
    logger = logging.getLogger(__name__)
    
    incomplete_tasks = []
    completed_count = 0
    
    for task in tasks:
        each_person_attn_maps_dir = task.attn_dir / f'layer_{task.layer}' / 'each_person_attn_maps'
        if each_person_attn_maps_dir.exists():
            completed_count += 1
        else:
            incomplete_tasks.append(task)
    
    logger.info(f"Found {completed_count} already completed tasks, "
                f"{len(incomplete_tasks)} tasks remaining")
    
    return incomplete_tasks


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Parallel processing for temporal attention focus analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  # Basic usage with automatic worker detection
  python temporal_attn_focus_parallel.py --input-dir /path/to/attention_maps --image-path-base /path/to/images
  
  # With specific number of workers and custom thresholds
  python temporal_attn_focus_parallel.py --input-dir /path/to/attention_maps --image-path-base /path/to/images --num-workers 8 --person-threshold 0.001
  
  # With visualization enabled and results output
  python temporal_attn_focus_parallel.py --input-dir /path/to/attention_maps --image-path-base /path/to/images --visualize-attention-maps --results-file results.json
        """
    )
    
    # Required arguments
    parser.add_argument("--input-dir", required=True,
                        help="Input directory containing multiple attention map directories")
    parser.add_argument("--image-path-base", required=True,
                        help="Base directory for finding original images")
    
    # Processing parameters
    parser.add_argument("--pattern", default="*attn",
                        help="Glob pattern to find attention map directories")
    parser.add_argument("--layer", type=int, default=23,
                        help="Layer to analyze")
    parser.add_argument("--person-threshold", type=float, default=0.0007,
                        help="Threshold for person attention maps")
    parser.add_argument("--gaze-threshold", type=float, default=0.0003,
                        help="Threshold for gaze attention maps")
    parser.add_argument("--kernel-size", type=int, default=3,
                        help="Kernel size for smoothing")
    parser.add_argument("--sigma", type=float, default=2,
                        help="Sigma for Gaussian smoothing")
    parser.add_argument("--visualize-attention-maps", action="store_true",
                        help="Visualize individual attention maps as overlays")
    
    # Parallel processing parameters
    parser.add_argument("--num-workers", type=int,
                        help="Number of worker processes (default: CPU count)")
    parser.add_argument("--chunk-size", type=int, default=1,
                        help="Number of tasks per chunk for load balancing")
    parser.add_argument("--skip-completed", action="store_true",
                        help="Skip tasks that have already been completed")
    
    # Output and logging
    parser.add_argument("--results-file", type=Path,
                        help="Path to save processing results JSON file")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Logging level")
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging(args.log_level)
    
    try:
        # Fix WSL paths if needed
        if 'wsl' in os.uname().release.lower():
            args.input_dir = fix_wsl_paths(args.input_dir)
            args.image_path_base = fix_wsl_paths(args.image_path_base)
        
        input_dir = Path(args.input_dir)
        image_path_base = Path(args.image_path_base)
        
        # Discover all processing tasks
        logger.info("Discovering processing tasks...")
        tasks = discover_processing_tasks(
            input_dir=input_dir,
            image_path_base=image_path_base,
            pattern=args.pattern,
            layer=args.layer,
            person_threshold=args.person_threshold,
            gaze_threshold=args.gaze_threshold,
            kernel_size=args.kernel_size,
            sigma=args.sigma,
            visualize_attention_maps=args.visualize_attention_maps
        )
        
        if not tasks:
            logger.warning("No tasks to process")
            return 0
        
        # Filter out completed tasks if requested
        if args.skip_completed:
            tasks = filter_completed_tasks(tasks)
            
            if not tasks:
                logger.info("All tasks have already been completed")
                return 0
        
        # Process tasks in parallel
        logger.info("Starting parallel processing...")
        results = process_tasks_parallel(
            tasks=tasks,
            num_workers=args.num_workers,
            chunk_size=args.chunk_size
        )
        
        # Save results if requested
        if args.results_file:
            save_processing_results(results, args.results_file)
            logger.info(f"Results saved to {args.results_file}")
        
        # Return non-zero exit code if there were failures
        if results['failed']:
            logger.error(f"Processing completed with {len(results['failed'])} failures")
            return 1
        
        logger.info("All tasks completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
