#!/home/alonz/llava/bin/python
"""
Launcher script for parallel temporal attention focus processing.

This script provides an easy-to-use interface for running the parallel processing
with common configurations and presets.
"""

import os
import sys
import argparse
from pathlib import Path
from multiprocessing import cpu_count
from typing import Dict, Any, Optional

# Add the current directory to the Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from temporal_attn_focus_parallel import main as parallel_main
except ImportError as e:
    print(f"Error importing parallel processing module: {e}")
    print("Make sure temporal_attn_focus_parallel.py is in the same directory")
    sys.exit(1)


class ProcessingConfig:
    """Configuration presets for different processing scenarios."""
    
    # Default configuration
    DEFAULT = {
        'layer': 23,
        'person_threshold': 0.0007,
        'gaze_threshold': 0.0003,
        'kernel_size': 3,
        'sigma': 2,
        'pattern': '*attn',
        'log_level': 'INFO',
        'skip_completed': False,
        'visualize_attention_maps': False
    }
    
    # High precision configuration
    HIGH_PRECISION = {
        **DEFAULT,
        'person_threshold': 0.0005,
        'gaze_threshold': 0.0002,
        'sigma': 1.5
    }
    
    # Fast processing configuration
    FAST = {
        **DEFAULT,
        'person_threshold': 0.001,
        'gaze_threshold': 0.0005,
        'kernel_size': 5,
        'sigma': 3
    }
    
    # Debug configuration with visualization
    DEBUG = {
        **DEFAULT,
        'log_level': 'DEBUG',
        'visualize_attention_maps': True,
        'skip_completed': False
    }


def get_optimal_workers() -> int:
    """Get optimal number of workers based on system resources."""
    cpu_cores = cpu_count()
    
    # Use 75% of available cores for optimal performance
    # while leaving some resources for the system
    optimal_workers = max(1, int(cpu_cores * 0.75))
    
    # Cap at 16 workers to avoid excessive overhead
    return min(optimal_workers, 16)


def validate_paths(input_dir: str, image_path_base: str) -> tuple[Path, Path]:
    """Validate and convert input paths."""
    input_path = Path(input_dir).resolve()
    image_path = Path(image_path_base).resolve()
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_path}")
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image base directory does not exist: {image_path}")
    
    if not image_path.is_dir():
        raise NotADirectoryError(f"Image base path is not a directory: {image_path}")
    
    return input_path, image_path


def print_system_info():
    """Print system information for debugging."""
    print(f"Python version: {sys.version}")
    print(f"CPU cores available: {cpu_count()}")
    print(f"Optimal workers: {get_optimal_workers()}")
    print(f"Current working directory: {Path.cwd()}")
    print()


def create_args_from_config(
    input_dir: str,
    image_path_base: str,
    config: Dict[str, Any],
    num_workers: Optional[int] = None,
    results_file: Optional[str] = None
) -> argparse.Namespace:
    """Create arguments namespace from configuration."""
    args = argparse.Namespace()
    
    # Set required arguments
    args.input_dir = str(input_dir)
    args.image_path_base = str(image_path_base)
    
    # Set configuration values
    for key, value in config.items():
        setattr(args, key, value)
    
    # Set optional arguments
    args.num_workers = num_workers or get_optimal_workers()
    args.chunk_size = 1
    args.results_file = Path(results_file) if results_file else None
    
    return args


def run_with_config(
    input_dir: str,
    image_path_base: str,
    config_name: str = "default",
    num_workers: Optional[int] = None,
    results_file: Optional[str] = None,
    **kwargs
) -> int:
    """
    Run parallel processing with a specific configuration.
    
    Args:
        input_dir: Directory containing attention map directories
        image_path_base: Base directory for finding original images
        config_name: Configuration preset name
        num_workers: Number of worker processes
        results_file: Path to save results JSON
        **kwargs: Additional configuration overrides
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Get base configuration
    config_map = {
        'default': ProcessingConfig.DEFAULT,
        'high_precision': ProcessingConfig.HIGH_PRECISION,
        'fast': ProcessingConfig.FAST,
        'debug': ProcessingConfig.DEBUG
    }
    
    if config_name.lower() not in config_map:
        available = ', '.join(config_map.keys())
        raise ValueError(f"Unknown configuration: {config_name}. Available: {available}")
    
    config = config_map[config_name.lower()].copy()
    
    # Apply any overrides
    config.update(kwargs)
    
    # Validate paths
    input_path, image_path = validate_paths(input_dir, image_path_base)
    
    # Create arguments
    args = create_args_from_config(
        input_path, image_path, config, num_workers, results_file
    )
    
    # Print configuration info
    print(f"Running with '{config_name}' configuration:")
    print(f"  Input directory: {input_path}")
    print(f"  Image base directory: {image_path}")
    print(f"  Number of workers: {args.num_workers}")
    print(f"  Layer: {args.layer}")
    print(f"  Person threshold: {args.person_threshold}")
    print(f"  Gaze threshold: {args.gaze_threshold}")
    print(f"  Pattern: {args.pattern}")
    print(f"  Log level: {args.log_level}")
    print(f"  Skip completed: {args.skip_completed}")
    print(f"  Visualize attention maps: {args.visualize_attention_maps}")
    if args.results_file:
        print(f"  Results file: {args.results_file}")
    print()
    
    # Build command line arguments for the parallel processing script
    cmd_args = [
        '--input-dir', str(input_path),
        '--image-path-base', str(image_path),
        '--layer', str(args.layer),
        '--person-threshold', str(args.person_threshold),
        '--gaze-threshold', str(args.gaze_threshold),
        '--kernel-size', str(args.kernel_size),
        '--sigma', str(args.sigma),
        '--pattern', args.pattern,
        '--log-level', args.log_level,
        '--num-workers', str(args.num_workers),
        '--chunk-size', str(args.chunk_size)
    ]
    
    # Add boolean flags
    if args.skip_completed:
        cmd_args.append('--skip-completed')
    if args.visualize_attention_maps:
        cmd_args.append('--visualize-attention-maps')
    if args.results_file:
        cmd_args.extend(['--results-file', str(args.results_file)])
    
    # Temporarily replace sys.argv for the main function
    original_argv = sys.argv
    try:
        sys.argv = ['temporal_attn_focus_parallel.py'] + cmd_args
        
        # Run the main function
        return parallel_main()
        
    finally:
        sys.argv = original_argv


def parse_launcher_arguments() -> argparse.Namespace:
    """Parse launcher-specific arguments."""
    parser = argparse.ArgumentParser(
        description="Easy launcher for parallel temporal attention focus processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Configuration presets:
  default       - Balanced settings for general use
  high_precision - Higher precision with lower thresholds
  fast          - Faster processing with relaxed thresholds
  debug         - Debug mode with visualization enabled

Examples:
  # Run with default settings
  python launch_parallel.py /path/to/attention_maps /path/to/images
  
  # Run with high precision preset and 8 workers
  python launch_parallel.py /path/to/attention_maps /path/to/images --config high_precision --workers 8
  
  # Run with debug mode and save results
  python launch_parallel.py /path/to/attention_maps /path/to/images --config debug --results results.json
        """
    )
    
    # Required positional arguments (make optional when using utility commands)
    parser.add_argument("input_dir", nargs='?',
                        help="Directory containing attention map directories")
    parser.add_argument("image_path_base", nargs='?',
                        help="Base directory for finding original images")
    
    # Configuration
    parser.add_argument("--config", choices=["default", "high_precision", "fast", "debug"],
                        default="default", help="Configuration preset to use")
    
    # Processing options
    parser.add_argument("--workers", type=int,
                        help="Number of worker processes (default: auto-detect)")
    parser.add_argument("--results", type=str,
                        help="Path to save processing results JSON file")
    
    # Override options
    parser.add_argument("--layer", type=int,
                        help="Override layer to analyze")
    parser.add_argument("--person-threshold", type=float,
                        help="Override person attention threshold")
    parser.add_argument("--gaze-threshold", type=float,
                        help="Override gaze attention threshold")
    parser.add_argument("--pattern", type=str,
                        help="Override glob pattern for finding directories")
    
    # Utility options
    parser.add_argument("--system-info", action="store_true",
                        help="Print system information and exit")
    parser.add_argument("--list-configs", action="store_true",
                        help="List available configuration presets and exit")
    
    return parser.parse_args()


def main() -> int:
    """Main launcher entry point."""
    args = parse_launcher_arguments()
    
    # Handle utility options
    if args.system_info:
        print_system_info()
        return 0
    
    if args.list_configs:
        print("Available configuration presets:")
        print()
        configs = {
            'default': ProcessingConfig.DEFAULT,
            'high_precision': ProcessingConfig.HIGH_PRECISION,
            'fast': ProcessingConfig.FAST,
            'debug': ProcessingConfig.DEBUG
        }
        
        for name, config in configs.items():
            print(f"{name}:")
            for key, value in config.items():
                print(f"  {key}: {value}")
            print()
        
        return 0
    
    # Validate required arguments for processing
    if not args.input_dir or not args.image_path_base:
        print("Error: input_dir and image_path_base are required for processing", file=sys.stderr)
        print("Use --help for usage information", file=sys.stderr)
        return 1
    
    # Prepare override arguments
    overrides = {}
    if args.layer is not None:
        overrides['layer'] = args.layer
    if args.person_threshold is not None:
        overrides['person_threshold'] = args.person_threshold
    if args.gaze_threshold is not None:
        overrides['gaze_threshold'] = args.gaze_threshold
    if args.pattern is not None:
        overrides['pattern'] = args.pattern
    
    try:
        return run_with_config(
            input_dir=args.input_dir,
            image_path_base=args.image_path_base,
            config_name=args.config,
            num_workers=args.workers,
            results_file=args.results,
            **overrides
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
