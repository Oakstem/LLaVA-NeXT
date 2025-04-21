import argparse
import subprocess
import os
from pathlib import Path
import sys
import re # Import re for sanitization
from datetime import datetime # Import datetime

# Add project root to sys.path to allow imports like 'from docs. ...'
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from docs.clip_eval import fix_wsl_paths, get_hostname

def main():
    parser = argparse.ArgumentParser(description="Sweep attention extraction across specified layers using srun.")
    parser.add_argument("--image-path", required=True, help="Path to the input image file or URL.")
    parser.add_argument("--prompt", required=True, help="Text prompt for the model.")
    parser.add_argument("--base-output-dir", default="llava_attention_sweep", help="Base directory to save layer-specific attention maps.")
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"], help="Attention implementation for extract_attention.py.")
    quant_group = parser.add_mutually_exclusive_group()
    quant_group.add_argument("--load-4bit", action="store_true", help="Load model in 4-bit quantization.")
    quant_group.add_argument("--load-8bit", action="store_true", help="Load model in 8-bit quantization.")
    parser.add_argument("--start-layer", type=int, default=0, help="Starting layer index for the sweep.")
    parser.add_argument("--end-layer", type=int, default=27, help="Ending layer index for the sweep (inclusive).")
    parser.add_argument("--python-executable", default=sys.executable, help="Path to the python executable to use.")

    # Add arguments for attention processing to pass through
    parser.add_argument("--attn-threshold", type=float, default=0.3, help="Threshold for binary attention map (0-1).")
    parser.add_argument("--opening-kernel", type=int, default=7, help="Kernel size for morphological opening.")
    parser.add_argument("--min-blob-area", type=int, default=40, help="Minimum pixel area for attention blobs.")
    parser.add_argument("--min-avg-attn", type=float, default=0.15, help="Minimum average attention within a blob (0-1).")
    parser.add_argument("--show-highest-attn-blob", action="store_true", help="Only visualize the blob with the highest average attention.")
    parser.add_argument("--dilate-highest-blob", type=int, default=0, help="Kernel size to dilate the highest attention blob (if shown). 0 or 1 means no dilation.")


    args = parser.parse_args()

    # --- Fix WSL Paths if needed ---
    # Let fix_wsl_paths handle the logic of whether conversion is necessary
    if 'wsl' in os.uname().release.lower():
        print("WSL detected. Applying fix_wsl_paths if necessary.")
        original_base_dir = args.base_output_dir
        original_image_path = args.image_path
        args.base_output_dir = fix_wsl_paths(args.base_output_dir)
        args.image_path = fix_wsl_paths(args.image_path)
        if args.base_output_dir != original_base_dir:
             print(f"Fixed base_output_dir: {args.base_output_dir}")
        if args.image_path != original_image_path:
             print(f"Fixed image_path: {args.image_path}")
    # ---

    # --- Create Timestamped Output Directory ---
    # Sanitize the prompt for use in the directory name
    sanitized_prompt = re.sub(r'[^\w\-]+', '_', args.prompt) # Replace non-word chars (allow hyphen) with _
    sanitized_prompt = sanitized_prompt.strip('_') # Remove leading/trailing underscores
    sanitized_prompt = sanitized_prompt[:30] # Limit length

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{timestamp}_{sanitized_prompt}" if sanitized_prompt else timestamp # Combine timestamp and prompt, handle empty prompt

    # Create the main output dir (e.g., llava_attention_sweep)
    main_output_dir = Path(args.base_output_dir)
    # Create the image-specific dir (e.g., llava_attention_sweep/visualization_frame_87965_attn_sweep)
    image_specific_dir = main_output_dir / f"{Path(args.image_path).stem}_attn_sweep"
    # Create the final timestamped run directory within the image-specific dir using the new name
    run_output_path = image_specific_dir / run_dir_name
    run_output_path.mkdir(exist_ok=True, parents=True)

    # script_to_run should be relative to the location of sweep_attention.py
    # or construct the full path relative to the project root
    script_to_run = project_root / "docs" / "extract_attention.py"


    # Use the timestamped run_output_path as the base for layer directories
    print(f"Starting attention sweep from layer {args.start_layer} to {args.end_layer}")
    print(f"Base output directory for this run: {run_output_path}") # Print the timestamped path
    print(f"Using python: {args.python_executable}")
    print(f"Script to run: {script_to_run}")


    for layer_idx in range(args.start_layer, args.end_layer + 1):
        # Create layer directory inside the timestamped run directory
        layer_output_dir = run_output_path / f"layer_{layer_idx:02d}"
        layer_output_dir.mkdir(exist_ok=True, parents=True)
        log_file_path = layer_output_dir / f"extract_layer_{layer_idx:02d}.log"

        print(f"\n--- Processing Layer {layer_idx} ---")
        print(f"Output directory: {layer_output_dir}")
        print(f"Log file: {log_file_path}")

        # Construct the command for extract_attention.py
        cmd_extract = [
            args.python_executable,
            str(script_to_run),
            "--image-path", args.image_path,
            "--prompt", args.prompt,
            "--output-dir", str(layer_output_dir),
            "--attn-implementation", args.attn_implementation,
            "--attn-layer-ind", str(layer_idx),
            # Add new processing args
            "--attn-threshold", str(args.attn_threshold),
            "--opening-kernel", str(args.opening_kernel),
            "--min-blob-area", str(args.min_blob_area),
            "--min-avg-attn", str(args.min_avg_attn),
            "--dilate-highest-blob", str(args.dilate_highest_blob)
        ]
        # Add boolean flags if set
        if args.load_4bit:
            cmd_extract.append("--load-4bit")
        if args.load_8bit:
            cmd_extract.append("--load-8bit")
        if args.show_highest_attn_blob:
            cmd_extract.append("--show-highest-attn-blob")

        # Construct the python command directly (removed srun)
        cmd_to_run = cmd_extract

        print(f"Executing command: {' '.join(cmd_to_run)}")

        try:
            # Execute the command, capture stdout/stderr, redirect stderr to stdout
            process = subprocess.run(
                cmd_to_run, # Use the direct python command
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Redirect stderr to stdout
                text=True,
                check=False # Don't raise exception on non-zero exit code immediately
            )

            # Write combined output to log file
            with open(log_file_path, 'w') as log_file:
                log_file.write(process.stdout)

            # Check return code and print status
            if process.returncode == 0:
                print(f"Layer {layer_idx} processing completed successfully.")
                print(f"Output saved to: {layer_output_dir}")
                print(f"Log saved to: {log_file_path}")
            else:
                print(f"Error processing layer {layer_idx}. Return code: {process.returncode}")
                print(f"Check log file for details: {log_file_path}")
                # Optionally break or continue on error
                # break

        # Removed FileNotFoundError for srun
        except Exception as e:
            print(f"An unexpected error occurred during script execution for layer {layer_idx}: {e}")
            # Log the exception as well
            with open(log_file_path, 'a') as log_file:
                 log_file.write(f"\n\n--- SCRIPT EXECUTION ERROR ---\n{e}")
            # Optionally break or continue
            # break

    print("\n--- Attention sweep finished ---")

if __name__ == "__main__":
    main()
