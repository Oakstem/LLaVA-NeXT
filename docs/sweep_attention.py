import argparse
import subprocess
import os
from pathlib import Path
import sys
import re # Import re for sanitization
from datetime import datetime # Import datetime
import time # Import time for ETA calculation

# Add project root to sys.path to allow imports like 'from docs. ...'
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import functions directly from extract_attention.py
from docs.extract_attention import load_model, process_image_and_prompt, fix_wsl_paths

def find_image_files(path, extensions, recursive=False):
    """
    Find all image files in the given path with specified extensions.

    Args:
        path (str or Path): Directory or file path
        extensions (list): List of valid file extensions (e.g., ['.jpg', '.png'])
        recursive (bool): Whether to search subdirectories recursively

    Returns:
        list: List of Path objects for all valid image files
    """
    path = Path(path)
    image_files = []

    # If path is a file, check if it's a valid image file
    if path.is_file():
        if path.suffix.lower() in extensions:
            image_files.append(path)
        return image_files

    # If path is a directory, find all image files
    if recursive:
        print(f"Recursively searching for images in {path}")
        for ext in extensions:
            image_files.extend(path.glob(f'**/*{ext}'))
    else:
        for ext in extensions:
            image_files.extend(path.glob(f'*{ext}'))

    return sorted(image_files)

def main():
    parser = argparse.ArgumentParser(description="Sweep attention extraction across specified layers using srun.")

    # Input path options
    parser.add_argument("--image-path", required=True, help="Path to the input image file or directory containing images.")
    parser.add_argument("--recursive", action="store_true", help="Recursively process images in subdirectories.")
    parser.add_argument("--image-extensions", default=".jpg,.jpeg,.png,.webp", help="Comma-separated list of image extensions to process (e.g., '.jpg,.png').")

    # Prompt options
    parser.add_argument("--prompt", default=None, help="Text prompt for the model.")

    # Output options
    parser.add_argument("--base-output-dir", default="llava_attention_sweep", help="Base directory to save layer-specific attention maps.")

    # Model options
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"], help="Attention implementation for extract_attention.py.")
    quant_group = parser.add_mutually_exclusive_group()
    quant_group.add_argument("--load-4bit", action="store_true", help="Load model in 4-bit quantization.")
    quant_group.add_argument("--load-8bit", action="store_true", help="Load model in 8-bit quantization.")

    # Layer selection options
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument("--layer", type=int, help="Extract attention for a specific layer only.")
    layer_group.add_argument("--start-layer", type=int, default=0, help="Starting layer index for the sweep.")
    parser.add_argument("--end-layer", type=int, default=27, help="Ending layer index for the sweep (inclusive).")

    parser.add_argument("--python-executable", default=sys.executable, help="Path to the python executable to use.")

    # Add arguments for attention processing to pass through
    parser.add_argument("--attn-threshold", type=float, default=0.001, help="Threshold for binary attention map (0-1).")
    parser.add_argument("--opening-kernel", type=int, default=50, help="Kernel size for morphological opening.")
    parser.add_argument("--min-blob-area", type=int, default=20, help="Minimum pixel area for attention blobs.")
    parser.add_argument("--min-avg-attn", type=float, default=0.15, help="Minimum average attention within a blob (0-1).")
    parser.add_argument("--show-highest-attn-blob", action="store_true", help="Only visualize the blob with the highest average attention.")
    parser.add_argument("--dilate-highest-blob", type=int, default=0, help="Kernel size to dilate the highest attention blob (if shown). 0 or 1 means no dilation.")


    args = parser.parse_args()

    # set default prompt if not provided
    if args.prompt is None:
        args.prompt = ("You are an expert vision assistant. Step 1 – Caption Provide **one concise sentence** that broadly describes the entire scene. Begin the line with: Caption:"
                       " Step 2 - Foreground people & gaze:"
                       " 1. Detect every person in the foreground of the image."
                       " 2. List them **left‑to‑right**. Number sequentially starting at 1. For each person output exactly **one line** in this format: Person {N}: {short description}, looking at {target description | “outside the frame” | “uncertain”}. "
                       "Output format (no extra lines, no prose other than what is specified): "
                       "Caption: {your one‑sentence scene description} Person 1: {short description}, looking at {short description} .\n"
                       " Person 2: {short description}, looking at {short description} .\n"
                       " Additional rules • Keep the phrase **“looking at”** unchanged. • {short description} = ≤ 6 words (e.g., “man in red jacket”). • If no foreground person is detected, write exactly: `No foreground people detected.` • If gaze cannot be determined, use “uncertain”."
                       " Do **not** output your reasoning or any extra text.")
        args.prompt = ("Caption: <one short sentence about the whole image>"
                       "For each large‑enough person in the foreground, from left to right:"
                       "Person <#>: <short description>, looking at <object description | outside of frame | uncertain>"
                       "If no foreground people: No foreground people detected.")
        args.prompt = ("Describe the image and where each person is looking in the following format:"
                       "Caption: <one short sentence about the whole image>"
                       "For each large‑enough person in the foreground, from left to right:"
                       "Person <#>: <short description>, looking at <object | outside of frame | uncertain>"
                       "If no foreground people: No foreground people detected."
                       "Always use 'looking at' to describe gaze direction.")
        args.prompt = ("Complete the sentence. The man is looking at")
        # args.prompt = ("Complete the sentence. The man is wearing a")
        # args.prompt = ("Complete the sentence. This area in the image has")

    # Handle layer selection logic
    if args.layer is not None:
        # If specific layer is provided, use it for both start and end
        start_layer = end_layer = args.layer
    else:
        # Otherwise use the range specified
        start_layer = args.start_layer
        end_layer = args.end_layer

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

    # Parse image extensions
    image_extensions = [ext.strip() if ext.startswith('.') else f'.{ext.strip()}'
                        for ext in args.image_extensions.split(',')]

    # Find all image files to process
    image_paths = find_image_files(args.image_path, image_extensions, args.recursive)

    if not image_paths:
        print(f"Error: No image files found with extensions {image_extensions} in {args.image_path}")
        return

    print(f"Found {len(image_paths)} image(s) to process")

    # --- Create Timestamped Output Directory ---
    # Sanitize the prompt for use in the directory name
    sanitized_prompt = re.sub(r'[^\w\-]+', '_', args.prompt) # Replace non-word chars (allow hyphen) with _
    sanitized_prompt = sanitized_prompt.strip('_') # Remove leading/trailing underscores
    sanitized_prompt = sanitized_prompt[:30] # Limit length

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{timestamp}_{sanitized_prompt}" if sanitized_prompt else timestamp # Combine timestamp and prompt, handle empty prompt

    # Create the main output dir (e.g., llava_attention_sweep)
    main_output_dir = Path(args.base_output_dir)
    main_output_dir.mkdir(exist_ok=True, parents=True)

    # Create the timestamped run directory
    run_output_path = main_output_dir / run_dir_name
    run_output_path.mkdir(exist_ok=True, parents=True)

    print(f"Base output directory for this run: {run_output_path}")

    # Load the model once to reuse across layers and images
    print(f"Loading model with attn_implementation='{args.attn_implementation}'")
    tokenizer, model, image_processor, max_length = load_model(
        attn_implementation=args.attn_implementation,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        attn_layer_ind=start_layer  # Initial layer, will be updated in the loop
    )

    if model is None:
        print("Error: Failed to load model")
        return

    model.eval()
    print("Model loaded successfully")

    # Process all images
    start_time = time.time()
    for img_idx, image_path in enumerate(image_paths):
        # Progress reporting
        percent_complete = (img_idx / len(image_paths)) * 100
        time_elapsed = time.time() - start_time

        # Calculate ETA if more than one image has been processed
        if img_idx > 0:
            time_per_image = time_elapsed / img_idx
            eta_seconds = time_per_image * (len(image_paths) - img_idx)
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
            progress_msg = f"Processing image {img_idx+1}/{len(image_paths)} ({percent_complete:.1f}%) - ETA: {eta_str}"
        else:
            progress_msg = f"Processing image {img_idx+1}/{len(image_paths)} ({percent_complete:.1f}%)"

        print(f"\n{'-'*80}")
        print(progress_msg)
        print(f"Image: {image_path}")

        # Create a specific output directory for this image
        rel_path = Path(image_path).relative_to(Path(args.image_path).parent) if Path(args.image_path).is_dir() else Path(image_path)
        image_output_dir = run_output_path / f"{rel_path.stem}_attn"

        # Process each layer for this image
        for layer_idx in range(start_layer, end_layer + 1):
            # Create layer directory inside the image directory
            layer_output_dir = image_output_dir / f"layer_{layer_idx:02d}"
            layer_output_dir.mkdir(exist_ok=True, parents=True)
            log_file_path = layer_output_dir / f"extract_layer_{layer_idx:02d}.log"

            print(f"  Processing Layer {layer_idx}")

            # Set the current layer index in the model config
            model.config.attn_layer_ind = layer_idx

            # Redirect stdout/stderr to capture logs
            original_stdout = sys.stdout
            original_stderr = sys.stderr

            # with open(log_file_path, 'w') as log_file:
            #     sys.stdout = log_file
            #     sys.stderr = log_file

            # Process the current layer
            process_image_and_prompt(
                str(image_path),
                args.prompt,
                model,
                tokenizer,
                image_processor,
                layer_output_dir,
                attn_threshold=args.attn_threshold,
                opening_kernel_size=args.opening_kernel,
                min_blob_area=args.min_blob_area,
                min_avg_attention=args.min_avg_attn,
                show_highest_attn_blob=args.show_highest_attn_blob,
                dilate_kernel_size=args.dilate_highest_blob
            )

            print(f"Layer {layer_idx} processing completed successfully.")

            sys.stdout = original_stdout
            sys.stderr = original_stderr

    total_time = time.time() - start_time
    print(f"\n{'-'*80}")
    print(f"All {len(image_paths)} images processed in {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    print("--- Attention sweep finished ---")

if __name__ == "__main__":
    main()
