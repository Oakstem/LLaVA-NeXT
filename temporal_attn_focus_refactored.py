#!/usr/bin/env python3
import os
import cv2
import glob
import torch
import argparse
import traceback
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from attn_utils import (
    fix_wsl_paths,
    find_persons_and_gaze_targets,
    process_persons_data,
    get_person_attention_maps,
    average_attention_maps,
    smooth_and_aggregate_attention_points,
    visualize_attention_with_centers
)

def visualize_individual_attention_maps(original_img_path, person_attn, gaze_target_attn, output_dir):
    """
    Visualize individual attention maps as overlays on the original image.

    Args:
        original_img_path: Path to the original image
        person_attn: Dictionary mapping person IDs to lists of person attention maps
        gaze_target_attn: Dictionary mapping person IDs to lists of gaze target attention maps
        output_dir: Base directory to save visualizations
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original image
    original_img = cv2.imread(str(original_img_path))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    img_height, img_width = original_img.shape[:2]

    # Create colormaps
    person_cmap = plt.cm.get_cmap('jet')
    gaze_cmap = plt.cm.get_cmap('inferno')

    # Process each person's attention maps
    for person_id, attn_maps in person_attn.items():
        # Create person-specific directory
        person_dir = output_dir / f"person_{person_id}_visualizations"
        person_dir.mkdir(exist_ok=True)

        # Process person attention maps
        if attn_maps:
            for i, attn_map_i in enumerate(attn_maps):
                token_idx = attn_map_i.get('token_id', None)
                token_text = attn_map_i.get('token_text', None)
                attn_map = attn_map_i.get('attention_map', None)
                if attn_map is None or token_idx is None or token_text is None:
                    continue
                # Normalize attention map
                const_factor = 100000
                attn_map = (const_factor * attn_map).astype(np.uint8)
                # Resize attention map to match image size
                attn_map_resized = cv2.resize(attn_map, (img_width, img_height))

                # Apply colormap and create overlay
                colored_map = person_cmap(attn_map_resized)[:, :, :3]  # Remove alpha channel

                # Create a blend of the original image and attention map
                alpha = 0.7  # Transparency factor
                overlay = original_img.copy() / 255.0  # Convert to 0-1 range
                overlay = overlay * (1-alpha) + colored_map * alpha

                # Create figure and add title
                plt.figure(figsize=(10, 8))
                plt.imshow(overlay)
                plt.title(f"Person {person_id} - Token: '{token_text}' (Index: {token_idx})")
                plt.axis('off')

                # Save the visualization
                save_path = person_dir / f"person_{person_id}_token_{token_idx}.png"
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()

        # Process gaze target attention maps
        if person_id in gaze_target_attn and gaze_target_attn[person_id]:
            for i, attn_map_i in enumerate(gaze_target_attn[person_id]):
                token_idx = attn_map_i.get('token_id', None)
                token_text = attn_map_i.get('token_text', None)
                attn_map = attn_map_i.get('attention_map', None)
                if attn_map is None or token_idx is None or token_text is None:
                    continue
                # Normalize attention map
                const_factor = 100000
                attn_map = (const_factor * attn_map).astype(np.uint8)
                # Resize attention map to match image size
                attn_map_resized = cv2.resize(attn_map, (img_width, img_height))

                # Apply colormap and create overlay
                colored_map = gaze_cmap(attn_map_resized)[:, :, :3]  # Remove alpha channel

                # Create a blend of the original image and attention map
                alpha = 0.7  # Transparency factor
                overlay = original_img.copy() / 255.0  # Convert to 0-1 range
                overlay = overlay * (1-alpha) + colored_map * alpha

                # Create figure and add title
                plt.figure(figsize=(10, 8))
                plt.imshow(overlay)
                plt.title(f"Gaze Target for Person {person_id} - Token: '{token_text}' (Index: {token_idx})")
                plt.axis('off')

                # Save the visualization
                save_path = person_dir / f"gaze_target_{person_id}_token_{token_idx}.png"
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()

    print(f"Individual attention map visualizations saved to {output_dir}")

def process_single_image(args):
    """Process a single image's attention maps"""
    result_dir = args.result_dir
    original_img_path = args.image_path
    layer = args.layer

    # Fix paths for WSL if needed
    if 'wsl' in os.uname().release.lower():
        result_dir = fix_wsl_paths(result_dir)
        original_img_path = fix_wsl_paths(original_img_path)

    # Set up paths
    layer_dir = Path(result_dir) / f'layer_{layer}'
    attn_maps_dir = layer_dir / 'attention_tensors'

    if not attn_maps_dir.exists():
        raise FileNotFoundError(f"Attention maps directory not found: {attn_maps_dir}")

    attn_map_files = sorted(attn_maps_dir.glob("*.pt"))

    if not attn_map_files:
        raise FileNotFoundError(f"No attention map files found in {attn_maps_dir}")

    # Load attention maps
    attn_maps = [torch.load(str(attn_map)) for attn_map in attn_map_files]

    # Extract token information
    full_text_arr = [[ind, val['token_text']] for ind, val in enumerate(attn_maps)]

    # Find persons and gaze targets
    persons_data = find_persons_and_gaze_targets(full_text_arr)
    persons_data = process_persons_data(persons_data, full_text_arr)

    # Extract attention maps for persons and gaze targets
    person_attn, gaze_target_attn = get_person_attention_maps(persons_data, attn_maps)

    # Visualize individual attention maps if requested
    if args.visualize_attention_maps:
        vis_output_dir = layer_dir / 'individual_attention_maps'
        visualize_individual_attention_maps(original_img_path, person_attn, gaze_target_attn, vis_output_dir)

    # Average the attention maps
    person_attn_avg, gaze_target_attn_avg = average_attention_maps(person_attn, gaze_target_attn)

    # Process attention maps for each person
    for person_id in person_attn_avg.keys():
        print(f"Processing person {person_id}")

        # Process person attention maps
        p_centers, p_smoothed_map = smooth_and_aggregate_attention_points(
            person_attn_avg[person_id],
            threshold=args.person_threshold,
            sigma=args.sigma
        )

        # Save visualization for person
        p_save_path = layer_dir / f"person_{person_id}_attn_map_smooth_centers.png"
        p_resized_centers, _ = visualize_attention_with_centers(
            original_img_path,
            p_smoothed_map,
            p_centers,
            str(p_save_path),
            threshold=args.person_threshold
        )

        # Save the centers and smoothed map to file
        person_results = {
            'centers': p_resized_centers,
            'smoothed_map': p_smoothed_map
        }
        p_results_path = layer_dir / 'each_person_attn_maps' / f"person_{person_id}" / f"person_{person_id}_attn_map_smooth_centers.pt"
        p_results_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(person_results, str(p_results_path))

        print(f"Person {person_id} has {len(p_centers)} attention centers saved to {p_results_path}")

        # Process gaze attention maps if available
        if person_id in gaze_target_attn_avg:
            g_centers, g_smoothed_map = smooth_and_aggregate_attention_points(
                gaze_target_attn_avg[person_id],
                threshold=args.gaze_threshold,
                kernel_size=args.kernel_size,
                sigma=args.sigma
            )

            # Save visualization for gaze
            g_save_path = layer_dir / f"gaze_target_{person_id}_attn_map_smooth_centers.png"
            g_resized_centers, _ = visualize_attention_with_centers(
                original_img_path,
                g_smoothed_map,
                g_centers,
                str(g_save_path),
                threshold=args.gaze_threshold
            )

            # Save the centers and smoothed map to file
            gaze_results = {
                'centers': g_resized_centers,
                'smoothed_map': g_smoothed_map
            }
            g_results_path = layer_dir / 'each_person_attn_maps' / f"person_{person_id}" / f"gaze_target_{person_id}_attn_map_smooth_centers.pt"
            g_results_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(gaze_results, str(g_results_path))

            print(f"Gaze target for person {person_id} has {len(g_centers)} attention centers saved to {g_results_path}")

def process_directory(args):
    """Process all images in a directory"""
    if 'wsl' in os.uname().release.lower():
        args.input_dir = fix_wsl_paths(args.input_dir)
        args.image_path_base = fix_wsl_paths(args.image_path_base)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find all attention map directories
    attention_dirs = []
    if args.pattern:
        # Use glob pattern to find directories
        pattern = str(input_dir / args.pattern)
        attention_dirs = [Path(p) for p in glob.glob(pattern) if os.path.isdir(p)]
    else:
        # Otherwise, search for layer directories in subdirectories
        for subdir in input_dir.iterdir():
            if subdir.is_dir():
                layer_dirs = list(subdir.glob(f"layer_{args.layer}"))
                if layer_dirs:
                    attention_dirs.append(subdir)

    if not attention_dirs:
        raise FileNotFoundError(f"No attention map directories found in {input_dir}")

    print(f"Found {len(attention_dirs)} attention map directories")

    # lets gather all the image paths
    image_paths = list(Path(args.image_path_base).glob("**/*.jpg"))

    # Process each directory
    for attn_dir in tqdm(attention_dirs, desc="Processing directories"):
        # Skip if the image path doesn't exist
        image_id = f"{attn_dir.name.split('_')[0]}.jpg"
        image_path = [p for p in image_paths if image_id in str(p)]
        if len(image_path) == 0 or not image_path[0].exists():
            print(f"Image not found for {attn_dir}: {image_path}")
            continue
        image_path = image_path[0]
        # Create arguments for processing the single image
        single_args = argparse.Namespace(
            result_dir=str(attn_dir),
            image_path=str(image_path),
            layer=args.layer,
            person_threshold=args.person_threshold,
            gaze_threshold=args.gaze_threshold,
            kernel_size=args.kernel_size,
            sigma=args.sigma,
            visualize_attention_maps=args.visualize_attention_maps
        )

        try:
            process_single_image(single_args)
        except Exception as e:
            print(f"Error processing {attn_dir}: {e}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Process attention maps for temporal attention focus analysis")

    # Add mode selection parameter
    parser.add_argument("--mode", choices=["single", "directory"], required=True,
                        help="Processing mode: single image or directory of images")

    # Single image parameters
    parser.add_argument("--result-dir", help="Directory containing attention maps for single image")
    parser.add_argument("--image-path", help="Path to the original image for single image mode")

    # Directory processing parameters
    parser.add_argument("--input-dir", help="Input directory containing multiple attention map directories")
    parser.add_argument("--pattern", help="Glob pattern to find attention map directories")
    parser.add_argument("--image-path-base", help="Base directory for finding original images")

    # Common parameters
    parser.add_argument("--layer", type=int, default=23, help="Layer to analyze")
    parser.add_argument("--person-threshold", type=float, default=0.0007, help="Threshold for person attention maps")
    parser.add_argument("--gaze-threshold", type=float, default=0.0003, help="Threshold for gaze attention maps")
    parser.add_argument("--kernel-size", type=int, default=3, help="Kernel size for smoothing")
    parser.add_argument("--sigma", type=float, default=2, help="Sigma for Gaussian smoothing")
    parser.add_argument("--visualize-attention-maps", action="store_true", default=False,
                        help="Visualize individual attention maps as overlays on the original image")

    args = parser.parse_args()

    # Validate arguments
    if args.mode == "single":
        if not args.result_dir or not args.image_path:
            parser.error("--result-dir and --image-path are required for single mode")
    elif args.mode == "directory":
        if not args.input_dir or not args.image_path_base:
            parser.error("--input-dir and --image-path-base are required for directory mode")

    return args

def main():
    """Main entry point"""
    args = parse_arguments()

    try:
        if args.mode == "single":
            process_single_image(args)
        elif args.mode == "directory":
            process_directory(args)
    except Exception as e:
        print(f"Error: {e}\n{traceback.format_exc()}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
