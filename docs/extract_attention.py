import os
import sys # Add sys import
import argparse
from datetime import datetime
from pathlib import Path
import socket
import warnings
import copy
from typing import Dict, List, Optional, Any, Tuple, Union
from gazefollow.gazefollow_utils import _pixel_to_token_indices_helper_anyres

# Add project root to sys.path to allow imports like 'from docs. ...' or 'from llava. ...'
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from io import BytesIO # Add BytesIO if handling URLs
import requests # Add requests if handling URLs
import cv2 # Add OpenCV import

from transformers import PreTrainedModel, PreTrainedTokenizer

# from docs.clip_eval import fix_wsl_paths # Revert to original import
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init # Optional: May speed up loading
def fix_wsl_paths(path: str) -> str:
    """Convert Windows paths to WSL paths if necessary."""
    # If path already starts with /mnt/, assume it's correct WSL format
    if path.startswith('/mnt/'):
        return path
    # Otherwise, attempt conversion from Windows format
    path = path.replace("\\", os.sep)
    drive_parts = path.split(os.sep)
    if len(drive_parts) > 0 and len(drive_parts[0]) > 1 and drive_parts[0][1] == ':':
        drive_letter = drive_parts[0][0].lower()
        # Reconstruct path starting from /mnt/<drive_letter>/...
        wsl_path = f'/mnt/{drive_letter}/' + os.sep.join(drive_parts[1:])
        return wsl_path
    else:
        # If it doesn't look like a Windows path either, return original
        return path
# Suppress warnings
# warnings.filterwarnings("ignore")

def setup_environment(hostname: str) -> None:
    """Setup environment variables based on hostname."""
    if 'psychology' in hostname:
        cache_dir = "/home/new_storage/HuggingFace_cache"
        os.environ.update({
            "HF_HOME": cache_dir,
            "TRANSFORMERS_CACHE": cache_dir,
            "HF_DATASETS_CACHE": cache_dir,
            "HF_TOKENIZERS_CACHE": cache_dir
        })

def get_hostname() -> str:
    """Get the current hostname."""
    hostname = socket.gethostname()
    print("Running on:", hostname)
    return hostname


# Modify load_model definition to accept quantization and attn_implementation args
# Keep the original definition accessible if needed
# _original_load_model = load_model
# Modify load_model signature to accept attn_layer_ind
def load_model(attn_implementation="sdpa", load_4bit=False, load_8bit=False, attn_layer_ind: int = -1) -> Tuple[PreTrainedTokenizer, PreTrainedModel, SigLipImageProcessor, int]:
    """Load and initialize the LLaVA model with optional quantization and specified attention layer index."""
    # TODO: Make model path configurable if needed
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov-chat"
    model_name = "llava_qwen"
    device_map = "auto"
    llava_model_args = {"multimodal": True}
    # Pass the attn_layer_ind to the model config
    custom_config = {'attn_layer_ind': attn_layer_ind}
    print(f"Setting custom config for model loading: {custom_config}")

    if load_4bit and load_8bit:
        raise ValueError("Cannot load in both 4-bit and 8-bit mode.")

    # Check if the specified attn_implementation is valid for the environment/model
    # Transformers might handle this internally, but good practice to be aware
    print(f"Attempting to load model with attn_implementation='{attn_implementation}', load_4bit={load_4bit}, load_8bit={load_8bit}")

    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        device_map=device_map,
        attn_implementation=attn_implementation,
        overwrite_config=custom_config,
        **llava_model_args
    )
    return tokenizer, model, image_processor, max_length


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    Load an image from the specified path or URL.
    Raises FileNotFoundError if local path doesn't exist.
    Raises requests.exceptions.RequestException for URL download errors.
    Raises PIL.UnidentifiedImageError (or similar) for invalid image data.
    """
    image_path = str(image_path)
    if image_path.startswith("http") or image_path.startswith("https"):
        # Directly attempt download and open, let exceptions propagate
        response = requests.get(image_path, stream=True)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        # Check existence first
        if not Path(image_path).exists():
             raise FileNotFoundError(f"Image file not found at {image_path}")
        # Directly attempt open, let exceptions propagate (e.g., PIL.UnidentifiedImageError)
        image = Image.open(image_path).convert("RGB")
    return image


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Function for creating a collage of attention maps with token titles
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def visualize_attention_collage(
    attention_maps: List[Tuple[Image.Image, str]],  # List of (attention_map, token_text) pairs
    # overlayed_image: Image.Image,
    output_path: Union[str, Path],
    grid_size: Tuple[int, int] = (5, 5),  # Grid layout (rows, cols)
) -> None:
    """
    Creates a collage of processed attention maps in a grid layout,
    with each map's title set to its corresponding word token.

    Args:
        attention_maps: List of tuples containing (attention_map, token_text)
        overlayed_image: The overlay PIL image
        output_path: Path to save the collage visualization
        grid_size: Tuple of (rows, cols) for the grid layout
        Other parameters: Same as visualize_processed_attention
    """
    rows, cols = grid_size
    total_maps = min(len(attention_maps), rows * cols)

    # Create figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.flatten()  # Convert 2D array of axes to 1D for easier indexing

    # Process each attention map
    for i in range(total_maps):
        attention_map, token_text = attention_maps[i]

        # Get the current axis
        ax = axes[i]


        # Display the blended image in the current subplot
        ax.imshow(attention_map)

        # Set the title to the token text
        ax.set_title(token_text, fontsize=12)

        # Remove axis ticks for cleaner display
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide any remaining empty subplots
    for i in range(total_maps, rows * cols):
        axes[i].axis('off')

    # Adjust layout and save
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved attention collage to: {output_path}")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# New function for processed attention visualization
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def visualize_processed_attention(
    attention_map: np.ndarray,
    original_image: Image.Image,
    output_path: Union[str, Path],
    threshold_value: float = 0.4,
    opening_kernel_size: int = 5,
    min_blob_area: int = 20,
    min_avg_attention: float = 0.2,
    # New parameters
    show_highest_attn_blob: bool = False,
    dilate_kernel_size: int = 0 # 0 or 1 means no dilation
) -> Image.Image:
    """
    Processes an attention map with thresholding, morphological opening,
    blob filtering, and visualizes the result overlaid on the original image.

    Args:
        attention_map: The raw 2D attention map (numpy array, expected range potentially 0-1 or similar).
        original_image: The original PIL image.
        output_path: Path to save the visualization.
        threshold_value: Value below which attention weights are set to 0 in the binary map.
        opening_kernel_size: Size of the kernel for morphological opening.
        min_blob_area: Minimum area (in pixels) for a blob to be kept.
        min_avg_attention: Minimum average attention value (from raw map) within a blob to be kept.
        show_highest_attn_blob: If True, only show the blob with the highest average attention.
        dilate_kernel_size: Kernel size for dilating the highest blob (if shown). 0 or 1 means no dilation.
    """

    # 1. Resize attention map to original image size
    # Ensure attention_map is a numpy array before proceeding
    if not isinstance(attention_map, np.ndarray):
        print(f"Error: attention_map is not a numpy array for {output_path}")
        return # Or raise TypeError

    map_img = Image.fromarray(attention_map.astype(np.float32))
    resized_map_img = map_img.resize(original_image.size, Image.Resampling.LANCZOS)
    resized_map = np.array(resized_map_img) # Fix indentation

    # Keep a copy of the resized raw map for average attention calculation
    raw_resized_map = resized_map.copy()

    binary_map = np.where(raw_resized_map >= threshold_value, 255, 0).astype(np.uint8)
    # Normalize the raw map to 0-1 range *before* thresholding for consistency
    if np.max(raw_resized_map) > np.min(raw_resized_map):
        norm_raw_map = (raw_resized_map - np.min(raw_resized_map)) / (np.max(raw_resized_map) - np.min(raw_resized_map))
    else:
        norm_raw_map = np.zeros_like(raw_resized_map)

    # 2. Thresholding on the *normalized* map
    # binary_map = np.where(norm_raw_map >= threshold_value, 255, 0).astype(np.uint8)

    # 3. Morphological Opening
    # kernel = np.ones((opening_kernel_size, opening_kernel_size), np.uint8) # Square kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel_size, opening_kernel_size)) # Circular kernel
    opened_map = cv2.morphologyEx(binary_map, cv2.MORPH_DILATE, kernel)

    # 4. Blob Filtering & Selection
    contours, _ = cv2.findContours(opened_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_blobs = [] # Store tuples of (avg_attention, contour)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_blob_area:
            # Create a mask for the current contour to calculate avg attention
            contour_mask = np.zeros_like(opened_map)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            masked_attention = norm_raw_map[contour_mask == 255]
            if masked_attention.size > 0:
                avg_attention = np.mean(masked_attention)
                if avg_attention >= min_avg_attention:
                    valid_blobs.append((avg_attention, contour))

    final_mask = np.zeros_like(opened_map)
    selected_contours = []
    if valid_blobs:
        if show_highest_attn_blob:
            # Sort by average attention descending
            valid_blobs.sort(key=lambda x: x[0], reverse=True)
            # Select the highest one
            selected_contours = [valid_blobs[0][1]]
        else:
            # Select all valid contours
            selected_contours = [blob[1] for blob in valid_blobs]

        # Draw selected contours onto the final mask
        cv2.drawContours(final_mask, selected_contours, -1, 255, thickness=cv2.FILLED)
    else: # Add else block for debugging
        print(f"DEBUG: No valid blobs found after filtering for {output_path.name}") # DEBUG print

    # 4.5 Optional Dilation (only if showing highest blob and kernel size > 1)
    if show_highest_attn_blob and dilate_kernel_size > 1 and selected_contours:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
        final_mask = cv2.dilate(final_mask, dilate_kernel, iterations=1)

    # 5. Visualization (using the final_mask)
    print(f"DEBUG: Found {len(selected_contours)} valid contours to visualize for {output_path.name}") # DEBUG print
    # Create the heatmap based on the *normalized raw map* but masked by the *final_mask*
    masked_norm_map = np.where(final_mask > 0, norm_raw_map, 0)
    heatmap_colors = cm.viridis(masked_norm_map)[:, :, :3] # Get RGB colors (0-1 float)
    heatmap_uint8 = (heatmap_colors * 255).astype(np.uint8) # Convert to uint8

    # Convert original PIL image to numpy array (RGB)
    original_np = np.array(original_image.convert('RGB'))

    # Create the blended image, starting with the original
    alpha = 0.5 # Blend factor (same as raw map)
    blended_np = original_np.copy()

    # Find where the final mask is non-zero (where blobs passed filters)
    mask_indices = final_mask > 0

    # Apply weighted blending *only* at the mask indices
    if np.any(mask_indices): # Check if there are any blobs to blend
        blended_np[mask_indices] = cv2.addWeighted(
            original_np[mask_indices],  # Source 1: Original image pixels
            1 - alpha,                  # Weight for source 1
            heatmap_uint8[mask_indices],# Source 2: Heatmap pixels
            alpha,                      # Weight for source 2
            0.0                         # Gamma (offset)
        )

    # Convert back to PIL Image
    overlay_img = Image.fromarray(blended_np)

    # 6. Save
    output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    print(f"Attempting to save processed attention map to: {output_path}") # DEBUG print
    overlay_img.save(output_path)
    return overlay_img

def save_raw_attention_tensor(
    attention_map: torch.Tensor,
    output_path: Union[str, Path],
    token_id: int,
    token_text: str,
    step_idx: int
) -> None:
    """
    Saves the raw attention tensor to a file using torch.save.

    Args:
        attention_map: The raw attention tensor
        output_path: Directory path to save the tensor file
        token_id: ID of the token this attention map corresponds to
        token_text: Text of the token this attention map corresponds to
        step_idx: Generation step index
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create filename with step and token info
    safe_token_text = "".join(c if c.isalnum() else "_" for c in token_text)
    if not safe_token_text:
        safe_token_text = f"tokenid_{token_id}"

    filename = f"attn_tensor_{step_idx:03d}_{safe_token_text}.pt"
    file_path = output_path / filename

    # Save metadata along with tensor
    data = {
        "attention_map": attention_map,  # Move to CPU and detach
        "token_id": token_id,
        "token_text": token_text,
        "step_idx": step_idx,
        "timestamp": str(datetime.now())
    }

    # Save using torch.save
    torch.save(data, file_path)
    print(f"Saved raw attention tensor to {file_path}")

# --- Helper functions for the generation loop ---
def prepare_model_inputs(step_idx, curr_input_ids, past_kvs, attn_inds, img_tens=None, img_szs=None):
    """Prepare inputs for the model's forward pass at the current generation step."""
    inputs = {
        "input_ids": curr_input_ids,
        "past_key_values": past_kvs,
        "use_cache": True,
        "output_attentions": True,
        "output_hidden_states": True,
        "atten_ids": attn_inds
    }

    # Only pass image data in the first step
    if step_idx == 0 and img_tens is not None:
        inputs["images"] = img_tens
        inputs["image_sizes"] = img_szs
        inputs["modalities"] = ["image"]

    return inputs

def extract_attention_map(step_idx, tokenizer, outputs, token_idx, attn_indices, image_start_idx, image_end_idx, grid_sz, n_patches):
    """Extract and prepare the attention map from model outputs."""
    # Get attention matrices from output
    attentions = outputs.attentions
    if attentions is None:
        return None, None

    # Process attention matrices
    selected_attentions = attentions[0][0]  # Get from first batch
    selected_attentions = selected_attentions.squeeze(0)  # Remove batch dimension
    avg_attentions = selected_attentions.mean(dim=0)  # Average across heads

    # Get attention from current token to image patches
    token_attention_to_image = avg_attentions[token_idx, image_start_idx:image_end_idx]

    # Verify shape matches expected number of patches
    if token_attention_to_image.shape[0] != n_patches:
        return None, None

    # Handle padding for square grid
    expected_elements = grid_sz * grid_sz
    if expected_elements > n_patches:
        padding_size = expected_elements - n_patches
        token_attention_to_image = torch.cat([
            token_attention_to_image,
            torch.zeros(padding_size, device=token_attention_to_image.device)
        ])
    elif expected_elements < n_patches:
        return None, None

    # Convert to numpy for visualization
    attention_map = token_attention_to_image.reshape(grid_sz, grid_sz).cpu().numpy()

    # # Extract token text for debugging
    # token_text = None
    # if len(attn_indices) > 0:
    #     txt_ids = torch.argmax(outputs.logits, dim=-1)[0]
    #     if step_idx == 0:  # First token
    #         txt_np = np.array([tokenizer.decode(val).strip() for val in txt_ids])
    #         resulted_description = ",".join(txt_np[attn_indices])
    #         token_text = resulted_description
    #     else:
    #         token_text = tokenizer.decode(txt_ids[0])

    return attention_map        #, token_text

def save_attention_visualizations(attention_map, image, token_id, token_text, step_idx,
                                 vis_raw_dir, vis_processed_dir, tensor_dir,
                                 attn_threshold, opening_kernel_size, min_blob_area,
                                 min_avg_attention, show_highest_blob, dilate_kernel_size,
                                 visualize_overlays=True, create_collage_maps=None):
    """Save attention visualizations and tensors."""
    # Skip if attention map is invalid
    if not isinstance(attention_map, np.ndarray) or attention_map.ndim != 2:
        print(f"[ERROR] Invalid attention map for step {step_idx}, skipping visualization.")
        return None

    # Generate safe token text for filenames
    safe_token_text = "".join(c if c.isalnum() else "_" for c in token_text)
    if not safe_token_text:
        safe_token_text = f"tokenid_{token_id}"

    # Create filenames
    token_filename = f"token_{step_idx:03d}_{safe_token_text}"
    raw_path = vis_raw_dir / f"{token_filename}.png"
    processed_path = vis_processed_dir / f"{token_filename}.png"

    # Save raw visualization if requested
    if visualize_overlays:
        # Convert map to image and resize to match original image
        map_img = Image.fromarray(attention_map.astype(np.float32))
        resized_map = np.array(map_img.resize(image.size, Image.Resampling.LANCZOS))

        # Normalize the map for visualization
        min_val, max_val = np.min(resized_map), np.max(resized_map)
        norm_map = np.zeros_like(resized_map) if max_val <= min_val else \
                  (resized_map - min_val) / (max_val - min_val)

        # Create heatmap overlay
        heatmap = cm.viridis(norm_map)[:, :, :3]
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        img_rgb = image if image.mode == "RGB" else image.convert("RGB")
        overlay_img = Image.blend(img_rgb, Image.fromarray(heatmap_uint8), alpha=0.5)
        overlay_img.save(raw_path)

    # Save raw tensor
    save_raw_attention_tensor(
        attention_map=attention_map,
        output_path=tensor_dir,
        token_id=token_id,
        token_text=token_text,
        step_idx=step_idx
    )

    # Create processed visualization if requested
    overlay_img = None
    if visualize_overlays:
        overlay_img = visualize_processed_attention(
            attention_map=attention_map,
            original_image=image,
            output_path=processed_path,
            threshold_value=attn_threshold,
            opening_kernel_size=opening_kernel_size,
            min_blob_area=min_blob_area,
            min_avg_attention=min_avg_attention,
            show_highest_attn_blob=show_highest_blob,
            dilate_kernel_size=dilate_kernel_size
        )

        # Add to collage collection if requested
        if create_collage_maps is not None and overlay_img is not None:
            create_collage_maps.append((overlay_img, token_text))

    return overlay_img

def process_image_and_prompt(
    image_path: Union[str, Path],
    mask_path: Optional[Union[str, Path]],
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    image_processor: SigLipImageProcessor,
    output_dir: Union[str, Path],
    # heatmap_threshold: float = 0.0004, # Old threshold, remove/replace
    # Add new parameters for processed visualization
    attn_threshold: float = 0.4,
    opening_kernel_size: int = 5,
    min_blob_area: int = 50,
    min_avg_attention: float = 0.2,
    # Add new params
    show_highest_attn_blob: bool = False,
    dilate_kernel_size: int = 0,
    # Add collage parameters
    create_collage: bool = True,
    collage_grid_rows: int = 3,
    collage_grid_cols: int = 4,
    visualize_attn_overlays: bool = True,
    return_masked_tokens: bool = False,
) -> str:
    """
    Process a single image and prompt to generate text and extract attention,
    saving both raw and processed attention maps.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load and process image
    image = load_image(image_path)
    image_tensor = process_images([image], image_processor, model.config) # Returns tensor directly
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    # Prepare conversation input - Use the correct conv template based on model
    model_name = get_model_name_from_path(model.config._name_or_path)
    # Infer conv template if possible, default to qwen_1_5 if unsure
    conv_template = "qwen_1_5" # Default
    if 'qwen' in model_name.lower():
        conv_template = "qwen_1_5"
    elif 'llava' in model_name.lower():
         # Add logic for other llava versions if needed, e.g. "vicuna_v1"
         # Check model.config to be sure
         pass # Keep qwen for llava-onevision-qwen

    print(f"Using conversation template: {conv_template}")

    # Construct the prompt with the image token
    # Check if the prompt already contains the image token
    if DEFAULT_IMAGE_TOKEN not in prompt:
        # Prepend the image token and a newline, standard format
        full_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
    else:
        # Assume user provided the prompt in the correct format
        full_prompt = prompt

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], full_prompt)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_question, tokenizer,
        IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    image_sizes = [image.size]

    # --- Manual Generation Loop for Attention Extraction ---
    print("Starting manual generation loop for attention extraction...")
    max_new_tokens = 256 # Set max generation length
    generated_ids = []
    all_attentions = [] # Store attentions from each step if needed later
    past_key_values = None
    current_input_ids = input_ids # Start with the initial prompt+image input

    # Prepare for attention visualization
    vis_output_dir_raw = output_dir / "attention_maps_raw"
    vis_output_dir_raw.mkdir(exist_ok=True, parents=True)
    vis_output_dir_processed = output_dir / "attention_maps_processed"
    vis_output_dir_processed.mkdir(exist_ok=True, parents=True)

    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, list): # Handle cases where eos_token_id might be a list
        eos_token_id = eos_token_id[0]

    # --- Determine image patch information before the loop ---
    # Get num_patches directly from the vision tower config using hasattr checks
    num_patches = 729 # Default fallback
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
        else:
            print("Warning: Could not determine number of patches automatically from vision tower attributes. Assuming 576 (24x24).")
    else:
        print("Warning: Could not get vision tower from model. Assuming 576 (24x24) patches.")

    print(f"Using number of image patches: {num_patches}")

    # Determine the start index of image features in the LLM input sequence
    # This depends on where IMAGE_TOKEN_INDEX is in the tokenized prompt
    image_token_indices_in_original = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0]
    if len(image_token_indices_in_original) == 0:
         print("Warning: IMAGE_TOKEN_INDEX not found in original input_ids. Assuming image features start at index 1.")
         image_token_start_index_in_llm = 1 # Fallback assumption (after BOS token)
    else:
         image_token_start_index_in_llm = image_token_indices_in_original[0].item()
         print(f"Image token index found at: {image_token_start_index_in_llm}")

    # Calculate the end index based on the determined num_patches
    image_token_end_index_in_llm = image_token_start_index_in_llm + num_patches

    # Determine grid size for visualization (assuming square grid)
    grid_size = int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        print(f"Warning: Number of patches ({num_patches}) is not a perfect square. Visualization might be inaccurate.")
        grid_size = int(np.ceil(np.sqrt(num_patches)))

    # Storage for collecting attention maps for the collage
    collected_maps = []  # Store (attention_map, token_text) pairs

    # todo: remove once done
    # mask_file = r"D:\Projects\data\gazefollow\train_gaze_segmentations\masks\gaze__00000466_masks.npy"
    # mask_file = r"D:\Projects\data\gazefollow\train_gaze_segmentations\masks\gaze__00000318_masks.npy"
    mask_file = fix_wsl_paths(mask_path)
    # with open(mask_file, "rb") as f:
    #     import pickle
    #     mask = pickle.load(f)
    mask = np.load(mask_file, allow_pickle=True).item()['masks'][0]
    # lets extract the coordinates of each pixel in the mask
    mask_coords = np.argwhere(mask)
    print(f"Mask coordinates shape: {mask_coords.shape}")
    vision_tower_config = model.get_vision_tower().config
    atten_indices, base_xy_indices = _pixel_to_token_indices_helper_anyres(mask_coords, image.size, possible_resolutions=model.config.image_grid_pinpoints)

    # atten_indices = atten_indices + np.arange(4750, 4753).tolist()       # todo: remove this line, its just for testing
    # atten_indices = atten_indices + np.arange(4740, 4749).tolist()       # todo: remove this line, its just for testing
    print(f"Attention mask shape: {len(atten_indices)}")

    # --- Generation Loop ---
    for i in range(max_new_tokens):
        with torch.inference_mode():
            # 1. Prepare inputs and run model
            model_inputs = prepare_model_inputs(
                i, current_input_ids, past_key_values, atten_indices,
                image_tensor, image_sizes
            )
            outputs = model(**model_inputs)

            # 2. Get next token (greedy decoding)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)

            # 3. Check for end of sequence
            if next_token_id.item() == eos_token_id:
                print("EOS token generated. Stopping.")
                break

            # 4. Store the generated token
            generated_ids.append(next_token_id.item())

            # 5. Extract attention map for visualization
            current_token_index = outputs.logits.shape[1] - 1
            attention_map = extract_attention_map(i, tokenizer,
                outputs, current_token_index, atten_indices,
                image_token_start_index_in_llm, image_token_end_index_in_llm,
                grid_size, num_patches
            )

            ## Print the debug text like in the original code
            if i == 0:
                txt_ids = torch.argmax(outputs.logits, dim=-1)[0]
                txt_np = np.array([tokenizer.decode(val).strip() for val in txt_ids])
                resulted_description = ",".join(txt_np[atten_indices])
                if return_masked_tokens:
                    return resulted_description
            else:
                resulted_description = tokenizer.decode(next_token_id.item())
            print(f"Resulted words from mask: {resulted_description}")

            # 6. Process and save attention map if valid
            if attention_map is not None:
                token_text = tokenizer.decode([next_token_id.item()]).strip()
                save_attention_visualizations(
                    attention_map, image, next_token_id.item(), token_text, i,
                    vis_output_dir_raw, vis_output_dir_processed, output_dir / "attention_tensors",
                    attn_threshold, opening_kernel_size, min_blob_area, min_avg_attention,
                    show_highest_attn_blob, dilate_kernel_size,
                    visualize_attn_overlays, collected_maps if create_collage else None
                )
            else:
                print(f"Warning: Could not extract valid attention map for step {i}. Skipping visualization.")

            # 7. Prepare for next iteration
            current_input_ids = next_token_id.unsqueeze(-1)
            past_key_values = outputs.past_key_values

            # Update attention indices
            if i == 0:
                initial_token_length = past_key_values[0][0].shape[2]
            input_token_length = past_key_values[0][0].shape[2]
            atten_indices = atten_indices + [input_token_length-1]
            print(f"Total token length: {input_token_length}")
            print(f"Current Attention indices: {atten_indices}")

    # After the generation loop, create collages from the collected maps
    if create_collage and collected_maps:
        print(f"Creating attention collages from {len(collected_maps)} collected maps")

        # Set up directory for collages
        collage_dir = output_dir / "attention_collages"
        collage_dir.mkdir(exist_ok=True, parents=True)

        # Calculate how many collages we need
        maps_per_collage = collage_grid_rows * collage_grid_cols
        num_collages = (len(collected_maps) + maps_per_collage - 1) // maps_per_collage  # Ceiling division

        for collage_idx in range(num_collages):
            start_idx = collage_idx * maps_per_collage
            end_idx = min(start_idx + maps_per_collage, len(collected_maps))

            collage_maps = collected_maps[start_idx:end_idx]
            output_path = collage_dir / f"attention_collage_{collage_idx + 1}.png"

            # Create the collage
            visualize_attention_collage(
                attention_maps=collage_maps,
                output_path=output_path,
                grid_size=(collage_grid_rows, collage_grid_cols),
            )

        print(f"Created {num_collages} attention map collages in {collage_dir}")

    # --- Decode final generated text ---
    text_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print("-" * 30)
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")
    print(f"Generated Response (Manual Loop): {text_output}")
    print("-" * 30)
    print(f"Finished. Raw attention maps saved in: {vis_output_dir_raw}")
    print(f"Finished. Processed attention maps saved in: {vis_output_dir_processed}")
    # --- End Manual Generation Loop ---


def main() -> int:
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Extract LLaVA attention maps for an image and prompt.")
    parser.add_argument("--image-path", required=True, help="Path to the input image file or URL.")
    parser.add_argument("--prompt", required=True, help="Text prompt for the model.")
    parser.add_argument("--output-dir", default="llava_attention_maps", help="Directory to save attention map images.")
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"], help="Attention implementation to use (eager might be needed for attention extraction).")
    # Add argument for attention layer index
    parser.add_argument("--attn-layer-ind", type=int, default=-1, help="Index of the attention layer to extract (-1 for last layer).")
    # Add arguments for attention processing
    parser.add_argument("--attn-threshold", type=float, default=0.3, help="Threshold for binary attention map (0-1).")
    parser.add_argument("--opening-kernel", type=int, default=7, help="Kernel size for morphological opening.") # Reduced default from 50
    parser.add_argument("--min-blob-area", type=int, default=40, help="Minimum pixel area for attention blobs.")
    parser.add_argument("--min-avg-attn", type=float, default=0.15, help="Minimum average attention within a blob (0-1).")
    # Add args for highest blob / dilation
    parser.add_argument("--show-highest-attn-blob", action="store_true", help="Only visualize the blob with the highest average attention.")
    parser.add_argument("--dilate-highest-blob", type=int, default=0, help="Kernel size to dilate the highest attention blob (if shown). 0 or 1 means no dilation.")

    quant_group = parser.add_mutually_exclusive_group()
    quant_group.add_argument("--load-4bit", action="store_true", help="Load model in 4-bit quantization.")
    quant_group.add_argument("--load-8bit", action="store_true", help="Load model in 8-bit quantization.")

    args = parser.parse_args()

    # Setup environment (optional, based on docs/clip_eval.py)
    # hostname = get_hostname()
    # setup_environment(hostname)

    # Optional: Disable init for faster loading
    # disable_torch_init()

    # SAFETY LOGIC: Check for mutually exclusive quantization flags
    if args.load_4bit and args.load_8bit:
        print("[ERROR] Cannot load in both 4-bit and 8-bit mode.")
        return 1

    # Load model - pass quantization and attn_implementation args
    print(f"Loading model with attn_implementation='{args.attn_implementation}', load_4bit={args.load_4bit}, load_8bit={args.load_8bit}...")
    tokenizer = model = image_processor = max_length = None
    model_load_failed = False

    tokenizer, model, image_processor, max_length = load_model(
        attn_implementation=args.attn_implementation,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        attn_layer_ind=args.attn_layer_ind # Pass the argument here
    )
    # Post-checks for model loading
    if model_load_failed or any(x is None for x in [tokenizer, model, image_processor]):
        print("[ERROR] Model loading failed. If attention extraction fails, try running again with --attn-implementation eager")
        return 1 # Indicate error
    model.eval()
    print("Model loaded.")

    # Output directory is already relative to CWD, no need to fix WSL path for it.
    # The fix_wsl_paths was likely intended for input image paths if given in Windows format.
    if 'wsl' in os.uname().release.lower():
        args.image_path = fix_wsl_paths(args.image_path)
        # args.output_dir = fix_wsl_paths(args.output_dir) # Output dir is relative, shouldn't need fixing
    # Use the base output directory specified by user, don't add image stem subfolder here
    output_dir_to_use = Path(args.output_dir)

    # Process the image and prompt
    process_image_and_prompt(
        args.image_path,
        args.prompt,
        model,
        tokenizer,
        image_processor,
        output_dir_to_use, # Use the potentially fixed path
        # Pass new arguments
        attn_threshold=args.attn_threshold,
        opening_kernel_size=args.opening_kernel,
        min_blob_area=args.min_blob_area,
        min_avg_attention=args.min_avg_attn,
        # Pass highest blob / dilation args
        show_highest_attn_blob=args.show_highest_attn_blob,
        dilate_kernel_size=args.dilate_highest_blob
    )
    return None


if __name__ == "__main__":
    import sys
    sys.exit(main()) # Return exit code
