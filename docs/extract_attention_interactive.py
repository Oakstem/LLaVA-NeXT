# %% [markdown]
# # Interactive LLaVA Attention Extraction Notebook
#
# This notebook provides an interactive environment for extracting attention maps from LLaVA models.
# It allows you to load a model once and then run multiple experiments with different images, masks, and prompts.
#
# ## Features:
# - Load model once, use multiple times
# - Easy configuration modification
# - Support for different input types (images, masks)
# - Flexible attention visualization parameters
# - Batch processing capabilities

# Todo: finished adding indexes of image and text tokens to the output, lets try to limit the attention on the image tokens only

# %%
# Setup and Imports
import os
import sys
import json
from datetime import datetime
from pathlib import Path
import copy
from typing import Dict, List, Optional, Any, Tuple, Union

# Add project root to sys.path
try:
    project_root = Path(__file__).resolve().parent.parent
except:
    project_root = Path(__name__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from io import BytesIO
import requests

from transformers import PreTrainedModel, PreTrainedTokenizer
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN, IGNORE_INDEX
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from gazefollow.gazefollow_utils import _pixel_to_token_indices_helper_anyres

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %
# Model Loading and Setup
def load_model_and_setup(
    model_path: str = "lmms-lab/llava-onevision-qwen2-7b-ov-chat",
    attn_implementation: str = "sdpa",
    load_4bit: bool = False,
    load_8bit: bool = False,
    attn_layer_ind: int = -1
) -> Tuple[PreTrainedTokenizer, PreTrainedModel, SigLipImageProcessor, int]:
    """
    Load and initialize the LLaVA model with specified configurations.
    This function should be called once at the beginning of your session.
    """
    print("Loading model and components...")

    model_name = "llava_qwen"
    device_map = "auto"
    llava_model_args = {"multimodal": True}
    custom_config = {'attn_layer_ind': attn_layer_ind}

    print(f"Model path: {model_path}")
    print(f"Attention implementation: {attn_implementation}")
    print(f"Custom config: {custom_config}")

    if load_4bit and load_8bit:
        raise ValueError("Cannot load in both 4-bit and 8-bit mode.")

    # Load the model components
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path, None, model_name,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        device_map=device_map,
        attn_implementation=attn_implementation,
        overwrite_config=custom_config,
        **llava_model_args
    )

    model.eval()
    print("✅ Model loaded successfully!")
    print(f"Model device: {model.device}")
    print(f"Max context length: {max_length}")

    return tokenizer, model, image_processor, max_length

# Load model once (modify these parameters as needed)
MODEL_CONFIG = {
    "model_path": "lmms-lab/llava-onevision-qwen2-7b-ov-chat",
    "attn_implementation": "sdpa",  # Change to "eager" if attention extraction fails
    "load_4bit": False,
    "load_8bit": False,
    "attn_layer_ind": 23  # -1 for last layer
}

# Load the model (this will take a few minutes)
tokenizer, model, image_processor, max_length = load_model_and_setup(**MODEL_CONFIG)

# %
# Helper Functions
def fix_wsl_paths(path: str) -> str:
    """Convert Windows paths to WSL paths if necessary."""
    if path.startswith('/mnt/'):
        return path
    path = path.replace("\\", os.sep)
    drive_parts = path.split(os.sep)
    if len(drive_parts) > 0 and len(drive_parts[0]) > 1 and drive_parts[0][1] == ':':
        drive_letter = drive_parts[0][0].lower()
        wsl_path = f'/mnt/{drive_letter}/' + os.sep.join(drive_parts[1:])
        return wsl_path
    else:
        return path

def load_image(image_path: Union[str, Path]) -> Image.Image:
    """Load an image from path or URL."""
    image_path = str(image_path)
    if image_path.startswith("http"):
        response = requests.get(image_path, stream=True)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found at {image_path}")
        image = Image.open(image_path).convert("RGB")
    return image

def load_mask_from_file(mask_path: Union[str, Path]) -> np.ndarray:
    """Load mask from numpy file."""
    mask_path = fix_wsl_paths(str(mask_path))
    if not Path(mask_path).exists():
        raise FileNotFoundError(f"Mask file not found at {mask_path}")

    mask_data = np.load(mask_path, allow_pickle=True)
    if isinstance(mask_data, np.ndarray) and mask_data.dtype == object:
        # Handle pickled dict format
        mask_dict = mask_data.item()
        if 'masks' in mask_dict:
            mask = mask_dict['masks'][0]  # Take first mask
        else:
            mask = mask_dict
    else:
        mask = mask_data

    return mask

def get_attention_indices_from_mask(mask: np.ndarray, image_size: Tuple[int, int], model_config) -> List[int]:
    """Convert mask pixels to token indices."""
    mask_coords = np.argwhere(mask)
    print(f"Mask coordinates shape: {mask_coords.shape}")

    # Get attention indices from pixel coordinates
    atten_indices, base_xy_indices = _pixel_to_token_indices_helper_anyres(
        mask_coords, image_size, possible_resolutions=model_config.image_grid_pinpoints,
        add_system_prompt_tokens= False,
        add_user_prompt_tokens= False,
        user_prompt_range=[1849, 1860]
    )

    return atten_indices

def save_raw_attention_tensor(attention_map: np.ndarray, full_attention, output_path: Path,
                             token_id: int, token_text: str, step_idx: int) -> None:
    """Save raw attention tensor to file."""
    output_path.mkdir(parents=True, exist_ok=True)

    safe_token_text = "".join(c if c.isalnum() else "_" for c in token_text)
    if not safe_token_text:
        safe_token_text = f"tokenid_{token_id}"

    filename = f"attn_tensor_{step_idx:03d}_{safe_token_text}.pt"
    file_path = output_path / filename

    data = {
        "attention_map": torch.from_numpy(attention_map) if isinstance(attention_map, np.ndarray) else attention_map,
        "full_attention": full_attention,
        "token_id": token_id,
        "token_text": token_text,
        "step_idx": step_idx,
        "timestamp": str(datetime.now())
    }

    torch.save(data, file_path)
    # print(f"Saved raw attention tensor to {file_path}")

def visualize_processed_attention(
    attention_map: np.ndarray,
    original_image: Image.Image,
    output_path: Union[str, Path],
    threshold_value: float = 0.4,
    opening_kernel_size: int = 5,
    min_blob_area: int = 20,
    min_avg_attention: float = 0.2,
    show_highest_attn_blob: bool = False,
    dilate_kernel_size: int = 0
) -> Image.Image:
    """Process and visualize attention map with filtering."""

    if not isinstance(attention_map, np.ndarray) or attention_map.ndim != 2:
        print(f"Error: Invalid attention map for {output_path}")
        return None

    # Resize attention map to match image size
    map_img = Image.fromarray(attention_map.astype(np.float32))
    resized_map_img = map_img.resize(original_image.size, Image.Resampling.LANCZOS)
    resized_map = np.array(resized_map_img)

    # Normalize the raw map
    if np.max(resized_map) > np.min(resized_map):
        norm_raw_map = (resized_map - np.min(resized_map)) / (np.max(resized_map) - np.min(resized_map))
    else:
        norm_raw_map = np.zeros_like(resized_map)

    # Binary thresholding
    binary_map = np.where(norm_raw_map >= threshold_value, 255, 0).astype(np.uint8)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel_size, opening_kernel_size))
    opened_map = cv2.morphologyEx(binary_map, cv2.MORPH_DILATE, kernel)

    # Blob filtering
    contours, _ = cv2.findContours(opened_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_blobs = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_blob_area:
            contour_mask = np.zeros_like(opened_map)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            masked_attention = norm_raw_map[contour_mask == 255]
            if masked_attention.size > 0:
                avg_attention = np.mean(masked_attention)
                if avg_attention >= min_avg_attention:
                    valid_blobs.append((avg_attention, contour))

    # Select blobs
    final_mask = np.zeros_like(opened_map)
    if valid_blobs:
        if show_highest_attn_blob:
            valid_blobs.sort(key=lambda x: x[0], reverse=True)
            selected_contours = [valid_blobs[0][1]]
        else:
            selected_contours = [blob[1] for blob in valid_blobs]

        cv2.drawContours(final_mask, selected_contours, -1, 255, thickness=cv2.FILLED)

        # Optional dilation
        if show_highest_attn_blob and dilate_kernel_size > 1:
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
            final_mask = cv2.dilate(final_mask, dilate_kernel, iterations=1)

    # Create visualization
    masked_norm_map = np.where(final_mask > 0, norm_raw_map, 0)
    heatmap_colors = cm.viridis(masked_norm_map)[:, :, :3]
    heatmap_uint8 = (heatmap_colors * 255).astype(np.uint8)

    original_np = np.array(original_image.convert('RGB'))
    blended_np = original_np.copy()

    mask_indices = final_mask > 0
    if np.any(mask_indices):
        alpha = 0.5
        blended_np[mask_indices] = cv2.addWeighted(
            original_np[mask_indices], 1 - alpha,
            heatmap_uint8[mask_indices], alpha, 0.0
        )

    overlay_img = Image.fromarray(blended_np)

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_img.save(output_path)
        # print(f"Saved processed attention to: {output_path}")

    return overlay_img

def visualize_attention_collage(
    attention_maps: List[Tuple[Image.Image, str]],
    output_path: Union[str, Path],
    grid_size: Tuple[int, int] = (3, 4)
) -> None:
    """Create a collage of attention maps."""
    rows, cols = grid_size
    total_maps = min(len(attention_maps), rows * cols)

    if total_maps == 0:
        print("No attention maps to create collage")
        return

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    if total_maps == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(total_maps):
        attention_map, token_text = attention_maps[i]
        ax = axes[i]
        ax.imshow(attention_map)
        ax.set_title(token_text, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty subplots
    for i in range(total_maps, rows * cols):
        axes[i].axis('off')

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved attention collage to: {output_path}")

# %%
# JSON Data Loading and Prompt Building Functions

def load_train_results_json(json_path: Union[str, Path]) -> Dict[str, Any]:
    """Load the train_results.json file."""
    json_path = fix_wsl_paths(str(json_path))
    if not Path(json_path).exists():
        raise FileNotFoundError(f"JSON file not found at {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded train_results.json with {len(data)} entries")
    return data

def build_prompt_with_subject_description(subject_description: str, base_prompt_template: str = "Complete the sentence. The {} is looking at") -> str:
    """Build a prompt using the subject description."""
    if not subject_description:
        # Fallback to generic prompt
        return "Complete the sentence. The person is looking at"

    # Clean the subject description
    subject_description = subject_description.strip()

    # Build the prompt
    prompt = base_prompt_template.format(subject_description)
    return prompt

def create_image_key_from_path(image_path: Union[str, Path]) -> str:
    """Create the JSON key from image path (e.g., '00000000/00000021.jpg' -> '00000000_00000021')."""
    image_path = Path(image_path)

    # Extract the folder and filename
    folder_name = image_path.parent.name
    filename = image_path.stem  # filename without extension

    # Create the key
    key = f"{folder_name}_{filename}"
    return key

def save_results_to_json(results: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Save the generation results to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {output_path}")

# %%
# Helper Functions for Generation Process

def _prepare_configs(generation_config: Optional[Dict], attention_config: Optional[Dict]) -> Tuple[Dict, Dict]:
    """Prepare and merge generation and attention configurations with defaults."""
    default_generation_config = {
        "max_new_tokens": 50,
        "temperature": 0.0,
        "do_sample": False
    }

    default_attention_config = {
        "attn_threshold": 0.4,
        "opening_kernel_size": 5,
        "min_blob_area": 50,
        "min_avg_attention": 0.2,
        "show_highest_attn_blob": False,
        "dilate_kernel_size": 0,
        "create_collage": True,
        "collage_grid_rows": 3,
        "collage_grid_cols": 4,
        "visualize_attn_overlays": True,
        "save_tensors": True
    }

    gen_config = {**default_generation_config, **(generation_config or {})}
    attn_config = {**default_attention_config, **(attention_config or {})}

    return gen_config, attn_config

def _setup_output_directories(output_dir: Union[str, Path]) -> Tuple[Path, Path, Path, Path, Path]:
    """Create and return all output directory paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    vis_output_dir_raw = output_dir / "attention_maps_raw"
    vis_output_dir_processed = output_dir / "attention_maps_processed"
    tensor_output_dir = output_dir / "attention_tensors"
    collage_output_dir = output_dir / "attention_collages"

    for dir_path in [vis_output_dir_raw, vis_output_dir_processed, tensor_output_dir, collage_output_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)

    return output_dir, vis_output_dir_raw, vis_output_dir_processed, tensor_output_dir, collage_output_dir

def _prepare_inputs(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    prompt: str,
    image_processor: SigLipImageProcessor,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel
) -> Tuple[Any, Any, torch.Tensor, List, List[int], Any]:
    """Load and prepare image, mask, and input tensors."""
    # Load image and mask
    image_path = fix_wsl_paths(str(image_path))
    mask_path = fix_wsl_paths(str(mask_path))

    image = load_image(image_path)
    mask = load_mask_from_file(mask_path)

    print(f"Image size: {image.size}")
    print(f"Mask shape: {mask.shape}")

    # Process image
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    # Get attention indices from mask
    atten_indices = get_attention_indices_from_mask(mask, image.size, model.config)
    print(f"Initial attention indices: {len(atten_indices)} tokens")

    # Prepare conversation
    conv_template = "qwen_1_5"  # Default for the model

    if DEFAULT_IMAGE_TOKEN not in prompt:
        full_prompt = f"{DEFAULT_IMAGE_TOKEN}\\n{prompt}"
    else:
        full_prompt = prompt

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], full_prompt)
    conv.append_message(conv.roles[1], None)        # 5 tokens are always added to the end of the user prompt
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_question, tokenizer,
        IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    image_sizes = [image.size]
    # tokenizer.decode(input_ids.cpu().numpy()[0][-11:-9])
    return image, mask, image_tensor, image_sizes, atten_indices, input_ids

def _determine_image_patch_info(model: PreTrainedModel, input_ids: torch.Tensor) -> Tuple[int, int, int, int]:
    """Determine image patch information and token indices."""
    # Determine image patch information
    num_patches = 729  # Default fallback
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

    print(f"Using number of image patches: {num_patches}")

    # Find image token indices
    image_token_indices_in_original = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0]
    if len(image_token_indices_in_original) == 0:
        image_token_start_index_in_llm = 1
    else:
        image_token_start_index_in_llm = image_token_indices_in_original[0].item()

    image_token_end_index_in_llm = image_token_start_index_in_llm + num_patches
    grid_size = int(np.sqrt(num_patches))

    if grid_size * grid_size != num_patches:
        grid_size = int(np.ceil(np.sqrt(num_patches)))

    print(f"Image token range: {image_token_start_index_in_llm} to {image_token_end_index_in_llm}")
    print(f"Grid size: {grid_size}x{grid_size}")

    return num_patches, grid_size, image_token_start_index_in_llm, image_token_end_index_in_llm

def _generate_next_token(
    model_inputs: Dict,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    gen_config: Dict
) -> Tuple[torch.Tensor, str, Any]:
    """Generate next token and return token info and model outputs."""
    outputs = model(**model_inputs)

    # Get next token
    next_token_logits = outputs.logits[:, -1, :]
    if gen_config.get("do_sample", False):
        # Apply temperature
        if gen_config.get("temperature", 1.) != 1.0:
            next_token_logits = next_token_logits / gen_config["temperature"]
        next_token_id = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
    else:
        next_token_id = torch.argmax(next_token_logits, dim=-1)

    token_text = tokenizer.decode([next_token_id.item()]).strip()

    return next_token_id, token_text, outputs

def _extract_and_process_attention(
    outputs: Any,
    next_token_id: torch.Tensor,
    token_text: str,
    step_idx: int,
    num_patches: int,
    grid_size: int,
    image_token_start_index_in_llm: int,
    image: Any,
    attn_config: Dict,
    vis_output_dir_raw: Path,
    vis_output_dir_processed: Path,
    tensor_output_dir: Path
) -> Optional[Any]:
    """Extract attention map and save visualizations. Returns processed image for collage or None."""
    if outputs.attentions is None:
        return None

    attentions = outputs.attentions
    selected_attentions = attentions[0][0].squeeze(0)  # Remove batch dimension
    avg_attentions = selected_attentions.mean(dim=0)  # Average across heads

    current_token_index = outputs.logits.shape[1] - 1
    token_attention_to_image = avg_attentions[current_token_index, image_token_start_index_in_llm:image_token_start_index_in_llm + num_patches]

    if token_attention_to_image.shape[0] != num_patches:
        return None

    # Handle padding for square grid
    expected_elements = grid_size * grid_size
    if expected_elements > num_patches:
        padding_size = expected_elements - num_patches
        token_attention_to_image = torch.cat([
            token_attention_to_image,
            torch.zeros(padding_size, device=token_attention_to_image.device)
        ])

    attention_map = token_attention_to_image.reshape(grid_size, grid_size).cpu().numpy()
    processed_img = None

    # Save visualizations
    if attn_config["visualize_attn_overlays"]:
        # Create safe filename
        safe_token_text = "".join(c if c.isalnum() else "_" for c in token_text)
        if not safe_token_text:
            safe_token_text = f"tokenid_{next_token_id.item()}"

        # Save raw attention
        raw_path = vis_output_dir_raw / f"token_{step_idx:03d}_{safe_token_text}.png"
        map_img = Image.fromarray(attention_map.astype(np.float32))
        resized_map = np.array(map_img.resize(image.size, Image.Resampling.LANCZOS))

        min_val, max_val = np.min(resized_map), np.max(resized_map)
        norm_map = np.zeros_like(resized_map) if max_val <= min_val else (resized_map - min_val) / (max_val - min_val)

        heatmap = cm.viridis(norm_map)[:, :, :3]
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        overlay_img = Image.blend(image, Image.fromarray(heatmap_uint8), alpha=0.5)
        overlay_img.save(raw_path)

        # Processed visualization
        processed_path = vis_output_dir_processed / f"token_{step_idx:03d}_{safe_token_text}.png"
        processed_img = visualize_processed_attention(
            attention_map=attention_map,
            original_image=image,
            output_path=processed_path,
            **{k: v for k, v in attn_config.items() if k in [
                'threshold_value', 'opening_kernel_size', 'min_blob_area',
                'min_avg_attention', 'show_highest_attn_blob', 'dilate_kernel_size'
            ]}
        )

    # Save tensor
    if attn_config["save_tensors"]:
        save_raw_attention_tensor(
            attention_map, attentions, tensor_output_dir,
            next_token_id.item(), token_text, step_idx
        )

    return processed_img

def _create_collages(
    collected_maps: List,
    attn_config: Dict,
    collage_output_dir: Path
) -> None:
    """Create attention collages from collected maps."""
    if not (attn_config["create_collage"] and collected_maps):
        return

    maps_per_collage = attn_config["collage_grid_rows"] * attn_config["collage_grid_cols"]
    num_collages = (len(collected_maps) + maps_per_collage - 1) // maps_per_collage

    for collage_idx in range(num_collages):
        start_idx = collage_idx * maps_per_collage
        end_idx = min(start_idx + maps_per_collage, len(collected_maps))
        collage_maps = collected_maps[start_idx:end_idx]

        collage_path = collage_output_dir / f"attention_collage_{collage_idx + 1}.png"
        visualize_attention_collage(
            collage_maps, collage_path,
            (attn_config["collage_grid_rows"], attn_config["collage_grid_cols"])
        )

# %%
# Main Generation Function with Attention Extraction (Refactored)
def run_generation_with_attention(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    prompt: str,
    output_dir: Union[str, Path],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    image_processor: SigLipImageProcessor,
    generation_config: Optional[Dict] = None,
    attention_config: Optional[Dict] = None,
    bias_strength: float = 0.0

) -> Dict[str, Any]:
    """
    Run generation with attention extraction for a given image, mask, and prompt.

    Returns a dictionary with generated text and output paths.
    """

    # Prepare configurations
    gen_config, attn_config = _prepare_configs(generation_config, attention_config)

    print(f"Processing image: {image_path}")
    print(f"Using mask: {mask_path}")
    print(f"Prompt: {prompt}")

    # Setup output directories
    output_dir, vis_output_dir_raw, vis_output_dir_processed, tensor_output_dir, collage_output_dir = _setup_output_directories(output_dir)

    # Load and prepare inputs
    image, mask, image_tensor, image_sizes, atten_indices, input_ids = _prepare_inputs(image_path, mask_path, prompt, image_processor, tokenizer, model)
    # temp boost coordinates
    boost_positions = atten_indices
    # Determine image patch information
    num_patches, grid_size, image_token_start_index_in_llm, image_token_end_index_in_llm = _determine_image_patch_info(model, input_ids)

    # Generation loop
    print("Starting generation with attention extraction...")
    max_new_tokens = gen_config["max_new_tokens"]
    generated_ids = []
    past_key_values = None
    current_input_ids = input_ids
    collected_maps = []  # For collage

    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]

    for i in range(max_new_tokens):
        with torch.inference_mode():
            # Prepare model inputs
            model_inputs = {
                "input_ids": current_input_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
                "output_attentions": True,
                "output_hidden_states": True,
                "atten_ids": None,
                "boost_positions": boost_positions,
                "bias_strength": bias_strength,
            }

            # Add image data only in first step
            if i == 0:
                model_inputs["images"] = image_tensor
                model_inputs["image_sizes"] = image_sizes
                model_inputs["modalities"] = ["image"]

            # Generate next token
            next_token_id, token_text, outputs = _generate_next_token(model_inputs, model, tokenizer, gen_config)

            # Check for EOS
            if next_token_id.item() == eos_token_id:
                print("EOS token generated. Stopping.")
                break

            generated_ids.append(next_token_id.item())

            # Extract and process attention
            processed_img = _extract_and_process_attention(
                outputs, next_token_id, token_text, i,
                num_patches, grid_size, image_token_start_index_in_llm,
                image, attn_config, vis_output_dir_raw, vis_output_dir_processed, tensor_output_dir
            )

            # Collect for collage
            if attn_config["create_collage"] and processed_img is not None:
                collected_maps.append((processed_img, token_text))

            # Print current result
            if i == 0:
                txt_ids = torch.argmax(outputs.logits, dim=-1)[0]
                txt_np = np.array([tokenizer.decode(val).strip() for val in txt_ids])
                resulted_description = ",".join(txt_np[boost_positions])
                print(f"Resulted words from mask: {resulted_description}")
            # else:
                # print(f"Generated token {i}: '{token_text}'")

            # Update for next iteration
            current_input_ids = next_token_id.unsqueeze(-1)
            past_key_values = outputs.past_key_values

            # Update attention indices
            if i == 0:
                initial_token_length = past_key_values[0][0].shape[2]
            input_token_length = past_key_values[0][0].shape[2]
            # boost_positions = boost_positions + [input_token_length - 1]

    # Create collages
    # _create_collages(collected_maps, attn_config, collage_output_dir)

    # Generate final text
    final_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print("\\n" + "="*50)
    print("GENERATION COMPLETE")
    print("="*50)
    print(f"Generated text: {final_text}")
    print(f"Generated {len(generated_ids)} tokens")
    print(f"Output directory: {output_dir}")

    return {
        "generated_text": final_text,
        "generated_tokens": generated_ids,
        "num_tokens": len(generated_ids),
        "output_directories": {
            "main": str(output_dir),
            "raw_attention": str(vis_output_dir_raw) if attn_config["visualize_attn_overlays"] else None,
            "processed_attention": str(vis_output_dir_processed) if attn_config["visualize_attn_overlays"] else None,
            "tensors": str(tensor_output_dir) if attn_config["save_tensors"] else None,
            "collages": str(collage_output_dir) if attn_config["create_collage"] and collected_maps else None
        },
        "config_used": {
            "generation": gen_config,
            "attention": attn_config
        }
    }

# %%
# Batch Processing Functions

def process_batch(
    batch_items: List[Dict[str, str]],
    base_output_dir: str,
    generation_config: Optional[Dict] = None,
    attention_config: Optional[Dict] = None
) -> List[Dict[str, Any]]:
    """
    Process multiple image/mask/prompt combinations.

    batch_items: List of dicts with keys 'image_path', 'mask_path', 'prompt', 'name'
    """
    results = []

    for i, item in enumerate(batch_items):
        print(f"\\n{'='*60}")
        print(f"Processing batch item {i+1}/{len(batch_items)}: {item.get('name', f'item_{i+1}')}")
        print('='*60)

        output_dir = Path(base_output_dir) / item.get('name', f'item_{i+1}')

        result = run_generation_with_attention(
            image_path=item['image_path'],
            mask_path=item['mask_path'],
            prompt=item['prompt'],
            output_dir=str(output_dir),
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            generation_config=generation_config,
            attention_config=attention_config,
            bias_strength= generation_config["bias_strength"]  # Bias strength for attention extraction
        )

        result['item_name'] = item.get('name', f'item_{i+1}')
        result['item_index'] = i
        results.append(result)

        # Brief summary
        if "error" not in result:
            print(f"✅ Completed: Generated {result['num_tokens']} tokens")
        else:
            print(f"❌ Failed: {result['error']}")

    return results

def process_batch_from_json(
    json_path: Union[str, Path],
    base_image_dir: Union[str, Path],
    base_mask_dir: Union[str, Path],
    base_output_dir: Union[str, Path],
    mask_filename_template: str = "gaze__{}_masks.npy",
    prompt_template: str = "Complete the sentence. The {} is looking at",
    generation_config: Optional[Dict] = None,
    attention_config: Optional[Dict] = None,
    limit_items: Optional[int] = None,
    filter_keys: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Process images using subject descriptions from JSON file.

    Args:
        json_path: Path to train_results.json
        base_image_dir: Base directory for images (e.g., "D:/Projects/data/gazefollow/train")
        base_mask_dir: Base directory for masks (e.g., "D:/Projects/data/gazefollow/train_gaze_segmentations/masks")
        base_output_dir: Base directory for outputs
        mask_filename_template: Template for mask filename (should contain {} for image key)
        prompt_template: Template for prompt (should contain {} for subject description)
        generation_config: Generation configuration
        attention_config: Attention configuration
        limit_items: Limit number of items to process (for testing)
        filter_keys: List of specific image keys to process (if None, process all)

    Returns:
        Dictionary with same structure as input JSON, but with additional fields:
        - generation_result: Contains the generated text and metadata
        - prompt_used: The actual prompt that was used
        - processing_timestamp: When the item was processed
    """
    # Load JSON data
    person_desc_data = load_train_results_json(json_path)

    # Filter data if needed
    if filter_keys:
        person_desc_data = {k: v for k, v in person_desc_data.items() if k in filter_keys}
        print(f"Filtered to {len(person_desc_data)} items based on filter_keys")

    if limit_items:
        items = list(person_desc_data.items())[:limit_items]
        person_desc_data = dict(items)
        print(f"Limited to {len(person_desc_data)} items for processing")

    base_image_dir = Path(base_image_dir)
    base_mask_dir = Path(base_mask_dir)
    base_output_dir = Path(base_output_dir)

    results = {}
    total_items = len(person_desc_data)

    for i, (image_key, subject_description) in enumerate(person_desc_data.items(), 1):
        print(f"\\n{'='*80}")
        print(f"Processing {i}/{total_items}: {image_key}")
        print('='*80)

        try:
            # Get subject description
            print(f"Subject description: '{subject_description}'")

            # Build prompt
            prompt = build_prompt_with_subject_description(subject_description, prompt_template)
            print(f"Generated prompt: '{prompt}'")

            # Construct file paths
            # Convert image_key back to path format (e.g., "00000000_00000021" -> "00000000/00000021.jpg")
            folder_name, filename = image_key.split('/')[-2:]
            image_path = base_image_dir / folder_name / f"{filename}"
            mask_path = base_mask_dir / mask_filename_template.format(filename.split('.')[0])

            image_path = Path(fix_wsl_paths(str(image_path)))
            mask_path = Path(fix_wsl_paths(str(mask_path)))

            print(f"Image path: {image_path}")
            print(f"Mask path: {mask_path}")

            # Check if files exist
            if not image_path.exists():
                error_msg = f"Image file not found: {image_path}"
                print(f"❌ {error_msg}")
                results[image_key] = {
                    "subject_description": subject_description,
                    "generation_result": {"error": error_msg},
                    "prompt_used": prompt
                }
                continue

            if not mask_path.exists():
                error_msg = f"Mask file not found: {mask_path}"
                print(f"❌ {error_msg}")
                results[image_key] = {
                    "subject_description": subject_description,
                    "generation_result": {"error": error_msg},
                    "prompt_used": prompt
                }
                continue

            # Set up output directory
            output_dir = base_output_dir / image_key

            # Run generation
            result = run_generation_with_attention(
                image_path=str(image_path),
                mask_path=str(mask_path),
                prompt=prompt,
                output_dir=str(output_dir),
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                generation_config=generation_config,
                attention_config=attention_config,
                bias_strength=generation_config.get("bias_strength", 4.5)  # Use default if not provided
            )

            # Store result
            results[image_key] = {
                "subject_description": subject_description,  # Keep original image info
                "generation_result": result,
                "prompt_used": prompt,
                "processing_timestamp": str(datetime.now())
            }

            # Brief summary
            if "error" not in result:
                print(f"✅ Completed: Generated '{result['generated_text']}'")
            else:
                print(f"❌ Failed: {result['error']}")

        except Exception as e:
            error_msg = f"Unexpected error processing {image_key}: {str(e)}"
            print(f"❌ {error_msg}")
            results[image_key] = {
                "subject_description": subject_description,
                "generation_result": {"error": error_msg},
                "prompt_used": prompt if 'prompt' in locals() else "N/A"
            }

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"generation_results_{timestamp}.json"
    results_path = base_output_dir / results_filename
    save_results_to_json(results, results_path)

    print(f"\\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print('='*80)
    print(f"Processed {len(results)} items")
    print(f"Results saved to: {results_path}")

    successful = sum(1 for r in results.values() if "error" not in r.get("generation_result", {}))
    failed = len(results) - successful
    print(f"Successful: {successful}, Failed: {failed}")

    return results

# %%
# Interactive Configuration and Usage
# This is the main cell you'll modify for different experiments

# Configuration for current run
all_results = {}
# for bias_i in np.linspace(2.5, 3.5, 5):     # good for hands detection
for bias_i in np.linspace(0., 5., 10):     
    print(f"\n{'='*60}")
    print(f"Running experiment with bias strength: {bias_i:.2f}")
    EXPERIMENT_CONFIG = {
        # Input paths (MODIFY THESE)
        "image_path": r"D:\Projects\data\gazefollow\train\00000000\00000010.jpg",  # Your image path
        # "mask_path": r"D:\Projects\data\gazefollow\train_gaze_segmentations\masks\gaze__00000318_masks.npy",  # Your mask path
        "mask_path": r"D:\Projects\data\gazefollow\train_gaze_segmentations\masks\gaze__00000010_masks.npy",  # Your mask path
        # "mask_path": r"D:\Projects\data\gazefollow\train_gaze_segmentations\masks\gaze__00000022_masks.npy",  # Your mask path
        # "prompt": "Complete the sentence. The tattooed man at wedding reception is looking at",  # Your prompt
        # "prompt": "Complete the sentence in a few words answer. The object is",  # Your prompt
        # "prompt": "Complete the sentence. Start your answer with 'The object in focus is'",  # Your prompt
        "prompt": "Complete the sentence. The person is looking at",  # Your prompt
        # "prompt": "Describe the image in 3 words.",  # Your prompt
        # "prompt": "Start the sentence with 'the object is'",  # Your prompt
        "output_dir": f"attention_output/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}_bias_{bias_i:.2f}",  # Output directory with timestamp and bias value

        # Generation configuration
        "generation_config": {
            "bias_strength": bias_i,  # Set to 0.0 for no bias, or adjust as needed
            "max_new_tokens": 50,
            "temperature": 0.0,
            "do_sample": False
        },

        # Attention visualization configuration
        "attention_config": {
            "attn_threshold": 0.4,
            "opening_kernel_size": 5,
            "min_blob_area": 50,
            "min_avg_attention": 0.2,
            "show_highest_attn_blob": False,
            "dilate_kernel_size": 0,
            "create_collage": True,
            "collage_grid_rows": 3,
            "collage_grid_cols": 4,
            "visualize_attn_overlays": True,
            "save_tensors": True
        }
    }

    # Run the experiment
    print("Starting attention extraction experiment...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = run_generation_with_attention(
        image_path=EXPERIMENT_CONFIG["image_path"],
        mask_path=EXPERIMENT_CONFIG["mask_path"],
        prompt=EXPERIMENT_CONFIG["prompt"],
        output_dir=EXPERIMENT_CONFIG["output_dir"],
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        generation_config=EXPERIMENT_CONFIG["generation_config"],
        attention_config=EXPERIMENT_CONFIG["attention_config"],
        bias_strength= EXPERIMENT_CONFIG["generation_config"]["bias_strength"]  # Bias strength for attention extraction
    )
    all_results[bias_i] = results

# Display results
if "error" not in results:
    print("\\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)
    print(f"Generated Text: {results['generated_text']}")
    print(f"Number of tokens: {results['num_tokens']}")
    print("\\nOutput Directories:")
    for key, path in results['output_directories'].items():
        if path:
            print(f"  {key}: {path}")
else:
    print(f"\\n❌ Experiment failed: {results['error']}")

# %%
# JSON-based Batch Processing Configuration
# Use this cell to process multiple images from the JSON file

JSON_BATCH_CONFIG = {
    # Path to the train_results.json file
    "json_path": r"D:\Projects\data\gazefollow\train_results.json",

    # Base directories
    "base_image_dir": r"D:\Projects\data\gazefollow\train",
    "base_mask_dir": r"D:\Projects\data\gazefollow\train_gaze_segmentations\masks",
    "base_output_dir": "attention_output/json_batch_processing",

    # File templates
    "mask_filename_template": "gaze__{}_masks.npy",
    # "prompt_template": "Complete the sentence. The {} is looking at",
    "prompt_template": "Complete the sentence. The person is looking at",
    # "prompt_template": "Complete the sentence in a few words answer. The object is",  # Your prompt
    # "prompt_template": "Complete the sentence in a short answer. The object in the center is",  # Your prompt

    # Processing options
    "limit_items": 200,  # Set to None to process all items, or a number to limit for testing
    "filter_keys": None,  # Set to list of specific keys to process, e.g., ["00000000_00000021", "00000000_00000025"]

    # Generation configuration
    "generation_config": {
        "bias_strength": 2.78,  # Set to 0.0 for no bias, or adjust as needed
        "max_new_tokens": 50,
        "temperature": 0.0,
        "do_sample": False
    },

    # Attention visualization configuration
    "attention_config": {
        "attn_threshold": 0.4,
        "opening_kernel_size": 5,
        "min_blob_area": 50,
        "min_avg_attention": 0.2,
        "show_highest_attn_blob": False,
        "dilate_kernel_size": 0,
        "create_collage": True,
        "collage_grid_rows": 3,
        "collage_grid_cols": 4,
        "visualize_attn_overlays": True,
        "save_tensors": True
    }
}

# Run JSON-based batch processing
print("Starting JSON-based batch processing...")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

batch_results = process_batch_from_json(
    json_path=JSON_BATCH_CONFIG["json_path"],
    base_image_dir=JSON_BATCH_CONFIG["base_image_dir"],
    base_mask_dir=JSON_BATCH_CONFIG["base_mask_dir"],
    base_output_dir=JSON_BATCH_CONFIG["base_output_dir"],
    mask_filename_template=JSON_BATCH_CONFIG["mask_filename_template"],
    prompt_template=JSON_BATCH_CONFIG["prompt_template"],
    generation_config=JSON_BATCH_CONFIG["generation_config"],
    attention_config=JSON_BATCH_CONFIG["attention_config"],
    limit_items=JSON_BATCH_CONFIG["limit_items"],
    filter_keys=JSON_BATCH_CONFIG["filter_keys"]
)

# Summary of results
print("\\n" + "="*80)
print("JSON BATCH PROCESSING SUMMARY")
print("="*80)
successful_count = 0
failed_count = 0

for image_key, result in batch_results.items():
    generation_result = result.get("generation_result", {})
    if "error" in generation_result:
        status = f"❌ Error: {generation_result['error']}"
        failed_count += 1
    else:
        status = f"✅ Success: '{generation_result['generated_text']}'"
        successful_count += 1

    subject_desc = result.get('subject_description', 'N/A')
    prompt_used = result.get('prompt_used', 'N/A')

    print(f"\\n{image_key}:")
    print(f"  Subject: {subject_desc}")
    print(f"  Prompt: {prompt_used}")
    print(f"  Result: {status}")

print(f"\\nTotal processed: {len(batch_results)}")
print(f"Successful: {successful_count}")
print(f"Failed: {failed_count}")

# %%
print("\\n🎉 Notebook setup complete! You can now:")
print("1. Modify EXPERIMENT_CONFIG and re-run the experiment cell")
print("2. Use JSON_BATCH_CONFIG for processing images with subject descriptions from JSON")
print("3. Use the manual batch processing cell for multiple images")
print("4. Adjust model configuration and reload if needed")
print("\\nNew Features Added:")
print("- JSON-based batch processing with subject descriptions")
print("- Automatic prompt generation using subject descriptions")
print("- Results saved in same structure as input JSON")

# %%
# Example Usage Instructions
"""
## How to use the new JSON-based processing:

1. **Single Image with JSON data:**
   ```python
   # Load the JSON data
   train_data = load_train_results_json("D:/Projects/data/gazefollow/train_results.json")

   # Get info for a specific image
   image_key = "00000000_00000021"
   image_info = get_image_info_from_json(train_data, image_key)

   # Build the prompt
   prompt = build_prompt_with_subject_description(image_info['subject_description'])
   print(f"Generated prompt: {prompt}")
   ```

2. **Batch processing with JSON data:**
   - Modify the JSON_BATCH_CONFIG above
   - Set limit_items to a small number for testing (e.g., 5)
   - Set filter_keys to specific image keys if needed
   - Run the JSON batch processing cell

3. **Expected output structure:**
   The results JSON will have the same keys as the input, but with additional fields:
   - generation_result: Contains the generated text and metadata
   - prompt_used: The actual prompt that was used
   - processing_timestamp: When the item was processed

4. **File naming conventions:**
   - Images: {base_image_dir}/{folder}/{filename}.jpg
   - Masks: {base_mask_dir}/gaze__{folder}_{filename}_masks.npy
   - Example: "00000000_00000021" → image: "00000000/00000021.jpg", mask: "gaze__00000000_00000021_masks.npy"
"""
