import os
import sys
import json
from datetime import datetime
from pathlib import Path
import copy
from typing import Dict, List, Optional, Any, Tuple, Union

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
import traceback

# Import our enhanced generation metrics
from generation_metrics import (
    ConfidenceMetrics, RepetitivityMetrics, TopKCandidateEvaluator,
    generate_next_token_with_evaluation, create_generation_summary,
    analyze_generation_quality, calculate_attention_correlation_from_similarity
)

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
        if 'static_masks' in mask_dict:
            mask = mask_dict['static_masks'][0]  # Take first mask
        elif 'masks' in mask_dict:
            mask = mask_dict['masks'][0]
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
    atten_indices, resized_mask = _pixel_to_token_indices_helper_anyres(
        mask_coords, image_size, possible_resolutions=model_config.image_grid_pinpoints,
        add_system_prompt_tokens= False,
        add_user_prompt_tokens= False,
        user_prompt_range=[1849, 1860]
    )

    return atten_indices, resized_mask

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


def visualize_embedding_similarity(
    text_token_embedding: torch.Tensor,
    image_token_embeddings: torch.Tensor,
    original_image: Image.Image,
    grid_size: int,
    output_path: Union[str, Path],
    region_radius: int = 3,
    top_k: int = 2,
    normalize: bool = True
) -> None:
    """
    Calculates and visualizes the semantic similarity between a text token's embedding
    and all image patch embeddings. Highlights the top-k similarity regions with area around them.

    Args:
        text_token_embedding: The feature embedding of the generated text token. Shape: [hidden_dim].
        image_token_embeddings: The feature embeddings of all image patches. Shape: [num_patches, hidden_dim].
        original_image: The original PIL image for the background.
        grid_size: The dimension of the square patch grid (e.g., 27 for a 27x27 grid).
        output_path: The path to save the visualization.
        region_radius: The radius around each top similarity value to include in visualization.
        top_k: The number of top similarity values to highlight (default: 3).
    """
    # 1. Calculate dot product similarity
    # text_token_embedding: [D], image_token_embeddings: [N, D] -> similarity_scores: [N]
    similarity_scores = torch.nn.functional.cosine_similarity(
        image_token_embeddings, text_token_embedding.unsqueeze(0), dim=1
    )

    # 2. Reshape into a 2D grid
    num_patches = image_token_embeddings.shape[0]
    expected_elements = grid_size * grid_size
    if expected_elements > num_patches:
        padding_size = expected_elements - num_patches
        similarity_scores = torch.cat([
            similarity_scores,
            torch.zeros(padding_size, device=similarity_scores.device)
        ])
    
    similarity_map = similarity_scores.reshape(grid_size, grid_size).cpu().numpy()
    
    # Find top-k similarity values and their positions
    flat_map = similarity_map.flatten()
    topk_indices = np.argpartition(flat_map, -top_k)[-top_k:]  # Get indices of top-k values
    topk_positions = [(idx // grid_size, idx % grid_size) for idx in topk_indices]
    
    # Create a mask for top-k regions with area around them
    topk_mask = np.zeros_like(similarity_map)
    
    for row, col in topk_positions:
        # Create circular region around each top value
        for r in range(max(0, row - region_radius), min(grid_size, row + region_radius + 1)):
            for c in range(max(0, col - region_radius), min(grid_size, col + region_radius + 1)):
                # Check if within circular radius
                distance = np.sqrt((r - row)**2 + (c - col)**2)
                if distance <= region_radius:
                    topk_mask[r, c] = 1
    
    # Apply mask to similarity map
    masked_similarity_map = similarity_map * topk_mask

    if normalize:
        # Normalize the masked similarity map
        min_val, max_val = np.min(masked_similarity_map), np.max(masked_similarity_map)
        if max_val > min_val:
            masked_similarity_map = (masked_similarity_map - min_val) / (max_val - min_val)
        else:
            masked_similarity_map = np.zeros_like(masked_similarity_map)

    # Resize heatmap to image size for overlay
    map_img = Image.fromarray(masked_similarity_map.astype(np.float32))
    resized_map_img = map_img.resize(original_image.size, Image.Resampling.LANCZOS)
    
    if normalize:
        # Normalize the resized map
        min_val, max_val = np.min(resized_map_img), np.max(resized_map_img)
        if max_val > min_val:
            resized_map_img = (resized_map_img - min_val) / (max_val - min_val)
        else:
            resized_map_img = np.zeros_like(resized_map_img)
    
    heatmap_colors = cm.inferno(np.array(resized_map_img))[:, :, :3]
    heatmap_uint8 = (heatmap_colors * 255).astype(np.uint8)

    # 4. Blend with original image and save
    overlay_img = Image.blend(original_image, Image.fromarray(heatmap_uint8), alpha=0.6)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_img.save(output_path)

    return masked_similarity_map


# JSON Data Loading and Prompt Building Functions

def load_train_results_json(json_path: Union[str, Path]) -> Dict[str, str]:
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
        # Convert all objects to JSON-serializable types
        serializable_results = _make_json_serializable(results)
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {output_path}")

# Helper Functions for Generation Process

def _prepare_configs(generation_config: Optional[Dict], attention_config: Optional[Dict]) -> Tuple[Dict, Dict]:
    """Prepare and merge generation and attention configurations with defaults."""
    default_generation_config = {
        "max_new_tokens": 50,
        "temperature": 0.1,
        "do_sample": True,
        "top_k": None  # None = disabled, int > 0 = enabled with specified value
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
        "save_tensors": False
    }

    gen_config = {**default_generation_config, **(generation_config or {})}
    attn_config = {**default_attention_config, **(attention_config or {})}

    return gen_config, attn_config

def _setup_output_directories(output_dir: Union[str, Path]) -> Tuple[Path, Path, Path, Path, Path, Path]:
    """Create and return all output directory paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    vis_output_dir_raw = output_dir / "attention_maps_raw"
    vis_output_dir_processed = output_dir / "attention_maps_processed"
    tensor_output_dir = output_dir / "attention_tensors"
    collage_output_dir = output_dir / "attention_collages"
    similarity_output_dir = output_dir / "embedding_similarity_overlays" # New directory

    for dir_path in [vis_output_dir_raw, vis_output_dir_processed, tensor_output_dir, collage_output_dir, similarity_output_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)

    return output_dir, vis_output_dir_raw, vis_output_dir_processed, tensor_output_dir, collage_output_dir, similarity_output_dir

def _prepare_inputs(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    prompt: str,
    image_processor: SigLipImageProcessor,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel
) -> Tuple[Any, Any, torch.Tensor, List, List[int], Any, torch.Tensor]:
    """Load and prepare image, mask, and input tensors."""
    person_mask = None
    person_mask_indices = None
    # Load image and mask
    image = load_image(image_path)
    mask = load_mask_from_file(mask_path)
    person_mask_path = str(mask_path).replace("gaze__", "person__")
    if Path(person_mask_path).exists():
        person_mask = load_mask_from_file(person_mask_path)


    print(f"Image size: {image.size}")
    print(f"Mask shape: {mask.shape}")


    # Process image
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    # Get attention indices from mask
    atten_indices, target_mask = get_attention_indices_from_mask(mask, image.size, model.config)
    if person_mask is not None:
        person_mask_indices, person_mask = get_attention_indices_from_mask(person_mask, image.size, model.config)
    print(f"Initial attention indices: {len(atten_indices)} tokens")
    masks = {'target_mask': target_mask, 'person_mask': person_mask}        # masks in [model's] input image resolution 
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
    return image, masks, image_tensor, image_sizes, atten_indices, person_mask_indices, input_ids

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
        # 1. Apply top-k filtering first (if enabled)
        if gen_config.get("top_k") is not None and gen_config.get("top_k") > 0:
            top_k = gen_config["top_k"]
            # Get top-k indices and values
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            # Create filtered logits tensor (set non-top-k to -inf)
            filtered_logits = torch.full_like(next_token_logits, float('-inf'))
            filtered_logits.scatter_(-1, top_k_indices, top_k_logits)
            next_token_logits = filtered_logits
        
        # 2. Apply temperature scaling (after top-k filtering)

        if gen_config.get("temperature", 1.) != 1.0:
            next_token_logits = next_token_logits / gen_config["temperature"]
        
        # 3. Sample from the (potentially filtered) distribution
        probs = torch.softmax(next_token_logits, dim=-1)
        if len(probs.shape) == 3:
            next_token_id = torch.multinomial(probs[0], 1)
        else:
            next_token_id = torch.multinomial(probs, 1)
    else:
        # Greedy decoding (no sampling)
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
    """Extract attention map and save visualizations. Returns processed image for collage and the raw attention map."""
    if outputs.attentions is None:
        return None, None

    attentions = outputs.attentions
    selected_attentions = attentions[0][0].squeeze(0)  # Remove batch dimension
    avg_attentions = selected_attentions.mean(dim=0)  # Average across heads

    current_token_index = outputs.logits.shape[1] - 1
    token_attention_to_image = avg_attentions[current_token_index, image_token_start_index_in_llm:image_token_start_index_in_llm + num_patches]

    if token_attention_to_image.shape[0] != num_patches:
        return None, None

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

    return processed_img, attention_map

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

def print_summary(generation_summary, quality_analysis, final_text: str, generated_ids: List[int], output_dir: Path) -> None:
    print("" + "="*50)
    print("GENERATION COMPLETE")
    print("="*50)
    print(f"Generated text: {final_text}")
    print(f"Generated {len(generated_ids)} tokens")
    print(f"Output directory: {output_dir}")

    # Print enhanced evaluation summary
    if generation_summary and quality_analysis:
        print("" + "="*50)
        print("ENHANCED EVALUATION SUMMARY")
        print("="*50)
        
        # Overall quality score
        overall_score = quality_analysis.get("overall_quality_score", 0)
        interpretation = quality_analysis.get("interpretation", {})
        quality_level = "Unknown"
        for level, is_level in interpretation.items():
            if is_level:
                quality_level = level.title()
                break
        
        print(f"Overall Quality Score: {overall_score:.2f}/10 ({quality_level})")
        
        # Key metrics
        gen_metrics = generation_summary.get("average_confidence", {})
        print(f"Average Confidence: {gen_metrics.get('confidence_score', 0):.3f}")
        print(f"Average Entropy: {gen_metrics.get('entropy', 0):.3f}")
        print(f"Confidence Trend: {generation_summary.get('confidence_trajectory', {}).get('trend', 'unknown')}")
        
        # Diversity metrics
        diversity = generation_summary.get("diversity_metrics", {})
        print(f"Type-Token Ratio: {diversity.get('type_token_ratio', 0):.3f}")
        print(f"Bigram Repetition: {diversity.get('repetition_penalty_2gram', 0):.3f}")
        
        # Decision points
        decision_points = generation_summary.get("decision_points", [])
        print(f"Decision Points (low confidence): {len(decision_points)}")
        
        if decision_points:
            print("Key Decision Points:")
            for dp in decision_points[:3]:  # Show first 3
                step = dp["step"]
                prob = dp["top1_probability"]
                alts = dp.get("alternatives", [])[:2]  # Show top 2 alternatives
                conf = dp["confidence"]
                print(f"  Step {step}: Confidence ({conf:.3f})")
                for alt in alts:
                    print(f"    {alt}")
        
        # Quality factors breakdown
        factors = quality_analysis.get("quality_factors", {})
        print(f"Quality Breakdown:")
        print(f"  Confidence Factor: {factors.get('confidence', 0):.2f}")
        print(f"  Diversity Factor: {factors.get('diversity', 0):.2f}")
        print(f"  Repetition Penalty: {factors.get('repetition', 0):.2f}")
        print(f"  Stability Factor: {factors.get('stability', 0):.2f}")
        print(f"  Attention Correlation with Mask Factor: {factors.get('attention_correlation', 0):.2f}")

    # Print Attention Correlation
    attention_correlation = generation_summary.get('average_attention_correlation', 0)
    if attention_correlation:
        print("" + "="*50)
        print("ATTENTION CORRELATION METRICS")
        print("="*50)
        print(f"Normalized Correlation Score: {attention_correlation.get('normalized_correlation_score', 0):.3f}")
        print(f"  - On-Target Attention Mean: {attention_correlation.get('target_attention_mean', 0):.4f}")
        print(f"  - Off-Target Attention Mean: {attention_correlation.get('off_target_attention_mean', 0):.4f}")
        print(f"  - Focus Ratio: {attention_correlation.get('attention_focus_ratio', 0):.2f}x")

# Summary and analysis of all bias sweep results

def _make_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    # Handle PyTorch tensors
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    else:
        return obj

def analyze_bias_sweep_results(all_results: Dict[float, Dict[str, Any]], base_output_dir: Union[str, Path], save_summary: bool = True) -> Dict[float, Dict[str, Any]]:
    """
    Analyze and summarize bias sweep experiment results.
    
    Args:
        all_results: Dictionary mapping bias strength to experiment results
        base_output_dir: Base output directory for saving summary
        save_summary: Whether to save the summary to file and print detailed results
        
    Returns:
        Dictionary of performance summary data
    """
    print(f"\n{'='*80}")
    print("BIAS SWEEP EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    # Analyze and sort results by performance
    performance_summary = []
    for bias_val, result in all_results.items():
        evaluation_summary = result.get("evaluation_summary", {})
        quality_analysis = result.get("quality_analysis", {})
        attention_correlation = result.get("attention_correlation", {})
        
        # Extract key performance metrics
        avg_confidence = evaluation_summary.get("average_confidence", {}).get("confidence_score", 0.0)
        avg_entropy = evaluation_summary.get("average_confidence", {}).get("entropy", float('inf'))
        correlation_score = attention_correlation.get("normalized_correlation_score", 0.0)
        generated_text = result.get("generated_text", "")
        num_tokens = result.get("num_tokens", 0)
        
        # Use existing overall quality score from quality_analysis
        overall_quality_score = quality_analysis.get("overall_quality_score", 0.0)
        
        performance_summary.append({
            'bias_strength': bias_val,
            'overall_quality_score': overall_quality_score,
            'avg_confidence': avg_confidence,
            'avg_entropy': avg_entropy,
            'correlation_score': correlation_score,
            'generated_text': generated_text,
            'num_tokens': num_tokens,
            'quality_analysis': quality_analysis
        })
    
    # Sort by overall quality score (descending - higher is better)
    performance_summary.sort(key=lambda x: x['overall_quality_score'], reverse=True)
    
    # Display top performers
    print(f"\nTOP 5 PERFORMING BIAS STRENGTHS:")
    print(f"{'Rank':<4} {'Bias':<6} {'Quality':<8} {'Confidence':<11} {'Entropy':<8} {'Correlation':<11} {'Tokens':<7} {'Generated Text':<30}")
    print("-" * 100)
    
    for i, result in enumerate(performance_summary[:5], 1):
        print(f"{i:<4} {result['bias_strength']:<6.2f} {result['overall_quality_score']:<8.2f} "
                f"{result['avg_confidence']:<11.3f} {result['avg_entropy']:<8.3f} "
                f"{result['correlation_score']:<11.3f} {result['num_tokens']:<7} "
                f"{result['generated_text'][:30]:<30}")
    
    # Display worst performers for comparison
    print(f"\nWORST 3 PERFORMING BIAS STRENGTHS:")
    print(f"{'Rank':<4} {'Bias':<6} {'Quality':<8} {'Confidence':<11} {'Entropy':<8} {'Correlation':<11} {'Tokens':<7} {'Generated Text':<30}")
    print("-" * 100)
    
    for i, result in enumerate(performance_summary[-3:], len(performance_summary)-2):
        print(f"{i:<4} {result['bias_strength']:<6.2f} {result['overall_quality_score']:<8.2f} "
                f"{result['avg_confidence']:<11.3f} {result['avg_entropy']:<8.3f} "
                f"{result['correlation_score']:<11.3f} {result['num_tokens']:<7} "
                f"{result['generated_text'][:30]:<30}")
    
    # Best bias strength recommendation
    best_result = performance_summary[0]
    print(f"\n🏆 RECOMMENDED BIAS STRENGTH: {best_result['bias_strength']:.2f}")
    print(f"   • Overall Quality Score: {best_result['overall_quality_score']:.2f}/10")
    print(f"   • Average Confidence: {best_result['avg_confidence']:.3f}")
    print(f"   • Average Entropy: {best_result['avg_entropy']:.3f}")
    print(f"   • Attention Correlation: {best_result['correlation_score']:.3f}")
    print(f"   • Generated Text: '{best_result['generated_text']}'")
    
    # Save detailed summary to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = Path(base_output_dir) / f"bias_sweep_summary_{timestamp}.json"
    summary_data = {
        "experiment_timestamp": timestamp,
        "bias_range": list(all_results.keys()),
        "performance_ranking": performance_summary,
        "best_bias_strength": best_result['bias_strength'],
        "summary_metrics": {
            "total_experiments": len(all_results),
            "best_quality_score": best_result['overall_quality_score'],
            "quality_score_range": [performance_summary[-1]['overall_quality_score'], performance_summary[0]['overall_quality_score']]
        }
    }
        
    if save_summary:
        # Convert to JSON-serializable format
        summary_data = _make_json_serializable(summary_data)
        
        with open(summary_path, 'w') as f:
            import json
            json.dump(summary_data, f, indent=2)
        
        print(f"\n📊 Detailed summary saved to: {summary_path}")
        print(f"{'='*80}")
    
    return performance_summary

# Batch processing helper functions
def prepare_person_desc_data(
    json_path: Union[str, Path],
    filter_keys: Optional[List[str]] = None,
    limit_items: Optional[int] = None
) -> Dict[str, str]:
    """Load and optionally filter and limit person description data."""
    json_path = fix_wsl_paths(str(json_path))
    data = load_train_results_json(json_path)
    if filter_keys:
        data = {k: v for k, v in data.items() if k in filter_keys}
    if limit_items:
        data = dict(list(data.items())[:limit_items])
    return data

def default_bias_range(
    min_bias: float = 1.0,
    max_bias: float = 4.0,
    steps: int = 4
) -> np.ndarray:
    """Return default bias sweep range."""
    return np.linspace(min_bias, max_bias, steps)

def prepare_batch_paths(
    image_key: str,
    base_image_dir: Union[str, Path],
    base_mask_dir: Union[str, Path],
    mask_filename_template: str = "gaze__{}_masks.npy"
) -> Tuple[Path, Path]:
    """Derive image and mask paths from image_key and validate existence."""
    base_image_dir = Path(fix_wsl_paths(str(base_image_dir)))
    base_mask_dir = Path(fix_wsl_paths(str(base_mask_dir)))
    parts = image_key.strip("/").split("/")
    folder, filename = parts[-2], parts[-1]
    image_path = base_image_dir / folder / filename
    mask_path = base_mask_dir / mask_filename_template.format(Path(filename).stem)
    if not image_path.exists() or not mask_path.exists():
        print(f"Image or mask not found for {image_key}")
        return None, None
    return image_path, mask_path

def create_experiment_config(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    prompt: str,
    output_dir: Union[str, Path],
    generation_config: Optional[Dict[str, Any]] = None,
    attention_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build base experiment config dict for bias sweep."""
    return {
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "prompt": prompt,
        "output_dir": str(output_dir),
        "generation_config": generation_config or {},
        "attention_config": attention_config or {},
    }

def save_image_results(
    results: Dict[str, Any],
    output_dir: Union[str, Path],
    prefix: str = "bias_sweep_results"
) -> Path:
    """Save individual image results to a timestamped JSON."""
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{prefix}_{timestamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_results_to_json(results, path)
    return path
def summarize_batch_results(
    all_image_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a summary of batch results containing only the prompt used,
    the generated prompt at the best bias strength, the best bias value,
    and the processing timestamp.
    """
    summary_results: Dict[str, Any] = {}
    for key, entry in all_image_results.items():
        best_bias = entry.get("best_bias_strength")
        gen_text = None
        if best_bias is not None and best_bias in entry.get("bias_sweep_results", {}):
            gen_text = entry["bias_sweep_results"][best_bias].get("generated_text")
        summary_results[key] = {
            "prompt_used": entry.get("prompt_used"),
            "generated_prompt": gen_text,
            "best_bias_strength": best_bias,
            "processing_timestamp": entry.get("processing_timestamp"),
        }
    return summary_results