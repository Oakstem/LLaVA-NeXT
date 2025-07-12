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

# Todo: 1. fix the focus ratio so it won't get results such as '2267577.26x'
# 2. idea - instead of trying to make the LM generate some text description based on our attention mask. lets just use the encoding vector of that region (we can average over several patches in the mask):
# eg: 'the person is looking at the <insert here the embedded vector of the mask region>'
# we're getting:     1. 'ceiling' (p=0.532)
                    # 2. 'hands' (p=0.175)
                    # 3. 'sky' (p=0.140)
# so we're very close to the correct answer (even though 'ceiling' is not that wrong)

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
import traceback

# Import our enhanced generation metrics
from generation_metrics import (
    ConfidenceMetrics, RepetitivityMetrics, TopKCandidateEvaluator,
    generate_next_token_with_evaluation, create_generation_summary,
    analyze_generation_quality
)

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
    threshold_value: float = 0.6
) -> None:
    """
    Calculates and visualizes the semantic similarity between a text token's embedding
    and all image patch embeddings.

    Args:
        text_token_embedding: The feature embedding of the generated text token. Shape: [hidden_dim].
        image_token_embeddings: The feature embeddings of all image patches. Shape: [num_patches, hidden_dim].
        original_image: The original PIL image for the background.
        grid_size: The dimension of the square patch grid (e.g., 27 for a 27x27 grid).
        output_path: The path to save the visualization.
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
    # threshold the similarity map to remove low values
    similarity_map = np.where(similarity_map >= threshold_value, similarity_map, 0)

    # Resize heatmap to image size for overlay
    map_img = Image.fromarray(similarity_map.astype(np.float32))
    resized_map_img = map_img.resize(original_image.size, Image.Resampling.LANCZOS)
    
    heatmap_colors = cm.inferno(np.array(resized_map_img))[:, :, :3]
    heatmap_uint8 = (heatmap_colors * 255).astype(np.uint8)

    # 4. Blend with original image and save
    overlay_img = Image.blend(original_image, Image.fromarray(heatmap_uint8), alpha=0.6)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_img.save(output_path)

    return similarity_map


# %%
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
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {output_path}")

# %%
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
        "save_tensors": True
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
) -> Tuple[Any, Any, torch.Tensor, List, List[int], Any]:
    """Load and prepare image, mask, and input tensors."""
    person_mask = None
    person_mask_indices = None
    # Load image and mask
    image = load_image(image_path)
    mask = load_mask_from_file(mask_path)
    person_mask_path = mask_path.replace("gaze__", "person__")
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
    atten_indices, resized_mask = get_attention_indices_from_mask(mask, image.size, model.config)
    if person_mask is not None:
        person_mask_indices, _ = get_attention_indices_from_mask(person_mask, image.size, model.config)
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
    return image, resized_mask, image_tensor, image_sizes, atten_indices, person_mask_indices, input_ids

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
    mask_path = fix_wsl_paths(str(mask_path))
    image_path = fix_wsl_paths(str(image_path))
    mask_embedding_path = mask_path.replace("masks.npy", "_target_embeddings.pt")

    if Path(mask_embedding_path).exists():
        target_mask_embedding = torch.load(mask_embedding_path)
        print(f"Using existing mask embedding from {mask_embedding_path}")
    else:
        target_mask_embedding = None

    print(f"Processing image: {image_path}")
    print(f"Using mask: {mask_path}")
    print(f"Prompt: {prompt}")

    # Setup output directories
    output_dir, vis_output_dir_raw, vis_output_dir_processed, tensor_output_dir, collage_output_dir, similarity_output_dir = _setup_output_directories(output_dir)

    # Load and prepare inputs
    image, resized_mask, image_tensor, image_sizes, atten_indices, person_mask_indices, input_ids = \
    _prepare_inputs(image_path, mask_path, prompt, image_processor, tokenizer, model)
    # temp boost coordinates
    boost_positions = {'gaze_source': person_mask_indices, 'gaze_target': atten_indices}
    # Determine image patch information
    num_patches, grid_size, image_token_start_index_in_llm, image_token_end_index_in_llm = _determine_image_patch_info(model, input_ids)

    # Enhanced generation loop with evaluation metrics
    print("Starting generation with attention extraction and advanced evaluation...")
    max_new_tokens = gen_config["max_new_tokens"]
    generated_ids = []
    past_key_values = None
    current_input_ids = input_ids
    collected_maps = []  # For collage

    # Initialize evaluation trackers
    confidence_tracker = ConfidenceMetrics()
    repetitivity_tracker = RepetitivityMetrics(window_size=10)
    candidate_evaluator = TopKCandidateEvaluator(k=5, tokenizer=tokenizer)
    all_step_metrics = []
    all_attention_maps = [] # To collect attention maps for correlation

    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]

    image_embeddings = None
    all_correlation_metrics = []
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
                "query_indices": attn_config.get("query_indices", None),
                "target_mask_embedding": target_mask_embedding,
            }

            # Add image data only in first step
            if i == 0:
                model_inputs["images"] = image_tensor
                model_inputs["image_sizes"] = image_sizes
                model_inputs["modalities"] = ["image"]

            # Enhanced generation with evaluation metrics
            try:
                next_token_id, token_text, outputs, evaluation_metrics = generate_next_token_with_evaluation(
                    model_inputs, model, tokenizer, gen_config,
                    confidence_tracker, repetitivity_tracker, candidate_evaluator, i
                )
                all_step_metrics.append(evaluation_metrics)
                
                # Print step details with metrics
                conf_score = evaluation_metrics["confidence"].get("confidence_score", 0)
                entropy = evaluation_metrics["confidence"].get("entropy", 0)
                print(f"Step {i}: '{token_text}' | Confidence: {conf_score:.3f} | Entropy: {entropy:.3f}")
                
                # Show top candidates for interesting steps (low confidence)
                if conf_score < 0.8 and len(evaluation_metrics["top_k_analysis"]["candidates"]) >= 3:
                    print(f"  Top alternatives:")
                    for j, candidate in enumerate(evaluation_metrics["top_k_analysis"]["candidates"][:3]):
                        print(f"    {j+1}. '{candidate['token_text']}' (p={candidate['probability']:.3f})")
                
            except Exception as e:
                print(f"Enhanced generation failed, falling back to standard generation: {e}")
                # Fallback to original generation method
                next_token_id, token_text, outputs = _generate_next_token(model_inputs, model, tokenizer, gen_config)

            # Check for EOS
            if next_token_id.item() == eos_token_id:
                print("EOS token generated. Stopping.")
                break

            generated_ids.append(next_token_id.item())

            # --- Semantic Similarity Visualization ---
            if outputs.hidden_states:
                last_hidden_state = outputs.hidden_states[-1].squeeze(0) # Shape: [Seq_Len, Hidden_Dim]
                text_embedding = last_hidden_state[-1] # Embedding of the new token
                if image_embeddings is None:
                    # store the image embeddings that exist only at the first step
                    image_embeddings = last_hidden_state[image_token_start_index_in_llm : image_token_start_index_in_llm + num_patches]

                safe_token_text = "".join(c if c.isalnum() else "_" for c in token_text)
                if not safe_token_text:
                    safe_token_text = f"tokenid_{next_token_id.item()}"
                
                sim_path = similarity_output_dir / f"similarity_{i:03d}_{safe_token_text}.png"
                similarity_map = visualize_embedding_similarity(
                    text_token_embedding=text_embedding,
                    image_token_embeddings=image_embeddings,
                    original_image=image,
                    grid_size=grid_size,
                    output_path=sim_path
                )

                # lets extract the image embeddings for the target resized_mask
                # mask_indices = np.where(resized_mask.flatten() > 0)[0]
                # if i==0 and len(mask_indices) > 0:
                #     mask_embeddings = image_embeddings[mask_indices, :].mean(dim=0)
                #     # save the mask embeddings
                #     torch.save(mask_embeddings, mask_embedding_path)
                #     print(f"Saved target mask embeddings to {mask_embedding_path}")



            # Extract and process attention
            processed_img, attention_map = _extract_and_process_attention(
                outputs, next_token_id, token_text, i,
                num_patches, grid_size, image_token_start_index_in_llm,
                image, attn_config, vis_output_dir_raw, vis_output_dir_processed, tensor_output_dir
            )
            if attention_map is not None:
                all_attention_maps.append(attention_map)

            # Collect for collage
            if attn_config["create_collage"] and processed_img is not None:
                collected_maps.append((processed_img, token_text))

            # Print current result
            if i == 0:
                txt_ids = torch.argmax(outputs.logits, dim=-1)[0]
                txt_np = np.array([tokenizer.decode(val).strip() for val in txt_ids])
                resulted_description = ",".join(txt_np[boost_positions['gaze_target']])
                resulted_source_description = ",".join(txt_np[boost_positions['gaze_source']]) if boost_positions['gaze_source'] is not None else "N/A"
                print(f"Resulted words from gaze target mask: {resulted_description}")
                print(f"Resulted words from gaze source mask: {resulted_source_description}")

            # Update for next iteration
            if next_token_id.dim() == 0:
                # Scalar tensor - convert to proper batch format
                current_input_ids = next_token_id.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1]
            else:
                # Already has some dimensions - ensure it's [batch_size, 1]
                current_input_ids = next_token_id.view(1, -1)
            past_key_values = outputs.past_key_values

            # Update attention indices
            if i == 0:
                initial_token_length = past_key_values[0][0].shape[2]
            input_token_length = past_key_values[0][0].shape[2]

            # Calculate attention correlation metric
            if all_attention_maps and len(atten_indices) > 0:
                attention_correlation = calculate_attention_correlation_from_similarity(
                    text_to_image_similarity_matrix=similarity_map,
                    attention_mask=resized_mask,
                )
                all_correlation_metrics.append(attention_correlation)

    # Generate final text
    final_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Generate comprehensive evaluation summary
    generation_summary = None
    quality_analysis = None
    if all_step_metrics:
        try:
            generation_summary = create_generation_summary(
                confidence_tracker, repetitivity_tracker, candidate_evaluator,
                final_text, all_step_metrics, all_correlation_metrics,
            )
            quality_analysis = analyze_generation_quality(generation_summary)
        except Exception as e:
            print(f"Error creating generation summary: {e}")

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
                print(f"  Step {step}: Low confidence ({prob:.3f})")
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


    return {
        "generated_text": final_text,
        "generated_tokens": generated_ids,
        "num_tokens": len(generated_ids),
        "output_directories": {
            "main": str(output_dir),
            "raw_attention": str(vis_output_dir_raw) if attn_config["visualize_attn_overlays"] else None,
            "processed_attention": str(vis_output_dir_processed) if attn_config["visualize_attn_overlays"] else None,
            "embedding_similarity": str(similarity_output_dir),
            "tensors": str(tensor_output_dir) if attn_config["save_tensors"] else None,
            "collages": str(collage_output_dir) if attn_config["create_collage"] and collected_maps else None
        },
        "config_used": {
            "generation": gen_config,
            "attention": attn_config
        },
        "evaluation_summary": generation_summary,
        "quality_analysis": quality_analysis,
        "attention_correlation": attention_correlation,
        "step_metrics": all_step_metrics
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
        print(f"{'='*60}")
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
        print(f"{'='*80}")
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
            traceback.print_exc()
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

    print(f"{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print('='*80)
    print(f"Processed {len(results)} items")
    print(f"Results saved to: {results_path}")

    successful = sum(1 for r in results.values() if "error" not in r.get("generation_result", {}))
    failed = len(results) - successful
    print(f"Successful: {successful}, Failed: {failed}")

    return results

# %%
# Additional Helper Functions for Enhanced Evaluation Analysis

# Analysis functions have been moved to generation_metrics.py
# Import them from there:
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from generation_metrics import print_detailed_step_analysis, compare_generation_configs, analyze_top_k_impact, calculate_attention_correlation_from_similarity

# %%
# Interactive Configuration and Usage
# This is the main cell you'll modify for different experiments

# Configuration for current run
all_results = {}
# for bias_i in np.linspace(2.5, 3.5, 5):     # good for hands detection
for bias_i in np.linspace(0., 5., 10):     
    print(f"{'='*60}")
    print(f"Running experiment with bias strength: {bias_i:.2f}")
    EXPERIMENT_CONFIG = {
        # Input paths (MODIFY THESE)
        "image_path": r"D:\Projects\data\gazefollow\train\00000000\00000022.jpg",  # Your image path
        # "mask_path": r"D:\Projects\data\gazefollow\train_gaze_segmentations\masks\gaze__00000318_masks.npy",  # Your mask path
        "mask_path": r"D:\Projects\data\gazefollow\train_gaze_segmentations\manual_masks\gaze__00000022_masks.npy",  # Your mask path
        "target_mask_embedding_path": r"D:\Projects\data\gazefollow\train_gaze_segmentations\masks\gaze__00000022_target_embeddings.pt",  # Your target mask embedding path
        # "mask_path": r"D:\Projects\data\gazefollow\train_gaze_segmentations\masks\gaze__00000022_masks.npy",  # Your mask path
        # "prompt": "Complete the sentence. The tattooed man at wedding reception is looking at",  # Your prompt
        # "prompt": "Complete the sentence in a few words answer. The object is",  # Your prompt
        # "prompt": "Complete the sentence. Start your answer with 'The object in focus is'",  # Your prompt
        "prompt": "Complete the sentence. The person is looking at _ which is",  # Your prompt
        # "prompt": "Describe the image in 3 words.",  # Your prompt
        # "prompt": "Start the sentence with 'the object is'",  # Your prompt
        "output_dir": f"attention_output/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}_bias_{bias_i:.2f}",  # Output directory with timestamp and bias value

        # Generation configuration
        "generation_config": {
            "bias_strength": bias_i,  # Set to 0.0 for no bias, or adjust as needed
            "max_new_tokens": 50,
            "temperature": 0.1,
            "do_sample": True,
            "top_k": 50,  # Set to None (disabled) or int > 0 (e.g., 50 for top-50 sampling)
            "output_hidden_states": True, # Ensure hidden states are output
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
            "save_tensors": True,
            "query_indices": {"gaze_source": [-5, -3], # taking the person indices from the end of the prompt, using range format
                              "gaze_target": [-3, 0]  # taking the attention indices from the end of the prompt
                            },
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
    print("" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)
    print(f"Generated Text: {results['generated_text']}")
    print(f"Number of tokens: {results['num_tokens']}")
    print("Output Directories:")
    for key, path in results['output_directories'].items():
        if path:
            print(f"  {key}: {path}")
else:
    print(f"❌ Experiment failed: {results['error']}")

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
        "bias_strength": 2.78,        "max_new_tokens": 50,
        "temperature": 0.0,
        "do_sample": False,
        "top_k": None,
        "output_hidden_states": True, # Ensure hidden states are output
    },
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
        "save_tensors": True,
        "query_indices": {"gaze_source": [-5, -3], # taking the person indices from the end of the prompt, using range format
                          "gaze_target": [-3, 0]  # taking the attention indices from the end of the prompt
                            },
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
print("5. Use helper functions for detailed analysis: print_detailed_step_analysis(results)")
print("6. Compare different configurations: compare_generation_configs(config_a, config_b, ..., run_generation_fn=run_generation_with_attention)")
print("7. Analyze top-k sampling impact: analyze_top_k_impact(image_path, mask_path, prompt, run_generation_fn=run_generation_with_attention)")
print("\\nEnhanced Features Available:")
print("- Advanced generation metrics with confidence, entropy, and logit gap analysis")
print("- Repetitivity and diversity tracking (type-token ratio, n-gram repetition)")
print("- Top-k candidate analysis at each generation step")
print("- Quality scoring and interpretation (excellent/good/fair/poor)")
print("- Decision point identification (low-confidence steps)")
print("- Generation trend analysis (confidence/entropy over time)")
print("- Comprehensive step-by-step evaluation reports")
print("\\nTop-K Sampling:")
print("  * Set 'top_k': None to disable (default)")
print("  * Set 'top_k': 50 for top-50 sampling (recommended values: 10-100)")
print("  * Works with temperature and attention biasing")
print("  * Use analyze_top_k_impact() to find optimal values")
print("\\nEvaluation Metrics:")
print("  * Confidence: entropy, logit gap, confidence score")
print("  * Diversity: type-token ratio, n-gram repetition, unique token ratio")
print("  * Quality: Overall score (0-10) with interpretation")
print("  * Decision Points: Steps where model had low confidence")

# %%
# Example Usage Instructions
"""
## How to use the enhanced generation and evaluation system:

1. **Basic usage with enhanced evaluation:**
   ```python
   results = run_generation_with_attention(
       image_path="path/to/image.jpg",
       mask_path="path/to/mask.npy", 
       prompt="Complete the sentence. The person is looking at",
       output_dir="output_dir",
       generation_config={"top_k": 50, "temperature": 0.8, "do_sample": True},
       attention_config=None  # Use defaults
   )
   
   # Results now include:
   # - evaluation_summary: Comprehensive generation analysis
   # - quality_analysis: Overall quality score and breakdown
   # - step_metrics: Detailed metrics for each generation step
   ```

2. **Detailed step analysis:**
   ```python
   # Print detailed step-by-step analysis
   print_detailed_step_analysis(results, max_steps=10)
   
   # Access specific metrics
   quality_score = results["quality_analysis"]["overall_quality_score"]
   confidence_trend = results["evaluation_summary"]["confidence_trajectory"]["trend"]
   decision_points = results["evaluation_summary"]["decision_points"]
   ```

3. **Compare different configurations:**
   ```python
   config_a = {"top_k": 50, "temperature": 0.8, "do_sample": True}
   config_b = {"top_k": None, "temperature": 0.5, "do_sample": True}
   
   comparison = compare_generation_configs(
       config_a, config_b, image_path, mask_path, prompt,
       run_generation_fn=run_generation_with_attention
   )
   ```

4. **Analyze top-k impact:**
   ```python
   # Test different top-k values automatically
   top_k_analysis = analyze_top_k_impact(
       image_path, mask_path, prompt, 
       top_k_values=[None, 10, 50, 100],
       run_generation_fn=run_generation_with_attention
   )
   ```

5. **Evaluation metrics available:**
   - **Confidence Metrics:**
     * entropy: Shannon entropy of probability distribution
     * logit_gap: Gap between top-2 logits (higher = more confident)
     * confidence_score: Composite confidence measure
   
   - **Repetitivity Metrics:**
     * type_token_ratio: Unique tokens / total tokens (higher = more diverse)
     * bigram_repetition: Ratio of repeated 2-grams
     * trigram_repetition: Ratio of repeated 3-grams
   
   - **Top-K Analysis:**
     * Top candidate probabilities and alternatives at each step
     * Decision points where model had low confidence
     * Probability mass distribution analysis

6. **Quality interpretation:**
   - **Excellent (8.0-10.0):** High confidence, good diversity, minimal repetition
   - **Good (6.0-8.0):** Generally confident with acceptable diversity
   - **Fair (4.0-6.0):** Mixed confidence or some repetition issues
   - **Poor (0.0-4.0):** Low confidence or significant repetition problems

7. **JSON-based batch processing:**
   - Modify the JSON_BATCH_CONFIG above
   - Set limit_items to a small number for testing (e.g., 5)
   - Set filter_keys to specific image keys if needed
   - Run the JSON batch processing cell
   - All enhanced metrics are automatically included in batch results

8. **Output structure:**
   The results JSON will have the same keys as the input, but with additional fields:
   - generation_result: Contains the generated text and all evaluation metadata
   - evaluation_summary: Comprehensive analysis including quality metrics
   - quality_analysis: Overall quality score and factor breakdown
   - step_metrics: Detailed metrics for each generation step
   - prompt_used: The actual prompt that was used
   - processing_timestamp: When the item was processed

9. **File naming conventions:**
   - Images: {base_image_dir}/{folder}/{filename}.jpg
   - Masks: {base_mask_dir}/gaze__{folder}_{filename}_masks.npy
   - Example: "00000000_00000021" → image: "00000000/00000021.jpg", mask: "gaze__00000000_00000021_masks.npy"

10. **Advanced usage tips:**
    - Use top_k=50 for balanced quality vs diversity
    - Use top_k=10 for more focused generation
    - Use top_k=None (disabled) for maximum diversity
    - Monitor decision_points to identify uncertainty in generation
    - Use confidence_trend to understand generation stability
    - Compare quality_scores across different configurations
"""
