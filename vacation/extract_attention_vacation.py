import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
import copy
import math
from typing import Dict, List, Optional, Any, Tuple, Union
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import PreTrainedModel, PreTrainedTokenizer
# Add project root to sys.path
try:
    project_root = Path(__file__).resolve().parent.parent
except NameError:
    project_root = Path.cwd().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
from gazefollow.extract_cls_token_image_similarity import draw_topk_similarity_overlay
from llava.conversation import conv_templates, SeparatorStyle
from generation_utils import (
    enable_inference_optimizations,
    load_model_and_setup,
    fix_wsl_paths,
    _prepare_configs,
    _setup_output_directories,
    _determine_image_patch_info,
    _extract_and_process_attention,
    create_similarity_visualization,
    calculate_correlation_metrics,
    update_generation_state,
    create_generation_results,
    initialize_generation_state,
    process_hidden_states_and_embeddings,
    load_image,
    save_mask_overlay_image,
    _mask_array_to_binary,
    # New gaze guidance functions
    select_token_by_gaze_correlation,
    generate_next_token_with_gaze_guidance,
    get_attention_indices_from_mask,
    process_images,
    calculate_coordinate_mapping,
    print_summary,
    analyze_bias_sweep_results,
    save_image_results,
    log_generation_step,
)

from generation_metrics import (
    ConfidenceMetrics, RepetitivityMetrics, TopKCandidateEvaluator,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN
)
from llava.mm_utils import tokenizer_image_token

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_PROMPT = """Complete the sentence in the following format, for example:
a tired guy in a hoodie → a guy in a gray hoodie and ripped jeans sitting on a worn wooden bench.  
a chic woman in a beige coat → a woman in a beige coat and ankle boots holding a phone.  
a techy man in a leather jacket → a man in a black leather jacket and glasses.  
a relaxed woman in a green sweater → a woman in a dark green sweater and black jeans carrying a tan shoulder bag.

The sentence: a _ → """

DEFAULT_TARGET_PROMPT = """Complete the sentence in the following format, for example:
a bulky backpack → a large black backpack with thick straps and a padded mesh back.
a glossy metal bottle → a tall stainless-steel bottle with a smooth reflective finish.
a warm croissant → a golden, flaky croissant with crisp layers and a soft buttery center.
a vintage camera → a compact silver-and-black camera with a textured grip and a chunky lens.
a loaded hotdog → a toasted bun filled with a browned sausage, topped with bright mustard and diced onions.
a cozy wool blanket → a thick cream-colored blanket with a soft woven pattern.

The sentence: an _ → """

def _prepare_inputs_vacation(
    image_path: Union[str, Path],
    target_bbox: List[int], # [xmin, ymin, xmax, ymax]
    prompt: str,
    image_processor: SigLipImageProcessor,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    insert_image_token: bool = True,
) -> Tuple[Any, Any, torch.Tensor, List, List[int], Any, torch.Tensor]:
    
    image = load_image(image_path)
    img_width, img_height = image.size
    
    # In this new mode, we use the bbox as the TARGET mask ("what to look at/describe").
    # We might not strictly need a "person_mask" (source) if we are just describing the entity.
    # However, GazeFollow logic often uses `person_mask` as query/bias.
    # If we want to describe the entity bounded by `target_bbox`, we should set `target_mask` = `target_bbox`.
    # And maybe `person_mask` = None? Or same?
    # Usually `target_mask` acts as the attention guidance.
    
    person_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    tx, ty, tx2, ty2 = target_bbox
    tx = max(0, min(tx, img_width))
    ty = max(0, min(ty, img_height))
    tx2 = max(0, min(tx2, img_width))
    ty2 = max(0, min(ty2, img_height))
    person_mask[ty:ty2, tx:tx2] = 1
        
    processed, patched_resized_before_pad_dim, patched_final_dim, patch_boxes = process_images([image], image_processor, model.config)
    
    if isinstance(processed, list):
        image_tensor = processed[0]
    else:
        image_tensor = processed
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(model.device, dtype=model.dtype)
    
    vision_tower = model.get_vision_tower() if hasattr(model, 'get_vision_tower') else None
    
    atten_indices = None
    if person_mask is not None:
        atten_indices, _ = get_attention_indices_from_mask(
            mask=person_mask,
            image_size=image.size,
            model_config=model.config,
            original_img_shape=image.size,
            patched_final_dim=patched_final_dim,
            patch_boxes=patch_boxes,
            patched_resized_before_pad_dim=patched_resized_before_pad_dim,
            vision_tower=vision_tower,
            apply_for_anyres_patches=False
        )
        
    person_mask_indices = None
    # No source mask indices
    # Prepare conversation
    conv_template = "qwen_1_5"  # Default for the model

    if insert_image_token and DEFAULT_IMAGE_TOKEN not in prompt:
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
    
    input_masks = {
        'person_mask': person_mask,
        'target_mask': person_mask,
        'person_mask_raw': person_mask,
        'target_mask_raw': person_mask
    }
    
    return image, input_masks, image_tensor, image_sizes, atten_indices, person_mask_indices, input_ids

def run_description_generation(
    image_path: str,
    bbox: List[int],
    prompt: str,
    output_dir: Union[str, Path],
    file_prefix: str,
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    image_processor: SigLipImageProcessor,
    generation_config: Optional[Dict] = None,
    attention_config: Optional[Dict] = None,
    save_mask_overlays: bool = False,
    bias_strength: float = 2.5,
    prev_run_last_hidden_state: Optional[torch.Tensor] = None,
    break_after_first_step: bool = False,
) -> Dict[str, Any]:
    
    gen_config, attn_config = _prepare_configs(generation_config, attention_config)
    
    output_directories = _setup_output_directories(output_dir)
    (out_path, vis_output_dir_raw, vis_output_dir_processed, 
     tensor_output_dir, collage_output_dir, similarity_output_dir) = output_directories
    
    (image, input_masks, image_tensor, image_sizes, atten_indices,
     person_mask_indices, input_ids) = _prepare_inputs_vacation(
        image_path,
        bbox,
        prompt,
        image_processor,
        tokenizer,
        model
    )
    
    if save_mask_overlays:
        mask_overlay_dir = Path(out_path) / "mask_overlays"
        save_mask_overlay_image(
            image=image,
            target_mask_raw=input_masks['target_mask_raw'],
            person_mask_raw=input_masks['person_mask_raw'],
            output_dir=mask_overlay_dir,
            filename_prefix=file_prefix, 
            alpha=0.4
        )
        
    boost_positions = {'gaze_source': atten_indices, 'gaze_target': None}
    
    num_patches, grid_size, image_token_start_index_in_llm, _ = _determine_image_patch_info(model, input_ids)
    
    state = initialize_generation_state(gen_config, tokenizer, input_ids)
    
    repr_layer_map = None
    repr_capture_layer_idx = None
    repr_inject_layer_idx = None
    if attn_config:
        repr_capture_layer_idx = attn_config.get("repr_capture_layer_idx", 20)
        repr_inject_layer_idx = attn_config.get("repr_inject_layer_idx", 0)
        repr_layer_map = attn_config.get("repr_layer_idx")

        fallback_layer = repr_layer_map
        if fallback_layer is None:
            fallback_layer = attn_config.get("layer_idx")
        if fallback_layer is None:
            fallback_layer = -1
        repr_layer_map = {
            "capture": repr_capture_layer_idx if repr_capture_layer_idx is not None else fallback_layer,
            "inject": repr_inject_layer_idx if repr_inject_layer_idx is not None else fallback_layer,
        }

    for i in range(state["max_new_tokens"]):
        with torch.inference_mode():
            model_inputs = {
                "input_ids": state["current_input_ids"],
                "past_key_values": state["past_key_values"],
                "use_cache": True,
                "output_attentions": True,
                "output_hidden_states": True,
                "atten_ids": None,
                "boost_positions": boost_positions,
                "include_image_inputs": True,
                "bias_strength": bias_strength,
                "target_mask_embedding": prev_run_last_hidden_state,
                "repr_layer_idx": repr_layer_map,
                "input_masks": input_masks,
                "base_image_token_inds": [image_token_start_index_in_llm, image_token_start_index_in_llm + num_patches],
            }
            if i == 0:
                model_inputs.update({"images": image_tensor, "image_sizes": image_sizes, "modalities": ["image"]})
                
            next_token_id, token_text, outputs, evaluation_metrics = generate_next_token_with_gaze_guidance(
                model_inputs, model, tokenizer, gen_config,
                state["confidence_tracker"], state["repetitivity_tracker"], 
                state["candidate_evaluator"], i,
                image_embeddings=state["image_embeddings"],
                target_mask=input_masks.get('target_mask')
            )
            
            processed_img, attention_map = _extract_and_process_attention(
                outputs, next_token_id, token_text, i, num_patches, grid_size, 
                image_token_start_index_in_llm, image, attn_config, 
                vis_output_dir_raw, vis_output_dir_processed, tensor_output_dir
            )

            state["all_step_metrics"].append(evaluation_metrics)

            # Process hidden states and extract embeddings
            _ = process_hidden_states_and_embeddings(
                outputs, attn_config, image_token_start_index_in_llm, 
                num_patches, model, state
            )
            update_generation_state(state, next_token_id, token_text, outputs, attention_map)
            
            if log_generation_step(i, token_text, evaluation_metrics, next_token_id, state["eos_token_id"]):
                break

            if break_after_first_step:
                print("Breaking after the first step as requested.")
                break
                
    results = create_generation_results(
        state, tokenizer, {"main": str(out_path)}, 
        gen_config, attn_config, None
    )
    
    # Save text
    if not break_after_first_step and "generated_text" in results:
        with open(Path(out_path) / "generated_text.txt", "w") as f:
            f.write(results["generated_text"])
            
    return results

def resolve_label(short_label: str) -> str:
    """Resolve P1 -> Person1, O1 -> Object1."""
    if short_label.startswith('P') and short_label[1:].isdigit():
        return f"Person{short_label[1:]}"
    if short_label.startswith('O') and short_label[1:].isdigit():
        return f"Object{short_label[1:]}"
    return short_label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default="vacation/test_annotations_complete.csv")
    parser.add_argument('--output_root', type=str, default="vacation/output")
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--limit_frames', type=int, default=None) # Limit number of frames
    parser.add_argument('--redo', action='store_true', help="Whether to redo existing outputs.")
    args = parser.parse_args()
    
    # Load DF
    if not os.path.exists(args.csv_path):
        print(f"Error: {args.csv_path} not found. Did you run parse_vacation_complete.py?")
        sys.exit(1)
        
    df = pd.read_csv(args.csv_path)
    
    # Load model
    tokenizer, model, image_processor, context_len = load_model_and_setup()
    
    # Group by frame
    frames_grouped = df.groupby(['video_id', 'frame_id'])
    
    frame_keys = list(frames_grouped.groups.keys())
    if args.limit_frames:
        frame_keys = frame_keys[:args.limit_frames]
        
    print(f"Processing {len(frame_keys)} frames...")
    
    for video_id, frame_id in tqdm(frame_keys):
        group = frames_grouped.get_group((video_id, frame_id))
        
        # 1. Map all entities in this frame
        entities = {}
        for _, row in group.iterrows():
            entities[row['bbx_label']] = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
            
        # Image path (assume same logic as before)
        # Note regarding frame_id offset: parser uses frame_id from file, which might match image name directly?
        # Parser script: `save_path = ... f"{frame_idx:06d}.png"`. 
        # And `frame_id` in annotation comes from `parts[5]`. 
        # Assuming frame_id in CSV matches the saved filename.
        # But wait, earlier I suspected offset. 
        # Let's try direct first. If 404, we can debug.
        # Actually parser script saves frames as `000001.png`, `000002.png` etc.
        # Check if frame_id in CSV is 1-based.
        # `NewAnt_217.txt`: frame_id starts at 0? `229 12 ... 0 ...` -> last col is 0.
        # Step 229 output shows frame_id 0, 1, 2...
        # So frame_id 0 corresponds to frame 1? 
        # `parse_vacation_test_frames` loop: frame_idx=1.
        # If annotations align with video start, then frame_id 0 is likely 000001.png.
        
        image_name = f"{int(frame_id) + 1:06d}.png" 
        image_path = Path("datasets/Vacation/frames") / str(video_id) / image_name
        full_image_path = project_root / image_path
        
        if not full_image_path.exists():
            # print(f"Image not found: {full_image_path}")
            continue
            
        processed_targets = set()
        
        # 2. Iterate Persons
        persons = group[group['bbx_label'].str.startswith('Person')]
        
        for _, row in persons.iterrows():
            person_label = row['bbx_label']
            person_bbox = entities[person_label]
            
            # Describe Person
            person_out_dir = Path(args.output_root) / f"{video_id}_{frame_id}" / f"{person_label}_description"
            if not person_out_dir.exists() or args.redo: # avoid re-doing if done? Or overwrite? User said "store results".
                 try:
                        # 1. Initial run to get hidden state
                        init_results = run_description_generation(
                            str(full_image_path),
                            person_bbox,
                            DEFAULT_PROMPT,
                            person_out_dir,
                            f"{person_label}",
                            model, tokenizer, image_processor,
                            generation_config={'max_new_tokens': 50, 'do_sample': False},
                            attention_config={'layer_idx': 23, 'return_full_hidden_states': True},
                            save_mask_overlays=True,
                            break_after_first_step=True
                        )
                        
                        first_step_hidden_state = init_results.get("first_step_hidden_state")

                        # 2. Main run with bias
                        run_description_generation(
                            str(full_image_path),
                            person_bbox,
                            DEFAULT_PROMPT,
                            person_out_dir,
                            f"{person_label}",
                            model, tokenizer, image_processor,
                            generation_config={'max_new_tokens': 50, 'do_sample': False},
                            attention_config={'layer_idx': 23},
                            save_mask_overlays=True,
                            bias_strength=2.5, # Or whatever bias needed
                            prev_run_last_hidden_state=first_step_hidden_state
                        )
                 except Exception as e:
                     print(f"Error describing {person_label} in {video_id}_{frame_id}: {e}")
            
            # Describe Target
            target_short_label = row['attention_focus']
            if pd.isna(target_short_label) or target_short_label in ['single', 'no_focus', 'NA']:
                continue
                
            target_label = resolve_label(target_short_label)
            
            # Only if target is Object (not a Person seen in this frame's entities list? or just starts with Object?)
            # Logic: "only if it's not already described as the people in step 1"
            # Since step 1 describes ALL persons in the frame, we should skip if target_label IS a person.
            if target_label.startswith('Person'):
                continue
                
            # If target is object, and exists in entities
            if target_label in entities:
                if target_label not in processed_targets:
                    output_label = target_label
                    target_out_dir = Path(args.output_root) / f"{video_id}_{frame_id}" / f"{output_label}_description"
                    
                    try:
                        # 1. Initial run
                        init_results = run_description_generation(
                            str(full_image_path),
                            entities[target_label],
                            DEFAULT_TARGET_PROMPT,
                            target_out_dir,
                            f"{output_label}",
                            model, tokenizer, image_processor,
                            generation_config={'max_new_tokens': 20, 'do_sample': False},
                            attention_config={'layer_idx': 23, 'return_full_hidden_states': True},
                            save_mask_overlays=True,
                            break_after_first_step=True
                        )

                        first_step_hidden_state = init_results.get("first_step_hidden_state")

                        # 2. Main run
                        run_description_generation(
                            str(full_image_path),
                            entities[target_label],
                            DEFAULT_TARGET_PROMPT,
                            target_out_dir,
                            f"{output_label}",
                            model, tokenizer, image_processor,
                            generation_config={'max_new_tokens': 20, 'do_sample': False},
                            attention_config={'layer_idx': 23},
                            save_mask_overlays=True,
                            bias_strength=2.5,
                            prev_run_last_hidden_state=first_step_hidden_state
                        )
                        processed_targets.add(target_label)
                    except Exception as e:
                        print(f"Error describing {output_label} in {video_id}_{frame_id}: {e}")
