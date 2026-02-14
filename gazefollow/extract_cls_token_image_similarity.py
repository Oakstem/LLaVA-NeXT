#!/usr/bin/env python3
"""Extract CLS hidden states and compare them against image embeddings from LLaVA."""

import argparse
import copy
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generation_utils import (  # noqa: E402
    enable_inference_optimizations,
    load_model_and_setup,
    load_image,
    fix_wsl_paths,
)
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare per-layer CLS embeddings with image embeddings obtained just after "
            "the LLaVA mm_projector."
        )
    )
    parser.add_argument(
        "--model-path",
        default="lmms-lab/llava-onevision-qwen2-7b-ov-chat",
        help="Path or hub ID of the checkpoint to load.",
    )
    parser.add_argument(
        "--model-base",
        default=None,
        help="Optional base checkpoint when loading a LoRA adapter.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional LoRA adapter weights to merge at inference.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Attention implementation hint forwarded to the model loader.",
    )
    parser.add_argument("--load-4bit", action="store_true", help="Load the model in 4-bit mode.")
    parser.add_argument("--load-8bit", action="store_true", help="Load the model in 8-bit mode.")
    parser.add_argument(
        "--prompt",
        # required=True,
        default="a woman in a white dress and sunglasses walking down the street.",
        # default="a woman standing near a stroller.",
        help="Natural language description used to pull CLS token hidden states.",
    )
    parser.add_argument(
        "--image-path",
        # required=True,
        default=r"D:\Projects\data\gazefollow\train\00000093\00093143.jpg",
        help="Path to the image that will be embedded and compared against CLS embeddings.",
    )
    parser.add_argument(
        "--output-dir",
        # required=True,
        default="./cls_image_similarity_outputs",
        help="Directory where all overlays and the summary JSON file will be written.",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Optional cap on the number of hidden-state layers to visualize.",
    )
    parser.add_argument(
        "--disable-optimizations",
        action="store_true",
        help="Skip enabling CUDA inference optimizations.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top-scoring image tokens per layer to visualize (must be >= 1).",
    )
    return parser.parse_args()


def resolve_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device"):
        return model.device  # type: ignore[return-value]
    return next(model.parameters()).device


def resolve_dtype(model: torch.nn.Module) -> torch.dtype:
    if hasattr(model, "dtype"):
        return model.dtype  # type: ignore[return-value]
    return next(model.parameters()).dtype


def ensure_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Prompt must be non-empty.")
    if DEFAULT_IMAGE_TOKEN in prompt:
        raise ValueError("Prompt must not contain image tokens (<image>) for CLS extraction.")
    return prompt


def prepare_multimodal_image_tensor(
    pil_image: Image.Image,
    image_processor,
    model,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Prepare the image tensor used for multimodal LLaVA forward passes."""
    processed = process_images([pil_image.copy()], image_processor, model.config)
    if isinstance(processed, tuple):
        image_tensor = processed[0]
    else:
        image_tensor = processed

    if isinstance(image_tensor, list):
        if not image_tensor:
            raise ValueError("Image processor returned an empty tensor list.")
        image_tensor = image_tensor[0]

    if isinstance(image_tensor, torch.Tensor) and image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError(f"Unexpected image tensor type: {type(image_tensor)}")

    device = resolve_device(model)
    dtype = resolve_dtype(model)
    image_tensor = image_tensor.to(device=device, dtype=dtype)
    return image_tensor, pil_image.size


def extract_cls_hidden_states(
    prompt: str,
    tokenizer,
    model,
) -> Dict[str, torch.Tensor]:
    conv_template = "qwen_1_5"

    # if insert_image_token and DEFAULT_IMAGE_TOKEN not in prompt:
    # full_prompt = f"{DEFAULT_IMAGE_TOKEN}\\n{prompt}"
    full_prompt = prompt

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], full_prompt)
    conv.append_message(conv.roles[1], None)        # 5 tokens are always added to the end of the user prompt
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_question, tokenizer,
        IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)


    # input_ids = tokenizer_image_token(
    #     prompt_text,
    #     tokenizer,
    #     IMAGE_TOKEN_INDEX,
    #     return_tensors="pt",
    # ).unsqueeze(0).to(resolve_device(model))

    attention_mask = torch.ones(
        input_ids.shape,
        dtype=torch.long,
        device=input_ids.device,
    )

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    if not outputs.hidden_states:
        raise RuntimeError("Model did not return hidden states. Enable output_hidden_states in config.")

    # cls_position = int(input_ids.shape[1] - 1)
    cls_position = int(input_ids.shape[1] - 6)
    cls_token_id = int(input_ids[0, cls_position].item())
    cls_token_str = tokenizer.decode([cls_token_id]).strip() or "<last-token>"
    cls_embeddings: List[torch.Tensor] = []
    for layer_hidden in outputs.hidden_states:
        cls_embeddings.append(layer_hidden[0, cls_position, :])

    return {
        "embeddings": cls_embeddings,
        "cls_position": cls_position,
        "cls_token_id": cls_token_id,
        "cls_token_str": cls_token_str,
    }


def prepare_image_embeddings(
    pil_image: Image.Image,
    image_processor,
    model,
) -> Dict[str, torch.Tensor]:
    pixel_values = image_processor.preprocess(pil_image.copy(), return_tensors="pt")["pixel_values"]
    if pixel_values.ndim == 3:
        pixel_values = pixel_values.unsqueeze(0)

    device = resolve_device(model)
    dtype = resolve_dtype(model)
    pixel_values = pixel_values.to(device=device, dtype=dtype)

    with torch.inference_mode():
        base_model = model.get_model()
        vision_tower = base_model.get_vision_tower()
        if vision_tower is None:
            raise RuntimeError("Loaded model does not expose a vision tower.")
        raw_features = vision_tower(pixel_values)
        if raw_features.ndim == 4:
            raw_features = raw_features.flatten(1, 2)
        post_projector = base_model.mm_projector(raw_features)

    def _squeeze_batch(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.squeeze(0) if tensor.dim() >= 3 and tensor.size(0) == 1 else tensor

    pre_proj = _squeeze_batch(raw_features)
    post_proj = _squeeze_batch(post_projector)

    return {
        "pre_projector": pre_proj,
        "post_projector": post_proj,
    }


def extract_decoder_image_hidden_states(
    tokenizer,
    model,
    image_tensor: torch.Tensor,
    image_size: Tuple[int, int],
) -> List[torch.Tensor]:
    """Run the full LLaVA model to capture decoder hidden states for image tokens."""
    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()
    input_ids = tokenizer_image_token(
        prompt_text,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(resolve_device(model))

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            images=image_tensor,
            image_sizes=[list(image_size)],
            modalities=["image"],
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            include_image_inputs=True,
        )

    hidden_states = outputs.hidden_states or ()
    if not hidden_states:
        raise RuntimeError("Decoder hidden states were not returned for the image pass.")

    base_model = model.get_model()
    vision_tower = base_model.get_vision_tower()
    num_patches = 729
    if vision_tower is not None and hasattr(vision_tower, "num_patches"):
        num_patches = int(vision_tower.num_patches)

    tokens_indexing = getattr(model, "tokens_indexing", None)
    if not isinstance(tokens_indexing, dict):
        raise RuntimeError("tokens_indexing metadata missing after multimodal forward pass.")
    image_index_list = tokens_indexing.get("image")
    if not image_index_list:
        raise RuntimeError("No image token indices found in tokens_indexing.")
    image_indices = image_index_list[0]
    if not torch.is_tensor(image_indices):
        image_indices = torch.as_tensor(image_indices, device=hidden_states[0].device)

    image_indices = image_indices.to(hidden_states[0].device)
    if image_indices.numel() < num_patches:
        raise RuntimeError(
            f"Expected at least {num_patches} image tokens but found {image_indices.numel()}."
        )
    selected_indices = image_indices[:num_patches].long()

    decoder_embeddings: List[torch.Tensor] = []
    for layer_hidden in hidden_states:
        layer_embeddings = layer_hidden[0, selected_indices, :].detach().clone()
        decoder_embeddings.append(layer_embeddings)
    return decoder_embeddings


def create_output_dirs(base_dir: Path) -> Tuple[Path, Path, str]:
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"cls_image_similarity_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"cls_image_similarity_{timestamp}_{suffix:02d}"
        suffix += 1

    overlay_dir = run_dir / "overlays"
    run_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, overlay_dir, timestamp


def draw_topk_similarity_overlay(
    image: Image.Image,
    grid_size: int,
    top_entries: Sequence[Dict[str, object]],
    output_path: Path,
) -> None:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    patch_w = image.width / grid_size
    patch_h = image.height / grid_size
    radius = max(6, int(0.30 * min(patch_w, patch_h)))

    color_palette = ["red", "orange", "yellow", "lime", "cyan", "magenta"]
    try:
        font = ImageFont.load_default()
        line_height = font.size + 2
    except OSError:
        font = None
        line_height = 14

    for entry in top_entries:
        grid_row = int(entry["grid_row"])
        grid_col = int(entry["grid_col"])
        score = float(entry["similarity"])
        rank = int(entry["rank"])

        center_x = (grid_col + 0.5) * patch_w
        center_y = (grid_row + 0.5) * patch_h

        bbox = [
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        ]
        color = color_palette[min(rank - 1, len(color_palette) - 1)]
        draw.ellipse(bbox, outline=color, width=max(2, radius // 3))

        label = f"#{rank} {score:.3f}"
        text_pos = (center_x + radius + 4, center_y - radius + (rank - 1) * line_height)
        draw.text(text_pos, label, fill=color, font=font, stroke_width=2, stroke_fill="black")

    overlay.save(output_path)


def compute_similarity_results(
    cls_embeddings: List[torch.Tensor],
    image_embeddings: Union[torch.Tensor, Sequence[torch.Tensor]],
    base_image: Image.Image,
    overlay_dir: Path,
    overlay_rel_dir: str,
    grid_size: Optional[int],
    top_k: int,
    max_layers: Optional[int] = None,
) -> Dict[str, object]:
    overlay_dir.mkdir(parents=True, exist_ok=True)

    if not cls_embeddings:
        return {"layers": [], "grid_size": grid_size}

    per_layer_embeddings: List[torch.Tensor]
    if isinstance(image_embeddings, torch.Tensor):
        per_layer_embeddings = [image_embeddings]
    else:
        per_layer_embeddings = list(image_embeddings)
    if not per_layer_embeddings:
        return {"layers": [], "grid_size": grid_size}

    limit = len(per_layer_embeddings)
    if max_layers is not None:
        limit = min(limit, max_layers)

    layer_results: List[Dict[str, object]] = []
    recorded_shapes: List[List[int]] = []

    def layer_label_from_index(index: int) -> str:
        return "embedding" if index == 0 else f"layer_{index - 1}"

    for image_layer_idx in range(limit):
        current_embeddings = per_layer_embeddings[image_layer_idx]
        if current_embeddings.dim() == 1:
            flattened = current_embeddings.unsqueeze(0)
        elif current_embeddings.dim() == 2:
            flattened = current_embeddings
        else:
            flattened = current_embeddings.reshape(-1, current_embeddings.shape[-1])

        recorded_shapes.append(list(current_embeddings.shape))

        device = flattened.device
        working_embeddings = flattened.to(dtype=torch.float32)
        normalized_image_embeddings = F.normalize(working_embeddings, p=2, dim=-1)

        current_grid = grid_size or int(math.sqrt(flattened.shape[0]))
        if current_grid * current_grid < flattened.shape[0]:
            current_grid = int(math.ceil(math.sqrt(flattened.shape[0])))

        for cls_layer_idx, cls_embedding in enumerate(cls_embeddings):
            cls_vector = cls_embedding.to(device, dtype=torch.float32)
            cls_vector = F.normalize(cls_vector, p=2, dim=-1)
            if normalized_image_embeddings.shape[-1] != cls_vector.shape[-1]:
                continue
            similarities = torch.matmul(normalized_image_embeddings, cls_vector).clamp(min=-1.0, max=1.0)
            usable_k = max(1, min(int(top_k), similarities.shape[0]))
            top_scores, top_indices = torch.topk(similarities, k=usable_k, dim=0)

            top_entries: List[Dict[str, object]] = []
            for rank, (score_tensor, idx_tensor) in enumerate(zip(top_scores, top_indices)):
                score = float(score_tensor.item())
                token_idx = int(idx_tensor.item())
                grid_row = token_idx // current_grid
                grid_col = token_idx % current_grid
                top_entries.append(
                    {
                        "rank": rank + 1,
                        "similarity": score,
                        "token_index": token_idx,
                        "grid_row": grid_row,
                        "grid_col": grid_col,
                        "grid_size": current_grid,
                    }
                )

            if not top_entries:
                continue

            primary_entry = top_entries[0]
            overlay_path = (
                overlay_dir / f"image_{image_layer_idx:02d}_cls_{cls_layer_idx:02d}_topk.png"
            )
            draw_topk_similarity_overlay(
                image=base_image,
                grid_size=current_grid,
                top_entries=top_entries,
                output_path=overlay_path,
            )
            overlay_rel_path = str(Path(overlay_rel_dir) / overlay_path.name)
            for entry in top_entries:
                entry["overlay_path"] = overlay_rel_path

            layer_results.append(
                {
                    "layer_index": image_layer_idx,
                    "layer_label": layer_label_from_index(image_layer_idx),
                    "cls_layer_index": cls_layer_idx,
                    "cls_layer_label": layer_label_from_index(cls_layer_idx),
                    "max_similarity": primary_entry["similarity"],
                    "best_token_index": primary_entry["token_index"],
                    "grid_row": primary_entry["grid_row"],
                    "grid_col": primary_entry["grid_col"],
                    "grid_size": current_grid,
                    "overlay_path": overlay_rel_path,
                    "top_similarities": top_entries,
                }
            )

    result: Dict[str, object] = {
        "layers": layer_results,
        "grid_size": grid_size,
    }
    if recorded_shapes:
        result["embedding_shape"] = recorded_shapes[0]
        if any(shape != recorded_shapes[0] for shape in recorded_shapes[1:]):
            result["per_layer_shapes"] = recorded_shapes

    return result


def save_results_json(
    output_path: Path,
    prompt: str,
    cls_info: Dict[str, object],
    model_path: str,
    image_path: str,
    timestamp: str,
    similarity_results: Dict[str, Dict[str, object]],
) -> None:
    payload = {
        "prompt": prompt,
        "model_path": model_path,
        "image_path": str(Path(image_path)),
        "timestamp": timestamp,
        "cls_token": {
            "id": cls_info["cls_token_id"],
            "label": cls_info["cls_token_str"],
            "position": cls_info["cls_position"],
        },
        "similarity_results": similarity_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main(args: argparse.Namespace) -> None:
    prompt = ensure_prompt(args.prompt)
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    if not args.disable_optimizations:
        enable_inference_optimizations()

    tokenizer, model, image_processor, _ = load_model_and_setup(
        model_path=args.model_path,
        attn_implementation=args.attn_implementation,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        model_base=args.model_base,
        adapter_path=args.adapter_path,
        attn_layer_ind=-1
    )

    cls_info = extract_cls_hidden_states(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
    )

    resolved_image_path = fix_wsl_paths(args.image_path)
    base_image = load_image(resolved_image_path)
    image_tensor, image_size = prepare_multimodal_image_tensor(
        base_image,
        image_processor,
        model,
    )

    embedding_sets = prepare_image_embeddings(
        base_image,
        image_processor,
        model,
    )
    decoder_layer_embeddings = extract_decoder_image_hidden_states(
        tokenizer=tokenizer,
        model=model,
        image_tensor=image_tensor,
        image_size=image_size,
    )
    embedding_sets["decoder_layers"] = decoder_layer_embeddings

    output_root = Path(fix_wsl_paths(args.output_dir))
    run_dir, overlay_dir, timestamp = create_output_dirs(output_root)
    similarity_results: Dict[str, Dict[str, object]] = {}
    for label, embeddings in embedding_sets.items():
        label_safe = label.replace(" ", "_")
        label_dir = overlay_dir / label_safe
        rel_dir = str(Path("overlays") / label_safe)
        similarity_results[label] = compute_similarity_results(
            cls_embeddings=cls_info["embeddings"],
            image_embeddings=embeddings,
            base_image=base_image,
            overlay_dir=label_dir,
            overlay_rel_dir=rel_dir,
            grid_size=None,
            top_k=args.top_k,
            max_layers=args.max_layers,
        )

    results_path = run_dir / "results.json"
    save_results_json(
        output_path=results_path,
        prompt=prompt,
        cls_info=cls_info,
        model_path=args.model_path,
        image_path=args.image_path,
        timestamp=timestamp,
        similarity_results=similarity_results,
    )

    total_overlays = sum(len(entry["layers"]) for entry in similarity_results.values())
    print(f"Saved {total_overlays} overlays to {overlay_dir}")
    print(f"Summary written to {results_path}")


if __name__ == "__main__":
    main(parse_args())
