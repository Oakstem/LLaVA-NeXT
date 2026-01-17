#!/usr/bin/env python3
"""Extract text and image embeddings plus layer-wise hidden states using a LLaVA checkpoint."""

import argparse
import copy
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from generation_utils import enable_inference_optimizations, load_image, load_model_and_setup
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token


DEFAULT_MODEL_PATH = "lmms-lab/llava-onevision-qwen2-7b-ov-chat"
DEFAULT_OUTPUT_ROOT = "embeddings"
DEFAULT_IMAGE_PROMPT = "Describe the image."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate token and image embeddings using LLaVA.")
    parser.add_argument("--text", default=None, help="Phrase to tokenize and embed. Use '<image>' placeholders if needed.")
    parser.add_argument("--image-path", default=None, help="Optional image path for extracting vision features.")
    parser.add_argument("--image-text", default=None, help="Optional text appended after the <image> token; defaults to 'Describe the image.' Set to empty string for no text.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Model or checkpoint path to load.")
    parser.add_argument("--model-base", default=None, help="Optional base model path when loading LoRA adapters.")
    parser.add_argument("--adapter-path", default=None, help="Optional LoRA adapter to merge at inference time.")
    parser.add_argument("--attn-implementation", default="sdpa", help="Attention implementation passed to the loader.")
    parser.add_argument("--load-4bit", action="store_true", help="Load the model with 4-bit quantization.")
    parser.add_argument("--load-8bit", action="store_true", help="Load the model with 8-bit quantization.")
    parser.add_argument("--disable-optimizations", action="store_true", help="Skip inference optimizations used by default.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT, help="Root directory for saved outputs (subdirs 'text' and 'image').")
    parser.add_argument("--output-path", default=None, help="Explicit path for text outputs (.pt).")
    parser.add_argument("--image-output-path", default=None, help="Explicit path for image outputs (.pt).")
    parser.add_argument("--offset-image-token", type=int, default=14, help="Token offset where image tokens start in the sequence.")
    return parser.parse_args()


def build_token_segments(token_ids: torch.Tensor) -> List[torch.Tensor]:
    """Mirror prepare_inputs_labels_for_multimodal by stripping IMAGE_TOKEN_INDEX tokens."""
    positions = torch.where(token_ids == IMAGE_TOKEN_INDEX)[0].tolist()
    if not positions:
        return [token_ids]

    boundaries = [-1] + positions + [token_ids.shape[0]]
    segments: List[torch.Tensor] = []
    for idx in range(len(boundaries) - 1):
        start = boundaries[idx] + 1
        end = boundaries[idx + 1]
        segments.append(token_ids[start:end])
    return segments


def build_conversation_prompt(prompt: str, tokenizer) -> Tuple[str, torch.Tensor]:
    """Format user prompt with the Qwen conversation template and tokenize."""
    conv_template = "qwen_1_5"

    if DEFAULT_IMAGE_TOKEN not in prompt:
        full_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
    else:
        full_prompt = prompt

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], full_prompt)
    conv.append_message(conv.roles[1], None)  # Reserve assistant turn.
    prompt_question = conv.get_prompt()

    prompt_ids = tokenizer_image_token(
        prompt_question,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )
    return prompt_question, prompt_ids


def resolve_output_path(explicit_path: Optional[str], root_dir: str, category: str) -> Path:
    if explicit_path:
        target = Path(explicit_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = Path(root_dir) / category
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{category}_embeddings_{timestamp}.pt"


def prepare_image_tensor(image_path: str, image_processor, model) -> Tuple[torch.Tensor, Tuple[int, int]]:
    pil_image = load_image(image_path)
    processed = process_images([pil_image], image_processor, model.config)

    if isinstance(processed, tuple):
        image_tensor = processed[0]
    else:
        image_tensor = processed

    if isinstance(image_tensor, list):
        if not image_tensor:
            raise ValueError("Image processor returned an empty list.")
        image_tensor = image_tensor[0]

    if isinstance(image_tensor, torch.Tensor) and image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError(f"Unexpected processed image type: {type(image_tensor)}")

    image_tensor = image_tensor.to(model.device, dtype=model.dtype)
    return image_tensor, pil_image.size


def main() -> None:
    args = parse_args()

    if not args.text and not args.image_path:
        raise ValueError("At least one of --text or --image-path must be provided.")

    if args.load_4bit and args.load_8bit:
        raise ValueError("Cannot enable both --load-4bit and --load-8bit simultaneously.")

    if not args.disable_optimizations:
        enable_inference_optimizations()

    tokenizer, model, image_processor, _ = load_model_and_setup(
        model_path=args.model_path,
        attn_implementation=args.attn_implementation,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        model_base=args.model_base,
        adapter_path=args.adapter_path,
    )

    model.eval()
    base_model = model.get_model()
    base_device = model.device if hasattr(model, "device") else next(model.parameters()).device

    if args.text:
        prompt_question, prompt_ids = build_conversation_prompt(args.text, tokenizer)
        if prompt_ids.ndim != 1:
            raise ValueError("tokenizer_image_token is expected to return a 1D tensor for a single prompt.")

        segments = build_token_segments(prompt_ids)
        tokens_for_embedding = torch.cat(segments, dim=0) if segments else prompt_ids.new_zeros((0,), dtype=torch.long)

        if tokens_for_embedding.numel() == 0:
            raise ValueError("No textual tokens remained after processing the prompt; cannot run language model forward pass.")

        tokens_for_embedding = tokens_for_embedding.to(base_device)
        input_ids = tokens_for_embedding.unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        with torch.no_grad():
            text_embeddings = base_model.embed_tokens(tokens_for_embedding)
            text_outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
            )

        embeddings_cpu = text_embeddings.detach().cpu()
        tokens_cpu = tokens_for_embedding.detach().cpu()
        hidden_states_cpu = [layer.detach().cpu() for layer in (text_outputs.hidden_states or [])]

        text_output_path = resolve_output_path(args.output_path, args.output_dir, "text")
        torch.save(
            {
                "text": args.text,
                "conversation_prompt": prompt_question,
                "input_ids": tokens_cpu,
                "embeddings": embeddings_cpu,
                "hidden_states": hidden_states_cpu,
            },
            text_output_path,
        )

        print(f"[text] Token count: {tokens_cpu.shape[0]}")
        print(f"[text] Embedding shape: {tuple(embeddings_cpu.shape)}")
        if hidden_states_cpu:
            print("[text] Hidden state shapes:")
            for idx, layer in enumerate(hidden_states_cpu):
                print(f"  Layer {idx}: {tuple(layer.shape)}")
        print(f"[text] Saved embeddings to {text_output_path}")

    if args.image_path:
        if image_processor is None:
            raise ValueError("Loaded model does not provide an image processor; cannot handle --image-path.")

        image_tensor, image_size = prepare_image_tensor(args.image_path, image_processor, model)

        vision_inputs = image_tensor
        vision_split_sizes: Optional[List[int]] = None
        if isinstance(vision_inputs, torch.Tensor) and vision_inputs.ndim == 5:
            batch, views = vision_inputs.shape[:2]
            vision_split_sizes = [views] * batch
            vision_inputs = vision_inputs.flatten(0, 1)
        elif isinstance(vision_inputs, list):
            stacked_inputs = []
            split_sizes_tmp: List[int] = []
            for element in vision_inputs:
                if not isinstance(element, torch.Tensor):
                    raise TypeError("Encountered non-tensor item in image list when preparing vision inputs.")
                if element.ndim == 4:
                    stacked_inputs.append(element)
                    split_sizes_tmp.append(element.shape[0])
                elif element.ndim == 3:
                    stacked_inputs.append(element.unsqueeze(0))
                    split_sizes_tmp.append(1)
                else:
                    raise ValueError(f"Unsupported tensor rank {element.ndim} in image list.")
            if not stacked_inputs:
                raise ValueError("Empty image list after processing.")
            vision_inputs = torch.cat(stacked_inputs, dim=0)
            vision_split_sizes = split_sizes_tmp
        else:
            vision_split_sizes = None

        vision_tower = base_model.get_vision_tower()
        base_token_count = getattr(vision_tower, "num_patches", 729)

        with torch.no_grad():
            vision_features = vision_tower(vision_inputs)
            projected_features = base_model.mm_projector(vision_features)
        
        # print the shape of the features
        print(f"Vision features shape: {vision_features.shape}")
        print(f"Projected features shape: {projected_features.shape}")

        prompt_suffix = DEFAULT_IMAGE_PROMPT if args.image_text is None else args.image_text
        prompt_suffix = (prompt_suffix or "").strip()
        image_prompt_question, image_prompt_ids = build_conversation_prompt(prompt_suffix, tokenizer)
        if image_prompt_ids.ndim != 1:
            raise ValueError("tokenizer_image_token is expected to return a 1D tensor for a single image prompt.")

        image_input_ids = image_prompt_ids.unsqueeze(0).to(base_device)
        image_attention_mask = torch.ones_like(image_input_ids, dtype=torch.long)

        # print the first 30 tokens
        decoded_tokens = []
        for key, val in enumerate(image_prompt_ids[:30].tolist()):
            if val == IMAGE_TOKEN_INDEX:
                decoded_tokens.append(f'{key}:{val}:<IMAGE_TOKEN>')
            else:
                decoded_str = tokenizer.decode([val], clean_up_tokenization_spaces=False)
                decoded_tokens.append(f'{key}:{val}:{decoded_str}')
        print(f"Token IDs for embedding (first 30): {decoded_tokens}")
        with torch.no_grad():
            image_outputs = model(
                input_ids=image_input_ids,
                attention_mask=image_attention_mask,
                images=image_tensor,
                image_sizes=[list(image_size)],
                use_cache=False,
                output_hidden_states=True,
            )

        vision_features_cpu = vision_features.detach().cpu()
        projected_features_cpu = projected_features.detach().cpu()
        if vision_features_cpu.dim() >= 3 and vision_features_cpu.shape[1] > base_token_count:
            vision_features_cpu = vision_features_cpu[:, :base_token_count]
        if projected_features_cpu.dim() >= 3 and projected_features_cpu.shape[1] > base_token_count:
            projected_features_cpu = projected_features_cpu[:, :base_token_count]
        vision_feature_groups = None
        projected_feature_groups = None
        if vision_split_sizes:
            vision_feature_groups = [chunk.clone() for chunk in torch.split(vision_features_cpu, vision_split_sizes, dim=0)]
            projected_feature_groups = [chunk.clone() for chunk in torch.split(projected_features_cpu, vision_split_sizes, dim=0)]
        image_hidden_states_cpu = [layer.detach().cpu() for layer in (image_outputs.hidden_states or [])]
        image_prompt_ids_cpu = image_prompt_ids.detach().cpu()

        # extract relevant hidden states corresponding to image tokens only
        # allow an optional offset for where image tokens start in the sequence
        offset = int(getattr(args, "offset_image_token", 14) or 0)
        if offset < 0:
            raise ValueError("offset_image_token must be non-negative.")

        if image_hidden_states_cpu:
            relevant_hidden_states: List[torch.Tensor] = []
            for layer in image_hidden_states_cpu:
                seq_len = layer.shape[1]
                if offset >= seq_len:
                    raise ValueError(f"offset_image_token ({offset}) >= sequence length ({seq_len}) for a hidden layer.")
                start = offset
                end = min(offset + base_token_count, seq_len)
                relevant_hidden_states.append(layer[:, start:end, :].clone())
                image_hidden_states_cpu = relevant_hidden_states

        image_output_path = resolve_output_path(args.image_output_path, args.output_dir, "image")
        torch.save(
            {
                "image_path": args.image_path,
                "original_image_size": image_size,
                "vision_split_sizes": vision_split_sizes,
                "vision_features": vision_features_cpu,
                "mm_projected_features": projected_features_cpu,
                "vision_feature_groups": vision_feature_groups,
                "mm_projected_feature_groups": projected_feature_groups,
                "prompt_text": prompt_suffix,
                "conversation_prompt": image_prompt_question,
                "prompt_input_ids": image_prompt_ids_cpu,
                "vision_hidden_states": image_hidden_states_cpu,
            },
            image_output_path,
        )

        print(f"[image] Vision encoder feature shape: {tuple(vision_features_cpu.shape)}")
        print(f"[image] Projector feature shape: {tuple(projected_features_cpu.shape)}")
        if vision_feature_groups:
            print(f"[image] Vision feature groups: {[tuple(chunk.shape) for chunk in vision_feature_groups]}")
        if image_hidden_states_cpu:
            print("[image] Hidden state shapes:")
            for idx, layer in enumerate(image_hidden_states_cpu):
                print(f"  Layer {idx}: {tuple(layer.shape)}")
        print(f"[image] Saved embeddings to {image_output_path}")


if __name__ == "__main__":
    main()
