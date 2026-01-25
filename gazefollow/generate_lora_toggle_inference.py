#!/usr/bin/env python3
"""Two-step LLaVA inference: LoRA-on for prompt A, LoRA-off for prompt B."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gazefollow.generate_vanilla_inference as gvi

from generation_utils import (
    enable_inference_optimizations,
    load_model_and_setup,
    fix_wsl_paths,
)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path

DEFAULT_PROMPT_A = "Describe where each person is looking in the image."
DEFAULT_PROMPT_A = "Describe in one short sentence where each person is looking in the image."
DEFAULT_PROMPT_B = "Based on each person's gaze direction, infer the social relationship between the people in the image."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two-step inference with LoRA enabled for the first prompt and disabled for the second."
    )
    parser.add_argument(
        "--model-path",
        default="lmms-lab/llava-onevision-qwen2-7b-ov-chat",
        help="Model or checkpoint path to load (defaults to LLaVA-OneVision 7B).",
    )
    parser.add_argument(
        "--model-base",
        default=None,
        help="Base model path when loading LoRA adapters.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="LoRA adapter path to activate for the first prompt.",
    )
    parser.add_argument(
        "--select-adapter",
        action="store_true",
        help="Interactively select adapter from recent checkpoints.",
    )
    parser.add_argument(
        "--adapter-name",
        default=None,
        help="Optional adapter name to activate when multiple adapters are present.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Attention implementation (e.g. 'sdpa', 'flash_attention_2').",
    )
    parser.add_argument("--load-4bit", action="store_true", help="Load the model with 4-bit quantization.")
    parser.add_argument("--load-8bit", action="store_true", help="Load the model with 8-bit quantization.")
    parser.add_argument("--image-path", required=True, help="Path to the input image.")

    parser.add_argument(
        "--image-aspect-ratio",
        default=gvi.DEFAULT_IMAGE_ASPECT_RATIO,
        help="Fallback image_aspect_ratio used when absent in the checkpoint.",
    )
    parser.add_argument(
        "--image-grid-pinpoints",
        default=gvi.DEFAULT_IMAGE_GRID_PINPOINTS_EXPR,
        help="Fallback image_grid_pinpoints used when absent in the checkpoint.",
    )

    parser.add_argument(
        "--prompt-a",
        "--prompt-first",
        dest="prompt_a",
        default=DEFAULT_PROMPT_A,
        help="First prompt (LoRA enabled).",
    )
    parser.add_argument(
        "--prompt-b",
        "--prompt-second",
        dest="prompt_b",
        default=DEFAULT_PROMPT_B,
        help="Second prompt to append after the first response (LoRA disabled).",
    )
    parser.add_argument(
        "--second-separator",
        default=" \n",
        help="Separator placed between the first response and prompt-b.",
    )

    parser.add_argument("--conv-template", default=None, help="Conversation template key override.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling value.")
    parser.add_argument("--num-beams", type=int, default=1, help="Beam search width.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling instead of greedy decoding.")
    parser.add_argument(
        "--log-topk-tokens",
        type=int,
        default=0,
        help="Compatibility flag for shared generation utilities (unused).",
    )
    parser.add_argument("--disable-optimizations", action="store_true", help="Skip CUDA inference optimizations.")
    parser.add_argument(
        "--save-output",
        default=None,
        help="Optional path to save a JSON payload with both responses.",
    )
    return parser.parse_args()


def ensure_adapter_selected(model, adapter_name: Optional[str]) -> None:
    if adapter_name:
        if not hasattr(model, "set_adapter"):
            raise ValueError("Adapter name provided but model does not support set_adapter().")
        model.set_adapter(adapter_name)


def main() -> None:
    args = parse_args()

    if args.load_4bit and args.load_8bit:
        raise ValueError("Cannot enable both --load-4bit and --load-8bit at the same time.")

    if not args.disable_optimizations:
        enable_inference_optimizations()

    adapter_path = args.adapter_path
    if args.select_adapter:
        adapter_path = gvi.prompt_for_adapter_path()

    if not adapter_path:
        raise ValueError("--adapter-path is required for LoRA toggle inference.")

    adapter_path = fix_wsl_paths(adapter_path)

    tokenizer, model, image_processor, _ = load_model_and_setup(
        model_path=args.model_path,
        attn_implementation=args.attn_implementation,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        model_base=args.model_base,
        adapter_path=adapter_path,
    )

    ensure_adapter_selected(model, args.adapter_name)

    if not hasattr(model, "disable_adapter"):
        raise ValueError("Loaded model does not expose disable_adapter(); ensure a PEFT LoRA adapter is loaded.")

    gvi.ensure_image_config(
        model,
        args.image_aspect_ratio,
        args.image_grid_pinpoints,
        override_aspect_ratio=args.image_aspect_ratio != gvi.DEFAULT_IMAGE_ASPECT_RATIO,
        override_grid_pinpoints=args.image_grid_pinpoints != gvi.DEFAULT_IMAGE_GRID_PINPOINTS_EXPR,
    )

    image_path = fix_wsl_paths(args.image_path)
    image_tensor, image_size = gvi.prepare_image_tensor(image_path, image_processor, model)

    model_name_source = model.config._name_or_path if hasattr(model.config, "_name_or_path") else args.model_path
    model_name = get_model_name_from_path(model_name_source)
    conv_name = gvi.determine_template(model_name, args.conv_template)

    conv_first = conv_templates[conv_name].copy()
    conv_first.tokenizer = tokenizer
    conv_second = conv_templates[conv_name].copy()
    conv_second.tokenizer = tokenizer

    base_gen_kwargs = gvi.build_generation_kwargs(args, tokenizer, image_tensor, image_size)

    print("Generating response for prompt A (LoRA enabled)...")
    response_a, _, _, _ = gvi.generate_turn(
        conv_first,
        args.prompt_a,
        tokenizer,
        model,
        image_tensor,
        image_size,
        base_gen_kwargs,
        include_image_token=True,
    )
    if not response_a.strip().endswith("."):
        response_a = response_a.strip() + "."
        
    combined_prompt = f"{response_a}{args.second_separator}{args.prompt_b}".strip()

    print("\nGenerating response for prompt B (LoRA disabled)...")
    with model.disable_adapter():
        response_b, _, _, _ = gvi.generate_turn(
            conv_second,
            combined_prompt,
            tokenizer,
            model,
            image_tensor,
            image_size,
            base_gen_kwargs,
            include_image_token=True,
        )

    print("\n=== Prompt A ===")
    print(args.prompt_a)
    print("\n=== Response A (LoRA) ===")
    print(response_a)
    print("\n=== Prompt B (with Response A inserted) ===")
    print(combined_prompt)
    print("\n=== Response B (Base) ===")
    print(response_b)

    if args.save_output:
        output_path = Path(args.save_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "image": image_path,
            "model_path": args.model_path,
            "adapter_path": adapter_path,
            "prompt_a": args.prompt_a,
            "response_a": response_a,
            "prompt_b": args.prompt_b,
            "combined_prompt": combined_prompt,
            "response_b": response_b,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved responses to {output_path}")


if __name__ == "__main__":
    main()
