#!/usr/bin/env python3
"""Minimal CLI for running vanilla LLaVA inference on an image and prompt.

This script reuses the shared model loading utilities from ``generation_utils``
so it supports optional LoRA adapters in addition to base checkpoints.
"""

import argparse
import json
import sys
from pathlib import Path
from threading import Thread
from typing import Callable, Optional, Tuple

import torch
from transformers import TextIteratorStreamer

# Ensure project root is importable (needed when running from subdirectories)
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generation_utils import (  # noqa: E402
    enable_inference_optimizations,
    load_model_and_setup,
    fix_wsl_paths,
    load_image,
)
from llava.mm_utils import (  # noqa: E402
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.constants import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vanilla LLaVA inference on a single image and prompt.")
    parser.add_argument("--model-path", default="lmms-lab/llava-onevision-qwen2-7b-ov-chat", help="Model or checkpoint path to load (defaults to LLaVA-OneVision 7B).")
    parser.add_argument("--model-base", default=None, help="Optional base model path when loading LoRA adapters.")
    parser.add_argument("--adapter-path", default=None, help="Optional LoRA adapter path to merge at inference time.")
    parser.add_argument("--attn-implementation", default="sdpa", help="Attention implementation passed to the loader (e.g. 'sdpa', 'flash_attention_2').")
    parser.add_argument("--load-4bit", action="store_true", help="Load the model with 4-bit quantization.")
    parser.add_argument("--load-8bit", action="store_true", help="Load the model with 8-bit quantization.")
    parser.add_argument("--image-path", required=True, help="Path to the input image.")
    
    # prompt_group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("--prompt", default="describe every person and where is he looking at?", help="User prompt to pair with the image.")
    # prompt_group.add_argument("--prompt-file", help="Path to a text file containing the prompt.")
    
    parser.add_argument("--conv-template", default=None, help="Conversation template key (defaults to qwen_1_5 for Qwen-style models).")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature (ignored if --do-sample is False).")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling value.")
    parser.add_argument("--num-beams", type=int, default=1, help="Number of beams for beam search decoding.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling instead of greedy decoding.")
    parser.add_argument("--disable-optimizations", action="store_true", help="Skip enabling CUDA inference optimizations.")
    parser.add_argument("--save-output", default=None, help="Optional path to save the generated text as JSON (with keys image, prompt, response).")
    parser.add_argument("--chat", action="store_true", help="Enter an interactive chat loop that reuses the loaded model and image.")
    parser.add_argument("--no-stream", action="store_true", help="Disable token streaming and only print responses after generation completes.")
    return parser.parse_args()

def prompt_for_adapter_path(training_outputs_dir: str = "training_outputs") -> Optional[str]:
    """Interactively prompt user to select a checkpoint or provide a custom path.
    
    Returns:
        Selected adapter path or None if user chooses no adapter.
    """
    print("=" * 50)
    print("Available recent checkpoints:")
    
    checkpoints_pattern = Path(training_outputs_dir).glob("llava-*/checkpoint-*")
    checkpoints = sorted(checkpoints_pattern, key=lambda p: p.stat().st_mtime, reverse=True)[:10]
    
    if not checkpoints:
        print("No checkpoints found.")
        return None
    
    for i, ckpt in enumerate(checkpoints):
        print(f"  [{i}] {ckpt}")
    
    print("  [-1] No adapter (base model only)")
    print("=" * 50)
    print(f"Default: [0] {checkpoints[0]}")
    
    try:
        user_input = input("Enter checkpoint number, -1 for no adapter, full path, or press Enter for default: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nUsing default checkpoint.")
        return str(checkpoints[0])
    
    if not user_input:
        selected = str(checkpoints[0])
        print(f"Selected: {selected}")
        return selected
    
    if user_input == "-1":
        print("Selected: No adapter (base model only)")
        return None
    
    if user_input.isdigit():
        idx = int(user_input)
        if 0 <= idx < len(checkpoints):
            selected = str(checkpoints[idx])
            print(f"Selected: {selected}")
            return selected
        print(f"Invalid index {idx}. Using default.")
        return str(checkpoints[0])
    
    return user_input

def read_prompt(prompt: Optional[str]) -> str:
    if prompt is not None:
        return prompt.strip()


def determine_template(model_name: str, override: Optional[str]) -> str:
    if override:
        if override not in conv_templates:
            available = ", ".join(sorted(conv_templates.keys()))
            raise ValueError(f"Conversation template '{override}' not found. Available: {available}")
        return override

    lowered = model_name.lower()
    if "qwen" in lowered:
        return "qwen_1_5"
    if "vicuna" in lowered:
        return "vicuna_v1"
    if "mpt" in lowered:
        return "mpt"
    # Default fallback works well for most modern checkpoints
    return "qwen_1_5"


def prepare_image_tensor(image_path: str, image_processor, model) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Prepare a single image tensor and return it with the original size."""
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


def build_generation_kwargs(args: argparse.Namespace, tokenizer, image_tensor: torch.Tensor, image_size: Tuple[int, int]):
    """Prepare generation kwargs shared across turns."""
    kwargs = {
        "images": image_tensor,
        "image_sizes": [list(image_size)],
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        # "eos_token_id": None,
    }

    if not args.do_sample:
        kwargs.pop("temperature", None)
        kwargs.pop("top_p", None)

    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return kwargs


def generate_turn(
    conv,
    prompt_text: str,
    tokenizer,
    model,
    image_tensor: torch.Tensor,
    image_size: Tuple[int, int],
    base_gen_kwargs: dict,
    include_image_token: bool,
    stream_printer: Optional[Callable[[str], None]] = None,
):
    """Generate a response for a single user turn and update the conversation."""

    prompt_text = prompt_text.strip()
    if not prompt_text:
        raise ValueError("Prompt text must be non-empty for generation.")

    if include_image_token and DEFAULT_IMAGE_TOKEN not in prompt_text:
        user_content = f"{DEFAULT_IMAGE_TOKEN}\n{prompt_text}"
    else:
        user_content = prompt_text

    conv.append_message(conv.roles[0], user_content)
    conv.append_message(conv.roles[1], None)
    prompt_for_tokenizer = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_for_tokenizer,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)

    gen_kwargs = base_gen_kwargs.copy()
    gen_kwargs["inputs"] = input_ids

    if stream_printer:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

        def run_generate():
            with torch.inference_mode():
                model.generate(**gen_kwargs)

        thread = Thread(target=run_generate)
        thread.start()

        collected_chunks = []
        for chunk in streamer:
            stream_printer(chunk)
            collected_chunks.append(chunk)

        thread.join()
        response = "".join(collected_chunks).strip()
    else:
        with torch.inference_mode():
            generation_output = model.generate(**gen_kwargs)

        if isinstance(generation_output, tuple):
            output_ids = generation_output[0]
        else:
            output_ids = generation_output

        generated_tokens = output_ids[0, input_ids.shape[1] :]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    conv.messages[-1][1] = response

    return response


def main() -> None:
    args = parse_args()

    if args.load_4bit and args.load_8bit:
        raise ValueError("Cannot enable both --load-4bit and --load-8bit at the same time.")

    if not args.disable_optimizations:
        enable_inference_optimizations()

    adapter_path = args.adapter_path
    if args.select_adapter:
        adapter_path = prompt_for_adapter_path()

    tokenizer, model, image_processor, _ = load_model_and_setup(
        model_path=args.model_path,
        attn_implementation=args.attn_implementation,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        model_base=args.model_base,
        adapter_path=fix_wsl_paths(adapter_path) if adapter_path else None,
    )

    prompt_text = read_prompt(args.prompt)
    image_path = fix_wsl_paths(args.image_path)

    image_tensor, image_size = prepare_image_tensor(image_path, image_processor, model)

    model_name_source = model.config._name_or_path if hasattr(model.config, "_name_or_path") else args.model_path
    model_name = get_model_name_from_path(model_name_source)
    conv_name = determine_template(model_name, args.conv_template)

    conv = conv_templates[conv_name].copy()
    conv.tokenizer = tokenizer

    base_gen_kwargs = build_generation_kwargs(args, tokenizer, image_tensor, image_size)
    stream_output = not args.no_stream

    if args.chat and not prompt_text:
        try:
            prompt_text = input("Enter initial user prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting without running inference.")
            return

    if not prompt_text:
        raise ValueError("A non-empty prompt is required to start the conversation.")

    def stream_printer(chunk: str) -> None:
        print(chunk, end="", flush=True)

    print("Generating response...")
    print("\n=== Model Response ===")
    if stream_output:
        response = generate_turn(
            conv,
            prompt_text,
            tokenizer,
            model,
            image_tensor,
            image_size,
            base_gen_kwargs,
            include_image_token=True,
            stream_printer=stream_printer,
        )
        print()
    else:
        response = generate_turn(
            conv,
            prompt_text,
            tokenizer,
            model,
            image_tensor,
            image_size,
            base_gen_kwargs,
            include_image_token=True,
        )
        print(response)

    if args.chat:
        while True:
            try:
                next_prompt = input("\nUser> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break

            if not next_prompt:
                continue

            if next_prompt.lower() in {"exit", "quit", "q"}:
                print("Exiting chat.")
                break

            if stream_output:
                print("\nAssistant> ", end="", flush=True)
                response = generate_turn(
                    conv,
                    next_prompt,
                    tokenizer,
                    model,
                    image_tensor,
                    image_size,
                    base_gen_kwargs,
                    include_image_token=False,
                    stream_printer=stream_printer,
                )
                print()
            else:
                response = generate_turn(
                    conv,
                    next_prompt,
                    tokenizer,
                    model,
                    image_tensor,
                    image_size,
                    base_gen_kwargs,
                    include_image_token=False,
                )
                print("\nAssistant>")
                print(response)

    if args.save_output:
        output_path = Path(args.save_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.chat:
            conversation_payload = [
                {"role": role, "message": message}
                for role, message in conv.messages
            ]
            payload = {
                "image": image_path,
                "conversation": conversation_payload,
            }
        else:
            payload = {
                "image": image_path,
                "prompt": prompt_text,
                "response": response,
            }

        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved response to {output_path}")


if __name__ == "__main__":
    main()
