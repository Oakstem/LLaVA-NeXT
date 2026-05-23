import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from extract_attention_direct_gt_masks import run_generation_with_attention
from generation_utils import (
    enable_inference_optimizations,
    fix_wsl_paths,
    load_model_and_setup,
    load_image,
)
from gazefollow.json_utils import make_json_safe


def _prepare_generation_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "bias_strength": args.bias_strength,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": args.do_sample,
        "top_k": args.top_k,
        "output_hidden_states": True,
        "save_debug_files": args.save_debug_files,
    }


def _prepare_attention_config(args: argparse.Namespace) -> Dict[str, Any]:
    attn_cfg: Dict[str, Any] = {
        "layer_idx": args.attn_layer_ind,
        "repr_layer_idx": args.repr_capture_layer,
        "repr_capture_layer_idx": args.repr_capture_layer,
        "repr_inject_layer_idx": args.repr_inject_layer,
        "query_indices": args.query_indices,
        "save_tensors": False,
    }
    return attn_cfg


def _prepare_guidance_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "top_k": args.guidance_top_k,
        "similarity_weight": args.similarity_weight,
        "probability_weight": args.probability_weight,
        "enable_after_step": args.guidance_enable_after_step,
        "enable_after_keyword": args.guidance_enable_after_keyword,
    }


def _default_query_indices() -> Dict[str, Any]:
    return {
        "gaze_source": [-6, -3],
        "gaze_target": [-3, 0],
    }


def _load_mask(mask_path: Path) -> np.ndarray:
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    data = np.load(mask_path, allow_pickle=True)
    if isinstance(data, np.ndarray):
        return data
    raise ValueError(f"Unexpected mask format in {mask_path}")


def _load_mask_metadata(mask_path: Path) -> Optional[Dict[str, Any]]:
    meta_path = mask_path.with_suffix(".json")
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _print_mask_info(mask_path: Path, metadata: Optional[Dict[str, Any]], mask: np.ndarray) -> None:
    stat = mask_path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    print(f"Mask file: {mask_path}")
    print(f" - Last modified: {modified}")
    print(f" - Shape: {list(mask.shape)} | dtype: {mask.dtype}")

    def _center_from_bounds(bounds: Dict[str, float]) -> Tuple[float, float]:
        x_min = bounds.get("x_min", 0.0)
        x_max = bounds.get("x_max", 0.0)
        y_min = bounds.get("y_min", 0.0)
        y_max = bounds.get("y_max", 0.0)
        return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)

    pixel_bounds = metadata.get("pixel_bounds") if metadata else None
    normalized_bounds = metadata.get("normalized_bounds") if metadata else None

    if pixel_bounds:
        center_px = _center_from_bounds(pixel_bounds)
        print(
            f" - Pixel bounds: {pixel_bounds} | center (px): "
            f"({center_px[0]:.1f}, {center_px[1]:.1f})"
        )
    else:
        coords = np.argwhere(mask > 0)
        if coords.size > 0:
            mean_yx = coords.mean(axis=0)
            print(
                f" - Pixel center (from mask): "
                f"({mean_yx[1]:.1f}, {mean_yx[0]:.1f})"
            )

    if normalized_bounds:
        center_norm = _center_from_bounds(normalized_bounds)
        print(
            f" - Normalized bounds: {normalized_bounds} | center (norm): "
            f"({center_norm[0]:.4f}, {center_norm[1]:.4f})"
        )


DEFAULT_PROMPT = """Complete the sentence in the following format, for example:
a tired guy in a hoodie → a guy in a gray hoodie and ripped jeans sitting on a worn wooden bench.  
a chic woman in a beige coat → a woman in a beige coat and ankle boots holding a phone.  
a techy man in a leather jacket → a man in a black leather jacket and glasses.  
a relaxed woman in a green sweater → a woman in a dark green sweater and black jeans carrying a tan shoulder bag.

The sentence: a _ → """
# DEFAULT_PROMPT = """Complete the sentence in the following format, for example:
# a scruffy guy in a tee → a guy with messy hair wearing a faded graphic t-shirt and loose jeans.
# a warm croissant → a golden, flaky croissant with crisp layers and a soft buttery center.
# a stylish woman in red → a woman with sleek hair wearing a bright red blazer and matching heels.
# a young child in overalls → a small child with curly hair wearing light denim overalls and a striped tee.
# a bulky backpack → a large black backpack with thick straps and a padded mesh back.
# a glossy metal bottle → a tall stainless-steel bottle with a smooth reflective finish.
# a worn-out notebook → a small notebook with frayed edges and a cracked leather cover.

# The sentence: a _ →"""

## Objects Only Prompt
# DEFAULT_PROMPT = """Complete the sentence in the following format, for example:
# a warm croissant → a golden, flaky croissant with crisp layers and a soft buttery center.
# a bulky backpack → a large black backpack with thick straps and a padded mesh back.
# a glossy metal bottle → a tall stainless-steel bottle with a smooth reflective finish.
# a worn-out notebook → a small notebook with frayed edges and a cracked leather cover.

# The sentence: a _ →"""
# DEFAULT_PROMPT = """Complete the sentence in the following format, for example:
# a scruffy guy in a tee → a guy with messy hair wearing a faded graphic t-shirt and loose jeans.
# a stylish woman in red → a woman with sleek hair wearing a bright red blazer and matching heels.
# a bulky backpack → a large black backpack with thick straps and a padded mesh back.
# a glossy metal bottle → a tall stainless-steel bottle with a smooth reflective finish.
# a nerdy man with glasses → a man with round glasses, a plaid button-up, and tucked-in chinos.
# a worn-out notebook → a small notebook with frayed edges and a cracked leather cover.

# The sentence: a _ →"""

def _resolve_image_path(
    explicit_path: Optional[str],
    mask_path: Path,
    metadata: Optional[Dict[str, Any]],
) -> Path:
    if explicit_path:
        return Path(fix_wsl_paths(explicit_path)).expanduser().resolve()
    if metadata and metadata.get("image_path"):
        return Path(fix_wsl_paths(metadata["image_path"])).expanduser().resolve()
    raise ValueError(
        "Image path not provided and mask metadata does not contain 'image_path'. "
        "Specify --image-path explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single-region extraction using a pre-defined mask",
    )
    parser.add_argument(
        "--image-path",
        default=None,
        help="Path to the input image. If omitted, inferred from mask metadata.",
    )
    parser.add_argument("--mask-path", default="region_masks/active_region_mask.npy", help="Binary mask file path.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used for generation.")
    parser.add_argument("--output-dir", default="attention_output/single_region_extraction", help="Directory to store outputs.")
    parser.add_argument("--bias-strength", type=float, default=2.5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--use-gaze-guidance", action="store_true", default=False)
    parser.add_argument("--save-debug-files", action="store_true", default=False)
    parser.add_argument(
        "--save-attention-mask-html",
        action="store_true",
        default=False,
        help="Save attention_mask_sequence.html visualization (disabled by default).",
    )
    parser.add_argument("--save-mask-overlays", action="store_true", default=False)
    parser.add_argument("--mask-overlay-alpha", type=float, default=0.4)
    parser.add_argument("--exclude-image-inputs", action="store_true", default=False)
    parser.add_argument("--break-after-first-step", action="store_true", default=False,
                        help="Stop after first decoding step to focus on embedding capture.")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--attn-layer-ind", type=int, default=-1)
    parser.add_argument("--model-path", default="lmms-lab/llava-onevision-qwen2-7b-ov-chat")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--load-4bit", action="store_true", default=False)
    parser.add_argument("--load-8bit", action="store_true", default=False)
    parser.add_argument("--repr-capture-layer", type=int, default=0)
    parser.add_argument("--repr-inject-layer", type=int, default=0)
    parser.add_argument("--query-indices", type=json.loads, default=None)
    parser.add_argument("--guidance-top-k", type=int, default=5)
    parser.add_argument("--similarity-weight", type=float, default=0.7)
    parser.add_argument("--probability-weight", type=float, default=0.3)
    parser.add_argument("--guidance-enable-after-step", type=int, default=0)
    parser.add_argument("--guidance-enable-after-keyword", type=str, default="looking")
    args = parser.parse_args()

    if args.query_indices is None:
        args.query_indices = _default_query_indices()

    enable_inference_optimizations()

    mask_path = Path(fix_wsl_paths(args.mask_path)).expanduser().resolve()
    metadata = _load_mask_metadata(mask_path)
    output_dir = Path(fix_wsl_paths(args.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = _resolve_image_path(args.image_path, mask_path, metadata)

    load_image(str(image_path))  # Ensure image exists/valid
    mask = _load_mask(mask_path)
    if mask.ndim > 2:
        mask = mask.squeeze()
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    _print_mask_info(mask_path, metadata, mask)

    tokenizer, model, image_processor, _ = load_model_and_setup(
        model_path=fix_wsl_paths(args.model_path),
        attn_implementation=args.attn_implementation,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        attn_layer_ind=args.attn_layer_ind,
        model_base=fix_wsl_paths(args.model_base) if args.model_base else None,
        adapter_path=fix_wsl_paths(args.adapter_path) if args.adapter_path else None,
    )

    generation_config = _prepare_generation_config(args)
    attention_config = _prepare_attention_config(args)
    guidance_config = _prepare_guidance_config(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"single_region_{timestamp}"

    mask_temp_path = run_dir / "region_mask.npy"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(mask_temp_path, mask)

    common_kwargs = {
        "image_path": str(image_path),
        "mask_path": str(mask_temp_path),
        "prompt": args.prompt,
        "model": model,
        "tokenizer": tokenizer,
        "image_processor": image_processor,
        "generation_config": generation_config,
        "attention_config": attention_config,
        "bias_strength": generation_config.get("bias_strength", 0),
        "use_gaze_guidance": args.use_gaze_guidance,
        "guidance_config": guidance_config,
        "save_debug_files": args.save_debug_files,
        "use_gt_gaze_csv": False,
        "save_mask_overlays": args.save_mask_overlays,
        "mask_overlay_alpha": args.mask_overlay_alpha,
        "include_image_inputs": not args.exclude_image_inputs,
        "filter_image_tokens_to_person_mask": False,
        "same_mask_for_person": True,
        "attention_mask_viz_dir": str(run_dir / "attention_masks")
        if args.save_attention_mask_html
        else False,
    }

    prev_hidden_state = None
    if not args.break_after_first_step:
        init_output_dir = run_dir / "initial_single_step"
        init_output_dir.mkdir(parents=True, exist_ok=True)
        print("Running initial single-step generation to capture hidden states for embedding patching...")
        init_results = run_generation_with_attention(
            output_dir=str(init_output_dir),
            break_after_first_step=True,
            **{**common_kwargs, "prompt": ""},
        )
        prev_hidden_state = init_results.get("first_step_hidden_state")
        if prev_hidden_state is None:
            print("⚠️ Warning: Initial run did not produce hidden states for embedding patching.")

    results = run_generation_with_attention(
        output_dir=str(run_dir),
        break_after_first_step=args.break_after_first_step,
        prev_run_last_hidden_state=prev_hidden_state,
        **common_kwargs,
    )

    summary_path = run_dir / "single_region_summary.json"
    payload = {
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "output_dir": str(run_dir),
        "generated_text": results.get("generated_text"),
        "num_tokens": results.get("num_tokens"),
        "evaluation_summary": results.get("evaluation_summary"),
        "quality_analysis": results.get("quality_analysis"),
        "attention_correlation": results.get("attention_correlation"),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(payload), f, indent=2)

    print("\nSingle region extraction complete.")
    print(f"Outputs saved to: {run_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
