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
    load_image,
    load_model_and_setup,
)
from gazefollow.json_utils import make_json_safe


# DEFAULT_PROMPT = """Complete the sentence:
# woman looking at man -> the woman in a tan coat is looking at the man in the navy sweater.
# girl looking at woman -> the girl with braided hair is looking at the woman in the black dress.
# man looking at phone -> the man in a gray hoodie is looking at the black smartphone in his hand.
# kid looking at toy -> the kid in green overalls is looking at the colorful toy on the floor.
# boy looking at dog -> the boy in a striped shirt is looking at the small brown dog.
# woman looking at cat -> the woman in a denim jacket is looking at the black cat on the sofa.
# man looking at hands -> the man in a white shirt is looking at his hands.
# girl looking at feet -> the girl in purple leggings is looking down at her feet.

# The sentence: the _ is looking at + -> """

DEFAULT_PROMPT = """Complete the sentence:
woman looking at man → the woman wearing a tan coat and black jeans is looking at the man in a navy sweater and dark trousers.
girl looking at woman → the girl in a yellow hoodie and denim shorts is looking at the woman in a black dress and brown boots.
man looking at phone → the man in a gray hoodie and blue jeans is looking at the black smartphone in his hand.
kid looking at toy → the kid in a striped tee and green overalls is looking at the colorful toy on the floor.
boy looking at dog → the boy in a red jacket and khaki pants is looking at the small brown dog.
woman looking at cat → the woman in a denim jacket and black leggings is looking at the black cat on the sofa.
man looking at hands → the man in a white shirt and dark jeans is looking at his hands.
girl looking at feet → the girl in a purple sweater and patterned leggings is looking down at her feet.

The sentence: the _ is looking at + → """

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


def _print_mask_info(label: str, mask_path: Path, metadata: Optional[Dict[str, Any]], mask: np.ndarray) -> None:
    stat = mask_path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{label} mask file: {mask_path}")
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


def _resolve_image_path(
    explicit_path: Optional[str],
    source_metadata: Optional[Dict[str, Any]],
    target_metadata: Optional[Dict[str, Any]],
) -> Path:
    if explicit_path:
        return Path(fix_wsl_paths(explicit_path)).expanduser().resolve()
    meta_candidates = [meta for meta in (target_metadata, source_metadata) if meta and meta.get("image_path")]
    if meta_candidates:
        resolved = [fix_wsl_paths(meta["image_path"]) for meta in meta_candidates]
        if len(set(resolved)) > 1:
            raise ValueError("Source and target mask metadata point to different images; please specify --image-path.")
        return Path(resolved[0]).expanduser().resolve()
    raise ValueError(
        "Image path not provided and neither mask metadata contains 'image_path'. "
        "Specify --image-path explicitly."
    )


def _prepare_mask(mask: np.ndarray) -> np.ndarray:
    processed = mask
    if processed.ndim > 2:
        processed = processed.squeeze()
    if processed.dtype != np.uint8:
        processed = processed.astype(np.uint8)
    return processed


def _build_prompt_for_model(prompt: str) -> str:
    prompt_for_model = prompt.replace("+", "_")
    underscore_count = prompt_for_model.count("_")
    if underscore_count < 2:
        raise ValueError(
            f"Prompt must contain two placeholders for patch insertion; found {underscore_count} underscores after conversion."
        )
    return prompt_for_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a dual-region extraction using two masks and two embedding patches",
    )
    parser.add_argument(
        "--image-path",
        default=None,
        help="Path to the input image. If omitted, inferred from either mask metadata.",
    )
    parser.add_argument(
        "--source-mask-path",
        default="region_masks/source_region_mask.npy",
        help="Binary mask file path for the gaze source (person) region.",
    )
    parser.add_argument(
        "--target-mask-path",
        default="region_masks/target_region_mask.npy",
        help="Binary mask file path for the gaze target region.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used for generation (placeholders at '_' and '+').")
    parser.add_argument("--output-dir", default="attention_output/dual_region_extraction", help="Directory to store outputs.")
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
    parser.add_argument(
        "--break-after-first-step",
        action="store_true",
        default=False,
        help="Stop after first decoding step to focus on embedding capture.",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--attn-layer-ind", type=int, default=-1)
    parser.add_argument("--model-path", default="lmms-lab/llava-onevision-qwen2-7b-ov-chat")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--load-4bit", action="store_true", default=False)
    parser.add_argument("--load-8bit", action="store_true", default=False)
    parser.add_argument("--repr-capture-layer", type=int, default=20)
    parser.add_argument("--repr-inject-layer", type=int, default=0)
    parser.add_argument("--query-indices", type=json.loads, default=None)
    parser.add_argument("--guidance-top-k", type=int, default=5)
    parser.add_argument("--similarity-weight", type=float, default=0.7)
    parser.add_argument("--probability-weight", type=float, default=0.3)
    parser.add_argument("--guidance-enable-after-step", type=int, default=0)
    parser.add_argument("--guidance-enable-after-keyword", type=str, default="looking")
    parser.add_argument(
        "--filter-image-tokens",
        action="store_true",
        default=False,
        help="Restrict image tokens to the source mask region.",
    )
    args = parser.parse_args()

    if args.query_indices is None:
        args.query_indices = _default_query_indices()

    enable_inference_optimizations()

    source_mask_path = Path(fix_wsl_paths(args.source_mask_path)).expanduser().resolve()
    target_mask_path = Path(fix_wsl_paths(args.target_mask_path)).expanduser().resolve()
    source_metadata = _load_mask_metadata(source_mask_path)
    target_metadata = _load_mask_metadata(target_mask_path)

    output_dir = Path(fix_wsl_paths(args.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = _resolve_image_path(args.image_path, source_metadata, target_metadata)
    load_image(str(image_path))

    source_mask = _prepare_mask(_load_mask(source_mask_path))
    target_mask = _prepare_mask(_load_mask(target_mask_path))

    if source_mask.shape != target_mask.shape:
        raise ValueError(
            f"Source mask shape {source_mask.shape} does not match target mask shape {target_mask.shape}."
        )

    _print_mask_info("Source", source_mask_path, source_metadata, source_mask)
    _print_mask_info("Target", target_mask_path, target_metadata, target_mask)

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
    run_dir = output_dir / f"dual_region_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    target_mask_temp_path = run_dir / "gaze__target_mask.npy"
    source_mask_temp_path = run_dir / "person__target_mask.npy"
    np.save(target_mask_temp_path, target_mask)
    np.save(source_mask_temp_path, source_mask)

    prompt_for_model = _build_prompt_for_model(args.prompt)

    common_kwargs = {
        "image_path": str(image_path),
        "mask_path": str(target_mask_temp_path),
        "prompt": prompt_for_model,
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
        "filter_image_tokens_to_person_mask": args.filter_image_tokens,
        "same_mask_for_person": False,
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
            print("Warning: Initial run did not produce hidden states for embedding patching.")

    results = run_generation_with_attention(
        output_dir=str(run_dir),
        break_after_first_step=args.break_after_first_step,
        prev_run_last_hidden_state=prev_hidden_state,
        **common_kwargs,
    )

    summary_path = run_dir / "dual_region_summary.json"
    payload = {
        "image_path": str(image_path),
        "source_mask_path": str(source_mask_path),
        "target_mask_path": str(target_mask_path),
        "run_source_mask_path": str(source_mask_temp_path),
        "run_target_mask_path": str(target_mask_temp_path),
        "prompt_requested": args.prompt,
        "prompt_used": prompt_for_model,
        "output_dir": str(run_dir),
        "generated_text": results.get("generated_text"),
        "num_tokens": results.get("num_tokens"),
        "evaluation_summary": results.get("evaluation_summary"),
        "quality_analysis": results.get("quality_analysis"),
        "attention_correlation": results.get("attention_correlation"),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(payload), f, indent=2)

    print("\nDual region extraction complete.")
    print(f"Outputs saved to: {run_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
