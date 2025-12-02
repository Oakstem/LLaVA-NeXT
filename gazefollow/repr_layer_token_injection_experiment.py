import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from generation_utils import save_image_results
def _clone_tokens_indexing(tokens_indexing: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Create a CPU-friendly copy of the model's tokens_indexing metadata.
    """
    if tokens_indexing is None:
        return None

    def _clone_value(value: Any) -> Any:
        if torch.is_tensor(value):
            if value.dim() == 0:
                return int(value.item())
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {k: _clone_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_clone_value(v) for v in value]
        return value

    return {key: _clone_value(val) for key, val in tokens_indexing.items()}


def run_repr_layer_image_token_injection_experiment(
    base_experiment_config: Dict[str, Any],
    model,
    tokenizer,
    image_processor,
    repr_layer_indices: List[int],
    bias_strength: Optional[float] = None,
    use_gaze_guidance: bool = True,
    guidance_config: Optional[Dict[str, Any]] = None,
    repr_target_layer_indices: Optional[List[int]] = None,
    enable_pairwise_sweep: bool = False,
    runs_per_combo: int = 2,
) -> Dict[str, Dict[str, Any]]:
    """
    Variation of the representation-layer sweep that alternates between full-image decoding
    and person-mask-filtered image tokens to study the impact of selective image context.
    The first run for each combination sends the full image to capture metadata, while
    every second run reuses cached hidden states and restricts the image tokens to the
    person mask selection during embedding construction.
    """
    if not repr_layer_indices:
        raise ValueError("repr_layer_indices must contain at least one layer index to sweep.")
    if runs_per_combo < 1:
        raise ValueError("runs_per_combo must be at least 1.")

    repr_layer_indices = [int(idx) for idx in repr_layer_indices]
    if repr_target_layer_indices is None and enable_pairwise_sweep:
        repr_target_layer_indices = repr_layer_indices
    target_indices = [int(idx) for idx in repr_target_layer_indices] if repr_target_layer_indices else repr_layer_indices

    combos: List[Tuple[int, int]]
    if enable_pairwise_sweep:
        combos = [
            (capture_idx, inject_idx)
            for capture_idx in repr_layer_indices
            for inject_idx in target_indices
        ]
    else:
        combos = [(idx, idx) for idx in repr_layer_indices]

    if not combos:
        raise ValueError("No representation layer combinations computed for sweep.")

    all_results: Dict[str, Dict[str, Any]] = {}
    base_output_dir = base_experiment_config.get("output_dir", "attention_output/repr_token_sweep")
    base_output_dir_path = Path(base_output_dir)
    base_output_dir_path.mkdir(parents=True, exist_ok=True)

    init_bias = bias_strength if bias_strength is not None else base_experiment_config.get(
        "generation_config", {}
    ).get("bias_strength", 0.0)

    effective_bias = bias_strength if bias_strength is not None else base_experiment_config.get(
        "generation_config", {}
    ).get("bias_strength", 0.0)

    required_cache_layers = sorted({idx for combo in combos for idx in combo})
    layer_hidden_state_cache: Dict[int, torch.Tensor] = {}
    print("Caching hidden states for token-injection sweep...")
    from gazefollow.extract_attention_direct_gt_masks import run_generation_with_attention

    if required_cache_layers:
        cache_layers_label = ", ".join(str(idx) for idx in required_cache_layers)
        print(f" - Capturing hidden states for layers [{cache_layers_label}] in a single initial run")
        cache_config = copy.deepcopy(base_experiment_config)
        cache_config["output_dir"] = f"{base_output_dir}/token_cache_all_layers"
        cache_config.setdefault("generation_config", {})["bias_strength"] = effective_bias
        attn_cfg = copy.deepcopy(cache_config.get("attention_config", {}))
        attn_cfg.pop("repr_capture_layer_idx", None)
        attn_cfg.pop("repr_inject_layer_idx", None)
        attn_cfg["repr_layer_idx"] = required_cache_layers[0]
        attn_cfg["return_full_hidden_states"] = True
        cache_config["attention_config"] = attn_cfg
        cache_results = run_generation_with_attention(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_path=cache_config["image_path"],
            mask_path=cache_config["mask_path"],
            prompt="",
            output_dir=cache_config["output_dir"],
            generation_config=cache_config["generation_config"],
            attention_config=cache_config.get("attention_config"),
            bias_strength=init_bias,
            break_after_first_step=True,
            use_gaze_guidance=use_gaze_guidance,
            guidance_config=guidance_config,
            save_debug_files=base_experiment_config.get("save_debug_files", False),
            use_gt_gaze_csv=cache_config.get("use_gt_gaze_csv", False),
            gt_gaze_csv_path=cache_config.get("gt_gaze_csv_path"),
            gt_gaze_mask_radius=cache_config.get("gt_gaze_mask_radius"),
            gt_gaze_mask_radius_ratio=cache_config.get("gt_gaze_mask_radius_ratio", 0.02),
            save_mask_overlays=cache_config.get("save_mask_overlays", False),
            mask_overlay_alpha=cache_config.get("mask_overlay_alpha", 0.4),
            use_target_insert_for_source=cache_config.get("use_target_insert_for_source", False),
        )
        all_hidden_states = cache_results.get("first_step_all_hidden_states")
        if not all_hidden_states:
            raise RuntimeError("Failed to retrieve full hidden state cache from initial run.")
        total_layers = len(all_hidden_states)
        for layer_idx in required_cache_layers:
            normalized_idx = layer_idx if layer_idx >= 0 else total_layers + layer_idx
            if normalized_idx < 0 or normalized_idx >= total_layers:
                raise ValueError(
                    f"Requested hidden-state layer {layer_idx} is out of range for {total_layers} total layers."
                )
            layer_hidden_state_cache[layer_idx] = all_hidden_states[normalized_idx]

    for capture_idx, inject_idx in combos:
        print(f"\n{'='*60}")
        print(f"Image token filtering sweep for repr layer combo: capture={capture_idx} inject={inject_idx}")
        experiment_config = copy.deepcopy(base_experiment_config)
        combo_label = f"cap{capture_idx}_inj{inject_idx}" if enable_pairwise_sweep else f"layer_{inject_idx}"
        experiment_config["output_dir"] = f"{base_output_dir}/{combo_label}"
        experiment_config.setdefault("generation_config", {})["bias_strength"] = effective_bias
        attn_cfg = copy.deepcopy(experiment_config.get("attention_config", {}))
        attn_cfg["repr_layer_idx"] = inject_idx
        if enable_pairwise_sweep:
            attn_cfg["repr_capture_layer_idx"] = capture_idx
            attn_cfg["repr_inject_layer_idx"] = inject_idx
        else:
            attn_cfg.pop("repr_capture_layer_idx", None)
            attn_cfg.pop("repr_inject_layer_idx", None)
        experiment_config["attention_config"] = attn_cfg

        if enable_pairwise_sweep:
            prev_hidden_state: Union[torch.Tensor, Dict[str, torch.Tensor]] = {
                "source": layer_hidden_state_cache[capture_idx],
                "target": layer_hidden_state_cache[inject_idx],
            }
        else:
            prev_hidden_state = layer_hidden_state_cache[inject_idx]

        captured_tokens_indexing: Optional[Dict[str, Any]] = None

        for run_iter in range(runs_per_combo):
            apply_filter = True
            run_label = f"{combo_label}_run{run_iter + 1}"
            run_output_dir = f"{experiment_config['output_dir']}/{run_label}"

            results = run_generation_with_attention(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                image_path=experiment_config["image_path"],
                mask_path=experiment_config["mask_path"],
                prompt=experiment_config["prompt"],
                output_dir=run_output_dir,
                generation_config=experiment_config["generation_config"],
                attention_config=experiment_config.get("attention_config"),
                bias_strength=experiment_config["generation_config"].get("bias_strength", 0.0),
                prev_run_last_hidden_state=prev_hidden_state,
                use_gaze_guidance=use_gaze_guidance,
                guidance_config=guidance_config,
                save_debug_files=base_experiment_config.get(
                    "save_debug_files",
                    base_experiment_config.get("generation_config", {}).get("save_debug_files", False)
                ),
                use_gt_gaze_csv=experiment_config.get("use_gt_gaze_csv", False),
                gt_gaze_csv_path=experiment_config.get("gt_gaze_csv_path"),
                gt_gaze_mask_radius=experiment_config.get("gt_gaze_mask_radius"),
                gt_gaze_mask_radius_ratio=experiment_config.get("gt_gaze_mask_radius_ratio", 0.02),
                save_mask_overlays=experiment_config.get("save_mask_overlays", False),
                mask_overlay_alpha=experiment_config.get("mask_overlay_alpha", 0.4),
                use_target_insert_for_source=experiment_config.get("use_target_insert_for_source", False),
                include_image_inputs=True,
                filter_image_tokens_to_person_mask=apply_filter,
            )
            results.pop("first_step_hidden_state", None)

            if run_iter == 0:
                captured_tokens_indexing = _clone_tokens_indexing(
                    getattr(model, "tokens_indexing", None)
                )
                if captured_tokens_indexing is not None:
                    results["captured_tokens_indexing"] = captured_tokens_indexing

            results["repr_capture_layer_idx"] = capture_idx if enable_pairwise_sweep else inject_idx
            results["repr_inject_layer_idx"] = inject_idx
            results["run_iteration"] = run_iter + 1
            results["image_token_filter_applied"] = apply_filter
            results["image_token_filter_source"] = "person_mask" if apply_filter else "none"
            results["combo_label"] = combo_label

            all_results[run_label] = results
            if apply_filter:
                print(f"Finished run {run_iter + 1} for {combo_label} with person-mask image filtering.")
            else:
                print(f"Finished run {run_iter + 1} for {combo_label} without image filtering.")

    if not all_results:
        print("Image token filtering sweep produced no results.")
        return {}

    print(f"\n{'='*80}")
    print("REPRESENTATION LAYER TOKEN INJECTION SWEEP SUMMARY")
    print(f"{'='*80}")

    performance_summary: List[Dict[str, Any]] = []
    for run_label, result in all_results.items():
        evaluation_summary = result.get("evaluation_summary") or {}
        quality_analysis = result.get("quality_analysis") or {}
        attention_correlation = result.get("attention_correlation") or {}

        avg_confidence = (evaluation_summary.get("average_confidence") or {}).get("confidence_score", 0.0)
        avg_entropy = (evaluation_summary.get("average_confidence") or {}).get("entropy", float("inf"))
        correlation_score = attention_correlation.get("normalized_correlation_score", 0.0)
        generated_text = result.get("generated_text", "") or ""
        num_tokens = result.get("num_tokens", 0)
        overall_quality_score = quality_analysis.get("overall_quality_score", 0.0) if quality_analysis else 0.0
        capture_idx = result.get("repr_capture_layer_idx")
        if capture_idx is None:
            capture_idx = result.get("repr_inject_layer_idx")
        inject_idx = result.get("repr_inject_layer_idx", capture_idx)

        performance_summary.append({
            "run_label": run_label,
            "repr_capture_layer_idx": capture_idx,
            "repr_inject_layer_idx": inject_idx,
            "overall_quality_score": overall_quality_score,
            "avg_confidence": avg_confidence,
            "avg_entropy": avg_entropy,
            "correlation_score": correlation_score,
            "generated_text": generated_text,
            "num_tokens": num_tokens,
            "image_token_filter_applied": result.get("image_token_filter_applied", False),
            "run_iteration": result.get("run_iteration"),
        })

    performance_summary.sort(key=lambda x: x["overall_quality_score"], reverse=True)

    if performance_summary:
        heading = "TOP 5 IMAGE-FILTERING RUNS"
        print(f"\n{heading}:")
        if enable_pairwise_sweep:
            header = (
                f"{'Rank':<4} {'Run':<18} {'Cap':<5} {'Inj':<5} {'Filter':<7} {'Quality':<8} "
                f"{'Conf':<7} {'Entropy':<8} {'Corr':<6} {'Tokens':<6}"
            )
        else:
            header = (
                f"{'Rank':<4} {'Run':<18} {'Layer':<7} {'Filter':<7} {'Quality':<8} "
                f"{'Conf':<7} {'Entropy':<8} {'Corr':<6} {'Tokens':<6}"
            )
        print(header)
        print("-" * len(header))
        for i, result in enumerate(performance_summary[:5], 1):
            inject_flag = "Y" if result.get("image_token_filter_applied") else "N"
            capture_idx = result.get("repr_capture_layer_idx")
            if capture_idx is None:
                capture_idx = result.get("repr_inject_layer_idx")
            inject_idx = result.get("repr_inject_layer_idx")
            if enable_pairwise_sweep:
                print(
                    f"{i:<4} {result['run_label']:<18} {capture_idx:<5} "
                    f"{inject_idx:<5} {inject_flag:<7} {result['overall_quality_score']:<8.2f} "
                    f"{result['avg_confidence']:<7.3f} {result['avg_entropy']:<8.3f} "
                    f"{result['correlation_score']:<6.3f} {result['num_tokens']:<6}"
                )
            else:
                print(
                    f"{i:<4} {result['run_label']:<18} {inject_idx:<7} "
                    f"{inject_flag:<7} {result['overall_quality_score']:<8.2f} "
                    f"{result['avg_confidence']:<7.3f} {result['avg_entropy']:<8.3f} "
                    f"{result['correlation_score']:<6.3f} {result['num_tokens']:<6}"
                )

        worst_start = max(len(performance_summary) - 3, 0)
        if worst_start < len(performance_summary):
            print(f"\nWORST {len(performance_summary) - worst_start} IMAGE-FILTERING RUNS:")
            print(header)
            print("-" * len(header))
            for i, result in enumerate(performance_summary[worst_start:], worst_start + 1):
                inject_flag = "Y" if result.get("image_token_filter_applied") else "N"
                capture_idx = result.get("repr_capture_layer_idx")
                if capture_idx is None:
                    capture_idx = result.get("repr_inject_layer_idx")
                inject_idx = result.get("repr_inject_layer_idx")
                if enable_pairwise_sweep:
                    print(
                        f"{i:<4} {result['run_label']:<18} {capture_idx:<5} "
                        f"{inject_idx:<5} {inject_flag:<7} {result['overall_quality_score']:<8.2f} "
                        f"{result['avg_confidence']:<7.3f} {result['avg_entropy']:<8.3f} "
                        f"{result['correlation_score']:<6.3f} {result['num_tokens']:<6}"
                    )
                else:
                    print(
                        f"{i:<4} {result['run_label']:<18} {inject_idx:<7} "
                        f"{inject_flag:<7} {result['overall_quality_score']:<8.2f} "
                        f"{result['avg_confidence']:<7.3f} {result['avg_entropy']:<8.3f} "
                        f"{result['correlation_score']:<6.3f} {result['num_tokens']:<6}"
                    )

        best_result = performance_summary[0]
        best_capture_idx = best_result.get("repr_capture_layer_idx")
        if best_capture_idx is None:
            best_capture_idx = best_result.get("repr_inject_layer_idx")
        best_inject_idx = best_result.get("repr_inject_layer_idx")
        recommended = (
            f"capture={best_capture_idx} -> inject={best_inject_idx}"
            if enable_pairwise_sweep else
            f"layer {best_inject_idx}"
        )
        print(f"\n🏆 RECOMMENDED CONFIGURATION: {recommended} (run={best_result['run_label']})")
        print(f"   • Overall Quality Score: {best_result['overall_quality_score']:.2f}/10")
        print(f"   • Average Confidence: {best_result['avg_confidence']:.3f}")
        print(f"   • Average Entropy: {best_result['avg_entropy']:.3f}")
        print(f"   • Attention Correlation: {best_result['correlation_score']:.3f}")
        print(f"   • Image Token Filter Applied: {'Yes' if best_result.get('image_token_filter_applied') else 'No'}")
        print(f"   • Generated Text: '{best_result['generated_text']}'")
    else:
        print("No image-filtering runs produced evaluation summaries.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_payload = {
        "timestamp": timestamp,
        "repr_layer_indices": repr_layer_indices,
        "repr_target_layer_indices": target_indices,
        "pairwise_sweep_enabled": enable_pairwise_sweep,
        "bias_strength_used": effective_bias,
        "runs_per_combo": runs_per_combo,
        "image_token_filter_results": all_results,
        "performance_summary": performance_summary,
    }
    saved_results_path = save_image_results(
        sweep_payload,
        base_output_dir_path,
        prefix="repr_layer_image_token_filter_results",
        append_mode=False
    )
    print(f"\n📁 Image token-filter sweep results saved to: {saved_results_path}")
    print(f"{'='*80}")

    return all_results
