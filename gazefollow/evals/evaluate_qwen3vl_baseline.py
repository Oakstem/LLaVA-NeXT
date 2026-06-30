#!/usr/bin/env python3
"""
Evaluate the base Qwen3-VL model on Gazefollow-style JSON datasets.

This mirrors evaluate_model_qwen3vl.py at the artifact/metric level, but uses a
Qwen3-VL instruction model for the answer generation step instead of a trained
LLaVA checkpoint. Grounding still defaults to Qwen3-VL-4B-Instruct.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from gazefollow.auto_phrase_grounding.qwen3vl_grounding import load_qwen3vl_model
from gazefollow.data_proc.add_in_out_labels import load_in_out_lookup
from gazefollow.evals.metric_utils import (
    filter_gaze_metrics,
    flatten_recomputed_metrics,
    summarize_metrics,
)
from evaluate_model_qwen3vl import (
    DEFAULT_TEST_CSV,
    DEFAULT_TRAIN_CSV,
    EvaluationSampleOutput,
    GazeEvaluationState,
    calculate_basic_metrics,
    extract_ground_truth_from_conversation,
    extract_prompt_from_conversation,
    infer_predicted_in_out,
    load_dataset,
    process_sample_with_qwen_grounding,
    wandb_history_key,
)
from log_wandb_evaluations import (
    DEFAULT_PROJECT,
    GENERATION_TABLE_COLUMNS,
    format_generation_sample,
    split_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL-8B-Instruct as a Gazefollow prediction baseline."
    )
    parser.add_argument(
        "--prediction-model-id",
        "--model-id",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Qwen3-VL model identifier used to generate Gazefollow predictions.",
    )
    parser.add_argument(
        "--prediction-device",
        default=None,
        help="Device map for the prediction model (defaults to 'auto').",
    )
    parser.add_argument("--dataset-json", required=True, help="Path to JSON dataset file.")
    parser.add_argument("--images-dir", required=True, help="Directory containing the images.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to evaluate.")
    parser.add_argument(
        "--prompt-override",
        default=None,
        help="Override the prompt used for Qwen generation instead of the dataset human prompt.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful vision-language assistant.",
        help="System prompt passed to the prediction model.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum generated answer tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling value.")
    parser.add_argument("--num-beams", type=int, default=1, help="Number of beams for beam search.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling instead of greedy decoding.")
    parser.add_argument("--output-dir", default="./evaluation_results", help="Directory for evaluation artifacts.")
    parser.add_argument("--save-predictions", action="store_true", help="Save predictions.json.")
    parser.add_argument(
        "--generate-model-results",
        dest="generate_model_results",
        action="store_true",
        help="Run Qwen3-VL grounding and save model_generation_results.json (default).",
    )
    parser.add_argument(
        "--no-generate-model-results",
        dest="generate_model_results",
        action="store_false",
        help="Only save text-generation metrics, without grounding.",
    )
    parser.set_defaults(generate_model_results=True)
    parser.add_argument(
        "--table-log-interval",
        type=int,
        default=25,
        help="Number of samples between generation table streaming logs (<=0 disables periodic streaming).",
    )
    parser.add_argument(
        "--table-log-file",
        default=None,
        help="Optional JSONL file for streamed generation rows (defaults to <output_dir>/generation_progress.jsonl).",
    )
    parser.add_argument(
        "--gaze-model-id",
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Qwen3-VL model identifier used for gaze target grounding.",
    )
    parser.add_argument("--gaze-box-threshold", type=float, default=0.20)
    parser.add_argument("--gaze-text-threshold", type=float, default=0.20)
    parser.add_argument("--gaze-device", default=None, help="Device map for grounding model (defaults to 'auto').")
    parser.add_argument("--gaze-max-new-tokens", type=int, default=192)
    parser.add_argument("--gaze-temperature", type=float, default=0.0)
    parser.add_argument("--gaze-iou-radius-ratio", type=float, default=0.05)
    parser.add_argument(
        "--in-out-labels-csv",
        default=None,
        help="Optional CSV with precomputed in/out labels. Defaults to train/test CSV by dataset name.",
    )
    parser.add_argument(
        "--log-to-wandb",
        dest="log_to_wandb",
        action="store_true",
        help="Log evaluation results to Weights & Biases (default).",
    )
    parser.add_argument("--no-log-to-wandb", dest="log_to_wandb", action="store_false")
    parser.set_defaults(log_to_wandb=True)
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", DEFAULT_PROJECT))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY") or None)
    parser.add_argument("--wandb-run-id", default=os.getenv("WANDB_RUN_ID") or None)
    parser.add_argument("--wandb-run-name", default=os.getenv("WANDB_RUN_NAME") or None)
    parser.add_argument("--wandb-metric-prefix", default=os.getenv("WANDB_METRIC_PREFIX", ""))
    parser.add_argument("--verbose", action="store_true", default=True)
    return parser.parse_args()


def resolve_image_path(sample: Dict[str, Any], images_dir: Path, dataset_json: Path) -> Optional[Path]:
    image_file = sample.get("image", "")
    if isinstance(image_file, list):
        image_file = image_file[0] if image_file else ""
    if not image_file and "image_path" in sample:
        image_file = sample.get("image_path", "")
    if not image_file:
        return None

    candidate = Path(str(image_file))
    if not candidate.is_absolute():
        candidate = images_dir / candidate
    if candidate.exists():
        return candidate

    if candidate.suffix:
        return None
    for ext in (".jpg", ".png", ".jpeg", ".bmp", ".gif", ".tiff"):
        alt = candidate.with_suffix(ext)
        if alt.exists():
            return alt

    fallback = dataset_json.parent / str(image_file)
    if fallback.exists():
        return fallback
    return None


def generate_qwen3vl_response(
    *,
    processor: Any,
    model: Any,
    image: Image.Image,
    prompt_text: str,
    system_prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    num_beams: int,
) -> str:
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    )

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(model.device)
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if num_beams and num_beams > 1:
        generation_kwargs["num_beams"] = num_beams
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)

    input_length = inputs.input_ids.shape[-1]
    generated_ids = generated[:, input_length:]
    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def generate_sample_outputs(args: argparse.Namespace, dataset_samples: List[Dict[str, Any]]) -> Tuple[List[EvaluationSampleOutput], List[Dict[str, Any]], Dict[str, Any]]:
    images_dir = Path(args.images_dir)
    dataset_json = Path(args.dataset_json)
    device_map = args.prediction_device or "auto"

    print(f"Loading prediction model ({args.prediction_model_id}) with device map {device_map}...")
    processor, model = load_qwen3vl_model(args.prediction_model_id, device_map=device_map)
    model.eval()

    predictions: List[str] = []
    ground_truths: List[str] = []
    sample_outputs: List[EvaluationSampleOutput] = []
    failed_samples: List[Dict[str, Any]] = []

    for idx, sample in enumerate(tqdm(dataset_samples, desc="Generating Qwen3-VL predictions")):
        sample_id = str(sample.get("id", idx))
        conversations = sample.get("conversations", [])
        dataset_prompt = extract_prompt_from_conversation(conversations)
        ground_truth = extract_ground_truth_from_conversation(conversations)
        if not ground_truth:
            failed_samples.append({"index": idx, "id": sample_id, "reason": "Missing ground truth"})
            continue

        prompt_used = args.prompt_override if args.prompt_override is not None else dataset_prompt
        prompt_source = "override" if args.prompt_override is not None else "dataset"
        if not prompt_used:
            failed_samples.append(
                {
                    "index": idx,
                    "id": sample_id,
                    "reason": "Missing prompt",
                    "dataset_prompt": dataset_prompt,
                    "prompt_override": args.prompt_override,
                }
            )
            continue

        image_path = resolve_image_path(sample, images_dir, dataset_json)
        if image_path is None:
            failed_samples.append({"index": idx, "id": sample_id, "reason": "Image not found"})
            continue

        image = Image.open(image_path).convert("RGB")
        try:
            prediction = generate_qwen3vl_response(
                processor=processor,
                model=model,
                image=image,
                prompt_text=prompt_used,
                system_prompt=args.system_prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
            )
        finally:
            image_size = image.size
            image.close()

        predictions.append(prediction)
        ground_truths.append(ground_truth)
        sample_outputs.append(
            EvaluationSampleOutput(
                index=idx,
                sample=sample,
                sample_id=sample_id,
                dataset_prompt=dataset_prompt,
                prompt_used=prompt_used,
                prompt_source=prompt_source,
                ground_truth=ground_truth,
                prediction=prediction,
                image_path=str(image_path),
                image_size=image_size,
                loss=None,
                predicted_in_out=infer_predicted_in_out(prediction),
                roi_eval=None,
            )
        )

        if (idx + 1) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = calculate_basic_metrics(predictions, ground_truths)
    metrics.update(
        {
            "eval_samples": len(dataset_samples),
            "eval_successful": len(predictions),
            "eval_failed": len(failed_samples),
            "eval_success_rate": len(predictions) / len(dataset_samples) if dataset_samples else 0.0,
        }
    )

    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return sample_outputs, failed_samples, metrics


def main() -> None:
    args = parse_args()
    if args.in_out_labels_csv is None:
        if "test" in args.dataset_json.lower():
            args.in_out_labels_csv = DEFAULT_TEST_CSV
            print(f"Using default test CSV for in/out labels: {DEFAULT_TEST_CSV}")
        else:
            args.in_out_labels_csv = DEFAULT_TRAIN_CSV
            print(f"Using default train CSV for in/out labels: {DEFAULT_TRAIN_CSV}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_log_interval = max(args.table_log_interval, 0)
    progress_log_path = Path(args.table_log_file) if args.table_log_file else output_dir / "generation_progress.jsonl"
    if args.generate_model_results:
        progress_log_path.parent.mkdir(parents=True, exist_ok=True)
        progress_log_path.unlink(missing_ok=True)

    print("=" * 60)
    print("Qwen3-VL Baseline Gazefollow Evaluation")
    print("=" * 60)
    print("\n1. Loading dataset...")
    dataset_samples = load_dataset(args.dataset_json, args.limit)

    print("\n2. Generating predictions...")
    start_time = time.time()
    sample_outputs, failed_samples, metrics_result = generate_sample_outputs(args, dataset_samples)

    images_dir = Path(args.images_dir)
    in_out_lookup = load_in_out_lookup(Path(args.in_out_labels_csv)) if args.in_out_labels_csv else None
    evaluation_state = GazeEvaluationState()
    gaze_device_map = args.gaze_device or "auto"
    gaze_processor = None
    gaze_model = None

    def ensure_gaze_resources() -> Tuple[Any, Any]:
        nonlocal gaze_processor, gaze_model
        if gaze_processor is None or gaze_model is None:
            print(f"Loading grounding model ({args.gaze_model_id}) with device map {gaze_device_map}...")
            gaze_processor, gaze_model = load_qwen3vl_model(args.gaze_model_id, device_map=gaze_device_map)
            gaze_model.eval()
        return gaze_processor, gaze_model

    if args.generate_model_results:
        print("\n3. Running grounding metrics...")
        for sample_output in tqdm(sample_outputs, desc="Grounding Qwen3-VL predictions"):
            process_sample_with_qwen_grounding(
                sample_output,
                args=args,
                images_dir=images_dir,
                state=evaluation_state,
                in_out_lookup=in_out_lookup,
                ensure_gaze_resources=ensure_gaze_resources,
                generate_model_results=True,
                gaze_device_map=gaze_device_map,
            )
            if evaluation_state.generation_rows_buffer:
                if table_log_interval <= 0 or len(evaluation_state.generation_rows_buffer) < table_log_interval:
                    continue
                progress_log_path.parent.mkdir(parents=True, exist_ok=True)
                with progress_log_path.open("a", encoding="utf-8") as handle:
                    for row in evaluation_state.generation_rows_buffer:
                        handle.write(json.dumps(row, ensure_ascii=False))
                        handle.write("\n")
                evaluation_state.generation_rows_buffer.clear()

        if evaluation_state.generation_rows_buffer:
            progress_log_path.parent.mkdir(parents=True, exist_ok=True)
            with progress_log_path.open("a", encoding="utf-8") as handle:
                for row in evaluation_state.generation_rows_buffer:
                    handle.write(json.dumps(row, ensure_ascii=False))
                    handle.write("\n")
            evaluation_state.generation_rows_buffer.clear()
    else:
        print("\n3. Skipping grounding metrics (--no-generate-model-results).")
        for sample_output in sample_outputs:
            evaluation_state.predictions_output.append(
                {
                    "id": sample_output.sample_id,
                    "image_path": sample_output.image_path,
                    "prompt": sample_output.prompt_used,
                    "ground_truth": sample_output.ground_truth,
                    "prompt_source": sample_output.prompt_source,
                    "prediction": sample_output.prediction,
                    "predicted_in_out": sample_output.predicted_in_out,
                    "success": True,
                }
            )

    evaluation_time = time.time() - start_time
    predictions_output = evaluation_state.predictions_output
    model_generation_records = evaluation_state.model_generation_records

    total_samples = len(dataset_samples)
    processed_samples = len(sample_outputs)
    successful_predictions = sum(1 for output in sample_outputs if output.prediction is not None)
    final_metrics = dict(metrics_result)
    final_metrics.update(
        {
            "total_samples": total_samples,
            "successfully_processed_samples": processed_samples,
            "successful_predictions": successful_predictions,
            "failed_samples": len(failed_samples),
            "success_rate": successful_predictions / total_samples if total_samples else 0.0,
            "processing_rate": processed_samples / total_samples if total_samples else 0.0,
            "evaluation_time_seconds": evaluation_time,
            "average_time_per_sample": evaluation_time / processed_samples if processed_samples else 0.0,
            "model_generation_samples": len(model_generation_records),
            "skipped_in_out_minus_one": len(evaluation_state.skipped_in_out_minus_one_samples),
        }
    )

    in_out_predictions: List[int] = []
    in_out_labels: List[int] = []
    for entry in predictions_output:
        gt_label = entry.get("gt_in_out")
        predicted_flag = entry.get("predicted_in_out")
        if isinstance(predicted_flag, int) and isinstance(gt_label, int):
            in_out_predictions.append(predicted_flag)
            in_out_labels.append(gt_label)

    metric_summary = None
    if model_generation_records:
        total_counts, filtered_counts = filter_gaze_metrics(
            model_generation_records,
            keep_metric=lambda record: record.get("gt_in_out") == 1 and record.get("predicted_in_out") == 1,
            mutate=True,
        )
        metric_summary = summarize_metrics(
            model_generation_records,
            in_out_predictions,
            in_out_labels,
            total_counts,
            filtered_counts,
        )
        final_metrics.update(flatten_recomputed_metrics(metric_summary))
        final_metrics["gaze_metrics"] = metric_summary.get("gaze_metrics")
        if "inout_precision" in metric_summary:
            final_metrics["inout_precision"] = metric_summary["inout_precision"]
        if "inout_confusion" in metric_summary:
            final_metrics["inout_confusion"] = metric_summary["inout_confusion"]
        final_metrics["samples_evaluated"] = metric_summary.get("samples_evaluated", 0)

    print("\nEvaluation summary:")
    print(f"  Total samples: {total_samples}")
    print(f"  Processed samples: {processed_samples}")
    print(f"  Successful predictions: {successful_predictions}")
    print(f"  Failed samples: {len(failed_samples)}")
    print(f"  Success rate: {final_metrics['success_rate']:.4f}")
    if metric_summary:
        l2_mean = ((metric_summary.get("gaze_metrics") or {}).get("gaze_l2_error") or {}).get("mean")
        if isinstance(l2_mean, (int, float)) and math.isfinite(l2_mean):
            print(f"  Gaze L2 mean: {l2_mean:.4f}")

    print(f"\n4. Saving results to {output_dir}...")
    metrics_file = output_dir / "metrics.json"
    with metrics_file.open("w", encoding="utf-8") as handle:
        json.dump(final_metrics, handle, indent=2, ensure_ascii=False)
    print(f"Metrics saved to: {metrics_file}")

    predictions_file = output_dir / "predictions.json"
    if args.save_predictions:
        with predictions_file.open("w", encoding="utf-8") as handle:
            json.dump(predictions_output, handle, indent=2, ensure_ascii=False)
        print(f"Detailed predictions saved to: {predictions_file}")

    model_generation_file: Optional[Path] = None
    if args.generate_model_results:
        model_generation_file = output_dir / "model_generation_results.json"
        with model_generation_file.open("w", encoding="utf-8") as handle:
            json.dump(model_generation_records, handle, indent=2, ensure_ascii=False)
        print(f"Model generation results saved to: {model_generation_file}")

    failed_file = output_dir / "failed_samples.json"
    if failed_samples:
        with failed_file.open("w", encoding="utf-8") as handle:
            json.dump(failed_samples, handle, indent=2, ensure_ascii=False)
        print(f"Failed samples saved to: {failed_file}")

    config_file = output_dir / "evaluation_config.json"
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "num_beams": args.num_beams,
    }
    config = {
        "prediction_model_id": args.prediction_model_id,
        "gaze_model_id": args.gaze_model_id,
        "dataset_json": args.dataset_json,
        "images_dir": args.images_dir,
        "prompt_override": args.prompt_override,
        "system_prompt": args.system_prompt,
        "generation_kwargs": generation_kwargs,
        "gaze_max_new_tokens": args.gaze_max_new_tokens,
        "gaze_box_threshold": args.gaze_box_threshold,
        "gaze_iou_radius_ratio": args.gaze_iou_radius_ratio,
        "in_out_labels_csv": args.in_out_labels_csv,
        "limit": args.limit,
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "table_log_interval": table_log_interval,
        "table_log_file": str(progress_log_path),
    }
    with config_file.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    print(f"Evaluation configuration saved to: {config_file}")

    if evaluation_state.dataset_updated and evaluation_state.dataset_gt_updates:
        from gazefollow.gaze_metrics import persist_ground_truth_updates

        persisted = persist_ground_truth_updates(Path(args.dataset_json), evaluation_state.dataset_gt_updates)
        if persisted:
            print(f"Persisted gaze ground truth for {persisted} samples to {args.dataset_json}")

    if args.log_to_wandb:
        wandb_run = None
        try:
            import wandb

            run_name = args.wandb_run_name or "qwen3vl-8b-baseline"
            init_kwargs: Dict[str, Any] = {
                "project": args.wandb_project,
                "name": run_name,
                "config": {**config, "evaluation_dir": str(output_dir)},
            }
            if args.wandb_entity:
                init_kwargs["entity"] = args.wandb_entity
            if args.wandb_run_id:
                init_kwargs["id"] = args.wandb_run_id
                init_kwargs["resume"] = "allow"
            wandb_run = wandb.init(**init_kwargs)
            scalar_metrics, non_scalar_metrics = split_metrics(final_metrics)
            prefix = str(args.wandb_metric_prefix or "").strip().strip("/")
            wandb_run.config.update(
                {
                    "metrics_path": str(metrics_file),
                    "config_path": str(config_file),
                    "predictions_path": str(predictions_file) if args.save_predictions else None,
                    "model_generation_path": str(model_generation_file) if model_generation_file else None,
                    **non_scalar_metrics,
                },
                allow_val_change=True,
            )
            wandb_run.log({wandb_history_key(prefix, key): value for key, value in scalar_metrics.items()})
            if model_generation_records:
                table = wandb.Table(columns=GENERATION_TABLE_COLUMNS)
                for record in model_generation_records:
                    for row in format_generation_sample(record):
                        table.add_data(*(row.get(column) for column in GENERATION_TABLE_COLUMNS))
                wandb_run.log({wandb_history_key(prefix, "generation_results"): table}, commit=False)
            print(f"Logged evaluation results to wandb run: {run_name}")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to log evaluation results to wandb: {exc}")
        finally:
            if wandb_run is not None:
                wandb_run.finish()

    if evaluation_state.missing_in_out_samples:
        print(f"Missing in_out labels for {len(evaluation_state.missing_in_out_samples)} samples.")
    if evaluation_state.skipped_in_out_minus_one_samples:
        print(f"Skipped in_out=-1 for {len(evaluation_state.skipped_in_out_minus_one_samples)} samples.")
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
