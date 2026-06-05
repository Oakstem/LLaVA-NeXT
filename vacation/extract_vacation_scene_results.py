#!/usr/bin/env python3
"""Run LLaVA video inference per Vacation scene and align results to annotations."""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gazefollow.extract_inference_representations import (  # noqa: E402
    DEFAULT_ADAPTER_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROMPTS,
    _build_multimodal_prompt,
    fix_wsl_paths,
    prepare_multimodal_video_tensor,
    resolve_device,
)
from vacation.analyze_vacation_video_results import (  # noqa: E402
    find_baseline,
    format_summary,
    load_scene_records,
    run_name,
    write_details_csv,
)


DEFAULT_ANNOTATIONS_CSV = "datasets/Vacation/test_annotations_with_scene_id.csv"
DEFAULT_VIDEOS_DIR = "datasets/Vacation/Videos"
DEFAULT_OUTPUT_CSV = "vacation_results/videos/test_annotations_with_scene_results.csv"
DEFAULT_PROMPT_FILE = "docs/vacation_event_level_gaze_communication_labels_concise.txt"
GAZE_LABELS = [
    "SingleGaze",
    "Mutual Gaze",
    "Gaze Aversion",
    "Gaze Following",
    "Joint Attention",
]
GAZE_LABEL_ALIASES = {
    "SingleGaze": ["SingleGaze", "Single Gaze", "Non-communicative", "Non communicative"],
    "Mutual Gaze": ["Mutual Gaze"],
    "Gaze Aversion": ["Gaze Aversion"],
    "Gaze Following": ["Gaze Following"],
    "Joint Attention": ["Joint Attention"],
}


def normalize_label_text(text: str) -> str:
    return re.sub(r"[\s-]+", "", text.casefold())


def extract_gaze_label(text: str) -> Optional[str]:
    normalized_text = normalize_label_text(text or "")
    matches: List[Tuple[int, str]] = []
    for label in GAZE_LABELS:
        for alias in GAZE_LABEL_ALIASES[label]:
            match_index = normalized_text.rfind(normalize_label_text(alias))
            if match_index >= 0:
                matches.append((match_index, label))
    return max(matches)[1] if matches else None


def relative_display_path(path: Path) -> str:
    resolved_path = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(resolved_path.relative_to(cwd))
    except ValueError:
        return os.path.relpath(resolved_path, cwd)


def parse_scene_number(scene_id: Any) -> int:
    text = str(scene_id)
    return int(text.rsplit("_", 1)[-1]) if "_" in text else int(text)


def extract_sequences_from_generate_output(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if hasattr(output, "sequences"):
        output = output.sequences
    if isinstance(output, torch.Tensor):
        return output[0] if output.ndim == 2 else output
    if isinstance(output, (list, tuple)) and output:
        first = output[0]
        return first[0] if isinstance(first, torch.Tensor) and first.ndim == 2 else first
    raise TypeError(f"Unsupported generation output type: {type(output)!r}")


def load_scene_frames(
    vr,
    start_frame: int,
    end_frame: int,
    max_frames_num: int,
) -> Tuple[np.ndarray, List[int]]:
    if max_frames_num < 1:
        raise ValueError("--max-frames-num must be positive.")
    total_frames = len(vr)
    if total_frames < 1:
        raise ValueError("Video contains no frames.")
    start_frame = max(0, min(int(start_frame), total_frames - 1))
    end_frame = max(start_frame, min(int(end_frame), total_frames - 1))
    num_frames = min(max_frames_num, end_frame - start_frame + 1)
    frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int).tolist()
    frames = vr.get_batch(frame_indices).asnumpy()
    return frames, frame_indices


def generate_scene_text(
    prompt: str,
    tokenizer,
    model,
    images,
    image_sizes: List[List[int]],
    conv_template: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: Optional[float],
    num_beams: int,
) -> str:
    input_ids = _build_multimodal_prompt(prompt, tokenizer, conv_template).to(resolve_device(model))
    gen_kwargs: Dict[str, Any] = {
        "inputs": input_ids,
        "images": images,
        "image_sizes": image_sizes,
        "modalities": ["video"],
        "do_sample": do_sample,
        "num_beams": num_beams,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        output = model.generate(**{key: value for key, value in gen_kwargs.items() if value is not None})
    output_ids = extract_sequences_from_generate_output(output)
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def build_scene_table(df: pd.DataFrame) -> pd.DataFrame:
    scene_rows = (
        df.groupby(["video_id", "scene_id"], sort=False)
        .agg(scene_start_frame=("frame_id", "min"), scene_end_frame=("frame_id", "max"))
        .reset_index()
    )
    scene_rows["scene_sort_key"] = scene_rows["scene_id"].map(parse_scene_number)
    return scene_rows.sort_values(["video_id", "scene_sort_key"]).drop(columns=["scene_sort_key"])


def run_vacation_scene_inference(args: argparse.Namespace) -> pd.DataFrame:
    from decord import VideoReader, cpu

    if args.prompt_file:
        args.prompt = Path(fix_wsl_paths(args.prompt_file)).read_text(encoding="utf-8").strip()
        if not args.prompt:
            raise ValueError(f"--prompt-file is empty: {args.prompt_file}")

    annotations_csv = Path(fix_wsl_paths(args.annotations_csv))
    videos_dir = Path(fix_wsl_paths(args.videos_dir))
    output_csv = Path(fix_wsl_paths(args.output_csv))

    df = pd.read_csv(annotations_csv)
    scene_table = build_scene_table(df)
    if args.video_ids:
        video_ids = {int(video_id) for video_id in args.video_ids}
        scene_table = scene_table[scene_table["video_id"].isin(video_ids)]
    if args.limit_scenes is not None:
        scene_table = scene_table.head(args.limit_scenes)

    for column in [
        "scene_start_frame",
        "scene_end_frame",
        "scene_sampled_frame_indices",
        "scene_generated_text",
        "scene_extracted_label",
    ]:
        if column not in df.columns:
            df[column] = None

    if not args.disable_optimizations:
        from generation_utils import enable_inference_optimizations

        enable_inference_optimizations()

    from generation_utils import load_model_and_setup

    tokenizer, model, image_processor, _ = load_model_and_setup(
        model_path=args.model_path,
        model_base=args.model_base,
        adapter_path=args.adapter_path,
        attn_implementation=args.attn_implementation,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        attn_layer_ind=-1,
    )

    processed = 0
    for video_id, video_scenes in scene_table.groupby("video_id", sort=False):
        video_path = videos_dir / f"{int(video_id)}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        vr = VideoReader(str(video_path), ctx=cpu(0))
        fps = float(vr.get_avg_fps())
        total_frames = len(vr)
        print(f"Video {video_id}: {len(video_scenes)} scenes, {total_frames} frames, {fps:.3f} fps", flush=True)

        for row in video_scenes.itertuples(index=False):
            frames, frame_indices = load_scene_frames(
                vr=vr,
                start_frame=int(row.scene_start_frame),
                end_frame=int(row.scene_end_frame),
                max_frames_num=args.max_frames_num,
            )
            images, image_sizes = prepare_multimodal_video_tensor(frames, image_processor, model)
            generated_text = generate_scene_text(
                prompt=args.prompt,
                tokenizer=tokenizer,
                model=model,
                images=images,
                image_sizes=image_sizes,
                conv_template=args.conv_template,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
            )

            mask = (df["video_id"] == row.video_id) & (df["scene_id"] == row.scene_id)
            df.loc[mask, "scene_start_frame"] = int(row.scene_start_frame)
            df.loc[mask, "scene_end_frame"] = int(row.scene_end_frame)
            df.loc[mask, "scene_sampled_frame_indices"] = json.dumps(frame_indices)
            df.loc[mask, "scene_generated_text"] = generated_text
            df.loc[mask, "scene_extracted_label"] = extract_gaze_label(generated_text)

            processed += 1
            print(
                f"[{processed}/{len(scene_table)}] video={row.video_id} scene={row.scene_id} "
                f"frames={row.scene_start_frame}-{row.scene_end_frame}: {generated_text[:120]!r}",
                flush=True,
            )
            if args.save_every and processed % args.save_every == 0:
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_csv, index=False)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def run_post_eval(args: argparse.Namespace) -> None:
    output_csv = Path(fix_wsl_paths(args.output_csv)).resolve()
    result_files = [output_csv]
    baseline_csv = args.eval_baseline_csv
    if baseline_csv:
        baseline_path = Path(fix_wsl_paths(baseline_csv)).resolve()
    else:
        candidates = sorted(
            path.resolve()
            for path in output_csv.parent.glob("test_annotations_with_scene_results*.csv")
            if path.is_file()
        )
        baseline_path = find_baseline(candidates, None).resolve()

    if baseline_path != output_csv:
        result_files.insert(0, baseline_path)

    source_files = {run_name(path): path for path in result_files}
    all_records = {name: load_scene_records(path) for name, path in source_files.items()}
    baseline_name = run_name(baseline_path)
    eval_output_dir = output_csv.parent / "analysis"
    summary_txt = (
        Path(fix_wsl_paths(args.eval_summary_txt))
        if args.eval_summary_txt
        else eval_output_dir / f"{output_csv.stem}_analysis_summary.txt"
    )
    details_csv = (
        Path(fix_wsl_paths(args.eval_details_csv))
        if args.eval_details_csv
        else eval_output_dir / f"{output_csv.stem}_analysis_details.csv"
    )

    summary = format_summary(all_records, baseline_name, source_files)
    print()
    print(summary, end="")
    summary_txt.parent.mkdir(parents=True, exist_ok=True)
    summary_txt.write_text(summary, encoding="utf-8")
    write_details_csv(details_csv, all_records, baseline_name, args.eval_include_generated_text)
    print(f"Wrote evaluation summary to {relative_display_path(summary_txt)}")
    print(f"Wrote evaluation details CSV to {relative_display_path(details_csv)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one LLaVA video inference per Vacation scene and write an annotation-aligned CSV."
    )
    parser.add_argument("--annotations-csv", default=DEFAULT_ANNOTATIONS_CSV)
    parser.add_argument("--videos-dir", default=DEFAULT_VIDEOS_DIR)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--prompt", default=DEFAULT_PROMPTS[0])
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE, help="Text file whose full contents are used as the prompt.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--conv-template", default="qwen_1_5")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-frames-num", type=int, default=16)
    parser.add_argument("--video-ids", nargs="+", default=None, help="Optional subset of Vacation video ids.")
    parser.add_argument("--limit-scenes", type=int, default=None, help="Optional cap for debugging.")
    parser.add_argument("--save-every", type=int, default=25, help="Write the CSV every N processed scenes; 0 disables intermediate writes.")
    parser.add_argument("--disable-optimizations", action="store_true")
    parser.add_argument("--skip-eval", action="store_true", help="Skip post-run GT/baseline evaluation.")
    parser.add_argument("--eval-baseline-csv", default=None, help="Baseline results CSV. Defaults to the baseline CSV in output dir.")
    parser.add_argument("--eval-summary-txt", default=None, help="Post-eval summary TXT path.")
    parser.add_argument("--eval-details-csv", default=None, help="Post-eval detailed CSV path.")
    parser.add_argument("--eval-include-generated-text", action="store_true", help="Include full generations in the post-eval CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_vacation_scene_inference(args)
    completed = df["scene_generated_text"].notna().sum()
    print(f"Wrote {completed}/{len(df)} annotation rows with scene results.")
    print(f"Saved results CSV to {relative_display_path(Path(fix_wsl_paths(args.output_csv)))}")
    if not args.skip_eval:
        run_post_eval(args)


if __name__ == "__main__":
    main()
