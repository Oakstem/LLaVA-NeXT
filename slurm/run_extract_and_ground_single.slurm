#!/bin/bash
#SBATCH --job-name=extract_ground_single
#SBATCH --output=logs/extract_ground_single_%j.out
#SBATCH --error=logs/extract_ground_single_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --nodes=1

# Repository root used for relative paths
PROJECT_ROOT="/galitylab/students/alonmardi/projects/LLaVA-NeXT"
export PROJECT_ROOT

# Python environment (must use an existing env)
PYTHON_BIN="/galitylab/students/alonmardi/.ground-uv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="/home/alonz/llava/bin/python"
fi
if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python env not found at /galitylab/students/alonmardi/llava/bin/python or /home/alonz/llava/bin/python"
    exit 1
fi

# Single-image configuration
IMAGE_PATH="${IMAGE_PATH:-/galitylab/students/alonmardi/gazefollow/train/00000000/00000119.jpg}"
MASK_PATH="${MASK_PATH:-}"
MASK_DIR="${MASK_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/steered_generation/single_test_${SLURM_JOB_ID}}"

# Retry configuration
RETRY_THRESHOLD="${RETRY_THRESHOLD:-0.16}"
RETRY_IOU_THRESHOLD="${RETRY_IOU_THRESHOLD:-0.2}"
RETRY_USE_BODY_BBOX="${RETRY_USE_BODY_BBOX:-false}"
RETRY_PERSON_BBOX_SCALE="${RETRY_PERSON_BBOX_SCALE:-}"
FINAL_RETRY_PERSON_BBOX_SCALE="${FINAL_RETRY_PERSON_BBOX_SCALE:-0.5}"

# Prompt configuration
USE_BODY_BBOX="${USE_BODY_BBOX:-true}"
USE_TARGET_INSERT_FOR_SOURCE="${USE_TARGET_INSERT_FOR_SOURCE:-false}"

# Create logs directory relative to the project root
mkdir -p "$PROJECT_ROOT/logs"

echo "=========================================="
echo "Extract & Ground Single Image Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""
echo "Image Settings:"
echo "  Image Path: $IMAGE_PATH"
echo "  Mask Path: ${MASK_PATH:-<auto>}"
echo "  Mask Dir: $MASK_DIR"
echo "  Output Dir: $OUTPUT_DIR"
echo "Retry Settings:"
echo "  Threshold: $RETRY_THRESHOLD"
echo "  IoU Threshold: $RETRY_IOU_THRESHOLD"
echo "  Retry Use Body BBox: $RETRY_USE_BODY_BBOX"
echo "  Retry Person BBox Scale: ${RETRY_PERSON_BBOX_SCALE:-<inherit>}"
echo "  Final Retry Person BBox Scale: $FINAL_RETRY_PERSON_BBOX_SCALE"
echo "Prompt Settings:"
echo "  Use Body BBox: $USE_BODY_BBOX"
echo "  Use Target Insert For Source: $USE_TARGET_INSERT_FOR_SOURCE"
echo "=========================================="

echo "Python: $PYTHON_BIN"
echo "Python Version:"
$PYTHON_BIN --version

echo "Machine IP Address:"
hostname -I

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Change to the project directory
cd "$PROJECT_ROOT"

# Ensure logs/output directories exist
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# Validate essential paths
if [ ! -f "$IMAGE_PATH" ]; then
    echo "ERROR: Image not found at $IMAGE_PATH"
    exit 1
fi

if [ -n "$MASK_PATH" ] && [ ! -f "$MASK_PATH" ]; then
    echo "ERROR: Mask file not found at $MASK_PATH"
    exit 1
fi

if [ -n "$MASK_DIR" ] && [ ! -d "$MASK_DIR" ]; then
    echo "ERROR: Mask directory not found at $MASK_DIR"
    exit 1
fi

# Environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "Executing single-image retry test..."
echo "=========================================="

$PYTHON_BIN - <<'PY'
import os
import sys
from pathlib import Path

from gazefollow.auto_phrase_grounding.extract_and_ground_pipeline import (
    DEFAULT_SOURCE_PROMPT,
    DEFAULT_TARGET_PROMPT,
    build_attention_config,
    build_generation_config,
    build_guidance_config,
    enable_inference_optimizations,
    load_image_tasks,
    load_qwen3vl_model,
    load_model_and_setup,
    fix_wsl_paths,
    parse_args,
    process_image_task_with_body_bbox_retry,
    save_json,
    _normalize_path,
)

project_root = Path(os.environ["PROJECT_ROOT"]) if "PROJECT_ROOT" in os.environ else Path.cwd()
image_path = os.environ["IMAGE_PATH"]
mask_path = os.environ.get("MASK_PATH") or None
mask_dir = os.environ.get("MASK_DIR") or None
output_dir = os.environ.get("OUTPUT_DIR") or "single_test"

retry_threshold = float(os.environ.get("RETRY_THRESHOLD", "0.16"))
retry_iou_threshold = float(os.environ.get("RETRY_IOU_THRESHOLD", "0.2"))
retry_use_body_bbox = os.environ.get("RETRY_USE_BODY_BBOX", "false").lower() == "true"
retry_person_bbox_scale = os.environ.get("RETRY_PERSON_BBOX_SCALE")
final_retry_person_bbox_scale = os.environ.get("FINAL_RETRY_PERSON_BBOX_SCALE")

use_body_bbox = os.environ.get("USE_BODY_BBOX", "true").lower() == "true"
use_target_insert_for_source = os.environ.get("USE_TARGET_INSERT_FOR_SOURCE", "false").lower() == "true"

argv = [
    "extract_and_ground_pipeline.py",
    "--mode",
    "single",
    "--image-path",
    image_path,
    "--output-dir",
    output_dir,
]
if mask_path:
    argv += ["--mask-path", mask_path]
if mask_dir:
    argv += ["--mask-dir", mask_dir]
if use_body_bbox:
    argv += ["--use-body-bbox"]
else:
    argv += ["--no-use-body-bbox"]
if use_target_insert_for_source:
    argv += ["--use-target-insert-for-source"]

sys.argv = argv
args = parse_args()

if args.use_target_insert_for_source:
    args.prompt = DEFAULT_TARGET_PROMPT
else:
    args.prompt = DEFAULT_SOURCE_PROMPT

mask_dir = _normalize_path(args.mask_dir) if args.mask_dir else None
tasks = load_image_tasks(
    mode=args.mode,
    image_path=args.image_path,
    mask_path=args.mask_path,
    image_list=args.image_list,
    mask_dir=mask_dir,
    mask_template=args.mask_template,
    fallback_mask_template=args.mask_fallback_template,
    limit=args.list_limit,
    image_id=args.image_id,
    require_mask=False,
)

output_dir = Path(_normalize_path(args.output_dir) or args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

llava_output_dir = Path(_normalize_path(args.llava_output_dir) if args.llava_output_dir else output_dir / "llava_runs")
llava_output_dir.mkdir(parents=True, exist_ok=True)

visualization_dir = None
if args.save_visualization:
    viz_dir = _normalize_path(args.visualization_dir) if args.visualization_dir else output_dir / "visualizations"
    visualization_dir = Path(viz_dir)
    visualization_dir.mkdir(parents=True, exist_ok=True)

enable_inference_optimizations()
tokenizer, model, image_processor, _ = load_model_and_setup(
    model_path=fix_wsl_paths(args.model_path),
    attn_implementation=args.attn_implementation,
    load_4bit=args.load_4bit,
    load_8bit=args.load_8bit,
    attn_layer_ind=args.attn_layer_index,
    model_base=fix_wsl_paths(str(_normalize_path(args.model_base))) if args.model_base else None,
    adapter_path=fix_wsl_paths(str(_normalize_path(args.adapter_path))) if args.adapter_path else None,
)

qwen_processor, qwen_model = load_qwen3vl_model(args.qwen_model_id, device_map=args.qwen_device_map)
generation_config = build_generation_config(args)
attention_config = build_attention_config(args)
guidance_config = build_guidance_config(args)
qwen_query_template = args.query_template if "{description}" in args.query_template else (
    args.query_template + " {description}"
)

task = tasks[0]
payload, best_error, best_iou, best_detection, used_retry, used_scaled_retry = process_image_task_with_body_bbox_retry(
    task,
    args=args,
    tokenizer=tokenizer,
    model=model,
    image_processor=image_processor,
    generation_config=generation_config,
    attention_config=attention_config,
    guidance_config=guidance_config,
    llava_output_dir=llava_output_dir,
    qwen_processor=qwen_processor,
    qwen_model=qwen_model,
    visualization_dir=visualization_dir,
    qwen_query_template=qwen_query_template,
    retry_threshold=retry_threshold,
    retry_iou_threshold=retry_iou_threshold,
    retry_use_body_bbox=retry_use_body_bbox,
    retry_person_bbox_scale=float(retry_person_bbox_scale) if retry_person_bbox_scale else None,
    final_retry_person_bbox_scale=float(final_retry_person_bbox_scale) if final_retry_person_bbox_scale else None,
)

payload.setdefault("artifacts", {})["best_error"] = best_error
payload.setdefault("artifacts", {})["best_iou"] = best_iou
payload.setdefault("artifacts", {})["used_retry"] = used_retry
payload.setdefault("artifacts", {})["used_scaled_retry"] = used_scaled_retry

per_image_path = output_dir / f"{task.image_id}.json"
save_json(payload, per_image_path)

summary_path = output_dir / "pipeline_results.json"
save_json(
    {
        "results": [payload],
        "failures": [],
        "num_results": 1,
        "num_failures": 0,
        "best_error": best_error,
        "best_iou": best_iou,
        "used_retry": used_retry,
        "used_scaled_retry": used_scaled_retry,
    },
    summary_path,
)

print(f"Saved results to {per_image_path}")
print(f"Saved summary to {summary_path}")
PY

exit_code=$?

echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code
