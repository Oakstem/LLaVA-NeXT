#!/bin/bash
#SBATCH --job-name=qwen3vl_dataset_eval
#SBATCH --output=logs/qwen3vl_dataset_eval_%j.out
#SBATCH --error=logs/qwen3vl_dataset_eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --nodes=1

###############################################################################
# Configuration (override via environment when submitting the job)
###############################################################################
DATASET_JSON="${DATASET_JSON:-/galitylab/students/alonmardi/projects/LLaVA-NeXT/training_datasets/20251007_165818_sgl_conversation_data_20251007_171523/val.json}"
IMAGES_DIR="${IMAGES_DIR:-/galitylab/students/alonmardi/gazefollow}"
DEFAULT_OUTPUT_DIR="./evaluation_results/qwen3vl_dataset_eval_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
LIMIT="${LIMIT:-}"  # Optional subset size for debugging

GAZE_MODEL_ID="${GAZE_MODEL_ID:-Qwen/Qwen3-VL-4B-Instruct}"
GAZE_BOX_THRESHOLD="${GAZE_BOX_THRESHOLD:-0.20}"
GAZE_TEXT_THRESHOLD="${GAZE_TEXT_THRESHOLD:-0.20}"
GAZE_IOU_RADIUS_RATIO="${GAZE_IOU_RADIUS_RATIO:-0.05}"
GAZE_DEVICE="${GAZE_DEVICE:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

SAVE_RECORDS="${SAVE_RECORDS:-true}"
USE_GPT_GAZE_TARGETS="${USE_GPT_GAZE_TARGETS:-false}"

WANDB="${WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-llava-dataset-eval}"
WANDB_ENTITY="${WANDB_ENTITY:-gylab}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_RUN_NAME_SUFFIX="${WANDB_RUN_NAME_SUFFIX:-}"

IN_OUT_LABELS_CSV="${IN_OUT_LABELS_CSV:-gazefollow/data/combined_description_results.csv}"

###############################################################################
# Setup and validation
###############################################################################
mkdir -p logs
mkdir -p "$(dirname "$OUTPUT_DIR")"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Qwen3-VL Dataset Evaluation Job"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node List: ${SLURM_JOB_NODELIST}"
echo "Start Time: $(date)"
echo ""
echo "Dataset JSON: $DATASET_JSON"
echo "Images Directory: $IMAGES_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Sample Limit: ${LIMIT:-none}"
echo ""
echo "Model Configuration:"
echo "  Qwen3-VL Model: $GAZE_MODEL_ID"
echo "  Box Threshold: $GAZE_BOX_THRESHOLD"
echo "  Text Threshold: $GAZE_TEXT_THRESHOLD"
echo "  IoU Radius Ratio: $GAZE_IOU_RADIUS_RATIO"
echo "  Device Override: ${GAZE_DEVICE:-auto}"
echo "  Max New Tokens: $MAX_NEW_TOKENS"
echo ""
echo "Records: $SAVE_RECORDS"
echo "Use GPT Gaze Targets: $USE_GPT_GAZE_TARGETS"
echo ""
echo "wandb Logging: $WANDB"
if [ "$WANDB" = "true" ]; then
    echo "  Project: $WANDB_PROJECT"
    echo "  Entity: ${WANDB_ENTITY:-default}"
    if [ -n "$WANDB_RUN_NAME" ]; then
        echo "  Run Name: $WANDB_RUN_NAME"
    fi
    if [ -n "$WANDB_RUN_NAME_SUFFIX" ]; then
        echo "  Run Name Suffix: $WANDB_RUN_NAME_SUFFIX"
    fi
fi
echo "=========================================="

if [ ! -f "$DATASET_JSON" ]; then
    echo "ERROR: Dataset JSON not found at $DATASET_JSON"
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "ERROR: Images directory not found at $IMAGES_DIR"
    exit 1
fi

if [ -n "$IN_OUT_LABELS_CSV" ] && [ ! -f "$IN_OUT_LABELS_CSV" ]; then
    echo "WARNING: In/out labels CSV not found at $IN_OUT_LABELS_CSV"
fi

###############################################################################
# Environment setup
###############################################################################
echo "Activating Python environment..."
source /galitylab/students/alonmardi/.ground-uv/bin/activate

which python
python --version

echo ""
echo "GPU Summary:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="/galitylab/students/alonmardi/projects/LLaVA-NeXT:${PYTHONPATH}"
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

cd /galitylab/students/alonmardi/projects/LLaVA-NeXT || exit 1

if [ ! -f "gazefollow/evaluate_dataset_qwen3vl.py" ]; then
    echo "ERROR: gazefollow/evaluate_dataset_qwen3vl.py not found in repository."
    exit 1
fi

###############################################################################
# Argument assembly
###############################################################################
ARGS=(
    "gazefollow/evaluate_dataset_qwen3vl.py"
    "--dataset-json" "$DATASET_JSON"
    "--images-dir" "$IMAGES_DIR"
    "--gaze-model-id" "$GAZE_MODEL_ID"
    "--gaze-box-threshold" "$GAZE_BOX_THRESHOLD"
    "--gaze-text-threshold" "$GAZE_TEXT_THRESHOLD"
    "--gaze-iou-radius-ratio" "$GAZE_IOU_RADIUS_RATIO"
    "--max-new-tokens" "$MAX_NEW_TOKENS"
)

if [ -n "$OUTPUT_DIR" ]; then
    ARGS+=("--output-dir" "$OUTPUT_DIR")
fi

if [ -n "$GAZE_DEVICE" ]; then
    ARGS+=("--gaze-device" "$GAZE_DEVICE")
fi

if [ -n "$LIMIT" ]; then
    ARGS+=("--limit" "$LIMIT")
fi

if [ "$SAVE_RECORDS" = "false" ]; then
    ARGS+=("--no-save-records")
fi

if [ "$USE_GPT_GAZE_TARGETS" = "true" ]; then
    ARGS+=("--use-gpt-gaze-targets")
fi

if [ -n "$IN_OUT_LABELS_CSV" ]; then
    ARGS+=("--in-out-labels-csv" "$IN_OUT_LABELS_CSV")
fi

if [ "$WANDB" = "false" ]; then
    ARGS+=("--no-log-to-wandb")
else
    if [ -n "$WANDB_PROJECT" ]; then
        ARGS+=("--wandb-project" "$WANDB_PROJECT")
    fi
    if [ -n "$WANDB_ENTITY" ]; then
        ARGS+=("--wandb-entity" "$WANDB_ENTITY")
    fi
    if [ -n "$WANDB_RUN_NAME" ]; then
        ARGS+=("--wandb-run-name" "$WANDB_RUN_NAME")
    fi
    if [ -n "$WANDB_RUN_NAME_SUFFIX" ]; then
        ARGS+=("--wandb-run-name-suffix" "$WANDB_RUN_NAME_SUFFIX")
    fi
fi

###############################################################################
# Execution
###############################################################################
echo "=========================================="
echo "Executing Qwen3-VL dataset evaluation:"
echo "python ${ARGS[*]}"
echo "=========================================="
echo ""

start_time=$(date +%s)

python "${ARGS[@]}"
exit_code=$?

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "=========================================="
echo "Evaluation Summary"
echo "=========================================="
echo "Completed at: $(date)"
echo "Exit Code: $exit_code"
echo "Runtime: ${duration} seconds ($((${duration} / 60)) minutes)"

metrics_path="$OUTPUT_DIR/metrics.json"
if [ $exit_code -eq 0 ]; then
    if [ -f "$metrics_path" ]; then
        echo ""
        echo "Key Metrics (from $(realpath "$metrics_path")):"
        export METRICS_PATH="$metrics_path"
        python - <<'PY'
import json
import os

metrics_path = os.environ.get("METRICS_PATH")
if not metrics_path or not os.path.exists(metrics_path):
    raise SystemExit

with open(metrics_path, "r", encoding="utf-8") as handle:
    metrics = json.load(handle)

ordered_keys = [
    "total_samples",
    "samples_with_ground_truth",
    "samples_with_person_descriptions",
    "samples_with_gaze_detections",
    "persons_with_gaze_detections",
    "failed_samples",
    "gaze_l2_error_mean",
    "gaze_l2_normalized_mean",
    "gaze_angular_error_mean",
    "gaze_iou_mean",
    "gaze_modified_l2_error_mean",
    "evaluation_time_seconds",
    "average_time_per_sample",
]

def format_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

for key in ordered_keys:
    if key in metrics:
        print(f"{key}: {format_value(metrics[key])}")
PY
    else
        echo "WARNING: Metrics file not found at $metrics_path"
    fi
    echo ""
    echo "Artifacts saved to: $OUTPUT_DIR"
else
    echo "Evaluation failed. Inspect logs for details."
fi

exit $exit_code
