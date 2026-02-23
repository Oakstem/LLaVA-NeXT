#!/bin/bash
#SBATCH --job-name=dataset_eval_qwen3vl
#SBATCH --output=logs/dataset_eval_qwen3vl_%j.out
#SBATCH --error=logs/dataset_eval_qwen3vl_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --nodes=1

# Configuration variables (override via environment if needed)
DATASET_JSON="${DATASET_JSON:-training_datasets/20251205_174741_sgl_conversation_data_20251214_204738/train.json}"
IMAGES_DIR="${IMAGES_DIR:-/galitylab/students/alonmardi/gazefollow}"
DATASET_BASENAME=$(basename "$DATASET_JSON" .json)
OUTPUT_DIR="${OUTPUT_DIR:-./dataset_evaluation_results/qwen3vl_${DATASET_BASENAME}_$(date +%Y%m%d_%H%M%S)}"
LIMIT="${LIMIT:-}"  # Optional subset for quick checks
RESUME_FROM="${RESUME_FROM:-dataset_evaluation_results/qwen3vl_train_20251213_223625/intermediate_results_latest.json}"  # Path to intermediate_results_latest.json

GAZE_MODEL_ID="${GAZE_MODEL_ID:-Qwen/Qwen3-VL-4B-Instruct}"
GAZE_BOX_THRESHOLD="${GAZE_BOX_THRESHOLD:-0.20}"
GAZE_TEXT_THRESHOLD="${GAZE_TEXT_THRESHOLD:-0.20}"
GAZE_IOU_RADIUS_RATIO="${GAZE_IOU_RADIUS_RATIO:-0.05}"
GAZE_DEVICE="${GAZE_DEVICE:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

SAVE_RECORDS="${SAVE_RECORDS:-true}"
SAVE_EVERY="${SAVE_EVERY:-50}"
WANDB="${WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-llava-dataset-eval}"
WANDB_ENTITY="${WANDB_ENTITY:-gylab}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_RUN_NAME_SUFFIX="${WANDB_RUN_NAME_SUFFIX:-}"
USE_GPT_GAZE_TARGETS="${USE_GPT_GAZE_TARGETS:-false}"
IN_OUT_LABELS_CSV="${IN_OUT_LABELS_CSV:-}"
GROUND_TRUTH_CSV="${GROUND_TRUTH_CSV:-}"

# Prepare directories
mkdir -p logs
if [ -n "$RESUME_FROM" ]; then
    OUTPUT_DIR="$(dirname "$RESUME_FROM")"
fi
mkdir -p "$(dirname "$OUTPUT_DIR")"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Dataset Evaluation (Qwen3-VL) Job Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_JOB_NODELIST}"
echo "Start Time: $(date)"
echo ""
echo "Dataset JSON: $DATASET_JSON"
echo "Images Directory: $IMAGES_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Sample Limit: ${LIMIT:-none}"
if [ -n "$RESUME_FROM" ]; then
    echo "Resuming from: $RESUME_FROM"
fi
echo ""
echo "Qwen3-VL:"
echo "  Model ID: $GAZE_MODEL_ID"
echo "  Box Threshold: $GAZE_BOX_THRESHOLD"
echo "  Text Threshold: $GAZE_TEXT_THRESHOLD"
echo "  IoU Radius Ratio: $GAZE_IOU_RADIUS_RATIO"
echo "  Device Override: ${GAZE_DEVICE:-auto}"
echo "  Max New Tokens: $MAX_NEW_TOKENS"
echo ""
echo "Record Saving: $SAVE_RECORDS (every $SAVE_EVERY samples)"
echo "wandb Logging: $WANDB"
echo "wandb Project: $WANDB_PROJECT"
echo "wandb Entity: ${WANDB_ENTITY:-default}"
if [ -n "$WANDB_RUN_NAME" ]; then
    echo "wandb Run Name: $WANDB_RUN_NAME"
fi
if [ -n "$WANDB_RUN_NAME_SUFFIX" ]; then
    echo "wandb Run Name Suffix: $WANDB_RUN_NAME_SUFFIX"
fi
if [ "$USE_GPT_GAZE_TARGETS" = "true" ]; then
    echo "Using GPT gaze targets instead of parsed text."
fi
if [ -n "$IN_OUT_LABELS_CSV" ]; then
    echo "In/Out labels CSV: $IN_OUT_LABELS_CSV"
fi
if [ -n "$GROUND_TRUTH_CSV" ]; then
    echo "Ground-truth CSV: $GROUND_TRUTH_CSV"
fi
echo "=========================================="

# Validate inputs
if [ ! -f "$DATASET_JSON" ]; then
    echo "ERROR: Dataset JSON not found: $DATASET_JSON"
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "ERROR: Images directory not found: $IMAGES_DIR"
    exit 1
fi

# Select approved Python environment
if [ -x "/galitylab/students/alonmardi/llava/bin/python" ]; then
    PYTHON_BIN="/galitylab/students/alonmardi/llava/bin/python"
elif [ -x "/home/alonz/llava/bin/python" ]; then
    PYTHON_BIN="/home/alonz/llava/bin/python"
else
    echo "ERROR: Could not find approved Python interpreter."
    exit 1
fi
echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" --version

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/galitylab/students/alonmardi/projects/LLaVA-NeXT:${PYTHONPATH}"
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

cd /galitylab/students/alonmardi/projects/LLaVA-NeXT || exit 1

if [ ! -f "gazefollow/evaluate_dataset_qwen3vl.py" ]; then
    echo "ERROR: gazefollow/evaluate_dataset_qwen3vl.py not found in repository."
    exit 1
fi

ARGS=(
    "--dataset-json" "$DATASET_JSON"
    "--images-dir" "$IMAGES_DIR"
    "--output-dir" "$OUTPUT_DIR"
    "--gaze-model-id" "$GAZE_MODEL_ID"
    "--gaze-box-threshold" "$GAZE_BOX_THRESHOLD"
    "--gaze-text-threshold" "$GAZE_TEXT_THRESHOLD"
    "--gaze-iou-radius-ratio" "$GAZE_IOU_RADIUS_RATIO"
    "--max-new-tokens" "$MAX_NEW_TOKENS"
    "--save-every" "$SAVE_EVERY"
)

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
if [ -n "$GROUND_TRUTH_CSV" ]; then
    ARGS+=("--ground-truth-csv" "$GROUND_TRUTH_CSV")
fi
if [ -n "$RESUME_FROM" ]; then
    ARGS+=("--resume-from" "$RESUME_FROM")
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

echo "=========================================="
echo "Running dataset evaluation command:"
echo "$PYTHON_BIN gazefollow/evaluate_dataset_qwen3vl.py ${ARGS[*]}"
echo "=========================================="

start_time=$(date +%s)

"$PYTHON_BIN" gazefollow/evaluate_dataset_qwen3vl.py "${ARGS[@]}"
exit_code=$?

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "=========================================="
echo "Dataset Evaluation Summary"
echo "=========================================="
echo "Completed at: $(date)"
echo "Exit Code: $exit_code"
echo "Runtime: ${duration} seconds ($((${duration} / 60)) minutes)"

if [ $exit_code -eq 0 ]; then
    metrics_path="$OUTPUT_DIR/metrics.json"
    if [ -f "$metrics_path" ]; then
        echo ""
        echo "Key Metrics:"
        export METRICS_PATH="$metrics_path"
        "$PYTHON_BIN" - <<'PY'
import json
import os

metrics_path = os.environ.get("METRICS_PATH")
if metrics_path and os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    keys = [
        "gaze_l2_error_mean",
        "gaze_l2_normalized_mean",
        "gaze_angular_error_mean",
        "gaze_iou_mean",
        "gaze_modified_l2_error_mean",
        "samples_with_gaze_detections",
        "persons_with_gaze_detections",
        "in_out_in_frame",
        "in_out_out_of_frame",
        "missing_in_out",
        "failed_samples",
    ]
    for key in keys:
        value = metrics.get(key)
        if value is None:
            continue
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
PY
    fi
    echo ""
    echo "Artifacts located in: $OUTPUT_DIR"
else
    echo "Evaluation failed. Inspect logs for details."
fi

exit $exit_code
