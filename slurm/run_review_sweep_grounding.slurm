#!/bin/bash
#SBATCH --job-name=review_sweep_grounding
#SBATCH --output=logs/review_sweep_grounding_%j.out
#SBATCH --error=logs/review_sweep_grounding_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu-tad-pool
#SBATCH --qos=owner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --nodes=1

PROJECT_ROOT="/galitylab/students/alonmardi/projects/LLaVA-NeXT"

PYTHON_BIN="/galitylab/students/alonmardi/.ground-uv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="/home/alonz/llava/bin/python"
fi
if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python env not found at /galitylab/students/alonmardi/llava/bin/python or /home/alonz/llava/bin/python"
    exit 1
fi

RESULTS_FILE="${RESULTS_FILE:-$PROJECT_ROOT/results/steered_generation/repr_layer_sweep_results_20260403_183355.json}"
MASK_META_FILE="${MASK_META_FILE:-$PROJECT_ROOT/region_masks/active_region_mask2.json}"
IMAGE_PATH="${IMAGE_PATH:-}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-4B-Instruct}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
TEMPERATURE="${TEMPERATURE:-0.0}"
LIMIT="${LIMIT:-}"
PRINT_TOP_K="${PRINT_TOP_K:-15}"
SAVE_OVERLAYS="${SAVE_OVERLAYS:-true}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/steered_generation/review_grounding_${SLURM_JOB_ID}}"

mkdir -p "$PROJECT_ROOT/logs"

echo "=========================================="
echo "Review Sweep Grounding Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Python: $PYTHON_BIN"
echo "Results File: $RESULTS_FILE"
echo "Mask Meta File: $MASK_META_FILE"
echo "Image Path: ${IMAGE_PATH:-<from mask metadata>}"
echo "Model ID: $MODEL_ID"
echo "Device Map: $DEVICE_MAP"
echo "Max New Tokens: $MAX_NEW_TOKENS"
echo "Temperature: $TEMPERATURE"
echo "Limit: ${LIMIT:-<all>}"
echo "Print Top K: $PRINT_TOP_K"
echo "Save Overlays: $SAVE_OVERLAYS"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="

echo "Python Version:"
"$PYTHON_BIN" --version

echo "Machine IP Address:"
hostname -I

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

cd "$PROJECT_ROOT" || exit 1
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "ERROR: Results file not found at $RESULTS_FILE"
    exit 1
fi

if [ ! -f "$MASK_META_FILE" ]; then
    echo "ERROR: Mask metadata file not found at $MASK_META_FILE"
    exit 1
fi

if [ -n "$IMAGE_PATH" ] && [ ! -f "$IMAGE_PATH" ]; then
    echo "ERROR: Image file not found at $IMAGE_PATH"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

ARGS=(
    "--results-file" "$RESULTS_FILE"
    "--mask-meta-file" "$MASK_META_FILE"
    "--model-id" "$MODEL_ID"
    "--device-map" "$DEVICE_MAP"
    "--output-dir" "$OUTPUT_DIR"
    "--max-new-tokens" "$MAX_NEW_TOKENS"
    "--temperature" "$TEMPERATURE"
    "--print-top-k" "$PRINT_TOP_K"
)

if [ -n "$IMAGE_PATH" ]; then
    ARGS+=("--image-path" "$IMAGE_PATH")
fi

if [ -n "$LIMIT" ]; then
    ARGS+=("--limit" "$LIMIT")
fi

if [ "$SAVE_OVERLAYS" = "true" ]; then
    ARGS+=("--save-overlays")
else
    ARGS+=("--no-save-overlays")
fi

echo "Executing command:"
echo "$PYTHON_BIN gazefollow/auto_phrase_grounding/review_sweep_grounding.py ${ARGS[*]}"
echo "=========================================="

"$PYTHON_BIN" gazefollow/auto_phrase_grounding/review_sweep_grounding.py "${ARGS[@]}"

exit_code=$?

echo ""
echo "=========================================="
echo "review_sweep_grounding finished with exit code $exit_code"
echo "End Time: $(date)"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="

exit $exit_code
