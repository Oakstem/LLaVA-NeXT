#!/bin/bash
#SBATCH --job-name=repr_layer_sweep
#SBATCH --output=logs/repr_layer_sweep_%j.out
#SBATCH --error=logs/repr_layer_sweep_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=gpu-tad-pool
#SBATCH --qos=owner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --nodes=1

PROJECT_ROOT="/galitylab/students/alonmardi/projects/LLaVA-NeXT"

PYTHON_BIN="/galitylab/students/alonmardi/llava/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="/home/alonz/llava/bin/python"
fi
if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python env not found at /galitylab/students/alonmardi/llava/bin/python or /home/alonz/llava/bin/python"
    exit 1
fi

MODEL_PATH="${MODEL_PATH:-lmms-lab/llava-onevision-qwen2-7b-ov-chat}"
MODEL_BASE="${MODEL_BASE:-}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
QUANTIZATION="${QUANTIZATION:-}"

IMAGE_PATH="${IMAGE_PATH:-/galitylab/students/alonmardi/gazefollow/train/00000094/00094313.jpg}"
MASK_PATH="${MASK_PATH:-region_masks/random_samples/00094313_region_mask.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/steered_generation/repr_layer_sweep_${SLURM_JOB_ID}}"

ATTN_LAYER_IND="${ATTN_LAYER_IND:-23}"
REPR_SWEEP_BIAS="${REPR_SWEEP_BIAS:-2.5}"
REPR_SWEEP_START_IDX="${REPR_SWEEP_START_IDX:-}"
REPR_SWEEP_LAYERS="${REPR_SWEEP_LAYERS:-}"
REPR_COMBO_SWEEP="${REPR_COMBO_SWEEP:-true}"
REPR_CAPTURE_LAYERS="${REPR_CAPTURE_LAYERS:-}"
REPR_INJECT_LAYERS="${REPR_INJECT_LAYERS:-}"
TOKEN_INJECTION_REPR_SWEEP="${TOKEN_INJECTION_REPR_SWEEP:-false}"
TOKEN_INJECTION_RUNS_PER_COMBO="${TOKEN_INJECTION_RUNS_PER_COMBO:-2}"

USE_GT_GAZE_CSV="${USE_GT_GAZE_CSV:-false}"
GT_GAZE_CSV_PATH="${GT_GAZE_CSV_PATH:-}"
USE_BODY_BBOX="${USE_BODY_BBOX:-false}"
USE_TARGET_INSERT_FOR_SOURCE="${USE_TARGET_INSERT_FOR_SOURCE:-false}"
SAVE_DEBUG_FILES="${SAVE_DEBUG_FILES:-true}"
SAVE_MASK_OVERLAYS="${SAVE_MASK_OVERLAYS:-false}"
EXCLUDE_IMAGE_INPUTS="${EXCLUDE_IMAGE_INPUTS:-false}"

USE_GAZE_GUIDANCE="${USE_GAZE_GUIDANCE:-false}"
GUIDANCE_TOP_K="${GUIDANCE_TOP_K:-10}"
SIMILARITY_WEIGHT="${SIMILARITY_WEIGHT:-0.7}"
PROBABILITY_WEIGHT="${PROBABILITY_WEIGHT:-0.3}"

mkdir -p "$PROJECT_ROOT/logs"

echo "=========================================="
echo "Representation Layer Sweep Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Python: $PYTHON_BIN"
echo "Model Path: $MODEL_PATH"
echo "Model Base: $MODEL_BASE"
echo "Adapter Path: $ADAPTER_PATH"
echo "Attention Implementation: $ATTN_IMPLEMENTATION"
echo "Quantization: $QUANTIZATION"
echo "Image Path: $IMAGE_PATH"
echo "Mask Path: $MASK_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Attention Layer: $ATTN_LAYER_IND"
echo "Repr Sweep Bias: $REPR_SWEEP_BIAS"
echo "Repr Sweep Start Idx: ${REPR_SWEEP_START_IDX:-<all>}"
echo "Repr Sweep Layers: ${REPR_SWEEP_LAYERS:-<all available>}"
echo "Repr Combo Sweep: $REPR_COMBO_SWEEP"
echo "Repr Capture Layers: ${REPR_CAPTURE_LAYERS:-<all available in combo mode>}"
echo "Repr Inject Layers: ${REPR_INJECT_LAYERS:-<all available in combo mode>}"
echo "Token Injection Repr Sweep: $TOKEN_INJECTION_REPR_SWEEP"
echo "Token Injection Runs Per Combo: $TOKEN_INJECTION_RUNS_PER_COMBO"
echo "Use GT Gaze CSV: $USE_GT_GAZE_CSV"
echo "GT Gaze CSV Path: ${GT_GAZE_CSV_PATH:-<default>}"
echo "Use Body BBox: $USE_BODY_BBOX"
echo "Use Target Insert For Source: $USE_TARGET_INSERT_FOR_SOURCE"
echo "Save Debug Files: $SAVE_DEBUG_FILES"
echo "Save Mask Overlays: $SAVE_MASK_OVERLAYS"
echo "Exclude Image Inputs: $EXCLUDE_IMAGE_INPUTS"
echo "Use Gaze Guidance: $USE_GAZE_GUIDANCE"
echo "Guidance Top-K: $GUIDANCE_TOP_K"
echo "Similarity Weight: $SIMILARITY_WEIGHT"
echo "Probability Weight: $PROBABILITY_WEIGHT"
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

if [ ! -f "$IMAGE_PATH" ]; then
    echo "ERROR: Image not found at $IMAGE_PATH"
    exit 1
fi

if [ ! -f "$MASK_PATH" ]; then
    echo "ERROR: Mask not found at $MASK_PATH"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

ARGS=(
    "--mode" "repr_sweep"
    "--model_path" "$MODEL_PATH"
    "--attn_implementation" "$ATTN_IMPLEMENTATION"
    "--image_path" "$IMAGE_PATH"
    "--mask_path" "$MASK_PATH"
    "--attn_layer_ind" "$ATTN_LAYER_IND"
    "--output_dir" "$OUTPUT_DIR"
    "--repr_sweep_bias" "$REPR_SWEEP_BIAS"
    "--guidance_top_k" "$GUIDANCE_TOP_K"
    "--similarity_weight" "$SIMILARITY_WEIGHT"
    "--probability_weight" "$PROBABILITY_WEIGHT"
)

if [ -n "$MODEL_BASE" ]; then
    ARGS+=("--model_base" "$MODEL_BASE")
fi

if [ -n "$ADAPTER_PATH" ]; then
    ARGS+=("--adapter_path" "$ADAPTER_PATH")
fi

if [ "$QUANTIZATION" = "4bit" ]; then
    ARGS+=("--load_4bit")
elif [ "$QUANTIZATION" = "8bit" ]; then
    ARGS+=("--load_8bit")
fi

if [ -n "$REPR_SWEEP_START_IDX" ]; then
    ARGS+=("--repr_sweep_start_idx" "$REPR_SWEEP_START_IDX")
fi

if [ -n "$REPR_SWEEP_LAYERS" ]; then
    read -r -a repr_sweep_layers_array <<< "$REPR_SWEEP_LAYERS"
    ARGS+=("--repr_sweep_layers" "${repr_sweep_layers_array[@]}")
fi

if [ "$REPR_COMBO_SWEEP" = "true" ]; then
    ARGS+=("--repr_combo_sweep")
fi

if [ -n "$REPR_CAPTURE_LAYERS" ]; then
    read -r -a repr_capture_layers_array <<< "$REPR_CAPTURE_LAYERS"
    ARGS+=("--repr_capture_layers" "${repr_capture_layers_array[@]}")
fi

if [ -n "$REPR_INJECT_LAYERS" ]; then
    read -r -a repr_inject_layers_array <<< "$REPR_INJECT_LAYERS"
    ARGS+=("--repr_inject_layers" "${repr_inject_layers_array[@]}")
fi

if [ "$TOKEN_INJECTION_REPR_SWEEP" = "true" ]; then
    ARGS+=("--token_injection_repr_sweep")
    ARGS+=("--token_injection_runs_per_combo" "$TOKEN_INJECTION_RUNS_PER_COMBO")
fi

if [ "$USE_GT_GAZE_CSV" = "true" ]; then
    ARGS+=("--use-gt-gaze-csv")
else
    ARGS+=("--no-use-gt-gaze-csv")
fi

if [ -n "$GT_GAZE_CSV_PATH" ]; then
    ARGS+=("--gt_gaze_csv_path" "$GT_GAZE_CSV_PATH")
fi

if [ "$USE_BODY_BBOX" = "true" ]; then
    ARGS+=("--use_body_bbox")
fi

if [ "$USE_TARGET_INSERT_FOR_SOURCE" = "true" ]; then
    ARGS+=("--use_target_insert_for_source")
fi

if [ "$SAVE_DEBUG_FILES" = "true" ]; then
    ARGS+=("--save_debug_files")
fi

if [ "$SAVE_MASK_OVERLAYS" = "true" ]; then
    ARGS+=("--save-mask-overlays")
fi

if [ "$EXCLUDE_IMAGE_INPUTS" = "true" ]; then
    ARGS+=("--exclude-image-inputs")
fi

if [ "$USE_GAZE_GUIDANCE" = "true" ]; then
    ARGS+=("--use_gaze_guidance")
fi

echo "Executing command:"
echo "$PYTHON_BIN gazefollow/extract_attention_direct_gt_masks.py ${ARGS[*]}"
echo "=========================================="

"$PYTHON_BIN" gazefollow/extract_attention_direct_gt_masks.py "${ARGS[@]}"

exit_code=$?
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code
