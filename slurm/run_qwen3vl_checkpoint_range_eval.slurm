#!/bin/bash
#SBATCH --job-name=qwen3vl_ckpt_range
#SBATCH --output=logs/qwen3vl_ckpt_range_%A_%a.out
#SBATCH --error=logs/qwen3vl_ckpt_range_%A_%a.err
#SBATCH --time=07:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --array=0-15
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alonmardi@mail.tau.ac.il

# Sweep checkpoints within a single training run. Provide:
#   TRAINING_RUN_DIR=training_outputs/llava-20251212_203210
#   START_CHECKPOINT=9000
#   END_CHECKPOINT=12000  # optional; leave empty to include all >= START_CHECKPOINT

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    set -euo pipefail
fi
echo "Job started at: $(date)"

PYTHON_BIN="${PYTHON_BIN:-/galitylab/students/alonmardi/.ground-uv/bin/python}"

# Model configuration
MODEL_PATH="${MODEL_PATH:-lmms-lab/llava-onevision-qwen2-7b-ov-chat}"
MODEL_BASE="${MODEL_BASE:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
LOAD_4BIT="${LOAD_4BIT:-false}"
LOAD_8BIT="${LOAD_8BIT:-false}"

# Dataset configuration
DATASET_JSON="${DATASET_JSON:-training_datasets/20251205_174741_sgl_conversation_data_20251214_204738/val_filtered.json}"
IMAGES_DIR="${IMAGES_DIR:-/galitylab/students/alonmardi/gazefollow}"
LIMIT="${LIMIT:-1000}"

# Generation controls
CONV_TEMPLATE="${CONV_TEMPLATE:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.9}"
NUM_BEAMS="${NUM_BEAMS:-1}"
DO_SAMPLE="${DO_SAMPLE:-false}"

# Output controls
OUTPUT_ROOT="${OUTPUT_ROOT:-./evaluation_results}"
RUN_TAG="${RUN_TAG:-}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-true}"
GENERATE_MODEL_RESULTS="${GENERATE_MODEL_RESULTS:-true}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PROMPT_OVERRIDE="${PROMPT_OVERRIDE:-}"
WANDB="${WANDB:-true}"

# Qwen3-VL grounding options
GAZE_MODEL_ID="${GAZE_MODEL_ID:-Qwen/Qwen3-VL-4B-Instruct}"
GAZE_BOX_THRESHOLD="${GAZE_BOX_THRESHOLD:-0.20}"
GAZE_TEXT_THRESHOLD="${GAZE_TEXT_THRESHOLD:-0.20}"
GAZE_DEVICE="${GAZE_DEVICE:-}"
GAZE_MAX_NEW_TOKENS="${GAZE_MAX_NEW_TOKENS:-192}"
GAZE_TEMPERATURE="${GAZE_TEMPERATURE:-0.0}"
GAZE_IOU_RADIUS_RATIO="${GAZE_IOU_RADIUS_RATIO:-0.05}"

# Checkpoint sweep selection
TRAINING_RUN_DIR="${TRAINING_RUN_DIR:-training_outputs/siglip_sweep_20260211_110537_002_dataset_train_filtered_20251224_203622_69k_train_filtered_20251224_203622_69k}"
START_CHECKPOINT="${START_CHECKPOINT:-4000}"
END_CHECKPOINT="${END_CHECKPOINT:-}"
CHECKPOINT_SPACING="${CHECKPOINT_SPACING:-1000}"

if [ -z "$TRAINING_RUN_DIR" ]; then
    echo "TRAINING_RUN_DIR is required (e.g., training_outputs/llava-20251212_203210)" >&2
    exit 1
fi

if ! [[ "$START_CHECKPOINT" =~ ^[0-9]+$ ]]; then
    echo "START_CHECKPOINT must be an integer, got: $START_CHECKPOINT" >&2
    exit 1
fi

if ! [[ "$CHECKPOINT_SPACING" =~ ^[0-9]+$ ]] || [ "$CHECKPOINT_SPACING" -le 0 ]; then
    echo "CHECKPOINT_SPACING must be a positive integer, got: $CHECKPOINT_SPACING" >&2
    exit 1
fi

if [ -n "$END_CHECKPOINT" ] && ! [[ "$END_CHECKPOINT" =~ ^[0-9]+$ ]]; then
    echo "END_CHECKPOINT must be empty or an integer, got: $END_CHECKPOINT" >&2
    exit 1
fi

if [ -n "$END_CHECKPOINT" ] && [ "$END_CHECKPOINT" -lt "$START_CHECKPOINT" ]; then
    echo "END_CHECKPOINT ($END_CHECKPOINT) must be >= START_CHECKPOINT ($START_CHECKPOINT)" >&2
    exit 1
fi

if [ $((START_CHECKPOINT % CHECKPOINT_SPACING)) -ne 0 ]; then
    echo "START_CHECKPOINT ($START_CHECKPOINT) must align to CHECKPOINT_SPACING ($CHECKPOINT_SPACING)" >&2
    exit 1
fi

if [ -n "$END_CHECKPOINT" ] && [ $((END_CHECKPOINT % CHECKPOINT_SPACING)) -ne 0 ]; then
    echo "END_CHECKPOINT ($END_CHECKPOINT) must align to CHECKPOINT_SPACING ($CHECKPOINT_SPACING)" >&2
    exit 1
fi

RUN_DIR="${TRAINING_RUN_DIR%/}"
if [ ! -d "$RUN_DIR" ]; then
    echo "Training run directory not found: $RUN_DIR" >&2
    exit 1
fi

mapfile -t CHECKPOINT_STEPS < <(
    find "$RUN_DIR" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' -printf '%P\n' |
        sed -E 's/^checkpoint-([0-9]+)$/\1/' |
        awk -v start="$START_CHECKPOINT" -v end="$END_CHECKPOINT" -v spacing="$CHECKPOINT_SPACING" '
            $0 ~ /^[0-9]+$/ {
                if ($0 >= start && (end == "" || $0 <= end) && ($0 % spacing) == 0) print $0
            }
        ' |
        sort -n
)

if [ "${#CHECKPOINT_STEPS[@]}" -eq 0 ]; then
    echo "No checkpoints found in $RUN_DIR matching range ${START_CHECKPOINT}-${END_CHECKPOINT:-inf}" >&2
    exit 1
fi

# Auto-submit with exact array size when run directly (not under SLURM).
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    task_count="${#CHECKPOINT_STEPS[@]}"
    array_spec=$((task_count - 1))
    echo "Submitting SLURM array with range 0-${array_spec}."
    exec sbatch --array="0-${array_spec}" --export=ALL "$0"
fi

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$INDEX" -ge "${#CHECKPOINT_STEPS[@]}" ]; then
    echo "Array index $INDEX has no matching checkpoint (found ${#CHECKPOINT_STEPS[@]} checkpoints); exiting."
    exit 0
fi

CHECKPOINT_STEP="${CHECKPOINT_STEPS[$INDEX]}"
ADAPTER_PATH="${ADAPTER_PATH_OVERRIDE:-${RUN_DIR}/checkpoint-${CHECKPOINT_STEP}}"

if [ ! -d "$ADAPTER_PATH" ]; then
    echo "Adapter path not found: $ADAPTER_PATH" >&2
    exit 1
fi

RUN_NAME="$(basename "$RUN_DIR")"
if [ -z "$RUN_TAG" ]; then
    RUN_TAG="$RUN_NAME"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/eval_qwen3vl_${RUN_TAG}_ckpt${CHECKPOINT_STEP}_${timestamp}}"

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

ARGS=()
ARGS+=("--model-path" "$MODEL_PATH")

if [ -n "$MODEL_BASE" ]; then
    ARGS+=("--model-base" "$MODEL_BASE")
fi

ARGS+=("--adapter-path" "$ADAPTER_PATH")

if [ -n "$ATTN_IMPLEMENTATION" ]; then
    ARGS+=("--attn-implementation" "$ATTN_IMPLEMENTATION")
fi

if [ "$LOAD_4BIT" = "true" ]; then
    ARGS+=("--load-4bit")
fi

if [ "$LOAD_8BIT" = "true" ]; then
    ARGS+=("--load-8bit")
fi

ARGS+=("--dataset-json" "$DATASET_JSON")
ARGS+=("--images-dir" "$IMAGES_DIR")
ARGS+=("--max-new-tokens" "$MAX_NEW_TOKENS")
ARGS+=("--temperature" "$TEMPERATURE")
ARGS+=("--top-p" "$TOP_P")
ARGS+=("--num-beams" "$NUM_BEAMS")

if [ "$DO_SAMPLE" = "true" ]; then
    ARGS+=("--do-sample")
fi

if [ -n "$CONV_TEMPLATE" ]; then
    ARGS+=("--conv-template" "$CONV_TEMPLATE")
fi

ARGS+=("--output-dir" "$OUTPUT_DIR")
ARGS+=("--batch-size" "$BATCH_SIZE")

if [ "$SAVE_PREDICTIONS" = "true" ]; then
    ARGS+=("--save-predictions")
fi

if [ "$GENERATE_MODEL_RESULTS" = "true" ]; then
    ARGS+=("--generate-model-results")
fi

if [ -n "$LIMIT" ]; then
    ARGS+=("--limit" "$LIMIT")
fi

if [ -n "$PROMPT_OVERRIDE" ]; then
    ARGS+=("--prompt-override" "$PROMPT_OVERRIDE")
fi

ARGS+=("--gaze-model-id" "$GAZE_MODEL_ID")
ARGS+=("--gaze-box-threshold" "$GAZE_BOX_THRESHOLD")
ARGS+=("--gaze-text-threshold" "$GAZE_TEXT_THRESHOLD")
ARGS+=("--gaze-max-new-tokens" "$GAZE_MAX_NEW_TOKENS")
ARGS+=("--gaze-temperature" "$GAZE_TEMPERATURE")
ARGS+=("--gaze-iou-radius-ratio" "$GAZE_IOU_RADIUS_RATIO")

if [ -n "$GAZE_DEVICE" ]; then
    ARGS+=("--gaze-device" "$GAZE_DEVICE")
fi

if [ "$WANDB" = "true" ]; then
    ARGS+=("--log-to-wandb")
fi

echo "=========================================="
echo "Executing Qwen3-VL evaluation:"
echo "Training run: ${RUN_NAME}"
echo "Checkpoint step: ${CHECKPOINT_STEP}"
echo "Adapter: ${ADAPTER_PATH}"
printf '%s ' "$PYTHON_BIN" "gazefollow/evals/evaluate_model_qwen3vl.py"
printf '%q ' "${ARGS[@]}"
echo ""
echo "Node hostname: $(hostname)"
echo "SLURM job id: ${SLURM_JOB_ID:-N/A}"
echo "SLURM array task id: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-N/A}"
echo "CPUs on node: ${SLURM_CPUS_ON_NODE:-N/A}"
echo "GPUs requested (job): ${SLURM_JOB_GPUS:-N/A}"
echo "GPUs visible: ${CUDA_VISIBLE_DEVICES:-N/A}"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU inventory:"
    nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader
else
    echo "GPU inventory: nvidia-smi not available"
fi
echo "Output dir: $OUTPUT_DIR"
echo "=========================================="

start_time=$(date +%s)

"$PYTHON_BIN" gazefollow/evals/evaluate_model_qwen3vl.py "${ARGS[@]}"
exit_code=$?

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "=========================================="
echo "Qwen3-VL evaluation finished with exit code $exit_code"
echo "Total runtime: ${duration}s ($((${duration} / 60)) minutes)"

if [ -d "$OUTPUT_DIR" ]; then
    echo "Artifacts:"
    ls -1 "$OUTPUT_DIR"
fi

echo "=========================================="

exit $exit_code
