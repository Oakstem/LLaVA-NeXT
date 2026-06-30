#!/bin/bash
#SBATCH --job-name=qwen3vl_baseline_eval
#SBATCH --output=logs/qwen3vl_baseline_eval_%j.out
#SBATCH --error=logs/qwen3vl_baseline_eval_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu-tad-pool
#SBATCH --qos=owner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alonmardi@mail.tau.ac.il

# Evaluate the base Qwen3-VL model directly on Gazefollow-style datasets.
# The prediction model defaults to Qwen3-VL-8B-Instruct, while gaze grounding
# defaults to Qwen3-VL-4B-Instruct.

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    set -euo pipefail
fi
echo "Job started at: $(date)"

PROJECT_DIR="${PROJECT_DIR:-/galitylab/students/alonmardi/projects/LLaVA-NeXT}"
PYTHON_BIN="${PYTHON_BIN:-/galitylab/students/alonmardi/llava/bin/python}"

# Dataset configuration
DATASET_JSON="${DATASET_JSON:-report_finals/datasets/validation/20251217/20251205_174741_sgl_conversation_data_20251214_204738/val_filtered.json}"
IMAGES_DIR="${IMAGES_DIR:-/galitylab/students/alonmardi/gazefollow}"
LIMIT="${LIMIT:-}"

# Qwen3-VL prediction model configuration
PREDICTION_MODEL_ID="${PREDICTION_MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
PREDICTION_DEVICE="${PREDICTION_DEVICE:-}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful vision-language assistant.}"
PROMPT_OVERRIDE="${PROMPT_OVERRIDE:-}"

# Generation controls
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"
NUM_BEAMS="${NUM_BEAMS:-1}"
DO_SAMPLE="${DO_SAMPLE:-false}"

# Qwen3-VL grounding configuration
GAZE_MODEL_ID="${GAZE_MODEL_ID:-Qwen/Qwen3-VL-4B-Instruct}"
GAZE_BOX_THRESHOLD="${GAZE_BOX_THRESHOLD:-0.20}"
GAZE_TEXT_THRESHOLD="${GAZE_TEXT_THRESHOLD:-0.20}"
GAZE_DEVICE="${GAZE_DEVICE:-}"
GAZE_MAX_NEW_TOKENS="${GAZE_MAX_NEW_TOKENS:-192}"
GAZE_TEMPERATURE="${GAZE_TEMPERATURE:-0.0}"
GAZE_IOU_RADIUS_RATIO="${GAZE_IOU_RADIUS_RATIO:-0.05}"
IN_OUT_LABELS_CSV="${IN_OUT_LABELS_CSV:-}"

# Output controls
OUTPUT_ROOT="${OUTPUT_ROOT:-./evaluation_results}"
RUN_TAG="${RUN_TAG:-qwen3vl_8b_baseline}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-true}"
GENERATE_MODEL_RESULTS="${GENERATE_MODEL_RESULTS:-true}"
TABLE_LOG_INTERVAL="${TABLE_LOG_INTERVAL:-25}"
WANDB="${WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-llava-model-eval}"
if [ "$WANDB_PROJECT" = "llava-gazefollow-finetune" ]; then
    echo "Overriding inherited training WANDB_PROJECT=$WANDB_PROJECT for model eval logging."
    WANDB_PROJECT="llava-model-eval"
    WANDB_RUN_ID=""
    WANDB_RUN_NAME=""
fi
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3vl-8b-baseline}"
WANDB_METRIC_PREFIX="${WANDB_METRIC_PREFIX:-}"

timestamp="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/eval_${RUN_TAG}_${timestamp}}"

mkdir -p logs
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"

ARGS=()
ARGS+=("--prediction-model-id" "$PREDICTION_MODEL_ID")
ARGS+=("--dataset-json" "$DATASET_JSON")
ARGS+=("--images-dir" "$IMAGES_DIR")
ARGS+=("--max-new-tokens" "$MAX_NEW_TOKENS")
ARGS+=("--temperature" "$TEMPERATURE")
ARGS+=("--top-p" "$TOP_P")
ARGS+=("--num-beams" "$NUM_BEAMS")
ARGS+=("--system-prompt" "$SYSTEM_PROMPT")
ARGS+=("--output-dir" "$OUTPUT_DIR")
ARGS+=("--table-log-interval" "$TABLE_LOG_INTERVAL")
ARGS+=("--gaze-model-id" "$GAZE_MODEL_ID")
ARGS+=("--gaze-box-threshold" "$GAZE_BOX_THRESHOLD")
ARGS+=("--gaze-text-threshold" "$GAZE_TEXT_THRESHOLD")
ARGS+=("--gaze-max-new-tokens" "$GAZE_MAX_NEW_TOKENS")
ARGS+=("--gaze-temperature" "$GAZE_TEMPERATURE")
ARGS+=("--gaze-iou-radius-ratio" "$GAZE_IOU_RADIUS_RATIO")

if [ -n "$PREDICTION_DEVICE" ]; then
    ARGS+=("--prediction-device" "$PREDICTION_DEVICE")
fi

if [ "$DO_SAMPLE" = "true" ]; then
    ARGS+=("--do-sample")
fi

if [ "$SAVE_PREDICTIONS" = "true" ]; then
    ARGS+=("--save-predictions")
fi

if [ "$GENERATE_MODEL_RESULTS" = "true" ]; then
    ARGS+=("--generate-model-results")
else
    ARGS+=("--no-generate-model-results")
fi

if [ -n "$LIMIT" ]; then
    ARGS+=("--limit" "$LIMIT")
fi

if [ -n "$PROMPT_OVERRIDE" ]; then
    ARGS+=("--prompt-override" "$PROMPT_OVERRIDE")
fi

if [ -n "$GAZE_DEVICE" ]; then
    ARGS+=("--gaze-device" "$GAZE_DEVICE")
fi

if [ -n "$IN_OUT_LABELS_CSV" ]; then
    ARGS+=("--in-out-labels-csv" "$IN_OUT_LABELS_CSV")
fi

if [ "$WANDB" = "true" ]; then
    ARGS+=("--log-to-wandb")
else
    ARGS+=("--no-log-to-wandb")
fi
if [ -n "$WANDB_PROJECT" ]; then
    ARGS+=("--wandb-project" "$WANDB_PROJECT")
fi
if [ -n "$WANDB_ENTITY" ]; then
    ARGS+=("--wandb-entity" "$WANDB_ENTITY")
fi
if [ -n "$WANDB_RUN_ID" ]; then
    ARGS+=("--wandb-run-id" "$WANDB_RUN_ID")
fi
if [ -n "$WANDB_RUN_NAME" ]; then
    ARGS+=("--wandb-run-name" "$WANDB_RUN_NAME")
fi
if [ -n "$WANDB_METRIC_PREFIX" ]; then
    ARGS+=("--wandb-metric-prefix" "$WANDB_METRIC_PREFIX")
fi

echo "=========================================="
echo "Executing Qwen3-VL baseline evaluation:"
printf '%s ' "$PYTHON_BIN" "gazefollow/evals/evaluate_qwen3vl_baseline.py"
printf '%q ' "${ARGS[@]}"
echo ""
echo "Project dir: $PROJECT_DIR"
echo "Node hostname: $(hostname)"
echo "SLURM job id: ${SLURM_JOB_ID:-N/A}"
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
echo "Prediction model: $PREDICTION_MODEL_ID"
echo "Grounding model: $GAZE_MODEL_ID"
echo "Output dir: $OUTPUT_DIR"
echo "Table log interval: $TABLE_LOG_INTERVAL"
echo "wandb run id: ${WANDB_RUN_ID:-N/A}"
echo "=========================================="

start_time=$(date +%s)

"$PYTHON_BIN" gazefollow/evals/evaluate_qwen3vl_baseline.py "${ARGS[@]}"
exit_code=$?

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "=========================================="
echo "Qwen3-VL baseline evaluation finished with exit code $exit_code"
echo "Total runtime: ${duration}s ($((${duration} / 60)) minutes)"

if [ -d "$OUTPUT_DIR" ]; then
    echo "Artifacts:"
    ls -1 "$OUTPUT_DIR"
fi

echo "=========================================="

exit $exit_code
