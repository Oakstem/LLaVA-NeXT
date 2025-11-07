#!/bin/bash
#SBATCH --job-name=describe_gaze_targets
#SBATCH --output=logs/describe_gaze_targets_%j.out
#SBATCH --error=logs/describe_gaze_targets_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --nodes=1

# Run gazefollow/auto_phrase_grounding/describe_gaze_targets.py with configurable arguments.
# Override the environment variables below to point at alternative datasets, output paths,
# or Qwen3-VL settings before submitting the job.

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    set -euo pipefail
fi

PYTHON_BIN="${PYTHON_BIN:-/galitylab/students/alonmardi/llava/bin/python}"

CSV_PATH="${CSV_PATH:-gazefollow/data/train_annotations_release.csv}"
IMAGES_ROOT="${IMAGES_ROOT:-/mnt/d/Projects/data/gazefollow}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-4B-Instruct}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TARGET_BOX_SIZE="${TARGET_BOX_SIZE:-60}"
LIMIT="${LIMIT:-}"
IMAGE_ID="${IMAGE_ID:-}"
USE_TARGET_POINT="${USE_TARGET_POINT:-false}"
USE_HEAD_BBOX="${USE_HEAD_BBOX:-false}"
GROUNDING_EVAL="${GROUNDING_EVAL:-true}"
BUILD_CONVERSATIONS="${BUILD_CONVERSATIONS:-false}"
CONVERSATION_OUTPUT_JSON="${CONVERSATION_OUTPUT_JSON:-}"
CONVERSATION_SAVE_INTERVAL="${CONVERSATION_SAVE_INTERVAL:-50}"
CONVERSATION_MAX_NORMALIZED_L2="${CONVERSATION_MAX_NORMALIZED_L2:-0.2}"
CONVERSATION_MAX_SOURCE_NORMALIZED_L2="${CONVERSATION_MAX_SOURCE_NORMALIZED_L2:-0.4}"
CONVERSATION_MAX_TARGET_NORMALIZED_L2="${CONVERSATION_MAX_TARGET_NORMALIZED_L2:-0.2}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
LOG_FILE="${LOG_FILE:-gazefollow/auto_phrase_grounding/describe_gaze_targets.log}"

timestamp="$(date +%Y%m%d_%H%M%S)"
OUTPUT_JSON="${OUTPUT_JSON:-gazefollow/auto_phrase_grounding/gaze_region_descriptions_${timestamp}.json}"

mkdir -p logs
output_dirname="$(dirname "$OUTPUT_JSON")"
mkdir -p "$output_dirname"
if [ -n "$LOG_FILE" ]; then
    mkdir -p "$(dirname "$LOG_FILE")"
fi
if [ -n "$CONVERSATION_OUTPUT_JSON" ]; then
    mkdir -p "$(dirname "$CONVERSATION_OUTPUT_JSON")"
fi

ARGS=()
ARGS+=("--csv-path" "$CSV_PATH")
ARGS+=("--images-root" "$IMAGES_ROOT")
ARGS+=("--model-id" "$MODEL_ID")
ARGS+=("--device-map" "$DEVICE_MAP")
ARGS+=("--max-new-tokens" "$MAX_NEW_TOKENS")
ARGS+=("--temperature" "$TEMPERATURE")
ARGS+=("--output-json" "$OUTPUT_JSON")
ARGS+=("--target-box-size" "$TARGET_BOX_SIZE")
ARGS+=("--save-interval" "$SAVE_INTERVAL")
ARGS+=("--log-file" "$LOG_FILE")

if [ -n "$LIMIT" ]; then
    ARGS+=("--limit" "$LIMIT")
fi

if [ -n "$IMAGE_ID" ]; then
    ARGS+=("--image-id" "$IMAGE_ID")
fi

if [ "$USE_TARGET_POINT" = "true" ]; then
    ARGS+=("--use-target-point")
fi

if [ "$USE_HEAD_BBOX" = "true" ]; then
    ARGS+=("--use-head-bbox")
fi

if [ "$GROUNDING_EVAL" = "true" ]; then
    ARGS+=("--grounding-eval")
fi

if [ "$BUILD_CONVERSATIONS" = "true" ]; then
    ARGS+=("--build-conversations")
fi

if [ -n "$CONVERSATION_OUTPUT_JSON" ]; then
    ARGS+=("--conversation-output-json" "$CONVERSATION_OUTPUT_JSON")
fi

ARGS+=("--conversation-save-interval" "$CONVERSATION_SAVE_INTERVAL")
ARGS+=("--conversation-max-normalized-l2" "$CONVERSATION_MAX_NORMALIZED_L2")
ARGS+=("--conversation-max-source-normalized-l2" "$CONVERSATION_MAX_SOURCE_NORMALIZED_L2")
ARGS+=("--conversation-max-target-normalized-l2" "$CONVERSATION_MAX_TARGET_NORMALIZED_L2")

echo "=========================================="
echo "Executing describe_gaze_targets run:"
printf '%s ' "$PYTHON_BIN" "gazefollow/auto_phrase_grounding/describe_gaze_targets.py"
printf '%q ' "${ARGS[@]}"
echo ""
echo "Output JSON: $OUTPUT_JSON"
if [ -n "$CONVERSATION_OUTPUT_JSON" ]; then
    echo "Conversation JSON: $CONVERSATION_OUTPUT_JSON"
fi
echo "=========================================="

start_time=$(date +%s)

"$PYTHON_BIN" gazefollow/auto_phrase_grounding/describe_gaze_targets.py "${ARGS[@]}"
exit_code=$?

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "=========================================="
echo "describe_gaze_targets finished with exit code $exit_code"
echo "Total runtime: ${duration}s ($((${duration} / 60)) minutes)"
echo "Result summary (if file exists):"
if [ -f "$OUTPUT_JSON" ]; then
    ls -lh "$OUTPUT_JSON"
fi
if [ -n "$CONVERSATION_OUTPUT_JSON" ] && [ -f "$CONVERSATION_OUTPUT_JSON" ]; then
    ls -lh "$CONVERSATION_OUTPUT_JSON"
fi
echo "=========================================="

exit $exit_code
