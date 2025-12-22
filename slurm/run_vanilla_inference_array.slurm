#!/bin/bash
#SBATCH --job-name=vanilla_inference_array
#SBATCH --output=logs/vanilla_inference_array_%A_%a.out
#SBATCH --error=logs/vanilla_inference_array_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-tad-pool
#SBATCH --qos=owner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --array=0-20

# if [[ -n "${SLURM_JOB_ID:-}" ]]; then
#     set -euo pipefail
# fi

MODEL_PATH="${MODEL_PATH:-lmms-lab/llava-onevision-qwen2-7b-ov-chat}"
MODEL_BASE="${MODEL_BASE:-}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-training_outputs/llava-20251220_225110}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
QUANTIZATION="${QUANTIZATION:-}"
IMAGE_PATH="${IMAGE_PATH:-/galitylab/students/alonmardi/projects/LLaVA-NeXT/baseline_images/39740.png}"
CONV_TEMPLATE="${CONV_TEMPLATE:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-1.1}"
TOP_P="${TOP_P:-0.9}"
NUM_BEAMS="${NUM_BEAMS:-1}"
DO_SAMPLE="${DO_SAMPLE:-false}"
DISABLE_OPTIMIZATIONS="${DISABLE_OPTIMIZATIONS:-false}"
CHAT_MODE="${CHAT_MODE:-false}"
NO_STREAM="${NO_STREAM:-false}"
DETECT_GAZE_BBOXES="${DETECT_GAZE_BBOXES:-false}"
IMAGE_ASPECT_RATIO_FALLBACK="${IMAGE_ASPECT_RATIO_FALLBACK:-anyres_max_4}"
IMAGE_GRID_PINPOINTS_FALLBACK="${IMAGE_GRID_PINPOINTS_FALLBACK:-(1x1),...,(2x2)}"
RUN_ID="${RUN_ID:-vanilla_array_$(basename "$CHECKPOINT_ROOT")}"
RESULTS_DIR="${RESULTS_DIR:-results/vanilla_inference_arrays/$RUN_ID}"

if [ -z "${PROMPT:-}" ]; then
    read -r -d '' PROMPT <<'EOF'
You are an expert vision assistant.
Step 1 - Caption
• Provide one concise sentence that broadly describes the entire scene.
• Begin the line with: Caption:
Step 2 - Foreground people & gaze
1. Detect every person in the image.
2. List them from left to right and number sequentially starting at 1.
For each person output exactly one line in this format:
Person {N}: {short description}, looking at {target | outside the frame | uncertain}
3. If there are more then 1 person, describe their social interactions.
Output format (no extra lines, no prose other than what is specified):
-------------------------------------------------
Caption: {your one-sentence scene description}
Person 1: {short description}, looking at ...
Person 2: {short description}, looking at ...
...
Social interactions: {describe any social interactions briefly}
-------------------------------------------------
Additional rules
• Keep the phrase "looking at" unchanged.
• {short description} must be 6 words or fewer (e.g., "man in red jacket").
• If no foreground person is detected, write exactly: No foreground people detected.
• If gaze cannot be determined, use "uncertain".
• Do not output your reasoning or any extra text.
EOF
fi

mkdir -p logs "$RESULTS_DIR"

PYTHON_BIN="${PYTHON_BIN:-/galitylab/students/alonmardi/llava/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="/home/alonz/llava/bin/python"
fi

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python environment not found; set PYTHON_BIN." >&2
    exit 1
fi

mapfile -t CHECKPOINT_NAMES < <(
    find "$CHECKPOINT_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V
)

if [ "${#CHECKPOINT_NAMES[@]}" -eq 0 ]; then
    echo "No checkpoints found under $CHECKPOINT_ROOT" >&2
    exit 1
fi

# Auto-submit with correct array size when not already running under SLURM.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    task_count="${#CHECKPOINT_NAMES[@]}"
    array_spec=$((task_count - 1))
    echo "Submitting SLURM array with range 0-${array_spec} for ${task_count} checkpoints."
    exec sbatch --array="0-${array_spec}" --export=ALL "$0"
fi

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$INDEX" -ge "${#CHECKPOINT_NAMES[@]}" ]; then
    echo "Array index $INDEX has no matching checkpoint (found ${#CHECKPOINT_NAMES[@]} checkpoints); exiting."
    exit 0
fi

CKPT_NAME="${CHECKPOINT_NAMES[$INDEX]}"
ADAPTER_PATH="${CHECKPOINT_ROOT%/}/${CKPT_NAME}"

echo "=========================================="
echo "Vanilla inference array task"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Checkpoint: ${ADAPTER_PATH}"
echo "Results dir: ${RESULTS_DIR}"
echo "=========================================="

echo "Activating virtual environment..."
source "$(dirname "$PYTHON_BIN")/activate"
export PYTHONPATH="/galitylab/students/alonmardi/projects/LLaVA-NeXT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

which python
python --version
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

cd /galitylab/students/alonmardi/projects/LLaVA-NeXT

ARGS=(
    "--model-path" "$MODEL_PATH"
    "--attn-implementation" "$ATTN_IMPLEMENTATION"
    "--image-path" "$IMAGE_PATH"
    "--image-aspect-ratio" "$IMAGE_ASPECT_RATIO_FALLBACK"
    "--image-grid-pinpoints" "$IMAGE_GRID_PINPOINTS_FALLBACK"
    "--prompt" "$PROMPT"
    "--max-new-tokens" "$MAX_NEW_TOKENS"
    "--temperature" "$TEMPERATURE"
    "--top-p" "$TOP_P"
    "--num-beams" "$NUM_BEAMS"
    "--adapter-path" "$ADAPTER_PATH"
    "--save-output" "$RESULTS_DIR/${CKPT_NAME}.json"
)

if [ -n "$MODEL_BASE" ]; then
    ARGS+=("--model-base" "$MODEL_BASE")
fi
if [ -n "$CONV_TEMPLATE" ]; then
    ARGS+=("--conv-template" "$CONV_TEMPLATE")
fi
if [ "$QUANTIZATION" = "4bit" ]; then
    ARGS+=("--load-4bit")
elif [ "$QUANTIZATION" = "8bit" ]; then
    ARGS+=("--load-8bit")
fi
if [ "$DO_SAMPLE" = "true" ]; then
    ARGS+=("--do-sample")
fi
if [ "$DISABLE_OPTIMIZATIONS" = "true" ]; then
    ARGS+=("--disable-optimizations")
fi
if [ "$CHAT_MODE" = "true" ]; then
    ARGS+=("--chat")
fi
if [ "$NO_STREAM" = "true" ]; then
    ARGS+=("--no-stream")
fi
if [ "$DETECT_GAZE_BBOXES" = "true" ]; then
    ARGS+=("--run-gdino")
fi

echo "Executing:"
printf '%s ' "$PYTHON_BIN" "generate_vanilla_inference.py"
printf '%q ' "${ARGS[@]}"
echo

"$PYTHON_BIN" generate_vanilla_inference.py "${ARGS[@]}"
status=$?

if [ $status -eq 0 ]; then
    python - "$RESULTS_DIR/${CKPT_NAME}.json" > "$RESULTS_DIR/${CKPT_NAME}.txt" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
prompt = data.get("prompt", "")
response = data.get("response", "")
print(f"prompt: {prompt}")
print("response:")
print(response)
PY
else
    echo "Checkpoint ${CKPT_NAME} failed with status $status" > "$RESULTS_DIR/${CKPT_NAME}.txt"
fi

echo "Finished with exit code $status"
exit $status
