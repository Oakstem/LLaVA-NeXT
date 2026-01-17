#!/bin/bash
#SBATCH --job-name=vanilla_inference_sweep
#SBATCH --output=logs/vanilla_inference_sweep_%j.out
#SBATCH --error=logs/vanilla_inference_sweep_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-tad-pool
#SBATCH --qos=owner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=12G
#SBATCH --nodes=1

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
RUN_ID="${RUN_ID:-vanilla_sweep_$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-results/vanilla_inference_sweeps/$RUN_ID}"

if [ -z "$PROMPT" ]; then
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
COMBINED_JSON="$RESULTS_DIR/combined.jsonl"
COMBINED_TXT="$RESULTS_DIR/combined.txt"
: > "$COMBINED_JSON"
: > "$COMBINED_TXT"

echo "=========================================="
echo "Vanilla Inference Sweep"
echo "Job ID: $SLURM_JOB_ID"
echo "Run ID: $RUN_ID"
echo "Checkpoint root: $CHECKPOINT_ROOT"
echo "Results dir: $RESULTS_DIR"
echo "=========================================="

PYTHON_BIN="/galitylab/students/alonmardi/llava/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="/home/alonz/llava/bin/python"
fi
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python environment not found."
    exit 1
fi

echo "Activating virtual environment..."
source "$(dirname "$PYTHON_BIN")/activate"
export PYTHONPATH="/galitylab/students/alonmardi/projects/LLaVA-NeXT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

which python
python --version
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

cd /galitylab/students/alonmardi/projects/LLaVA-NeXT

BASE_ARGS=(
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
)

if [ -n "$MODEL_BASE" ]; then
    BASE_ARGS+=("--model-base" "$MODEL_BASE")
fi
if [ -n "$CONV_TEMPLATE" ]; then
    BASE_ARGS+=("--conv-template" "$CONV_TEMPLATE")
fi
if [ "$QUANTIZATION" = "4bit" ]; then
    BASE_ARGS+=("--load-4bit")
elif [ "$QUANTIZATION" = "8bit" ]; then
    BASE_ARGS+=("--load-8bit")
fi
if [ "$DO_SAMPLE" = "true" ]; then
    BASE_ARGS+=("--do-sample")
fi
if [ "$DISABLE_OPTIMIZATIONS" = "true" ]; then
    BASE_ARGS+=("--disable-optimizations")
fi
if [ "$CHAT_MODE" = "true" ]; then
    BASE_ARGS+=("--chat")
fi
if [ "$NO_STREAM" = "true" ]; then
    BASE_ARGS+=("--no-stream")
fi
if [ "$DETECT_GAZE_BBOXES" = "true" ]; then
    BASE_ARGS+=("--run-gdino")
fi

shopt -s nullglob
CHECKPOINTS=("$CHECKPOINT_ROOT"/checkpoint-*)
shopt -u nullglob
if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "No checkpoints found under $CHECKPOINT_ROOT"
    exit 1
fi
IFS=$'\n' CHECKPOINTS=($(printf "%s\n" "${CHECKPOINTS[@]}" | sort -V))
unset IFS
TOTAL=${#CHECKPOINTS[@]}

echo "Found $TOTAL checkpoints."

for idx in "${!CHECKPOINTS[@]}"; do
    ckpt="${CHECKPOINTS[$idx]}"
    ckpt_name=$(basename "$ckpt")
    output_json="$RESULTS_DIR/${ckpt_name}.json"

    echo "------------------------------------------"
    echo "[$((idx + 1))/$TOTAL] Running $ckpt"
    echo "Saving to $output_json"

    python generate_vanilla_inference.py "${BASE_ARGS[@]}" --adapter-path "$ckpt" --save-output "$output_json"
    status=$?

    if [ $status -ne 0 ]; then
        echo "Checkpoint $ckpt failed with status $status" | tee -a "$COMBINED_TXT"
        continue
    fi

    python - "$ckpt" "$output_json" >> "$COMBINED_JSON" <<'PY'
import json
import sys
from pathlib import Path

ckpt_path = sys.argv[1]
output_path = Path(sys.argv[2])
data = json.loads(output_path.read_text(encoding="utf-8"))
data["checkpoint"] = ckpt_path
print(json.dumps(data))
PY

    {
        echo "=== ${ckpt_name} ==="
        python - "$output_json" <<'PY'
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
        echo
    } >> "$COMBINED_TXT"
done

echo "=========================================="
echo "Sweep complete."
echo "Combined JSONL: $COMBINED_JSON"
echo "Combined TXT:   $COMBINED_TXT"
echo "=========================================="
