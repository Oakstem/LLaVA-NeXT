#!/bin/bash
#SBATCH --job-name=vacation_scene_results
#SBATCH --output=logs/vacation_scene_results_%j.out
#SBATCH --error=logs/vacation_scene_results_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-tad-pool
#SBATCH --qos=owner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alonmardi@mail.tau.ac.il

PROJECT_ROOT="${PROJECT_ROOT:-/galitylab/students/alonmardi/projects/LLaVA-NeXT}"

PYTHON_BIN="${PYTHON_BIN:-/galitylab/students/alonmardi/llava/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="/home/alonz/llava/bin/python"
fi
if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python env not found at /galitylab/students/alonmardi/llava/bin/python or /home/alonz/llava/bin/python"
    exit 1
fi

MODEL_PATH="${MODEL_PATH:-lmms-lab/llava-onevision-qwen2-7b-ov-chat}"
MODEL_BASE="${MODEL_BASE:-}"
ADAPTER_PATH="${ADAPTER_PATH:-training_outputs/llava-20260420_030228/checkpoint-6000}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
QUANTIZATION="${QUANTIZATION:-}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"

ANNOTATIONS_CSV="${ANNOTATIONS_CSV:-datasets/Vacation/test_annotations_with_scene_id.csv}"
VIDEOS_DIR="${VIDEOS_DIR:-datasets/Vacation/Videos}"
OUTPUT_CSV="${OUTPUT_CSV:-vacation_results/videos/test_annotations_with_scene_results.csv}"
PROMPT="${PROMPT:-}"
if [ -z "${PROMPT_FILE+x}" ]; then
    if [ -n "$PROMPT" ]; then
        PROMPT_FILE=""
    else
        PROMPT_FILE="docs/vacation_event_level_gaze_communication_labels_concise.txt"
    fi
fi

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"
DO_SAMPLE="${DO_SAMPLE:-false}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-}"
NUM_BEAMS="${NUM_BEAMS:-1}"
DISABLE_OPTIMIZATIONS="${DISABLE_OPTIMIZATIONS:-false}"

MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-16}"
VIDEO_IDS="${VIDEO_IDS:-}"
LIMIT_SCENES="${LIMIT_SCENES:-}"
SAVE_EVERY="${SAVE_EVERY:-25}"

mkdir -p "$PROJECT_ROOT/logs"

echo "=========================================="
echo "Vacation Scene Results Extraction Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Python: $PYTHON_BIN"
echo "Project Root: $PROJECT_ROOT"
echo "Model Path: $MODEL_PATH"
echo "Model Base: ${MODEL_BASE:-<none>}"
echo "Adapter Path: ${ADAPTER_PATH:-<none>}"
echo "Attention Implementation: $ATTN_IMPLEMENTATION"
echo "Quantization: ${QUANTIZATION:-<none>}"
echo "Conv Template: $CONV_TEMPLATE"
echo "Annotations CSV: $ANNOTATIONS_CSV"
echo "Videos Dir: $VIDEOS_DIR"
echo "Output CSV: $OUTPUT_CSV"
echo "Prompt Override: ${PROMPT:-<none>}"
echo "Prompt File: ${PROMPT_FILE:-<none>}"
echo "Max New Tokens: $MAX_NEW_TOKENS"
echo "Do Sample: $DO_SAMPLE"
echo "Temperature: $TEMPERATURE"
echo "Top-p: ${TOP_P:-<default>}"
echo "Num Beams: $NUM_BEAMS"
echo "Max Frames: $MAX_FRAMES_NUM"
echo "Video IDs: ${VIDEO_IDS:-<all>}"
echo "Limit Scenes: ${LIMIT_SCENES:-<none>}"
echo "Save Every: $SAVE_EVERY"
echo "Disable Optimizations: $DISABLE_OPTIMIZATIONS"
echo "=========================================="

echo "Python Version:"
"$PYTHON_BIN" --version

echo "Machine IP Address:"
hostname -I

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

cd "$PROJECT_ROOT" || exit 1
mkdir -p logs
mkdir -p "$(dirname "$OUTPUT_CSV")"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

ARGS=(
    "--annotations-csv" "$ANNOTATIONS_CSV"
    "--videos-dir" "$VIDEOS_DIR"
    "--output-csv" "$OUTPUT_CSV"
    "--model-path" "$MODEL_PATH"
    "--attn-implementation" "$ATTN_IMPLEMENTATION"
    "--conv-template" "$CONV_TEMPLATE"
    "--max-new-tokens" "$MAX_NEW_TOKENS"
    "--temperature" "$TEMPERATURE"
    "--num-beams" "$NUM_BEAMS"
    "--max-frames-num" "$MAX_FRAMES_NUM"
    "--save-every" "$SAVE_EVERY"
)

if [ -n "$MODEL_BASE" ]; then
    ARGS+=("--model-base" "$MODEL_BASE")
fi

if [ -n "$ADAPTER_PATH" ]; then
    ARGS+=("--adapter-path" "$ADAPTER_PATH")
fi

if [ "$QUANTIZATION" = "4bit" ]; then
    ARGS+=("--load-4bit")
elif [ "$QUANTIZATION" = "8bit" ]; then
    ARGS+=("--load-8bit")
fi

if [ -n "$PROMPT" ]; then
    ARGS+=("--prompt" "$PROMPT")
fi

if [ -n "$PROMPT_FILE" ]; then
    ARGS+=("--prompt-file" "$PROMPT_FILE")
fi

if [ "$DO_SAMPLE" = "true" ]; then
    ARGS+=("--do-sample")
fi

if [ -n "$TOP_P" ]; then
    ARGS+=("--top-p" "$TOP_P")
fi

if [ "$DISABLE_OPTIMIZATIONS" = "true" ]; then
    ARGS+=("--disable-optimizations")
fi

if [ -n "$VIDEO_IDS" ]; then
    read -r -a video_ids_array <<< "$VIDEO_IDS"
    ARGS+=("--video-ids" "${video_ids_array[@]}")
fi

if [ -n "$LIMIT_SCENES" ]; then
    ARGS+=("--limit-scenes" "$LIMIT_SCENES")
fi

echo "Executing command:"
echo "$PYTHON_BIN gazefollow/extract_vacation_scene_results.py ${ARGS[*]}"
echo "=========================================="

"$PYTHON_BIN" gazefollow/extract_vacation_scene_results.py "${ARGS[@]}"

exit_code=$?
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code
