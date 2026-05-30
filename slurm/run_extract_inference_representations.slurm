#!/bin/bash
#SBATCH --job-name=extract_repr
#SBATCH --output=logs/extract_repr_%j.out
#SBATCH --error=logs/extract_repr_%j.err
#SBATCH --time=6:00:00
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

IMAGE_PATH="${IMAGE_PATH:-}"
VIDEO_PATH="${VIDEO_PATH:-/galitylab/students/alonmardi/Sherlock.S01E01.A.Study.in.Pink.mkv}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PROMPT="${PROMPT:-}"
PROMPTS_FILE="${PROMPTS_FILE:-}"
LAYER_INDICES="${LAYER_INDICES:-20 28}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"
DO_SAMPLE="${DO_SAMPLE:-false}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-}"
TOP_K="${TOP_K:-}"
DISABLE_OPTIMIZATIONS="${DISABLE_OPTIMIZATIONS:-false}"

MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-16}"
WINDOW_DURATION_SECONDS="${WINDOW_DURATION_SECONDS:-5.0}"
WINDOW_STRIDE_SECONDS="${WINDOW_STRIDE_SECONDS:-1.5}"
OFFSET_SECONDS="${OFFSET_SECONDS:-0.0}"
END_SECONDS="${END_SECONDS:-3000}"
LIMIT="${LIMIT:-}"

mkdir -p "$PROJECT_ROOT/logs"

echo "=========================================="
echo "Inference Representation Extraction Job"
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
echo "Image Path: ${IMAGE_PATH:-<video mode>}"
echo "Video Path: ${VIDEO_PATH:-<image mode>}"
echo "Output Path: ${OUTPUT_PATH:-<default>}"
echo "Output Dir: ${OUTPUT_DIR:-<default>}"
echo "Prompt Override: ${PROMPT:-<default>}"
echo "Prompts File: ${PROMPTS_FILE:-<none>}"
echo "Layer Indices: $LAYER_INDICES"
echo "Max New Tokens: $MAX_NEW_TOKENS"
echo "Do Sample: $DO_SAMPLE"
echo "Temperature: $TEMPERATURE"
echo "Top-p: ${TOP_P:-<default>}"
echo "Top-k: ${TOP_K:-<default>}"
echo "Max Frames: $MAX_FRAMES_NUM"
echo "Window Duration Seconds: $WINDOW_DURATION_SECONDS"
echo "Window Stride Seconds: $WINDOW_STRIDE_SECONDS"
echo "Offset Seconds: $OFFSET_SECONDS"
echo "End Seconds: $END_SECONDS"
echo "Limit: ${LIMIT:-<none>}"
echo "=========================================="

echo "Python Version:"
"$PYTHON_BIN" --version

echo "Machine IP Address:"
hostname -I

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

cd "$PROJECT_ROOT" || exit 1
mkdir -p logs
if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

ARGS=(
    "--model-path" "$MODEL_PATH"
    "--attn-implementation" "$ATTN_IMPLEMENTATION"
    "--conv-template" "$CONV_TEMPLATE"
    "--max-new-tokens" "$MAX_NEW_TOKENS"
    "--temperature" "$TEMPERATURE"
    "--max-frames-num" "$MAX_FRAMES_NUM"
    "--window-duration-seconds" "$WINDOW_DURATION_SECONDS"
    "--window-stride-seconds" "$WINDOW_STRIDE_SECONDS"
    "--offset-seconds" "$OFFSET_SECONDS"
    "--end-seconds" "$END_SECONDS"
)

if [ -n "$IMAGE_PATH" ]; then
    ARGS+=("--image-path" "$IMAGE_PATH")
    if [ -n "$OUTPUT_PATH" ]; then
        ARGS+=("--output-path" "$OUTPUT_PATH")
    fi
else
    ARGS+=("--video-path" "$VIDEO_PATH")
    if [ -n "$OUTPUT_DIR" ]; then
        ARGS+=("--output-dir" "$OUTPUT_DIR")
    fi
fi

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

if [ -n "$PROMPTS_FILE" ]; then
    ARGS+=("--prompts-file" "$PROMPTS_FILE")
fi

if [ -n "$LAYER_INDICES" ]; then
    read -r -a layer_indices_array <<< "$LAYER_INDICES"
    ARGS+=("--layer-indices" "${layer_indices_array[@]}")
fi

if [ "$DO_SAMPLE" = "true" ]; then
    ARGS+=("--do-sample")
fi

if [ -n "$TOP_P" ]; then
    ARGS+=("--top-p" "$TOP_P")
fi

if [ -n "$TOP_K" ]; then
    ARGS+=("--top-k" "$TOP_K")
fi

if [ "$DISABLE_OPTIMIZATIONS" = "true" ]; then
    ARGS+=("--disable-optimizations")
fi

if [ -n "$LIMIT" ]; then
    ARGS+=("--limit" "$LIMIT")
fi

echo "Executing command:"
echo "$PYTHON_BIN gazefollow/extract_inference_representations.py ${ARGS[*]}"
echo "=========================================="

"$PYTHON_BIN" gazefollow/extract_inference_representations.py "${ARGS[@]}"

exit_code=$?
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code
