#!/bin/bash
#SBATCH --job-name=text_embeddings
#SBATCH --output=logs/text_embeddings_%j.out
#SBATCH --error=logs/text_embeddings_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-tad-pool
#SBATCH --qos=owner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --nodes=1

# Configuration variables - adjust via environment overrides when submitting
TEXT_INPUT="${TEXT_INPUT:-laptop screen with photo of a couple on it}"
IMAGE_PATH="${IMAGE_PATH:-/galitylab/students/alonmardi/gazefollow/train/00000024/00024026.jpg}"
IMAGE_TEXT="${IMAGE_TEXT:-}"
TEXT_OUTPUT_PATH="${TEXT_OUTPUT_PATH:-}"
IMAGE_OUTPUT_PATH="${IMAGE_OUTPUT_PATH:-}"
MODEL_PATH="${MODEL_PATH:-lmms-lab/llava-onevision-qwen2-7b-ov-chat}"
MODEL_BASE="${MODEL_BASE:-}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
QUANTIZATION="${QUANTIZATION:-}"  # Options: 4bit, 8bit, or empty for full precision
DISABLE_OPTIMIZATIONS="${DISABLE_OPTIMIZATIONS:-false}"
OUTPUT_DIR="${OUTPUT_DIR:-embeddings}"

mkdir -p logs

echo "=========================================="
echo "Embedding Extraction Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Model Path: $MODEL_PATH"
echo "Model Base: $MODEL_BASE"
echo "Adapter Path: $ADAPTER_PATH"
echo "Attention Implementation: $ATTN_IMPLEMENTATION"
echo "Quantization: $QUANTIZATION"
echo "Disable Optimizations: $DISABLE_OPTIMIZATIONS"
echo "Output Dir: $OUTPUT_DIR"
echo "Text Input: ${TEXT_INPUT:-<none>}"
echo "Image Path: ${IMAGE_PATH:-<none>}"
echo "Image Prompt Text: ${IMAGE_TEXT:-<default>}"
echo "Text Output Path: ${TEXT_OUTPUT_PATH:-auto}"
echo "Image Output Path: ${IMAGE_OUTPUT_PATH:-auto}"
echo "=========================================="

echo "Activating virtual environment..."
source /galitylab/students/alonmardi/llava/bin/activate

which python
python --version

echo "Machine IP Address:"
hostname -I

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/galitylab/students/alonmardi/projects/LLaVA-NeXT:$PYTHONPATH"

cd /galitylab/students/alonmardi/projects/LLaVA-NeXT

ARGS=(
    "--model-path" "$MODEL_PATH"
    "--attn-implementation" "$ATTN_IMPLEMENTATION"
    "--output-dir" "$OUTPUT_DIR"
)

if [ -n "$TEXT_INPUT" ]; then
    ARGS+=("--text" "$TEXT_INPUT")
fi

if [ -n "$IMAGE_PATH" ]; then
    ARGS+=("--image-path" "$IMAGE_PATH")
fi

if [ -n "$IMAGE_TEXT" ]; then
    ARGS+=("--image-text" "$IMAGE_TEXT")
fi

if [ -n "$MODEL_BASE" ]; then
    ARGS+=("--model-base" "$MODEL_BASE")
fi

if [ -n "$ADAPTER_PATH" ]; then
    ARGS+=("--adapter-path" "$ADAPTER_PATH")
fi

if [ -n "$TEXT_OUTPUT_PATH" ]; then
    ARGS+=("--output-path" "$TEXT_OUTPUT_PATH")
fi

if [ -n "$IMAGE_OUTPUT_PATH" ]; then
    ARGS+=("--image-output-path" "$IMAGE_OUTPUT_PATH")
fi

if [ "$QUANTIZATION" = "4bit" ]; then
    ARGS+=("--load-4bit")
elif [ "$QUANTIZATION" = "8bit" ]; then
    ARGS+=("--load-8bit")
fi

if [ "$DISABLE_OPTIMIZATIONS" = "true" ]; then
    ARGS+=("--disable-optimizations")
fi

if [ -z "$TEXT_INPUT" ] && [ -z "$IMAGE_PATH" ]; then
    echo "ERROR: Neither TEXT_INPUT nor IMAGE_PATH provided; nothing to process."
    exit 1
fi

echo "Running command:"
echo "python generate_text_embeddings.py ${ARGS[*]}"

python generate_text_embeddings.py "${ARGS[@]}"

STATUS=$?
echo "Job finished with status: $STATUS"
echo "End Time: $(date)"
