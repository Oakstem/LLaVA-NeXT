#!/bin/bash
#SBATCH --job-name=vanilla_inference
#SBATCH --output=logs/vanilla_inference_%j.out
#SBATCH --error=logs/vanilla_inference_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=10G
#SBATCH --nodes=1

# Configuration variables - modify these as needed
MODEL_PATH="${MODEL_PATH:-lmms-lab/llava-onevision-qwen2-7b-ov-chat}"
MODEL_BASE="${MODEL_BASE:-}"
# ADAPTER_PATH="${ADAPTER_PATH:-/galitylab/students/alonmardi/projects/LLaVA-NeXT/training_outputs/checkpoint-13000}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
QUANTIZATION="${QUANTIZATION:-}"  # Options: 4bit, 8bit, none

# Inference parameters
IMAGE_PATH="${IMAGE_PATH:-/galitylab/students/alonmardi/projects/LLaVA-NeXT/baseline_images/39740.png}"
PROMPT="${PROMPT:-where the woman is looking at?}"
CONV_TEMPLATE="${CONV_TEMPLATE:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.8}"
NUM_BEAMS="${NUM_BEAMS:-1}"
DO_SAMPLE="${DO_SAMPLE:-false}"
DISABLE_OPTIMIZATIONS="${DISABLE_OPTIMIZATIONS:-false}"
SAVE_OUTPUT="${SAVE_OUTPUT:-}"
CHAT_MODE="${CHAT_MODE:-true}"
NO_STREAM="${NO_STREAM:-true}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "=========================================="
echo "Vanilla Inference Job Information"
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
echo "Image Path: $IMAGE_PATH"
echo "Prompt: $PROMPT"
echo "Max New Tokens: $MAX_NEW_TOKENS"
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
echo "Num Beams: $NUM_BEAMS"
echo "Do Sample: $DO_SAMPLE"
echo "Chat Mode: $CHAT_MODE"
echo "Save Output: $SAVE_OUTPUT"
echo "=========================================="

# Load any required modules (uncomment and modify as needed)
# module load cuda/11.8
# module load python/3.10

# Activate virtual environment
echo "Activating virtual environment..."
source /galitylab/students/alonmardi/llava/bin/activate

# Verify activation
which python
python --version

# Print machine IP address
echo "Machine IP Address:"
hostname -I

# Verify GPU availability
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/galitylab/students/alonmardi/projects/LLaVA-NeXT:$PYTHONPATH"

# Change to the project directory
cd /galitylab/students/alonmardi/projects/LLaVA-NeXT

# Build command arguments
ARGS=(
    "--model-path" "$MODEL_PATH"
    "--attn-implementation" "$ATTN_IMPLEMENTATION"
    "--image-path" "$IMAGE_PATH"
    "--prompt" "$PROMPT"
    "--max-new-tokens" "$MAX_NEW_TOKENS"
    "--temperature" "$TEMPERATURE"
    "--top-p" "$TOP_P"
    "--num-beams" "$NUM_BEAMS"
)

# Add optional model arguments
if [ -n "$MODEL_BASE" ]; then
    ARGS+=("--model-base" "$MODEL_BASE")
fi

if [ -n "$ADAPTER_PATH" ]; then
    ARGS+=("--adapter-path" "$ADAPTER_PATH")
fi

if [ -n "$CONV_TEMPLATE" ]; then
    ARGS+=("--conv-template" "$CONV_TEMPLATE")
fi

# Add quantization options
if [ "$QUANTIZATION" = "4bit" ]; then
    ARGS+=("--load-4bit")
elif [ "$QUANTIZATION" = "8bit" ]; then
    ARGS+=("--load-8bit")
fi

# Add sampling option
if [ "$DO_SAMPLE" = "true" ]; then
    ARGS+=("--do-sample")
fi

# Add optimization option
if [ "$DISABLE_OPTIMIZATIONS" = "true" ]; then
    ARGS+=("--disable-optimizations")
fi

# Add chat mode option
if [ "$CHAT_MODE" = "true" ]; then
    ARGS+=("--chat")
fi

# Add streaming option
if [ "$NO_STREAM" = "true" ]; then
    ARGS+=("--no-stream")
fi

# Add save output option
if [ -n "$SAVE_OUTPUT" ]; then
    ARGS+=("--save-output" "$SAVE_OUTPUT")
fi

# Print the command that will be executed
echo "Executing command:"
echo "python generate_vanilla_inference.py ${ARGS[*]}"
echo "=========================================="

# Run the script
python generate_vanilla_inference.py "${ARGS[@]}"

exit_code=$?
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code