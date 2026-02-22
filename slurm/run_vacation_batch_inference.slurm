#!/bin/bash
#SBATCH --job-name=vacation_batch_infer
#SBATCH --output=logs/vacation_batch_infer_%j.out
#SBATCH --error=logs/vacation_batch_infer_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=20G
#SBATCH --nodes=1

# Configuration variables - modify these as needed
MODEL_PATH="${MODEL_PATH:-lmms-lab/llava-onevision-qwen2-7b-ov-chat}"
MODEL_BASE="${MODEL_BASE:-}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
#ADAPTER_PATH="${ADAPTER_PATH:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

# Extract timestamp and checkpoint from ADAPTER_PATH for output filename
if [ -n "$ADAPTER_PATH" ]; then
    TIMESTAMP=$(echo "$ADAPTER_PATH" | grep -oP 'llava-\K\d{8}_\d{6}')
    CHECKPOINT=$(echo "$ADAPTER_PATH" | grep -oP 'checkpoint-\K\d+')
    OUTPUT_JSON="${OUTPUT_JSON:-vacation_results/llava_vacation_general_results_${TIMESTAMP}_checkpoint-${CHECKPOINT}_$(date +%Y%m%d_%H%M%S).json}"
else
    OUTPUT_JSON="${OUTPUT_JSON:-vacation_results/baseline_llava_vacation_general_results_$(date +%Y%m%d_%H%M%S).json}"
fi
QUANTIZATION="${QUANTIZATION:-}"  # Options: 4bit, 8bit, none

# Batch inference parameters
IMAGE_PATH="${IMAGE_PATH:-}"
ANNOTATIONS_FILE="${ANNOTATIONS_FILE:-datasets/Vacation/test_annotations.csv}"
FRAMES_DIR="${FRAMES_DIR:-datasets/Vacation/frames}"
OUTPUT_JSON="${OUTPUT_JSON:-vacation_results/llava_vacation_general_results_20260114_101635_checkpoint-5000_$(date +%Y%m%d_%H%M%S).json}"
PROMPT="${PROMPT:-}"
SKIP_STEP="${SKIP_STEP:-10}"
LIMIT="${LIMIT:-100}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"
RESUME="${RESUME:-false}"
RANDOMIZE="${RANDOMIZE:-true}"
SEED="${SEED:-120825}"

# Model/generation parameters
CONV_TEMPLATE="${CONV_TEMPLATE:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
DISABLE_OPTIMIZATIONS="${DISABLE_OPTIMIZATIONS:-false}"
TWO_STEP_INFERENCE="${TWO_STEP_INFERENCE:-true}"
STEP2_KEEP_ADAPTER="${STEP2_KEEP_ADAPTER:-false}"
STEP2_USES_IMAGE="${STEP2_USES_IMAGE:-false}"
STEP2_TEMPERATURE="${STEP2_TEMPERATURE:-}"

# GPT extraction parameters
ENABLE_GPT_EXTRACTION="${ENABLE_GPT_EXTRACTION:-false}"
GPT_MODEL="${GPT_MODEL:-gpt-5-nano}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "=========================================="
echo "Vacation Batch Inference Job Information"
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
if [ -n "$IMAGE_PATH" ]; then
    echo "Mode: Single image"
    echo "Image Path: $IMAGE_PATH"
else
    echo "Mode: Batch"
    echo "Annotations File: $ANNOTATIONS_FILE"
    echo "Frames Dir: $FRAMES_DIR"
fi
echo "Output JSON: $OUTPUT_JSON"
echo "Skip Step: $SKIP_STEP"
echo "Limit: $LIMIT"
echo "Checkpoint Interval: $CHECKPOINT_INTERVAL"
echo "Resume: $RESUME"
echo "Randomize: $RANDOMIZE"
echo "Seed: $SEED"
echo "Prompt: $PROMPT"
echo "Max New Tokens: $MAX_NEW_TOKENS"
echo "Temperature: $TEMPERATURE"
echo "Disable Optimizations: $DISABLE_OPTIMIZATIONS"
echo "Two-Step Inference: $TWO_STEP_INFERENCE"
echo "Step2 Keep Adapter: $STEP2_KEEP_ADAPTER"
echo "Step2 Uses Image: $STEP2_USES_IMAGE"
echo "Step2 Temperature: ${STEP2_TEMPERATURE:-<default --temperature>}"
echo "Enable GPT Extraction: $ENABLE_GPT_EXTRACTION"
echo "GPT Model: $GPT_MODEL"
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
    "--output-json" "$OUTPUT_JSON"
    "--model-path" "$MODEL_PATH"
    "--attn-implementation" "$ATTN_IMPLEMENTATION"
    "--max-new-tokens" "$MAX_NEW_TOKENS"
    "--temperature" "$TEMPERATURE"
)

if [ -n "$IMAGE_PATH" ]; then
    ARGS+=("--image-path" "$IMAGE_PATH")
else
    ARGS+=(
        "--annotations-file" "$ANNOTATIONS_FILE"
        "--frames-dir" "$FRAMES_DIR"
        "--skip-step" "$SKIP_STEP"
        "--checkpoint-interval" "$CHECKPOINT_INTERVAL"
    )
fi

# Add optional args
if [ -n "$PROMPT" ]; then
    ARGS+=("--prompt" "$PROMPT")
fi

if [ -z "$IMAGE_PATH" ] && [ -n "$LIMIT" ]; then
    ARGS+=("--limit" "$LIMIT")
fi

if [ -z "$IMAGE_PATH" ] && [ "$RESUME" = "true" ]; then
    ARGS+=("--resume")
fi

if [ -z "$IMAGE_PATH" ] && [ "$RANDOMIZE" = "true" ]; then
    ARGS+=("--randomize")
fi

if [ -z "$IMAGE_PATH" ] && [ -n "$SEED" ]; then
    ARGS+=("--seed" "$SEED")
fi

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

# Add optimization option
if [ "$DISABLE_OPTIMIZATIONS" = "true" ]; then
    ARGS+=("--disable-optimizations")
fi

# Add two-step inference option
if [ "$TWO_STEP_INFERENCE" = "true" ]; then
    ARGS+=("--two-step-inference")
fi
if [ "$STEP2_KEEP_ADAPTER" = "true" ]; then
    ARGS+=("--step2-keep-adapter")
fi
if [ "$STEP2_USES_IMAGE" = "true" ]; then
    ARGS+=("--step2-uses-image")
fi
if [ -n "$STEP2_TEMPERATURE" ]; then
    ARGS+=("--step2-temperature" "$STEP2_TEMPERATURE")
fi

# Add GPT extraction options
if [ "$ENABLE_GPT_EXTRACTION" = "true" ]; then
    ARGS+=("--enable-gpt-extraction" "--gpt-model" "$GPT_MODEL")
fi

# Print the command that will be executed
echo "Executing command:"
echo "python vacation/vacation_batch_inference.py ${ARGS[*]}"
echo "=========================================="

# Run the script
python vacation/vacation_batch_inference.py "${ARGS[@]}"
exit_code=$?
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code
