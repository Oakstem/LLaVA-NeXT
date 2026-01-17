#!/bin/bash
#SBATCH --job-name=vanilla_inference
#SBATCH --output=logs/vanilla_inference_%j.out
#SBATCH --error=logs/vanilla_inference_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-tad-pool
#SBATCH --qos=owner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=10G
#SBATCH --nodes=1

# Configuration variables - modify these as needed
MODEL_PATH="${MODEL_PATH:-lmms-lab/llava-onevision-qwen2-7b-ov-chat}"
MODEL_BASE="${MODEL_BASE:-}"
# ADAPTER_PATH="${ADAPTER_PATH:-/galitylab/students/alonmardi/projects/LLaVA-NeXT/training_outputs/llava-20250922_124635/checkpoint-12000}"
# ADAPTER_PATH="${ADAPTER_PATH:-/galitylab/students/alonmardi/projects/LLaVA-NeXT/training_outputs/llava-20251007_011312/checkpoint-10000}"
ADAPTER_PATH="${ADAPTER_PATH:--1}"        # set to -1 for baseline (no adapter)
# training_outputs/llava-20251207_192205/checkpoint-8000        # quite elaborate in our long prompt but wrong
# training_outputs/llava-20251207_192147/checkpoint-8000
# Prompt for adapter path if not set
if [ -z "$ADAPTER_PATH" ]; then
    echo "=========================================="
    echo "Available recent checkpoints:"
    
    # Store checkpoints in an array
    mapfile -t checkpoints < <(ls -dt /galitylab/students/alonmardi/projects/LLaVA-NeXT/training_outputs/llava-*/checkpoint-* 2>/dev/null | head -10)
    
    if [ ${#checkpoints[@]} -eq 0 ]; then
        echo "No checkpoints found."
        ADAPTER_PATH=""
    else
        for i in "${!checkpoints[@]}"; do
            echo "  [$i] ${checkpoints[$i]}"
        done
        
        echo "  [-1] No adapter (base model only)"
        echo "=========================================="
        echo "Default: [0] ${checkpoints[0]}"
        read -p "Enter checkpoint number, -1 for no adapter, full path, or press Enter for default: " user_input
        
        if [ -z "$user_input" ]; then
            # User pressed Enter, use the most recent checkpoint
            ADAPTER_PATH="${checkpoints[0]}"
            echo "Selected: $ADAPTER_PATH"
        elif [ "$user_input" = "-1" ]; then
            # User explicitly chose no adapter
            ADAPTER_PATH=""
            echo "Selected: No adapter (base model only)"
        elif [[ "$user_input" =~ ^[0-9]+$ ]] && [ "$user_input" -lt "${#checkpoints[@]}" ]; then
            # User entered a valid number
            ADAPTER_PATH="${checkpoints[$user_input]}"
            echo "Selected: $ADAPTER_PATH"
        else
            # User entered a full path
            ADAPTER_PATH="$user_input"
        fi
    fi
fi

if [ "$ADAPTER_PATH" = "-1" ]; then
    ADAPTER_PATH=""
fi

ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
QUANTIZATION="${QUANTIZATION:-}"  # Options: 4bit, 8bit, none

# Inference parameters
IMAGE_PATH="${IMAGE_PATH:-/galitylab/students/alonmardi/projects/LLaVA-NeXT/baseline_images/39740.png}"
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
# PROMPT="${PROMPT:-Describe the scene and then where every person in the image is looking.}"
# PROMPT="${PROMPT:-Describe where every person in the image is looking.}"
# PROMPT="${PROMPT:-Complete the sentence: the woman is looking at ...}"
CONV_TEMPLATE="${CONV_TEMPLATE:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-1.1}"
TOP_P="${TOP_P:-0.9}"
NUM_BEAMS="${NUM_BEAMS:-1}"
DO_SAMPLE="${DO_SAMPLE:-false}"
DISABLE_OPTIMIZATIONS="${DISABLE_OPTIMIZATIONS:-false}"
SAVE_OUTPUT="${SAVE_OUTPUT:-true}"
CHAT_MODE="${CHAT_MODE:-true}"
NO_STREAM="${NO_STREAM:-false}"
DETECT_GAZE_BBOXES="${DETECT_GAZE_BBOXES:-false}"
IMAGE_ASPECT_RATIO_FALLBACK="${IMAGE_ASPECT_RATIO_FALLBACK:-anyres_max_4}"
IMAGE_GRID_PINPOINTS_FALLBACK="${IMAGE_GRID_PINPOINTS_FALLBACK:-(1x1),...,(2x2)}"  # "(1x1),...,(6x6)"
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
echo "Detect Gaze Bboxes: $DETECT_GAZE_BBOXES"
echo "Fallback Image Aspect Ratio: $IMAGE_ASPECT_RATIO_FALLBACK"
echo "Fallback Image Grid Pinpoints: $IMAGE_GRID_PINPOINTS_FALLBACK"
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
    "--image-aspect-ratio" "$IMAGE_ASPECT_RATIO_FALLBACK"
    "--image-grid-pinpoints" "$IMAGE_GRID_PINPOINTS_FALLBACK"
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

# Add gaze detection option
if [ "$DETECT_GAZE_BBOXES" = "true" ]; then
    ARGS+=("--run-gdino")
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
