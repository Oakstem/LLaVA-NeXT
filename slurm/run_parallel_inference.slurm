#!/bin/bash
#SBATCH --job-name=parallel_inference
#SBATCH --output=logs/parallel_inference_%j.out
#SBATCH --error=logs/parallel_inference_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --nodes=1

# Configuration variables - modify these as needed
MODEL_PATH="${MODEL_PATH:-lmms-lab/llava-onevision-qwen2-7b-ov-chat}"
QUANTIZATION="${QUANTIZATION:-4bit}"  # Options: 4bit, none
ATTN_LAYER_IND="${ATTN_LAYER_IND:-23}"

# Worker configuration
NUM_WORKERS="${NUM_WORKERS:-4}"

# Data source arguments
BASE_IMAGE_DIR="${BASE_IMAGE_DIR:-/galitylab/students/alonmardi/gazefollow/train}"
BASE_MASK_DIR="${BASE_MASK_DIR:-/galitylab/students/alonmardi/gazefollow/train_gaze_segmentations/small_masks}"
JSON_PATH="${JSON_PATH:-}"
USE_PERSON_DESCRIPTIONS="${USE_PERSON_DESCRIPTIONS:-false}"
LIMIT_ITEMS="${LIMIT_ITEMS:-}"

# Generation arguments
PROMPT="${PROMPT:-You are provided with embeddings representing people or objects in an image. Your task is to describe each embedding and where it is looking clearly and succinctly in the following exact format: 'The _ [description of the person] is looking at _ [description of the object or person]. Repeat the sentence.' Make sure to include 'looking at' in each sentence and that each description accurately captures key visual attributes (e.g., age, gender, clothing, appearance for objects or people; type, color, state for objects) in no more than one short phrase.}"

# Bias sweep arguments
BIAS_MIN="${BIAS_MIN:-1.0}"
BIAS_MAX="${BIAS_MAX:-6.0}"
BIAS_STEPS="${BIAS_STEPS:-3}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-parallel_inference_$(date +%Y%m%d_%H%M%S)}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "=========================================="
echo "Parallel Inference Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Model Path: $MODEL_PATH"
echo "Quantization: $QUANTIZATION"
echo "Attention Layer: $ATTN_LAYER_IND"
echo "Number of Workers: $NUM_WORKERS"
echo "Base Image Dir: $BASE_IMAGE_DIR"
echo "Base Mask Dir: $BASE_MASK_DIR"
echo "JSON Path: $JSON_PATH"
echo "Use Person Descriptions: $USE_PERSON_DESCRIPTIONS"
echo "Limit Items: $LIMIT_ITEMS"
echo "Bias Range: $BIAS_MIN to $BIAS_MAX ($BIAS_STEPS steps)"
echo "Output Directory: $OUTPUT_DIR"
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
    "--model_path" "$MODEL_PATH"
    "--attn_layer_ind" "$ATTN_LAYER_IND"
    "--output_dir" "$OUTPUT_DIR"
    "--num_workers" "$NUM_WORKERS"
    "--base_image_dir" "$BASE_IMAGE_DIR"
    "--base_mask_dir" "$BASE_MASK_DIR"
    "--prompt" "$PROMPT"
    "--bias_min" "$BIAS_MIN"
    "--bias_max" "$BIAS_MAX"
    "--bias_steps" "$BIAS_STEPS"
)

# Add quantization options
if [ "$QUANTIZATION" = "4bit" ]; then
    ARGS+=("--load_4bit")
fi

# Add optional arguments
if [ -n "$JSON_PATH" ]; then
    ARGS+=("--json_path" "$JSON_PATH")
fi

if [ "$USE_PERSON_DESCRIPTIONS" = "true" ]; then
    ARGS+=("--use_person_descriptions")
fi

if [ -n "$LIMIT_ITEMS" ]; then
    ARGS+=("--limit_items" "$LIMIT_ITEMS")
fi

# Print the command that will be executed
echo "Executing command:"
echo "python parallel_inference_simple.py ${ARGS[*]}"
echo "=========================================="

# Run the script
python parallel_inference_simple.py "${ARGS[@]}"

exit_code=$?
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code