#!/bin/bash
#SBATCH --job-name=attention_extraction
#SBATCH --output=logs/attention_extraction_%j.out
#SBATCH --error=logs/attention_extraction_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --nodes=1

# Configuration variables - modify these as needed
MODE="${MODE:-batch}"  # Options: single, batch, sweep
MODEL_PATH="${MODEL_PATH:-lmms-lab/llava-onevision-qwen2-7b-ov-chat}"
QUANTIZATION="${QUANTIZATION:-}"  # Options: 4bit, 8bit, none
ATTN_LAYER_IND="${ATTN_LAYER_IND:-23}"

# Single mode arguments
IMAGE_PATH="${IMAGE_PATH:-/galitylab/students/alonmardi/gazefollow/train/00000000/00000001.jpg}"
MASK_PATH="${MASK_PATH:-/galitylab/students/alonmardi/gazefollow/train_gaze_segmentations/small_masks/gaze__00000001_masks.npy}"
PROMPT="${PROMPT:-You are provided with embeddings representing people or objects in an image. Your task is to describe each embedding and where it is looking clearly and succinctly in the following exact format: 'The _ [description of the person] is looking at  _ [description of the object or person]. Repeat the sentence.' Make sure to include 'looking at' in each sentence and that each description accurately captures key visual attributes (e.g., age, gender, clothing, appearance for objects or people; type, color, state for objects) in no more than one short phrase.}"

# Batch mode arguments
JSON_PATH="${JSON_PATH:-}"
BASE_IMAGE_DIR="${BASE_IMAGE_DIR:-/galitylab/students/alonmardi/gazefollow/train}"
BASE_MASK_DIR="${BASE_MASK_DIR:-/galitylab/students/alonmardi/gazefollow/train_gaze_segmentations/small_masks/small_masks}"
LIMIT_ITEMS="${LIMIT_ITEMS:-}"
RESUME_FROM_DIR="${RESUME_FROM_DIR:-}"
USE_PERSON_DESCRIPTIONS="${USE_PERSON_DESCRIPTIONS:-false}"
START_FROM="${START_FROM:-00004859}"

# Bias sweep arguments
BIAS_MIN="${BIAS_MIN:-1.0}"
BIAS_MAX="${BIAS_MAX:-6.0}"
BIAS_STEPS="${BIAS_STEPS:-3}"

# Gaze guidance arguments
USE_GAZE_GUIDANCE="${USE_GAZE_GUIDANCE:-false}"
GUIDANCE_TOP_K="${GUIDANCE_TOP_K:-10}"
SIMILARITY_WEIGHT="${SIMILARITY_WEIGHT:-0.7}"
PROBABILITY_WEIGHT="${PROBABILITY_WEIGHT:-0.3}"

# Beam search arguments
USE_BEAM_SEARCH="${USE_BEAM_SEARCH:-false}"
NUM_BEAMS="${NUM_BEAMS:-1}"
LENGTH_PENALTY="${LENGTH_PENALTY:-1.0}"
EARLY_STOPPING="${EARLY_STOPPING:-false}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-attention_extraction_$(date +%Y%m%d_%H%M%S)}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "=========================================="
echo "Attention Extraction Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Mode: $MODE"
echo "Model Path: $MODEL_PATH"
echo "Quantization: $QUANTIZATION"
echo "Attention Layer: $ATTN_LAYER_IND"
echo "Output Directory: $OUTPUT_DIR"
if [ "$MODE" = "single" ]; then
    echo "Image Path: $IMAGE_PATH"
    echo "Mask Path: $MASK_PATH"
elif [ "$MODE" = "batch" ]; then
    echo "Base Image Dir: $BASE_IMAGE_DIR"
    echo "Base Mask Dir: $BASE_MASK_DIR"
    echo "JSON Path: $JSON_PATH"
    echo "Limit Items: $LIMIT_ITEMS"
    echo "Resume From: $RESUME_FROM_DIR"
    echo "Use Person Descriptions: $USE_PERSON_DESCRIPTIONS"
    echo "Start From: $START_FROM"
elif [ "$MODE" = "sweep" ]; then
    echo "Bias Range: $BIAS_MIN to $BIAS_MAX ($BIAS_STEPS steps)"
fi
echo "Use Gaze Guidance: $USE_GAZE_GUIDANCE"
echo "Use Beam Search: $USE_BEAM_SEARCH"
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
    "--mode" "$MODE"
    "--model_path" "$MODEL_PATH"
    "--attn_implementation" "sdpa"
    "--attn_layer_ind" "$ATTN_LAYER_IND"
    "--output_dir" "$OUTPUT_DIR"
)

# Add quantization options
if [ "$QUANTIZATION" = "4bit" ]; then
    ARGS+=("--load_4bit")
elif [ "$QUANTIZATION" = "8bit" ]; then
    ARGS+=("--load_8bit")
fi

# Add mode-specific arguments
if [ "$MODE" = "single" ]; then
    ARGS+=("--image_path" "$IMAGE_PATH")
    ARGS+=("--mask_path" "$MASK_PATH")
    ARGS+=("--prompt" "$PROMPT")
elif [ "$MODE" = "batch" ]; then
    ARGS+=("--base_image_dir" "$BASE_IMAGE_DIR")
    ARGS+=("--base_mask_dir" "$BASE_MASK_DIR")
    ARGS+=("--prompt" "$PROMPT")
    
    if [ -n "$JSON_PATH" ]; then
        ARGS+=("--json_path" "$JSON_PATH")
    fi
    
    if [ -n "$LIMIT_ITEMS" ]; then
        ARGS+=("--limit_items" "$LIMIT_ITEMS")
    fi
    
    if [ -n "$RESUME_FROM_DIR" ] && [ "$RESUME_FROM_DIR" != "false" ]; then
        ARGS+=("--resume_from_dir" "$RESUME_FROM_DIR")
    fi
    
    if [ "$USE_PERSON_DESCRIPTIONS" = "true" ]; then
        ARGS+=("--use_person_descriptions")
    fi
    
    if [ -n "$START_FROM" ]; then
        ARGS+=("--start_from" "$START_FROM")
    fi
    
    # Add bias sweep arguments for batch mode
    ARGS+=("--bias_min" "$BIAS_MIN")
    ARGS+=("--bias_max" "$BIAS_MAX")
    ARGS+=("--bias_steps" "$BIAS_STEPS")
    
elif [ "$MODE" = "sweep" ]; then
    ARGS+=("--image_path" "$IMAGE_PATH")
    ARGS+=("--mask_path" "$MASK_PATH")
    ARGS+=("--prompt" "$PROMPT")
    ARGS+=("--bias_min" "$BIAS_MIN")
    ARGS+=("--bias_max" "$BIAS_MAX")
    ARGS+=("--bias_steps" "$BIAS_STEPS")
fi

# Add gaze guidance arguments
if [ "$USE_GAZE_GUIDANCE" = "true" ]; then
    ARGS+=("--use_gaze_guidance")
    ARGS+=("--guidance_top_k" "$GUIDANCE_TOP_K")
    ARGS+=("--similarity_weight" "$SIMILARITY_WEIGHT")
    ARGS+=("--probability_weight" "$PROBABILITY_WEIGHT")
fi

# Add beam search arguments
if [ "$USE_BEAM_SEARCH" = "true" ]; then
    ARGS+=("--use_beam_search")
    ARGS+=("--num_beams" "$NUM_BEAMS")
    ARGS+=("--length_penalty" "$LENGTH_PENALTY")
    
    if [ "$EARLY_STOPPING" = "true" ]; then
        ARGS+=("--early_stopping")
    fi
fi

# Print the command that will be executed
echo "Executing command:"
echo "python extract_attention_interactive_refactored.py ${ARGS[*]}"
echo "=========================================="

# Run the script
python extract_attention_interactive_refactored.py "${ARGS[@]}"

exit_code=$?
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code