#!/bin/bash
#SBATCH --job-name=llava_extraction
#SBATCH --output=logs/llava_extraction_%j.out
#SBATCH --error=logs/llava_extraction_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --nodes=1

# Configuration variables - modify these as needed
IMAGE_PATH="${IMAGE_PATH:-/galitylab/students/alonmardi/gazefollow/train/00000000}"
GAZEFOLLOW_DATASET="${GAZEFOLLOW_DATASET:-/galitylab/students/alonmardi/gazefollow}"
OUTPUT_DIR="${OUTPUT_DIR:-llava_attention_results}"
START_LAYER="${START_LAYER:-0}"
END_LAYER="${END_LAYER:-27}"
QUANTIZATION="${QUANTIZATION:-4bit}"  # Options: 4bit, 8bit, none
RECURSIVE="${RECURSIVE:-true}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "=========================================="
echo "LLaVA Extraction Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Image Path: $IMAGE_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Layer Range: $START_LAYER to $END_LAYER"
echo "Quantization: $QUANTIZATION"
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
    "--image-path" "$IMAGE_PATH"
    "--gazefollow_dataset" "$GAZEFOLLOW_DATASET"
    "--base-output-dir" "$OUTPUT_DIR"
    "--start-layer" "$START_LAYER"
    "--end-layer" "$END_LAYER"
    "--attn-implementation" "sdpa"
    "--attn-threshold" "0.001"
    "--opening-kernel" "50"
    "--min-blob-area" "20"
    "--min-avg-attn" "0.15"
    "--show-highest-attn-blob"
)

# Add recursive flag if enabled
if [ "$RECURSIVE" = "true" ]; then
    ARGS+=("--recursive")
fi

# Add quantization options
if [ "$QUANTIZATION" = "4bit" ]; then
    ARGS+=("--load-4bit")
elif [ "$QUANTIZATION" = "8bit" ]; then
    ARGS+=("--load-8bit")
fi

# Print the command that will be executed
echo "Executing command:"
echo "python docs/llava_extract_ppl_loc_and_gaze.py ${ARGS[*]}"
echo "=========================================="

# Run the script
python docs/llava_extract_ppl_loc_and_gaze.py "${ARGS[@]}"

exit_code=$?
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code
