#!/bin/bash
#SBATCH --job-name=detect_gaze_targets
#SBATCH --output=logs/detect_gaze_targets_%j.out
#SBATCH --error=logs/detect_gaze_targets_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --nodes=1

# Configuration variables - modify these as needed
IMAGE_PATH="${IMAGE_PATH:-/galitylab/students/alonmardi/projects/LLaVA-NeXT/baseline_images/39740.png}"
DESCRIPTION_FILE="${DESCRIPTION_FILE:-}"
DESCRIPTION_TEXT="${DESCRIPTION_TEXT:-Caption: A man and a woman sitting on a sofa in a living room, with the woman holding a purse.\n\nPerson 1: Woman in blue dress, looking at purse.\nPerson 2: Man in black shirt, looking at someone or something outside the frame.\nSocial interactions: The woman and man appear to be engaged in a conversation or activity.\n\nNo foreground people detected.}"
MODEL_ID="${MODEL_ID:-IDEA-Research/grounding-dino-base}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.3}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.25}"
OUTPUT="${OUTPUT:-}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "=========================================="
echo "Detect Gaze Targets Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Image Path: $IMAGE_PATH"
echo "Description File: $DESCRIPTION_FILE"
echo "Description Text: $DESCRIPTION_TEXT"
echo "Model ID: $MODEL_ID"
echo "Box Threshold: $BOX_THRESHOLD"
echo "Text Threshold: $TEXT_THRESHOLD"
echo "Output: $OUTPUT"
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
    "--model-id" "$MODEL_ID"
    "--box-threshold" "$BOX_THRESHOLD"
    "--text-threshold" "$TEXT_THRESHOLD"
)

# Add description source (file or text)
if [ -n "$DESCRIPTION_FILE" ]; then
    ARGS+=("--description-file" "$DESCRIPTION_FILE")
elif [ -n "$DESCRIPTION_TEXT" ]; then
    ARGS+=("--description-text" "$DESCRIPTION_TEXT")
else
    echo "ERROR: Either DESCRIPTION_FILE or DESCRIPTION_TEXT must be provided."
    exit 1
fi

# Add output option
if [ -n "$OUTPUT" ]; then
    ARGS+=("--output" "$OUTPUT")
fi

# Print the command that will be executed
echo "Executing command:"
echo "python gazefollow/auto_phrase_grounding/detect_gaze_targets.py ${ARGS[*]}"
echo "=========================================="

# Run the script
python gazefollow/auto_phrase_grounding/detect_gaze_targets.py "${ARGS[@]}"

exit_code=$?
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code
