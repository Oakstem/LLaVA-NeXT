#!/bin/sh
export PYTHONPATH=/home/ai_center/ai_users/alonmardi/projects/LLaVA-NeXT:$PYTHONPATH

# Set Matplotlib config/cache directory
export MPLCONFIGDIR="/home/ai_center/ai_users/alonmardi/.cache/matplotlib"
mkdir -p $MPLCONFIGDIR

#SBATCH --job-name=attn_focus
#SBATCH --output=attn_focus_%j.log
#SBATCH --error=attn_focus_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Change to working directory
cd $SLURM_SUBMIT_DIR

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Specify input parameters
PYTHON_EXEC="/home/ai_center/ai_users/alonmardi/miniconda/envs/llava/bin/python"

# Replace these with actual paths from your environment or with parameters passed to the sbatch command
IMAGE_PATH="/home/ai_center/ai_data/alonmardi/gazefollow/test2/"
RESULT_DIR="/home/ai_center/ai_data/alonmardi/gazefollow/llava_attention_gazefollow_sweep_run/20250506_010714_Describe_the_image_and_where_e"
PATTERN="*_attn_sweep"
LAYER=23
#THRESHOLD=${3:-0.0007}
SIGMA=2
OUTPUT_DIR="/home/ai_center/ai_data/alonmardi/gazefollow/llava_attention_gazefollow_sweep_run/"
LOG_FILE="${OUTPUT_DIR}/run_log.txt"

# Create necessary directories
mkdir -p $(dirname "$RESULT_DIR")

# Echo parameters for logging
echo "Running with parameters:"
echo "RESULT_DIR: $RESULT_DIR"
#echo "IMG_PATH: $IMG_PATH"
#echo "THRESHOLD: $THRESHOLD"
echo "SIGMA: $SIGMA"

# Run the wrapper script
$PYTHON_EXEC temporal_attn_focus_refactored.py \
    --mode "directory" \
    --sigma "$SIGMA" \
    --input-dir "$RESULT_DIR" \
    --image-path-base "$IMAGE_PATH" \
    #--pattern "$PATTERN" \
    --layer "$LAYER" 2>&1 | tee "$LOG_FILE"

echo "Script finished."
