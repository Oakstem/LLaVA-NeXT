#!/bin/bash
#SBATCH --job-name=llava_extraction
#SBATCH --output=logs/llava_extraction_%j.out
#SBATCH --error=logs/llava_extraction_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Load any required modules (uncomment and modify as needed)
# module load cuda/11.8
# module load python/3.10

# Activate virtual environment
source /galitylab/students/alonmardi/llava/bin/activate

# Verify GPU availability
echo "GPU Information:"
nvidia-smi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/galitylab/students/alonmardi/projects/LLaVA-NeXT:$PYTHONPATH"

# Change to the project directory
cd /galitylab/students/alonmardi/projects/LLaVA-NeXT

# Run the script with example parameters
# Modify these parameters according to your needs:
python docs/llava_extract_ppl_loc_and_gaze.py \
    --image-path "/galitylab/students/alonmardi/gazefollow/train" \
    --recursive \
    --gazefollow_dataset "/galitylab/students/alonmardi/gazefollow" \
    --base-output-dir "/galitylab/students/alonmardi/gazefollow/results/ppl_dec_loc/llava_attention_results" \
    --layer 23 \
    --start-from "00000001" \
    --skip-processed "ppl_loc_processed.txt"

echo "Job completed at: $(date)"
