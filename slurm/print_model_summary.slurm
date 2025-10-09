#!/bin/bash
#SBATCH --job-name=print_model_summary
#SBATCH --output=logs/print_model_summary_%j.out
#SBATCH --error=logs/print_model_summary_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting model summary printing at $(date)"

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
nvidia-smi

# Set environment variables
export PYTHONPATH="/galitylab/students/alonmardi/projects/LLaVA-NeXT:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Change to project directory
cd /galitylab/students/alonmardi/projects/LLaVA-NeXT

echo ""
echo "=== RUNNING MODEL SUMMARY SCRIPT ==="
echo ""

# Run the model summary script
python print_model_summary.py

echo ""
echo "Model summary printing completed at $(date)"
