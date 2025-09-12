#!/bin/bash
#SBATCH --job-name=llava_gazefollow
#SBATCH --output=logs/llava_gazefollow_%j.out
#SBATCH --error=logs/llava_gazefollow_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules (adjust based on your cluster)
module load cuda/11.8
module load python/3.10

# Activate virtual environment
source ~/llava/bin/activate

# Set environment variables and run training
cd /mnt/d/Projects/LLaVA-NeXT && \
export PYTHONPATH="${PYTHONPATH}:/mnt/d/Projects/LLaVA-NeXT" && \
~/llava/bin/python llava/train/train.py \
  --model_name_or_path "lmms-lab/llava-onevision-qwen2-7b-ov-chat" \
  --data_path "/mnt/d/Projects/data/gazefollow/results/valid_runs/sgl_conversation_data.json" \
  --output_dir "/mnt/d/Projects/LLaVA-NeXT/training_outputs" \
  --lora_enable True \
  --lora_r 4 \
  --lora_alpha 16 \
  --lora_target_modules "lm_head" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --save_steps 100 \
  --logging_steps 10 \
  --image_folder "/mnt/d/Projects/data/gazefollow"

echo "Gazefollow training completed at $(date)"