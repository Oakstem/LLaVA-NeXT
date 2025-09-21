#!/bin/bash
#SBATCH --job-name=llava_gazefollow
#SBATCH --output=logs/llava_gazefollow_%j.out
#SBATCH --error=logs/llava_gazefollow_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=22:00:00
#SBATCH --partition=gpu

# Create logs directory if it doesn't exist
mkdir -p logs

# # Load necessary modules (adjust based on your cluster)
# module load cuda/11.8
# module load python/3.10

echo "Activating virtual environment..."
source /galitylab/students/alonmardi/llava/bin/activate
# Activate virtual environment
# source ~/llava/bin/activate

# Verify activation
which python
python --version


# Print machine IP address
echo "Machine IP Address:"
hostname -I

# Verify GPU availability
echo "GPU Information:"
nvidia-smi

export PYTHONPATH="/galitylab/students/alonmardi/projects/LLaVA-NeXT:$PYTHONPATH"
# Set environment variables and run training
cd /galitylab/students/alonmardi/projects/LLaVA-NeXT && \
python llava/train/train.py \
  --model_name_or_path "lmms-lab/llava-onevision-qwen2-7b-ov-chat" \
  --data_path "/galitylab/students/alonmardi/gazefollow/results/json_results/sgl_conversation_data.json" \
  --output_dir "/galitylab/students/alonmardi/projects/LLaVA-NeXT/training_outputs/$(date +%Y%m%d_%H%M%S)" \
  --lora_enable True \
  --lora_r 4 \
  --lora_alpha 16 \
  --lora_target_modules "lm_head" \
  --mm_tunable_parts "mm_mlp_adapter,mm_language_model" \
  --num_train_epochs 4 \
  --per_device_train_batch_size 1 \
  --save_steps 1000 \
  --logging_steps 100 \
  --image_folder "/galitylab/students/alonmardi/gazefollow" \
  --bf16    

echo "Gazefollow training completed at $(date)"