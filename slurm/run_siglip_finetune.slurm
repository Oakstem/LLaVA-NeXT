#!/bin/bash
#SBATCH --job-name=llava_siglip_finetune
#SBATCH --output=logs/llava_siglip_finetune_%j.out
#SBATCH --error=logs/llava_siglip_finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Create logs directory if it doesn't exist
mkdir -p logs

# Environment setup
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

echo "Activating virtual environment..."
source ~/llava/bin/activate

# Verify activation
which python
python --version

# Print machine IP address
echo "Machine IP Address:"
hostname -I

# Verify GPU availability
echo "GPU Information:"
nvidia-smi

# Model configuration
LLM_VERSION="Qwen/Qwen2-7B-Instruct" 
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# Training configuration
PROMPT_VERSION="qwen_1_5"
RUN_NAME="llava-onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-si_stage_$(date +%Y%m%d_%H%M%S)"
PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-7b-ov-chat"  # Starting from pretrained checkpoint

echo "RUN_NAME: ${RUN_NAME}"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"

# Set environment variables and run training
export PYTHONPATH="/mnt/d/Projects/LLaVA-NeXT:$PYTHONPATH"
cd /mnt/d/Projects/LLaVA-NeXT

# Number of GPUs for this node
NUM_GPUS=8

ACCELERATE_CPU_AFFINITY=1 ~/llava/bin/python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --use_env \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path /mnt/d/Projects/LLaVA-NeXT/datasets/sgl_conversation_data.json \
    --image_folder /mnt/d/Projects/LLaVA-NeXT/datasets/images \
    --video_folder /mnt/d/Projects/LLaVA-NeXT/datasets/videos \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir /mnt/d/Projects/LLaVA-NeXT/training_outputs/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32

echo "SigLIP finetuning completed at $(date)"