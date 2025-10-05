#!/bin/bash
#SBATCH --job-name=llava_direct_finetune_siglip
#SBATCH --output=logs/llava_direct_finetune_siglip_%j.out
#SBATCH --error=logs/llava_direct_finetune_siglip_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting LLaVA direct finetune with SigLIP at $(date)"

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

# Set environment variables for distributed training
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export PYTHONPATH="/galitylab/students/alonmardi/projects/LLaVA-NeXT:$PYTHONPATH"

# Fix NUMA affinity issues in SLURM environment
export ACCELERATE_DISABLE_NUMA=1
export TOKENIZERS_PARALLELISM=false

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.6
export CUDA_LAUNCH_BLOCKING=0

PROJECT_NAME="llava-gazefollow-finetune"

# Model configuration - use the pretrained LLaVA model directly
CKPT_PATH="lmms-lab/llava-onevision-qwen2-7b-ov-chat"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"

# Training configuration
PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="llava-$(date +%Y%m%d_%H%M%S)"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

# Use the pretrained model directly
CKPT_PATH="lmms-lab/llava-onevision-qwen2-7b-ov-chat"

# ===== TRAINING PARAMETERS =====
# Data configuration
DATA_PATH="/galitylab/students/alonmardi/projects/LLaVA-NeXT/datasets/splits/20251004_172624_sgl_conversation_data_20251005_003422/train.json"
IMAGE_FOLDER="/galitylab/students/alonmardi/gazefollow"
OUTPUT_BASE_DIR="/galitylab/students/alonmardi/projects/LLaVA-NeXT/training_outputs"
DEEPSPEED_CONFIG="scripts/zero3.json"

# Model architecture parameters
MM_TUNABLE_PARTS="mm_vision_resampler"
MM_VISION_TOWER_LR="1e-6"
MM_PROJECTOR_TYPE="mlp2x_gelu"
MM_VISION_SELECT_LAYER="-2"
MM_USE_IM_START_END="False"
MM_USE_IM_PATCH_TOKEN="False"
GROUP_BY_MODALITY_LENGTH="True"
IMAGE_ASPECT_RATIO="anyres_max_9"
IMAGE_GRID_PINPOINTS="(1x1),...,(6x6)"
MM_PATCH_MERGE_TYPE="spatial_unpad"

# LoRA parameters
LORA_ENABLE="True"
LORA_R="4"
LORA_ALPHA="16"

# Training hyperparameters
NUM_TRAIN_EPOCHS="2"
PER_DEVICE_TRAIN_BATCH_SIZE="2"
PER_DEVICE_EVAL_BATCH_SIZE="4"
GRADIENT_ACCUMULATION_STEPS="2"
GRADIENT_CHECKPOINTING="True"
LEARNING_RATE="5e-6"
WEIGHT_DECAY="0."
WARMUP_RATIO="0.03"
LR_SCHEDULER_TYPE="cosine"

# Evaluation and saving
EVALUATION_STRATEGY="steps"
EVAL_STEPS="1000000"    # never evaluate during training
SAVE_STRATEGY="steps"
SAVE_STEPS="5000"
SAVE_TOTAL_LIMIT="20"
LOGGING_STEPS="100"

# Precision and performance
BF16="True"
TF32="True"
MODEL_MAX_LENGTH="32768"

# Set distributed training variables for single node
export NUM_GPUS=1
export NNODES=1
export RANK=0
export ADDR="localhost"
export PORT=$((29500 + ($SLURM_JOB_ID % 1000)))

# Change to project directory and run training
cd /galitylab/students/alonmardi/projects/LLaVA-NeXT

# Set WANDB environment variables
export WANDB_NAME=$MID_RUN_NAME
export WANDB_PROJECT=$PROJECT_NAME


# Print all training parameters before starting
echo "=== TRAINING PARAMETERS ==="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Model: ${CKPT_PATH}"
echo "Vision Tower: ${VISION_MODEL_VERSION}"
echo "Prompt Version: ${PROMPT_VERSION}"
echo "Run Name: $MID_RUN_NAME"
echo "Output Directory: ${OUTPUT_BASE_DIR}/${MID_RUN_NAME}"
echo ""
echo "=== DATA CONFIGURATION ==="
echo "Data Path: ${DATA_PATH}"
echo "Image Folder: ${IMAGE_FOLDER}"
echo "DeepSpeed Config: ${DEEPSPEED_CONFIG}"
echo ""
echo "=== MODEL ARCHITECTURE ==="
echo "MM Tunable Parts: ${MM_TUNABLE_PARTS}"
echo "MM Vision Tower LR: ${MM_VISION_TOWER_LR}"
echo "MM Projector Type: ${MM_PROJECTOR_TYPE}"
echo "MM Vision Select Layer: ${MM_VISION_SELECT_LAYER}"
echo "MM Use IM Start End: ${MM_USE_IM_START_END}"
echo "MM Use IM Patch Token: ${MM_USE_IM_PATCH_TOKEN}"
echo "Group by Modality Length: ${GROUP_BY_MODALITY_LENGTH}"
echo "Image Aspect Ratio: ${IMAGE_ASPECT_RATIO}"
echo "Image Grid Pinpoints: ${IMAGE_GRID_PINPOINTS}"
echo "MM Patch Merge Type: ${MM_PATCH_MERGE_TYPE}"
echo ""
echo "=== LORA CONFIGURATION ==="
echo "LoRA Enable: ${LORA_ENABLE}"
echo "LoRA r: ${LORA_R}"
echo "LoRA alpha: ${LORA_ALPHA}"
echo ""
echo "=== TRAINING HYPERPARAMETERS ==="
echo "Epochs: ${NUM_TRAIN_EPOCHS}"
echo "Train Batch Size: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "Eval Batch Size: ${PER_DEVICE_EVAL_BATCH_SIZE}"
echo "Gradient Accumulation Steps: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Gradient Checkpointing: ${GRADIENT_CHECKPOINTING}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Weight Decay: ${WEIGHT_DECAY}"
echo "Warmup Ratio: ${WARMUP_RATIO}"
echo "LR Scheduler: ${LR_SCHEDULER_TYPE}"
echo ""
echo "=== EVALUATION & SAVING ==="
echo "Evaluation Strategy: ${EVALUATION_STRATEGY}"
echo "Eval Steps: ${EVAL_STEPS}"
echo "Save Strategy: ${SAVE_STRATEGY}"
echo "Save Steps: ${SAVE_STEPS}"
echo "Save Total Limit: ${SAVE_TOTAL_LIMIT}"
echo "Logging Steps: ${LOGGING_STEPS}"
echo ""
echo "=== PRECISION & PERFORMANCE ==="
echo "BF16: ${BF16}"
echo "TF32: ${TF32}"
echo "Model Max Length: ${MODEL_MAX_LENGTH}"
echo "========================="

# Print SLURM environment variables
echo "=== SLURM ENVIRONMENT ==="
env | grep SLURM | sort
echo "========================"

ACCELERATE_DISABLE_NUMA=1 TOKENIZERS_PARALLELISM=false torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --mm_tunable_parts=${MM_TUNABLE_PARTS} \
    --mm_vision_tower_lr=${MM_VISION_TOWER_LR} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type ${MM_PROJECTOR_TYPE} \
    --mm_vision_select_layer ${MM_VISION_SELECT_LAYER} \
    --mm_use_im_start_end ${MM_USE_IM_START_END} \
    --mm_use_im_patch_token ${MM_USE_IM_PATCH_TOKEN} \
    --group_by_modality_length ${GROUP_BY_MODALITY_LENGTH} \
    --image_aspect_ratio ${IMAGE_ASPECT_RATIO} \
    --image_grid_pinpoints ${IMAGE_GRID_PINPOINTS} \
    --mm_patch_merge_type ${MM_PATCH_MERGE_TYPE} \
    --lora_enable ${LORA_ENABLE} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --bf16 ${BF16} \
    --run_name $MID_RUN_NAME \
    --output_dir ${OUTPUT_BASE_DIR}/${MID_RUN_NAME} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --evaluation_strategy ${EVALUATION_STRATEGY} \
    --eval_steps ${EVAL_STEPS} \
    --save_strategy ${SAVE_STRATEGY} \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
    --logging_steps ${LOGGING_STEPS} \
    --tf32 ${TF32} \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING}
    # --dataloader_num_workers 12 \
    # --lazy_preprocess True \
    # --report_to wandb \
    # --dataloader_drop_last True \
    # --attn_implementation sdpa \
    # --optim "adamw_torch_fused" \
    # --max_grad_norm 1.0 \
    # --save_safetensors True

echo "Direct finetune training completed at $(date)"