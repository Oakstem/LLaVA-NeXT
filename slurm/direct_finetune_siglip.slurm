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
# export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL
export PYTHONPATH="/galitylab/students/alonmardi/projects/LLaVA-NeXT:$PYTHONPATH"

# Fix NUMA affinity issues in SLURM environment
export ACCELERATE_DISABLE_NUMA=1
export TOKENIZERS_PARALLELISM=false

# Memory optimization settings
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.6
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Additional memory optimizations
# export TORCH_CUDNN_V8_API_ENABLED=1
# export CUDA_MEMORY_FRACTION=0.95

PROJECT_NAME="llava-gazefollow-finetune"

# Model configuration - use the pretrained LLaVA model directly
CKPT_PATH="lmms-lab/llava-onevision-qwen2-7b-ov-chat"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"

# Training configuration
PROMPT_VERSION="qwen_1_5"

# Resume from checkpoint (optional)
# Set to a previous run's output directory name to resume training, e.g., "llava-20241007_123456"
# The trainer will automatically resume from the latest checkpoint in that directory
# Leave empty to start a fresh training run with a new output directory
RESUME_FROM_CHECKPOINT=""

# Set run name based on whether we're resuming or starting fresh
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    MID_RUN_NAME="$RESUME_FROM_CHECKPOINT"
    echo "Resuming training: ${MID_RUN_NAME}"
else
    MID_RUN_NAME="llava-$(date +%Y%m%d_%H%M%S)"
    echo "Starting fresh training: ${MID_RUN_NAME}"
fi

# Use the pretrained model directly
CKPT_PATH="lmms-lab/llava-onevision-qwen2-7b-ov-chat"

# ===== TRAINING PARAMETERS =====
# Data configuration
DATA_PATH="training_datasets/20251007_165818_sgl_conversation_data_20251007_171523/train.json"
# DATA_PATH="training_datasets/20251102/train_dataset_localization_results_normalized_filtered_conversations.json"
# DATA_PATH="training_datasets/20251205_174741_sgl_conversation_data_20251206_191724/train.json"
# DATA_PATH="training_datasets/train_filtered_20251224_203622_69k.json"
IMAGE_FOLDER="/galitylab/students/alonmardi/gazefollow"
ROI_CANDIDATES_CSV="training_datasets/stage1b_negatives/stage1b_yolo_negatives.csv"
ROI_MAX_POSITIVES="1"
ROI_MAX_NEGATIVES="8"
ROI_POSITIVE_RADIUS_RATIO="0.08"
ROI_CANDIDATES_RECOMPUTE_POSITIVES="True"
ROI_CANDIDATES_POSITIVE_CENTER_DISTANCE_THRESHOLD="0.1"
OUTPUT_BASE_DIR="/galitylab/students/alonmardi/projects/LLaVA-NeXT/training_outputs"
# DEEPSPEED_CONFIG="scripts/zero3.json"
DEEPSPEED_CONFIG="scripts/zero1.json"

# Model architecture parameters, NO SPACES ALLOWED!
# MM_TUNABLE_PARTS="mm_projector,model.layers[-8:].self_attn.{q_proj,v_proj,o_proj},model.layers[-8:].mlp,"
# MM_TUNABLE_PARTS="mm_projector,model.layers[-20:].self_attn.{q_proj,v_proj},model.layers[-20:].mlp"
MM_TUNABLE_PARTS="mm_projector,lm_head"
# MM_TUNABLE_PARTS="mm_projector,model.vision_tower.vision_tower.vision_model.encoder.layers[-20:].self_attn.{q_proj,v_proj},model.vision_tower.vision_tower.vision_model.encoder.layers[-20:].mlp"
#"mm_projector,model.layers[-4:].self_attn.{q_proj,v_proj},model.layers[-4:].mlp,vision_tower.vision_model.encoder.layers[-4:].self_attn.{q_proj,v_proj},vision_tower.vision_model.encoder.layers[-4:].mlp"
MM_VISION_TOWER_LR="1e-6"
MM_PROJECTOR_LR="5e-5"
MM_PROJECTOR_TYPE="mlp2x_gelu"
MM_VISION_SELECT_LAYER="-2"
MM_USE_IM_START_END="False"
MM_USE_IM_PATCH_TOKEN="True"
GROUP_BY_MODALITY_LENGTH="True"
IMAGE_ASPECT_RATIO="anyres_max_4" #"anyres_max_9"
IMAGE_GRID_PINPOINTS="(1x1),...,(2x2)"
# IMAGE_GRID_PINPOINTS="(1x1),...,(6x6)"
MM_PATCH_MERGE_TYPE="spatial_unpad"

# LoRA parameters
LORA_ENABLE="True"
LORA_R="4"
LORA_ALPHA="32"
LORA_DROPOUT="0.05"
MM_PROJECTOR_FULL_FINETUNE="True"

# Training hyperparameters
MAX_STEPS="1500"
PER_DEVICE_TRAIN_BATCH_SIZE="1"  # Reduced from 2 to 1
PER_DEVICE_EVAL_BATCH_SIZE="2"  # Reduced from 4 to 2
GRADIENT_ACCUMULATION_STEPS="8"  # Increased to maintain effective batch size
GRADIENT_CHECKPOINTING="False"
LEARNING_RATE="2e-6"
WEIGHT_DECAY="0.01"
WARMUP_RATIO="0.03"
LR_SCHEDULER_TYPE="cosine"
ROI_CONTRASTIVE_ENABLE="True"
ROI_CONTRASTIVE_WEIGHT="0.2"
ROI_CONTRASTIVE_TEMPERATURE="0.1"
ROI_CONTRASTIVE_DIM="512"
ROI_CONTRASTIVE_PHRASE="looking at"
ROI_CONTRASTIVE_WARMUP_RATIO="0.1"
ROI_CONTRASTIVE_RADIUS_RATIO="0.08"
ROI_POS_EMBED_DIM="64"
ROI_CONTRASTIVE_OOF_ENABLE="True"
ROI_CONTRASTIVE_OOF_TEXT="looking at someone or something outside the frame"
ROI_CONTRASTIVE_OOF_EXTRA_TEXTS="looking towards the camera||looking at the camera"
ROI_CONTRASTIVE_OOF_WEIGHT="0.5"
ROI_CONTRASTIVE_USE_TRUE_OOF_FRAMES="True"
ROI_CONTRASTIVE_PREVIEW_SAMPLES="5"
ROI_CONTRASTIVE_PREVIEW_TOPK="5"

# Evaluation and saving
EVALUATION_STRATEGY="no"
EVAL_STEPS="1000000"    # never evaluate during training
SAVE_STRATEGY="steps"
SAVE_STEPS="1000"
SAVE_TOTAL_LIMIT="20"
LOGGING_STEPS="100"

# Precision and performance
BF16="True"
TF32="True"
MODEL_MAX_LENGTH="6144"   #"32768"

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
export WANDB_JOB_ID="${SLURM_JOB_ID}"


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
echo "ROI Candidates CSV: ${ROI_CANDIDATES_CSV}"
echo "ROI Max Positives: ${ROI_MAX_POSITIVES}"
echo "ROI Max Negatives: ${ROI_MAX_NEGATIVES}"
echo "ROI Positive Radius Ratio: ${ROI_POSITIVE_RADIUS_RATIO}"
echo "ROI Candidates Recompute Positives: ${ROI_CANDIDATES_RECOMPUTE_POSITIVES}"
echo "ROI Candidates Positive Center Distance Threshold: ${ROI_CANDIDATES_POSITIVE_CENTER_DISTANCE_THRESHOLD}"
echo "DeepSpeed Config: ${DEEPSPEED_CONFIG}"
echo ""
echo "=== MODEL ARCHITECTURE ==="
echo "MM Tunable Parts: ${MM_TUNABLE_PARTS}"
echo "MM Vision Tower LR: ${MM_VISION_TOWER_LR}"
echo "MM Projector LR: ${MM_PROJECTOR_LR}"
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
echo "LoRA dropout: ${LORA_DROPOUT}"
echo ""
echo "=== TRAINING HYPERPARAMETERS ==="
echo "Max Steps: ${MAX_STEPS}"
echo "Train Batch Size: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "Eval Batch Size: ${PER_DEVICE_EVAL_BATCH_SIZE}"
echo "Gradient Accumulation Steps: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Gradient Checkpointing: ${GRADIENT_CHECKPOINTING}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Weight Decay: ${WEIGHT_DECAY}"
echo "Warmup Ratio: ${WARMUP_RATIO}"
echo "LR Scheduler: ${LR_SCHEDULER_TYPE}"
echo "ROI Contrastive Enable: ${ROI_CONTRASTIVE_ENABLE}"
echo "ROI Contrastive Weight: ${ROI_CONTRASTIVE_WEIGHT}"
echo "ROI Contrastive Temperature: ${ROI_CONTRASTIVE_TEMPERATURE}"
echo "ROI Contrastive Dim: ${ROI_CONTRASTIVE_DIM}"
echo "ROI Contrastive Phrase: ${ROI_CONTRASTIVE_PHRASE}"
echo "ROI Contrastive Warmup Ratio: ${ROI_CONTRASTIVE_WARMUP_RATIO}"
echo "ROI Contrastive Radius Ratio: ${ROI_CONTRASTIVE_RADIUS_RATIO}"
echo "ROI Position Embed Dim: ${ROI_POS_EMBED_DIM}"
echo "ROI Contrastive OOF Enable: ${ROI_CONTRASTIVE_OOF_ENABLE}"
echo "ROI Contrastive OOF Text: ${ROI_CONTRASTIVE_OOF_TEXT}"
echo "ROI Contrastive OOF Extra Texts: ${ROI_CONTRASTIVE_OOF_EXTRA_TEXTS}"
echo "ROI Contrastive OOF Weight: ${ROI_CONTRASTIVE_OOF_WEIGHT}"
echo "ROI Contrastive Use True OOF Frames: ${ROI_CONTRASTIVE_USE_TRUE_OOF_FRAMES}"
echo "ROI Contrastive Preview Samples: ${ROI_CONTRASTIVE_PREVIEW_SAMPLES}"
echo "ROI Contrastive Preview TopK: ${ROI_CONTRASTIVE_PREVIEW_TOPK}"
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

# Check if resuming from existing checkpoint
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    echo "=== RESUME TRAINING ==="
    echo "Output directory: ${OUTPUT_BASE_DIR}/${MID_RUN_NAME}"
    echo "Trainer will auto-resume from latest checkpoint in this directory"
    echo "========================"
fi

ACCELERATE_DISABLE_NUMA=1 TOKENIZERS_PARALLELISM=false torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --roi_candidates_csv ${ROI_CANDIDATES_CSV} \
    --roi_max_positives ${ROI_MAX_POSITIVES} \
    --roi_max_negatives ${ROI_MAX_NEGATIVES} \
    --roi_positive_radius_ratio ${ROI_POSITIVE_RADIUS_RATIO} \
    --roi_candidates_recompute_positives ${ROI_CANDIDATES_RECOMPUTE_POSITIVES} \
    --roi_candidates_positive_center_distance_threshold ${ROI_CANDIDATES_POSITIVE_CENTER_DISTANCE_THRESHOLD} \
    --mm_tunable_parts=${MM_TUNABLE_PARTS} \
    --mm_vision_tower_lr=${MM_VISION_TOWER_LR} \
    --mm_projector_lr=${MM_PROJECTOR_LR} \
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
    --lora_dropout ${LORA_DROPOUT} \
    --bf16 ${BF16} \
    --run_name $MID_RUN_NAME \
    --output_dir ${OUTPUT_BASE_DIR}/${MID_RUN_NAME} \
    --max_steps ${MAX_STEPS} \
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
    --roi_contrastive_enable ${ROI_CONTRASTIVE_ENABLE} \
    --roi_contrastive_weight ${ROI_CONTRASTIVE_WEIGHT} \
    --roi_contrastive_temperature ${ROI_CONTRASTIVE_TEMPERATURE} \
    --roi_contrastive_dim ${ROI_CONTRASTIVE_DIM} \
    --roi_contrastive_phrase "${ROI_CONTRASTIVE_PHRASE}" \
    --roi_contrastive_warmup_ratio ${ROI_CONTRASTIVE_WARMUP_RATIO} \
    --roi_contrastive_radius_ratio ${ROI_CONTRASTIVE_RADIUS_RATIO} \
    --roi_pos_embed_dim ${ROI_POS_EMBED_DIM} \
    --roi_contrastive_oof_enable ${ROI_CONTRASTIVE_OOF_ENABLE} \
    --roi_contrastive_oof_text "${ROI_CONTRASTIVE_OOF_TEXT}" \
    --roi_contrastive_oof_extra_texts "${ROI_CONTRASTIVE_OOF_EXTRA_TEXTS}" \
    --roi_contrastive_oof_weight ${ROI_CONTRASTIVE_OOF_WEIGHT} \
    --roi_contrastive_use_true_oof_frames ${ROI_CONTRASTIVE_USE_TRUE_OOF_FRAMES} \
    --roi_contrastive_preview_samples ${ROI_CONTRASTIVE_PREVIEW_SAMPLES} \
    --roi_contrastive_preview_topk ${ROI_CONTRASTIVE_PREVIEW_TOPK} \
    --logging_steps ${LOGGING_STEPS} \
    --tf32 ${TF32} \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
    --optim "adamw_torch_fused" \
    --dataloader_num_workers 8 \
    --dataloader_prefetch_factor 2 \
    --dataloader_pin_memory True \
    --persistent_workers=True \
    --mm_projector_full_finetune ${MM_PROJECTOR_FULL_FINETUNE} \
    --report_to wandb \
    # --deepspeed ${DEEPSPEED_CONFIG} \
    # --attn_implementation flash_attention_2 \
    # --dataloader_num_workers 10 \
    # --lazy_preprocess True \
    # --report_to wandb \
    # --dataloader_drop_last True \

    
    # --max_grad_norm 1.0 \
    # --save_safetensors True

echo "Direct finetune training completed at $(date)"
