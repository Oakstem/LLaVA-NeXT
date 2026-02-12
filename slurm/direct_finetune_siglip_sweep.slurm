#!/bin/bash
#SBATCH --job-name=llava_direct_finetune_siglip_sweep
#SBATCH --output=logs/llava_direct_finetune_siglip_sweep_%A_%a.out
#SBATCH --error=logs/llava_direct_finetune_siglip_sweep_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-tad-pool
#SBATCH --qos=owner

set -u
mkdir -p logs

# ------------------------
# Sweep configuration
# ------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/galitylab/students/alonmardi/projects/LLaVA-NeXT}"
if [ ! -d "$PROJECT_ROOT" ]; then
    PROJECT_ROOT="$(pwd)"
fi

PROJECT_NAME="${PROJECT_NAME:-llava-gazefollow-finetune}"
CKPT_PATH="${CKPT_PATH:-lmms-lab/llava-onevision-qwen2-7b-ov-chat}"
VISION_MODEL_VERSION="${VISION_MODEL_VERSION:-google/siglip-so400m-patch14-384}"
PROMPT_VERSION="${PROMPT_VERSION:-qwen_1_5}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/galitylab/students/alonmardi/gazefollow}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/galitylab/students/alonmardi/projects/LLaVA-NeXT/training_outputs}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero1.json}"
USE_DEEPSPEED="${USE_DEEPSPEED:-false}"

MM_PROJECTOR_TYPE="${MM_PROJECTOR_TYPE:-mlp2x_gelu}"
MM_VISION_SELECT_LAYER="${MM_VISION_SELECT_LAYER:--2}"
MM_USE_IM_START_END="${MM_USE_IM_START_END:-False}"
MM_USE_IM_PATCH_TOKEN="${MM_USE_IM_PATCH_TOKEN:-True}"
GROUP_BY_MODALITY_LENGTH="${GROUP_BY_MODALITY_LENGTH:-True}"
IMAGE_ASPECT_RATIO="${IMAGE_ASPECT_RATIO:-anyres_max_4}"
IMAGE_GRID_PINPOINTS="${IMAGE_GRID_PINPOINTS:-(1x1),...,(2x2)}"
MM_PATCH_MERGE_TYPE="${MM_PATCH_MERGE_TYPE:-spatial_unpad}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
MM_VISION_TOWER_LR="${MM_VISION_TOWER_LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
ROI_CONTRASTIVE_ENABLE="${ROI_CONTRASTIVE_ENABLE:-False}"
ROI_CONTRASTIVE_WEIGHT="${ROI_CONTRASTIVE_WEIGHT:-0.1}"
ROI_CONTRASTIVE_TEMPERATURE="${ROI_CONTRASTIVE_TEMPERATURE:-0.07}"
ROI_CONTRASTIVE_DIM="${ROI_CONTRASTIVE_DIM:-512}"
ROI_CONTRASTIVE_PHRASE="${ROI_CONTRASTIVE_PHRASE:-looking at}"
ROI_CONTRASTIVE_WARMUP_RATIO="${ROI_CONTRASTIVE_WARMUP_RATIO:-0.1}"
ROI_CONTRASTIVE_RADIUS_RATIO="${ROI_CONTRASTIVE_RADIUS_RATIO:-0.08}"
EVALUATION_STRATEGY="${EVALUATION_STRATEGY:-no}"
EVAL_STEPS="${EVAL_STEPS:-1000000}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"
LOGGING_STEPS="${LOGGING_STEPS:-100}"
BF16="${BF16:-True}"
TF32="${TF32:-True}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-6144}"

BASE_DATA_PATH="${BASE_DATA_PATH:-training_datasets/20251007_165818_sgl_conversation_data_20251007_171523/train.json}"
BASE_MM_TUNABLE_PARTS="${BASE_MM_TUNABLE_PARTS:-mm_projector,lm_head}"
BASE_MM_TUNABLE_PARTS+=",model.layers[-8:].self_attn.{q_proj,v_proj,o_proj},model.layers[-8:].mlp"
BASE_MM_TUNABLE_PARTS+=",model.vision_tower.vision_tower.vision_model.encoder.layers[-8:].self_attn.{q_proj,v_proj,o_proj},model.vision_tower.vision_tower.vision_model.encoder.layers[-8:].mlp"
BASE_LORA_ENABLE="${BASE_LORA_ENABLE:-True}"
BASE_LORA_R="${BASE_LORA_R:-4}"
BASE_LORA_ALPHA="${BASE_LORA_ALPHA:-16}"
BASE_LORA_DROPOUT="${BASE_LORA_DROPOUT:-0.05}"
BASE_MM_PROJECTOR_FULL_FINETUNE="${BASE_MM_PROJECTOR_FULL_FINETUNE:-True}"

# Run-set toggles
RUNSET_LORA_RANKS="${RUNSET_LORA_RANKS:-false}"
RUNSET_VISION_TOWER_ONLY="${RUNSET_VISION_TOWER_ONLY:-false}"
RUNSET_VISION_TOWER_LAST8="${RUNSET_VISION_TOWER_LAST8:-false}"
RUNSET_VISION_TOWER_LAST16="${RUNSET_VISION_TOWER_LAST16:-false}"
RUNSET_VISION_LAST8_PLUS_MM_LMHEAD="${RUNSET_VISION_LAST8_PLUS_MM_LMHEAD:-false}"
RUNSET_VISION_LAST16_PLUS_MM_LMHEAD="${RUNSET_VISION_LAST16_PLUS_MM_LMHEAD:-false}"
RUNSET_MM_PLUS_LMHEAD_ONLY="${RUNSET_MM_PLUS_LMHEAD_ONLY:-false}"
RUNSET_FULL_LLM="${RUNSET_FULL_LLM:-false}"
RUNSET_LLM_LAST8="${RUNSET_LLM_LAST8:-false}"
RUNSET_LLM_LAST8_PLUS_MM_LMHEAD="${RUNSET_LLM_LAST8_PLUS_MM_LMHEAD:-false}"
RUNSET_LLM_LAST8_PLUS_VISION_LAST8="${RUNSET_LLM_LAST8_PLUS_VISION_LAST8:-false}"
RUNSET_LORA_DROPOUTS="${RUNSET_LORA_DROPOUTS:-false}"
RUNSET_DATASETS="${RUNSET_DATASETS:-true}"

# Sweep values
LORA_RANK_SWEEP=(16 32 64)
LORA_DROPOUT_SWEEP=(0.05 0.1 0.2)
DATASET_SWEEP_PATHS=(
    "training_datasets/20251007_165818_sgl_conversation_data_20251007_171523/train.json"
    "training_datasets/train_filtered_20251224_203622_69k.json"
    "training_datasets/20260207_184610_sgl_conversation_data_20260207_193554/train.json"
)

VT_LAST_8_SPEC="model.vision_tower.vision_tower.vision_model.encoder.layers[-8:].self_attn.{q_proj,v_proj,o_proj},model.vision_tower.vision_tower.vision_model.encoder.layers[-8:].mlp"
VT_LAST_16_SPEC="model.vision_tower.vision_tower.vision_model.encoder.layers[-16:].self_attn.{q_proj,v_proj,o_proj},model.vision_tower.vision_tower.vision_model.encoder.layers[-16:].mlp"
LLM_LAST_8_SPEC="model.layers[-8:].self_attn.{q_proj,v_proj,o_proj},model.layers[-8:].mlp"

NUM_GPUS=1
NNODES=1
RANK=0
ADDR="localhost"

# Array submission controls
AUTO_SUBMIT_ARRAY="${AUTO_SUBMIT_ARRAY:-true}"
MAX_PARALLEL_ARRAY_JOBS="${MAX_PARALLEL_ARRAY_JOBS:-0}" # 0 means no % cap
SWEEP_NAME="${SWEEP_NAME:-siglip_sweep_$(date +%Y%m%d_%H%M%S)}"

slugify() {
    echo "$1" | tr ' /,[]:(){}' '_' | tr -s '_' | sed 's/^_//;s/_$//'
}

dataset_tag() {
    local data_path="$1"
    local base_name
    local parent_name
    base_name="$(basename "$data_path")"
    parent_name="$(basename "$(dirname "$data_path")")"
    if [ "$base_name" = "train.json" ]; then
        echo "$parent_name"
    else
        echo "${base_name%.*}"
    fi
}

dataset_note_path() {
    local data_path="$1"
    local note_path="$data_path"
    note_path="${note_path#training_datasets/}"
    note_path="${note_path#/}"
    echo "$note_path"
}

RUN_SET_NAMES=()
RUN_VARIANTS=()
RUN_PARTS=()
RUN_LORA_R=()
RUN_LORA_DROPOUT=()
RUN_DATA_PATHS=()
RUN_LORA_ENABLE=()
RUN_MM_PROJECTOR_FULL=()

add_run() {
    RUN_SET_NAMES+=("$1")
    RUN_VARIANTS+=("$2")
    RUN_PARTS+=("$3")
    RUN_LORA_R+=("$4")
    RUN_LORA_DROPOUT+=("$5")
    RUN_DATA_PATHS+=("$6")
    RUN_LORA_ENABLE+=("${7:-$BASE_LORA_ENABLE}")
    RUN_MM_PROJECTOR_FULL+=("${8:-$BASE_MM_PROJECTOR_FULL_FINETUNE}")
}

build_runs() {
    if [ "$RUNSET_LORA_RANKS" = "true" ]; then
        for rank in "${LORA_RANK_SWEEP[@]}"; do
            add_run "lora_rank" "r${rank}" "$BASE_MM_TUNABLE_PARTS" "$rank" "$BASE_LORA_DROPOUT" "$BASE_DATA_PATH"
        done
    fi

    if [ "$RUNSET_VISION_TOWER_ONLY" = "true" ]; then
        add_run "vision_only" "all" "mm_vision_tower" "$BASE_LORA_R" "$BASE_LORA_DROPOUT" "$BASE_DATA_PATH"
    fi

    if [ "$RUNSET_VISION_TOWER_LAST8" = "true" ]; then
        add_run "vision_last8" "all" "$VT_LAST_8_SPEC" "$BASE_LORA_R" "$BASE_LORA_DROPOUT" "$BASE_DATA_PATH"
    fi

    if [ "$RUNSET_VISION_TOWER_LAST16" = "true" ]; then
        add_run "vision_last16" "all" "$VT_LAST_16_SPEC" "$BASE_LORA_R" "$BASE_LORA_DROPOUT" "$BASE_DATA_PATH"
    fi

    if [ "$RUNSET_VISION_LAST8_PLUS_MM_LMHEAD" = "true" ]; then
        add_run "vision_last8_plus_mm_lmhead" "all" "$VT_LAST_8_SPEC,mm_projector,lm_head" "$BASE_LORA_R" "$BASE_LORA_DROPOUT" "$BASE_DATA_PATH"
    fi

    if [ "$RUNSET_VISION_LAST16_PLUS_MM_LMHEAD" = "true" ]; then
        add_run "vision_last16_plus_mm_lmhead" "all" "$VT_LAST_16_SPEC,mm_projector,lm_head" "$BASE_LORA_R" "$BASE_LORA_DROPOUT" "$BASE_DATA_PATH"
    fi

    if [ "$RUNSET_MM_PLUS_LMHEAD_ONLY" = "true" ]; then
        add_run "mm_lmhead_only" "all" "mm_projector,lm_head" "$BASE_LORA_R" "$BASE_LORA_DROPOUT" "$BASE_DATA_PATH"
    fi

    if [ "$RUNSET_FULL_LLM" = "true" ]; then
        add_run "llm_full" "all" "mm_language_model" "$BASE_LORA_R" "$BASE_LORA_DROPOUT" "$BASE_DATA_PATH"
    fi

    if [ "$RUNSET_LLM_LAST8" = "true" ]; then
        add_run "llm_last8" "all" "$LLM_LAST_8_SPEC" "$BASE_LORA_R" "$BASE_LORA_DROPOUT" "$BASE_DATA_PATH"
    fi

    if [ "$RUNSET_LLM_LAST8_PLUS_MM_LMHEAD" = "true" ]; then
        add_run "llm_last8_plus_mm_lmhead" "all" "$LLM_LAST_8_SPEC,mm_projector,lm_head" "$BASE_LORA_R" "$BASE_LORA_DROPOUT" "$BASE_DATA_PATH"
    fi

    if [ "$RUNSET_LLM_LAST8_PLUS_VISION_LAST8" = "true" ]; then
        add_run "llm_last8_plus_vision_last8" "all" "$LLM_LAST_8_SPEC,$VT_LAST_8_SPEC,mm_projector,lm_head" "$BASE_LORA_R" "$BASE_LORA_DROPOUT" "$BASE_DATA_PATH"
    fi

    if [ "$RUNSET_LORA_DROPOUTS" = "true" ]; then
        for dropout in "${LORA_DROPOUT_SWEEP[@]}"; do
            add_run "lora_dropout" "d${dropout}" "$BASE_MM_TUNABLE_PARTS" "$BASE_LORA_R" "$dropout" "$BASE_DATA_PATH"
        done
    fi

    if [ "$RUNSET_DATASETS" = "true" ]; then
        for data_path in "${DATASET_SWEEP_PATHS[@]}"; do
            add_run "dataset" "$(dataset_tag "$data_path")" "$BASE_MM_TUNABLE_PARTS" "$BASE_LORA_R" "$BASE_LORA_DROPOUT" "$data_path"
        done
    fi
}

build_runs
TOTAL_RUNS="${#RUN_SET_NAMES[@]}"

if [ "$TOTAL_RUNS" -eq 0 ]; then
    echo "No runs configured. Check RUNSET_* toggles."
    exit 1
fi

# Submitter mode: submit an array and exit.
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "Sweep name: $SWEEP_NAME"
    echo "Configured runs: $TOTAL_RUNS"

    if [ "$AUTO_SUBMIT_ARRAY" != "true" ]; then
        echo "AUTO_SUBMIT_ARRAY=false. Submit manually with:"
        echo "  sbatch --array=0-$((TOTAL_RUNS - 1)) slurm/direct_finetune_siglip_sweep.slurm"
        echo ""
        echo "Configured array tasks:"
        for i in "${!RUN_SET_NAMES[@]}"; do
            cfg_data_id="$(dataset_tag "${RUN_DATA_PATHS[$i]}")"
            cfg_run_name="${SWEEP_NAME}_$(printf '%03d' "$((i + 1))")_$(slugify "${RUN_SET_NAMES[$i]}_${RUN_VARIANTS[$i]}_${cfg_data_id}")"
            echo "  [$i] ${cfg_run_name}"
            echo "      sbatch --array=${i} --export=ALL,SWEEP_NAME=${SWEEP_NAME} slurm/direct_finetune_siglip_sweep.slurm"
        done
        exit 0
    fi

    array_range="0-$((TOTAL_RUNS - 1))"
    if [ "$MAX_PARALLEL_ARRAY_JOBS" != "0" ]; then
        array_range="${array_range}%${MAX_PARALLEL_ARRAY_JOBS}"
    fi

    echo "Submitting array: $array_range"
    sbatch --array="$array_range" --export=ALL,SWEEP_NAME="$SWEEP_NAME" "$0"
    exit $?
fi

# Worker mode: execute only the selected run.
TASK_ID="$SLURM_ARRAY_TASK_ID"
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "$TOTAL_RUNS" ]; then
    echo "SLURM_ARRAY_TASK_ID=$TASK_ID out of range [0, $((TOTAL_RUNS - 1))]"
    exit 1
fi

set_name="${RUN_SET_NAMES[$TASK_ID]}"
variant="${RUN_VARIANTS[$TASK_ID]}"
mm_tunable_parts="${RUN_PARTS[$TASK_ID]}"
lora_r="${RUN_LORA_R[$TASK_ID]}"
lora_dropout="${RUN_LORA_DROPOUT[$TASK_ID]}"
data_path="${RUN_DATA_PATHS[$TASK_ID]}"
lora_enable="${RUN_LORA_ENABLE[$TASK_ID]}"
mm_projector_full_finetune="${RUN_MM_PROJECTOR_FULL[$TASK_ID]}"

PYTHON_BIN="/galitylab/students/alonmardi/llava/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="/home/alonz/llava/bin/python"
fi
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python environment not found in /galitylab/... or /home/alonz/..."
    exit 1
fi
source "$(dirname "$PYTHON_BIN")/activate"

cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_P2P_LEVEL=NVL
export ACCELERATE_DISABLE_NUMA=1
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0

SLURM_JOB_ID_SAFE="${SLURM_JOB_ID:-0}"
PORT=$((29500 + (SLURM_JOB_ID_SAFE % 1000) + TASK_ID))

mkdir -p "$OUTPUT_BASE_DIR"
export WANDB_PROJECT="$PROJECT_NAME"
export WANDB_JOB_ID="${SLURM_JOB_ID_SAFE}_${TASK_ID}"

run_prefix="$(printf '%03d' "$((TASK_ID + 1))")"
data_id="$(dataset_tag "$data_path")"
data_note="$(dataset_note_path "$data_path")"
run_name="${SWEEP_NAME}_${run_prefix}_$(slugify "${set_name}_${variant}_${data_id}")"
output_dir="${OUTPUT_BASE_DIR}/${run_name}"
cfg_note="set=${set_name}|var=${variant}|parts=${mm_tunable_parts}|lora=${lora_enable},r=${lora_r},drop=${lora_dropout}|mmproj_full=${mm_projector_full_finetune}|roi=${ROI_CONTRASTIVE_ENABLE},w=${ROI_CONTRASTIVE_WEIGHT},t=${ROI_CONTRASTIVE_TEMPERATURE},d=${ROI_CONTRASTIVE_DIM},wr=${ROI_CONTRASTIVE_WARMUP_RATIO},rr=${ROI_CONTRASTIVE_RADIUS_RATIO}|data=${data_note}"

export WANDB_NAME="$run_name"
export WANDB_NOTES="$cfg_note"

echo "============================================================"
echo "Array task: ${TASK_ID}/${TOTAL_RUNS}"
echo "Run name: ${run_name}"
echo "W&B notes: ${cfg_note}"
echo "Output: ${output_dir}"
echo "============================================================"

cmd=(
    torchrun
    --nproc_per_node="${NUM_GPUS}"
    --nnodes="${NNODES}"
    --node_rank="${RANK}"
    --master_addr="${ADDR}"
    --master_port="${PORT}"
    llava/train/train_mem.py
    --model_name_or_path "${CKPT_PATH}"
    --version "${PROMPT_VERSION}"
    --data_path "${data_path}"
    --image_folder "${IMAGE_FOLDER}"
    "--mm_tunable_parts=${mm_tunable_parts}"
    --mm_vision_tower_lr "${MM_VISION_TOWER_LR}"
    --vision_tower "${VISION_MODEL_VERSION}"
    --mm_projector_type "${MM_PROJECTOR_TYPE}"
    --mm_vision_select_layer "${MM_VISION_SELECT_LAYER}"
    --mm_use_im_start_end "${MM_USE_IM_START_END}"
    --mm_use_im_patch_token "${MM_USE_IM_PATCH_TOKEN}"
    --group_by_modality_length "${GROUP_BY_MODALITY_LENGTH}"
    --image_aspect_ratio "${IMAGE_ASPECT_RATIO}"
    --image_grid_pinpoints "${IMAGE_GRID_PINPOINTS}"
    --mm_patch_merge_type "${MM_PATCH_MERGE_TYPE}"
    --lora_enable "${lora_enable}"
    --lora_r "${lora_r}"
    --lora_alpha "${BASE_LORA_ALPHA}"
    --lora_dropout "${lora_dropout}"
    --bf16 "${BF16}"
    --run_name "${run_name}"
    --output_dir "${output_dir}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --evaluation_strategy "${EVALUATION_STRATEGY}"
    --eval_steps "${EVAL_STEPS}"
    --save_strategy "${SAVE_STRATEGY}"
    --save_steps "${SAVE_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --learning_rate "${LEARNING_RATE}"
    --weight_decay "${WEIGHT_DECAY}"
    --warmup_ratio "${WARMUP_RATIO}"
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
    --roi_contrastive_enable "${ROI_CONTRASTIVE_ENABLE}"
    --roi_contrastive_weight "${ROI_CONTRASTIVE_WEIGHT}"
    --roi_contrastive_temperature "${ROI_CONTRASTIVE_TEMPERATURE}"
    --roi_contrastive_dim "${ROI_CONTRASTIVE_DIM}"
    --roi_contrastive_phrase "${ROI_CONTRASTIVE_PHRASE}"
    --roi_contrastive_warmup_ratio "${ROI_CONTRASTIVE_WARMUP_RATIO}"
    --roi_contrastive_radius_ratio "${ROI_CONTRASTIVE_RADIUS_RATIO}"
    --logging_steps "${LOGGING_STEPS}"
    --tf32 "${TF32}"
    --model_max_length "${MODEL_MAX_LENGTH}"
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
    --optim "adamw_torch_fused"
    --dataloader_num_workers 8
    --dataloader_prefetch_factor 2
    --dataloader_pin_memory True
    --persistent_workers True
    --mm_projector_full_finetune "${mm_projector_full_finetune}"
    --report_to wandb
)

if [ "$USE_DEEPSPEED" = "true" ]; then
    cmd+=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

"${cmd[@]}"
status=$?

echo "Run finished with status: $status"
exit "$status"
