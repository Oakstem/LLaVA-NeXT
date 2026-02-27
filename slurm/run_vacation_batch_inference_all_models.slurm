#!/bin/bash
#SBATCH --job-name=vacation_batch_all
#SBATCH --output=logs/vacation_batch_all_%A_%a.out
#SBATCH --error=logs/vacation_batch_all_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --array=0-50

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/galitylab/students/alonmardi/projects/LLaVA-NeXT}"
TRAINING_OUTPUTS_ROOT="${TRAINING_OUTPUTS_ROOT:-$REPO_ROOT/training_outputs}"
RUN_NAME_PATTERN="${RUN_NAME_PATTERN:-*}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-1000}"
CKPT_START="${CKPT_START:-6000}"
CKPT_END="${CKPT_END:-10000}"
CKPT_STEP="${CKPT_STEP:-$CHECKPOINT_STEP}"
JOB_SCRIPT="${JOB_SCRIPT:-$REPO_ROOT/slurm/run_vacation_batch_inference.slurm}"
TWO_STEP_INFERENCE="${TWO_STEP_INFERENCE:-true}"
STEP2_KEEP_ADAPTER="${STEP2_KEEP_ADAPTER:-true}"
STEP2_USES_IMAGE="${STEP2_USES_IMAGE:-true}"
STEP2_TEMPERATURE="${STEP2_TEMPERATURE:-}"
RESUME_MISSING_ONLY="${RESUME_MISSING_ONLY:-false}"

RUN_TAG="${RUN_TAG:-${SLURM_ARRAY_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
if [[ -n "$CKPT_START" || -n "$CKPT_END" ]]; then
    if [[ -z "$CKPT_START" || -z "$CKPT_END" ]]; then
        echo "Both CKPT_START and CKPT_END must be set for ranged checkpoints." >&2
        exit 1
    fi
    CKPT_MODE="range"
    OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/vacation_results/vacation_batch_ckpt-${CKPT_START}-${CKPT_END}_step-${CKPT_STEP}_${RUN_TAG}}"
else
    CKPT_MODE="single"
    OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/vacation_results/vacation_batch_ckpt-${CHECKPOINT_STEP}_${RUN_TAG}}"
fi
RUN_LIST_FILE="${RUN_LIST_FILE:-$OUTPUT_DIR/checkpoint_paths.txt}"
MISSING_LIST_FILE="${MISSING_LIST_FILE:-$OUTPUT_DIR/checkpoint_paths.missing.txt}"
TASK_LIST_FILE="${TASK_LIST_FILE:-$RUN_LIST_FILE}"

mkdir -p logs "$OUTPUT_DIR"

collect_checkpoints() {
    local run
    local tmp_file
    tmp_file="$(mktemp "${RUN_LIST_FILE}.tmp.XXXXXX")"
    mapfile -t runs < <(
        find "$TRAINING_OUTPUTS_ROOT" -maxdepth 1 -mindepth 1 -type d -name "$RUN_NAME_PATTERN" -printf '%f\n' | sort
    )
    for run in "${runs[@]}"; do
        if [ "$CKPT_MODE" = "range" ]; then
            for step in $(seq "$CKPT_START" "$CKPT_STEP" "$CKPT_END"); do
                local ckpt_path="$TRAINING_OUTPUTS_ROOT/$run/checkpoint-$step"
                if [ -d "$ckpt_path" ]; then
                    echo "$ckpt_path" >> "$tmp_file"
                fi
            done
        else
            local ckpt_path="$TRAINING_OUTPUTS_ROOT/$run/checkpoint-$CHECKPOINT_STEP"
            if [ -d "$ckpt_path" ]; then
                echo "$ckpt_path" >> "$tmp_file"
            fi
        fi
    done
    sort -u "$tmp_file" > "$RUN_LIST_FILE"
    rm -f "$tmp_file"
}

output_json_for_checkpoint() {
    local adapter_path="$1"
    local run_name
    local ckpt_tag
    run_name="$(basename "$(dirname "$adapter_path")")"
    ckpt_tag="$(basename "$adapter_path")"
    echo "$OUTPUT_DIR/${run_name}_${ckpt_tag}.json"
}

collect_missing_checkpoints() {
    local tmp_file
    local adapter_path
    local output_json
    tmp_file="$(mktemp "${MISSING_LIST_FILE}.tmp.XXXXXX")"
    mapfile -t checkpoint_paths < "$RUN_LIST_FILE"
    for adapter_path in "${checkpoint_paths[@]}"; do
        output_json="$(output_json_for_checkpoint "$adapter_path")"
        if [ ! -s "$output_json" ]; then
            echo "$adapter_path" >> "$tmp_file"
        fi
    done
    sort -u "$tmp_file" > "$MISSING_LIST_FILE"
    rm -f "$tmp_file"
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if [ -s "$RUN_LIST_FILE" ]; then
        echo "Using existing run list file: $RUN_LIST_FILE"
    else
        collect_checkpoints
    fi

    if [ "$RESUME_MISSING_ONLY" = "true" ]; then
        collect_missing_checkpoints
        TASK_LIST_FILE="$MISSING_LIST_FILE"
    else
        TASK_LIST_FILE="$RUN_LIST_FILE"
    fi

    task_count=$(wc -l < "$TASK_LIST_FILE" | tr -d ' ')
    if [ "$task_count" -eq 0 ]; then
        if [ "$RESUME_MISSING_ONLY" = "true" ]; then
            echo "No missing checkpoints to process under $OUTPUT_DIR"
        elif [ "$CKPT_MODE" = "range" ]; then
            echo "No checkpoints found for checkpoint-${CKPT_START}..${CKPT_END} (step ${CKPT_STEP}) under $TRAINING_OUTPUTS_ROOT" >&2
        else
            echo "No checkpoints found for checkpoint-$CHECKPOINT_STEP under $TRAINING_OUTPUTS_ROOT" >&2
        fi
        exit 0
    fi
    array_spec=$((task_count - 1))
    if [ "$RESUME_MISSING_ONLY" = "true" ]; then
        total_count=$(wc -l < "$RUN_LIST_FILE" | tr -d ' ')
        echo "Resume mode: submitting missing checkpoints only ($task_count/$total_count)"
    fi
    echo "Submitting SLURM array 0-$array_spec"
    exec sbatch --array="0-${array_spec}" --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",RUN_LIST_FILE="$RUN_LIST_FILE",TASK_LIST_FILE="$TASK_LIST_FILE",MISSING_LIST_FILE="$MISSING_LIST_FILE",RESUME_MISSING_ONLY="$RESUME_MISSING_ONLY",CHECKPOINT_STEP="$CHECKPOINT_STEP",CKPT_START="$CKPT_START",CKPT_END="$CKPT_END",CKPT_STEP="$CKPT_STEP",TRAINING_OUTPUTS_ROOT="$TRAINING_OUTPUTS_ROOT",TWO_STEP_INFERENCE="$TWO_STEP_INFERENCE",STEP2_KEEP_ADAPTER="$STEP2_KEEP_ADAPTER",STEP2_USES_IMAGE="$STEP2_USES_IMAGE",STEP2_TEMPERATURE="$STEP2_TEMPERATURE" "$0"
fi

if [ ! -s "$TASK_LIST_FILE" ]; then
    if [ "${SLURM_ARRAY_TASK_ID:-}" = "0" ]; then
        if [ "$RESUME_MISSING_ONLY" = "true" ]; then
            if [ ! -s "$RUN_LIST_FILE" ]; then
                collect_checkpoints
            fi
            collect_missing_checkpoints
            TASK_LIST_FILE="$MISSING_LIST_FILE"
        else
            collect_checkpoints
            TASK_LIST_FILE="$RUN_LIST_FILE"
        fi
    else
        for _ in $(seq 1 60); do
            if [ -s "$TASK_LIST_FILE" ]; then
                break
            fi
            sleep 1
        done
    fi
fi
if [ ! -s "$TASK_LIST_FILE" ]; then
    echo "Task list file is missing or empty: $TASK_LIST_FILE" >&2
    exit 1
fi

mapfile -t CHECKPOINT_PATHS < "$TASK_LIST_FILE"
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$INDEX" -ge "${#CHECKPOINT_PATHS[@]}" ]; then
    echo "No checkpoint for array index $INDEX (found ${#CHECKPOINT_PATHS[@]} checkpoints)"
    exit 0
fi

ADAPTER_PATH="${CHECKPOINT_PATHS[$INDEX]}"
RUN_NAME="$(basename "$(dirname "$ADAPTER_PATH")")"
CKPT_TAG="$(basename "$ADAPTER_PATH")"
OUTPUT_JSON="$(output_json_for_checkpoint "$ADAPTER_PATH")"

export ADAPTER_PATH
export OUTPUT_JSON

echo "=========================================="
echo "Vacation batch inference array task"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Checkpoint: $ADAPTER_PATH"
echo "Output JSON: $OUTPUT_JSON"
echo "=========================================="

bash "$JOB_SCRIPT"
