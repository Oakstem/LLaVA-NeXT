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
CKPT_START="${CKPT_START:-5000}"
CKPT_END="${CKPT_END:-5000}"
CKPT_STEP="${CKPT_STEP:-$CHECKPOINT_STEP}"
JOB_SCRIPT="${JOB_SCRIPT:-$REPO_ROOT/slurm/run_vacation_batch_inference.slurm}"
TWO_STEP_INFERENCE="${TWO_STEP_INFERENCE:-true}"
SECOND_STEP_KEEP_ADAPTER="${SECOND_STEP_KEEP_ADAPTER:-false}"
SECOND_STEP_USES_IMAGE="${SECOND_STEP_USES_IMAGE:-false}"

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

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    collect_checkpoints
    task_count=$(wc -l < "$RUN_LIST_FILE" | tr -d ' ')
    if [ "$task_count" -eq 0 ]; then
        if [ "$CKPT_MODE" = "range" ]; then
            echo "No checkpoints found for checkpoint-${CKPT_START}..${CKPT_END} (step ${CKPT_STEP}) under $TRAINING_OUTPUTS_ROOT" >&2
        else
            echo "No checkpoints found for checkpoint-$CHECKPOINT_STEP under $TRAINING_OUTPUTS_ROOT" >&2
        fi
        exit 1
    fi
    array_spec=$((task_count - 1))
    echo "Submitting SLURM array 0-$array_spec"
    exec sbatch --array="0-${array_spec}" --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",RUN_LIST_FILE="$RUN_LIST_FILE",CHECKPOINT_STEP="$CHECKPOINT_STEP",CKPT_START="$CKPT_START",CKPT_END="$CKPT_END",CKPT_STEP="$CKPT_STEP",TRAINING_OUTPUTS_ROOT="$TRAINING_OUTPUTS_ROOT",TWO_STEP_INFERENCE="$TWO_STEP_INFERENCE",SECOND_STEP_KEEP_ADAPTER="$SECOND_STEP_KEEP_ADAPTER",SECOND_STEP_USES_IMAGE="$SECOND_STEP_USES_IMAGE" "$0"
fi

if [ ! -s "$RUN_LIST_FILE" ]; then
    if [ "${SLURM_ARRAY_TASK_ID:-}" = "0" ]; then
        collect_checkpoints
    else
        for _ in $(seq 1 60); do
            if [ -s "$RUN_LIST_FILE" ]; then
                break
            fi
            sleep 1
        done
    fi
fi
if [ ! -s "$RUN_LIST_FILE" ]; then
    echo "Run list file is missing or empty: $RUN_LIST_FILE" >&2
    exit 1
fi

mapfile -t CHECKPOINT_PATHS < "$RUN_LIST_FILE"
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$INDEX" -ge "${#CHECKPOINT_PATHS[@]}" ]; then
    echo "No checkpoint for array index $INDEX (found ${#CHECKPOINT_PATHS[@]} checkpoints)"
    exit 0
fi

ADAPTER_PATH="${CHECKPOINT_PATHS[$INDEX]}"
RUN_NAME="$(basename "$(dirname "$ADAPTER_PATH")")"
CKPT_TAG="$(basename "$ADAPTER_PATH")"
OUTPUT_JSON="$OUTPUT_DIR/${RUN_NAME}_${CKPT_TAG}.json"

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
