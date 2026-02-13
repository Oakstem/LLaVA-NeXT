#!/bin/bash
#SBATCH --job-name=qwen3vl_neg_detection_arr
#SBATCH --output=logs/qwen3vl_neg_detection_arr_%A_%a.out
#SBATCH --error=logs/qwen3vl_neg_detection_arr_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu-tad-pool
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --array=0-0

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/galitylab/students/alonmardi/projects/LLaVA-NeXT}"
if [ ! -d "$PROJECT_ROOT" ]; then
    PROJECT_ROOT="$(pwd)"
fi

PYTHON_BIN="${PYTHON_BIN:-/galitylab/students/alonmardi/llava/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="/home/alonz/llava/bin/python"
fi
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python environment not found in /galitylab/... or /home/alonz/..." >&2
    exit 1
fi

JOB_SCRIPT="${JOB_SCRIPT:-$PROJECT_ROOT/slurm/run_qwen3vl_stage1b_negatives_detection.slurm}"
if [ ! -f "$JOB_SCRIPT" ] && [ -f "$PROJECT_ROOT/$JOB_SCRIPT" ]; then
    JOB_SCRIPT="$PROJECT_ROOT/$JOB_SCRIPT"
fi
if [ ! -f "$JOB_SCRIPT" ]; then
    echo "Base job script not found: $JOB_SCRIPT" >&2
    exit 1
fi

DATASET_JSON="${DATASET_JSON:-training_datasets/20251102/train_dataset_localization_results_normalized_filtered_conversations.json}"
IMAGES_DIR="${IMAGES_DIR:-/galitylab/students/alonmardi/gazefollow}"

CHUNK_SIZE="${CHUNK_SIZE:-10000}"
BASE_START_INDEX="${BASE_START_INDEX:-${START_INDEX_BASE:-0}}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-}"

RUN_TAG="${RUN_TAG:-${SLURM_ARRAY_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
OUTPUT_DIR="${OUTPUT_DIR:-training_datasets/stage1b_negatives/array_${RUN_TAG}}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-stage1b_qwen3vl_negatives}"
if [ "${OUTPUT_DIR#/}" = "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR"
fi

is_nonnegative_int() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

if ! is_nonnegative_int "$BASE_START_INDEX"; then
    echo "BASE_START_INDEX must be a non-negative integer, got: $BASE_START_INDEX" >&2
    exit 1
fi
if ! is_nonnegative_int "$CHUNK_SIZE" || [ "$CHUNK_SIZE" -eq 0 ]; then
    echo "CHUNK_SIZE must be a positive integer, got: $CHUNK_SIZE" >&2
    exit 1
fi
if [ -n "$MAX_CONCURRENT_TASKS" ] && (! is_nonnegative_int "$MAX_CONCURRENT_TASKS" || [ "$MAX_CONCURRENT_TASKS" -eq 0 ]); then
    echo "MAX_CONCURRENT_TASKS must be a positive integer when set, got: $MAX_CONCURRENT_TASKS" >&2
    exit 1
fi

resolve_dataset_path() {
    local dataset_path="$1"
    if [ -f "$dataset_path" ]; then
        printf "%s" "$dataset_path"
        return 0
    fi
    if [ -f "$PROJECT_ROOT/$dataset_path" ]; then
        printf "%s" "$PROJECT_ROOT/$dataset_path"
        return 0
    fi
    return 1
}

compute_total_samples() {
    local dataset_path
    dataset_path="$(resolve_dataset_path "$DATASET_JSON")" || {
        echo "Dataset file not found: $DATASET_JSON (also checked $PROJECT_ROOT/$DATASET_JSON)" >&2
        return 1
    }
    "$PYTHON_BIN" - "$dataset_path" <<'PY'
import csv
import json
import sys
from pathlib import Path

dataset_path = Path(sys.argv[1])
if dataset_path.suffix.lower() == ".csv":
    with dataset_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        print(sum(1 for _ in reader))
else:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"Dataset JSON must be a list, got {type(data)}")
    print(len(data))
PY
}

mkdir -p "$PROJECT_ROOT/logs" "$OUTPUT_DIR"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    if [ -z "$TOTAL_SAMPLES" ]; then
        TOTAL_SAMPLES="$(compute_total_samples)"
    fi
    if ! is_nonnegative_int "$TOTAL_SAMPLES"; then
        echo "TOTAL_SAMPLES must be a non-negative integer, got: $TOTAL_SAMPLES" >&2
        exit 1
    fi

    remaining_samples=$((TOTAL_SAMPLES - BASE_START_INDEX))
    if [ "$remaining_samples" -le 0 ]; then
        echo "No samples to process: TOTAL_SAMPLES=$TOTAL_SAMPLES, BASE_START_INDEX=$BASE_START_INDEX"
        exit 0
    fi

    task_count=$(((remaining_samples + CHUNK_SIZE - 1) / CHUNK_SIZE))
    array_spec="0-$((task_count - 1))"
    if [ -n "$MAX_CONCURRENT_TASKS" ]; then
        array_spec="${array_spec}%${MAX_CONCURRENT_TASKS}"
    fi

    echo "Submitting SLURM array with range ${array_spec}"
    echo "TOTAL_SAMPLES=${TOTAL_SAMPLES}, CHUNK_SIZE=${CHUNK_SIZE}, BASE_START_INDEX=${BASE_START_INDEX}"
    exec sbatch \
        --array="$array_spec" \
        --export=ALL,TOTAL_SAMPLES="$TOTAL_SAMPLES",BASE_START_INDEX="$BASE_START_INDEX",CHUNK_SIZE="$CHUNK_SIZE",OUTPUT_DIR="$OUTPUT_DIR",OUTPUT_PREFIX="$OUTPUT_PREFIX",PYTHON_BIN="$PYTHON_BIN",PROJECT_ROOT="$PROJECT_ROOT",JOB_SCRIPT="$JOB_SCRIPT",DATASET_JSON="$DATASET_JSON",IMAGES_DIR="$IMAGES_DIR" \
        "$0"
fi

if [ -z "$TOTAL_SAMPLES" ]; then
    TOTAL_SAMPLES="$(compute_total_samples)"
fi
if ! is_nonnegative_int "$TOTAL_SAMPLES"; then
    echo "TOTAL_SAMPLES must be a non-negative integer, got: $TOTAL_SAMPLES" >&2
    exit 1
fi

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
START_INDEX=$((BASE_START_INDEX + INDEX * CHUNK_SIZE))
remaining_samples=$((TOTAL_SAMPLES - START_INDEX))

if [ "$remaining_samples" -le 0 ]; then
    echo "Array index $INDEX starts at $START_INDEX, beyond TOTAL_SAMPLES=$TOTAL_SAMPLES; exiting."
    exit 0
fi

LIMIT="$CHUNK_SIZE"
if [ "$remaining_samples" -lt "$LIMIT" ]; then
    LIMIT="$remaining_samples"
fi

END_INDEX=$((START_INDEX + LIMIT - 1))
PART_TAG="$(printf '%07d_%07d' "$START_INDEX" "$END_INDEX")"
OUTPUT_CSV="${OUTPUT_DIR}/${OUTPUT_PREFIX}_${PART_TAG}.csv"

export PROJECT_ROOT
export PYTHON_BIN
export DATASET_JSON
export IMAGES_DIR
export START_INDEX
export LIMIT
export OUTPUT_CSV

echo "=========================================="
echo "Qwen3VL Stage-1b Negatives Array Task"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Start Index: $START_INDEX"
echo "Limit: $LIMIT"
echo "End Index: $END_INDEX"
echo "Output CSV: $OUTPUT_CSV"
echo "=========================================="

bash "$JOB_SCRIPT"
