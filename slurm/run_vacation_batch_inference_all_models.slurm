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
ARRAY_MAX_CONCURRENT="${ARRAY_MAX_CONCURRENT:-}"
RESUME_REQUIRE_SUCCESS_MARKER="${RESUME_REQUIRE_SUCCESS_MARKER:-false}"

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
if [ "${OUTPUT_DIR#/}" = "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR"
fi
RUN_LIST_FILE="${RUN_LIST_FILE:-$OUTPUT_DIR/checkpoint_paths.txt}"
MISSING_LIST_FILE="${MISSING_LIST_FILE:-$OUTPUT_DIR/checkpoint_paths.missing.txt}"
TASK_LIST_FILE="${TASK_LIST_FILE:-$RUN_LIST_FILE}"
STATUS_DIR="${STATUS_DIR:-$OUTPUT_DIR/task_status}"
TASK_LOG_DIR="${TASK_LOG_DIR:-$OUTPUT_DIR/task_logs}"
TASK_MANIFEST_FILE="${TASK_MANIFEST_FILE:-$OUTPUT_DIR/task_manifest.tsv}"

mkdir -p logs "$OUTPUT_DIR" "$STATUS_DIR" "$TASK_LOG_DIR"

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

manifest_index_for_checkpoint() {
    local adapter_path="$1"
    if [ ! -s "$TASK_MANIFEST_FILE" ]; then
        return 1
    fi
    awk -F'\t' -v p="$adapter_path" '$2 == p { print $1; exit }' "$TASK_MANIFEST_FILE"
}

checkpoint_has_success_marker() {
    local adapter_path="$1"
    local manifest_index
    local end_file
    local exit_code
    manifest_index="$(manifest_index_for_checkpoint "$adapter_path" || true)"
    if [ -z "$manifest_index" ]; then
        return 1
    fi
    end_file="$STATUS_DIR/task_${manifest_index}.end"
    if [ ! -s "$end_file" ]; then
        return 1
    fi
    exit_code="$(awk -F'=' '/^exit_code=/{print $2; exit}' "$end_file")"
    [ "$exit_code" = "0" ]
}

collect_missing_checkpoints() {
    local tmp_file
    local adapter_path
    local output_json
    local manifest_index
    tmp_file="$(mktemp "${MISSING_LIST_FILE}.tmp.XXXXXX")"
    mapfile -t checkpoint_paths < "$RUN_LIST_FILE"
    for adapter_path in "${checkpoint_paths[@]}"; do
        output_json="$(output_json_for_checkpoint "$adapter_path")"
        if [ ! -s "$output_json" ]; then
            echo "$adapter_path" >> "$tmp_file"
            continue
        fi
        if [ "$RESUME_REQUIRE_SUCCESS_MARKER" = "true" ]; then
            manifest_index="$(manifest_index_for_checkpoint "$adapter_path" || true)"
            if [ -n "$manifest_index" ] && ! checkpoint_has_success_marker "$adapter_path"; then
                echo "$adapter_path" >> "$tmp_file"
            fi
        fi
    done
    sort -u "$tmp_file" > "$MISSING_LIST_FILE"
    rm -f "$tmp_file"
}

write_task_manifest() {
    local list_file="$1"
    local tmp_file
    local index
    local adapter_path
    tmp_file="$(mktemp "${TASK_MANIFEST_FILE}.tmp.XXXXXX")"
    : > "$tmp_file"
    index=0
    while IFS= read -r adapter_path; do
        [ -n "$adapter_path" ] || continue
        printf "%s\t%s\t%s\n" "$index" "$adapter_path" "$(output_json_for_checkpoint "$adapter_path")" >> "$tmp_file"
        index=$((index + 1))
    done < "$list_file"
    mv "$tmp_file" "$TASK_MANIFEST_FILE"
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
    array_range="0-${array_spec}"
    if [ -n "$ARRAY_MAX_CONCURRENT" ]; then
        if ! [[ "$ARRAY_MAX_CONCURRENT" =~ ^[0-9]+$ ]] || [ "$ARRAY_MAX_CONCURRENT" -le 0 ]; then
            echo "ARRAY_MAX_CONCURRENT must be a positive integer, got: $ARRAY_MAX_CONCURRENT" >&2
            exit 1
        fi
        array_range="${array_range}%${ARRAY_MAX_CONCURRENT}"
    fi
    if [ "$RESUME_MISSING_ONLY" = "true" ]; then
        total_count=$(wc -l < "$RUN_LIST_FILE" | tr -d ' ')
        echo "Resume mode: submitting missing checkpoints only ($task_count/$total_count)"
    fi
    write_task_manifest "$TASK_LIST_FILE"
    echo "Task manifest: $TASK_MANIFEST_FILE"
    echo "Task status dir: $STATUS_DIR"
    echo "Task logs dir: $TASK_LOG_DIR"
    echo "Submitting SLURM array $array_range"
    exec sbatch --array="$array_range" --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",RUN_LIST_FILE="$RUN_LIST_FILE",TASK_LIST_FILE="$TASK_LIST_FILE",MISSING_LIST_FILE="$MISSING_LIST_FILE",TASK_MANIFEST_FILE="$TASK_MANIFEST_FILE",STATUS_DIR="$STATUS_DIR",TASK_LOG_DIR="$TASK_LOG_DIR",RESUME_MISSING_ONLY="$RESUME_MISSING_ONLY",ARRAY_MAX_CONCURRENT="$ARRAY_MAX_CONCURRENT",RESUME_REQUIRE_SUCCESS_MARKER="$RESUME_REQUIRE_SUCCESS_MARKER",CHECKPOINT_STEP="$CHECKPOINT_STEP",CKPT_START="$CKPT_START",CKPT_END="$CKPT_END",CKPT_STEP="$CKPT_STEP",TRAINING_OUTPUTS_ROOT="$TRAINING_OUTPUTS_ROOT",TWO_STEP_INFERENCE="$TWO_STEP_INFERENCE",STEP2_KEEP_ADAPTER="$STEP2_KEEP_ADAPTER",STEP2_USES_IMAGE="$STEP2_USES_IMAGE",STEP2_TEMPERATURE="$STEP2_TEMPERATURE" "$0"
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
TASK_START_FILE="$STATUS_DIR/task_${INDEX}.start"
TASK_END_FILE="$STATUS_DIR/task_${INDEX}.end"
TASK_CHILD_OUT="$TASK_LOG_DIR/task_${INDEX}.child.out"
TASK_CHILD_ERR="$TASK_LOG_DIR/task_${INDEX}.child.err"
mkdir -p "$STATUS_DIR" "$TASK_LOG_DIR"
{
    echo "timestamp=$(date -Is)"
    echo "event=start"
    echo "job_id=${SLURM_JOB_ID:-N/A}"
    echo "array_job_id=${SLURM_ARRAY_JOB_ID:-N/A}"
    echo "task_id=${SLURM_ARRAY_TASK_ID:-N/A}"
    echo "host=$(hostname)"
    echo "task_list_file=$TASK_LIST_FILE"
} > "$TASK_START_FILE"
record_task_exit() {
    local exit_code=$?
    {
        echo "timestamp=$(date -Is)"
        echo "event=end"
        echo "job_id=${SLURM_JOB_ID:-N/A}"
        echo "array_job_id=${SLURM_ARRAY_JOB_ID:-N/A}"
        echo "task_id=${SLURM_ARRAY_TASK_ID:-N/A}"
        echo "exit_code=$exit_code"
    } > "$TASK_END_FILE"
}
trap record_task_exit EXIT
if [ "$INDEX" -ge "${#CHECKPOINT_PATHS[@]}" ]; then
    echo "No checkpoint for array index $INDEX (found ${#CHECKPOINT_PATHS[@]} checkpoints)"
    exit 0
fi

ADAPTER_PATH="${CHECKPOINT_PATHS[$INDEX]}"
RUN_NAME="$(basename "$(dirname "$ADAPTER_PATH")")"
CKPT_TAG="$(basename "$ADAPTER_PATH")"
OUTPUT_JSON="$(output_json_for_checkpoint "$ADAPTER_PATH")"

{
    echo "adapter_path=$ADAPTER_PATH"
    echo "output_json=$OUTPUT_JSON"
    echo "child_stdout=$TASK_CHILD_OUT"
    echo "child_stderr=$TASK_CHILD_ERR"
} >> "$TASK_START_FILE"

export ADAPTER_PATH
export OUTPUT_JSON

echo "=========================================="
echo "Vacation batch inference array task"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Checkpoint: $ADAPTER_PATH"
echo "Output JSON: $OUTPUT_JSON"
echo "Child stdout log: $TASK_CHILD_OUT"
echo "Child stderr log: $TASK_CHILD_ERR"
echo "=========================================="

set +e
bash "$JOB_SCRIPT" > "$TASK_CHILD_OUT" 2> "$TASK_CHILD_ERR"
child_exit_code=$?
set -e
echo "Child script exit code: $child_exit_code"
exit "$child_exit_code"
