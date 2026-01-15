#!/bin/bash
#SBATCH --job-name=vanilla_latest_array
#SBATCH --output=logs/vanilla_latest_array_%A_%a.out
#SBATCH --error=logs/vanilla_latest_array_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-tad-pool
#SBATCH --qos=owner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --array=0-20

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    set -euo pipefail
    trap 'echo "Error on line $LINENO: $BASH_COMMAND" >&2' ERR
fi

REPO_ROOT="${REPO_ROOT:-/galitylab/students/alonmardi/projects/LLaVA-NeXT}"
if [ ! -d "$REPO_ROOT" ]; then
    echo "Repo root not found: $REPO_ROOT" >&2
    exit 1
fi
cd "$REPO_ROOT"

LATEST_COUNT="${1:-${LATEST_COUNT:-10}}"
if ! [[ "$LATEST_COUNT" =~ ^[0-9]+$ ]] || [ "$LATEST_COUNT" -lt 1 ]; then
    echo "LATEST_COUNT must be a positive integer." >&2
    exit 1
fi

TRAINING_OUTPUTS_ROOT="${TRAINING_OUTPUTS_ROOT:-$REPO_ROOT/training_outputs}"
RUN_NAME_PATTERN="${RUN_NAME_PATTERN:-llava-*}"
CHECKPOINT_START="${CHECKPOINT_START:-3000}"
CHECKPOINT_END="${CHECKPOINT_END:-4000}"
DEBUG="${DEBUG:-false}"

MODEL_PATH="${MODEL_PATH:-lmms-lab/llava-onevision-qwen2-7b-ov-chat}"
MODEL_BASE="${MODEL_BASE:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
QUANTIZATION="${QUANTIZATION:-}"
IMAGE_PATH="${IMAGE_PATH:-$REPO_ROOT/baseline_images/39740.png}"
CONV_TEMPLATE="${CONV_TEMPLATE:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-1.1}"
TOP_P="${TOP_P:-0.9}"
NUM_BEAMS="${NUM_BEAMS:-1}"
DO_SAMPLE="${DO_SAMPLE:-false}"
DISABLE_OPTIMIZATIONS="${DISABLE_OPTIMIZATIONS:-false}"
CHAT_MODE="${CHAT_MODE:-false}"
NO_STREAM="${NO_STREAM:-false}"
DETECT_GAZE_BBOXES="${DETECT_GAZE_BBOXES:-false}"
IMAGE_ASPECT_RATIO_FALLBACK="${IMAGE_ASPECT_RATIO_FALLBACK:-anyres_max_4}"
IMAGE_GRID_PINPOINTS_FALLBACK="${IMAGE_GRID_PINPOINTS_FALLBACK:-(1x1),...,(2x2)}"

if [ -n "${ARRAY_JOB_ID:-}" ]; then
    RUN_ID="vanilla_latest_${LATEST_COUNT}_${ARRAY_JOB_ID}"
elif [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    RUN_ID="vanilla_latest_${LATEST_COUNT}_${SLURM_ARRAY_JOB_ID}"
elif [ -z "${RUN_ID:-}" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        RUN_ID="vanilla_latest_${LATEST_COUNT}_${SLURM_JOB_ID}"
    else
        RUN_ID="vanilla_latest_${LATEST_COUNT}_$(date +%Y%m%d_%H%M%S)"
    fi
fi

RESULTS_DIR="${RESULTS_DIR:-evaluation_results/${RUN_ID}}"
AGGREGATED_JSON="${AGGREGATED_JSON:-evaluation_results/${RUN_ID}_combined.json}"
AUTO_AGGREGATE="${AUTO_AGGREGATE:-true}"

echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "CWD: $(pwd)"
echo "SLURM job id: ${SLURM_JOB_ID:-N/A}"
echo "SLURM array job id: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "SLURM array task id: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "RUN_ID: ${RUN_ID}"
echo "LATEST_COUNT: ${LATEST_COUNT}"
echo "CHECKPOINT_START: ${CHECKPOINT_START:-}"
echo "CHECKPOINT_END: ${CHECKPOINT_END:-}"

# if [ -z "${PROMPT:-}" ]; then
#     read -r -d '' PROMPT <<'EOF'
# You are an expert vision assistant.
# Step 1 - Caption
# - Provide one concise sentence that broadly describes the entire scene.
# - Begin the line with: Caption:
# Step 2 - Foreground people & gaze
# 1. Detect every person in the image.
# 2. List them from left to right and number sequentially starting at 1.
# For each person output exactly one line in this format:
# Person {N}: {short description}, looking at {target | outside the frame | uncertain}
# 3. If there are more then 1 person, describe their social interactions.
# Output format (no extra lines, no prose other than what is specified):
# -------------------------------------------------
# Caption: {your one-sentence scene description}
# Person 1: {short description}, looking at ...
# Person 2: {short description}, looking at ...
# ...
# Social interactions: {describe any social interactions briefly}
# -------------------------------------------------
# Additional rules
# - Keep the phrase "looking at" unchanged.
# - {short description} must be 6 words or fewer (e.g., "man in red jacket").
# - If no foreground person is detected, write exactly: No foreground people detected.
# - If gaze cannot be determined, use "uncertain".
# - Do not output your reasoning or any extra text.
# EOF
# fi

# For each person in the image, describe who they are and what they are looking at.
# Then classify the social interaction using one of the following categories:
# Non-communicative gaze: no clear gaze-based interaction between people.
# Follow: one person is looking at another person, while the other person is looking elsewhere (not back at them).
# Mutual gaze: two people are looking directly at each other.
# Joint attention: two or more people are looking at the same external object (but not at each other).
if [ -z "${PROMPT:-}" ]; then
    PROMPT="$(cat <<'EOF'
For each person in the image:
Briefly describe who they are (role/appearance) and what they are looking at (another person, an object, or off e.g., “off-screen left”).
Then choose exactly one social interaction label using the rules below in this priority order:
Priority order (apply top to bottom):
Mutual gaze: at least two people are looking at each other (A→B and B→A).
If this is true, the label must be Mutual gaze, even if you think they share attention.
Single: one person looks at another (A→B) while the other looks elsewhere (B→≠A).
Joint attention: two or more people are looking at the same external target (same object/location), and that target is not any person.
Non-communicative gaze: none of the above apply; no clear gaze-based interaction.
Important constraint:
Joint attention never applies when the shared target is a person. If people are looking at each other, that is Mutual gaze, not Joint attention.
EOF
)"
fi

PYTHON_BIN="${PYTHON_BIN:-/galitylab/students/alonmardi/llava/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="/home/alonz/llava/bin/python"
fi
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python environment not found; set PYTHON_BIN." >&2
    exit 1
fi

if [ -n "$CHECKPOINT_START" ] && ! [[ "$CHECKPOINT_START" =~ ^[0-9]+$ ]]; then
    echo "CHECKPOINT_START must be an integer (step number)." >&2
    exit 1
fi
if [ -n "$CHECKPOINT_END" ] && ! [[ "$CHECKPOINT_END" =~ ^[0-9]+$ ]]; then
    echo "CHECKPOINT_END must be an integer (step number)." >&2
    exit 1
fi
if [ -n "$CHECKPOINT_START" ] && [ -n "$CHECKPOINT_END" ] && [ "$CHECKPOINT_END" -lt "$CHECKPOINT_START" ]; then
    echo "CHECKPOINT_END must be >= CHECKPOINT_START." >&2
    exit 1
fi

collect_checkpoints() {
    local run
    local -a run_dirs=()
    CHECKPOINT_PATHS=()
    mapfile -t run_dirs < <(
        find "$TRAINING_OUTPUTS_ROOT" -maxdepth 1 -mindepth 1 -type d -name "$RUN_NAME_PATTERN" -printf '%T@ %f\n' |
            sort -nr |
            awk -v limit="$LATEST_COUNT" 'NR<=limit {print $2}'
    )
    if [ "$DEBUG" = "true" ]; then
        echo "Selected runs (${#run_dirs[@]}): ${run_dirs[*]}"
    fi

    for run in "${run_dirs[@]}"; do
        local -a run_ckpts=()
        mapfile -t run_ckpts < <(
            find "$TRAINING_OUTPUTS_ROOT/$run" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' -printf '%p\n' |
                sort -V
        )
        if [ -z "$CHECKPOINT_START" ] && [ -z "$CHECKPOINT_END" ]; then
            CHECKPOINT_PATHS+=("${run_ckpts[@]}")
            if [ "$DEBUG" = "true" ]; then
                echo "Run ${run}: ${#run_ckpts[@]} checkpoints (no filtering)"
            fi
            continue
        fi

        local -a filtered_ckpts=()
        for ckpt in "${run_ckpts[@]}"; do
            local step="${ckpt##*/}"
            step="${step#checkpoint-}"
            if ! [[ "$step" =~ ^[0-9]+$ ]]; then
                continue
            fi
            if [ -n "$CHECKPOINT_START" ] && [ "$step" -lt "$CHECKPOINT_START" ]; then
                continue
            fi
            if [ -n "$CHECKPOINT_END" ] && [ "$step" -gt "$CHECKPOINT_END" ]; then
                continue
            fi
            filtered_ckpts+=("$ckpt")
        done
        CHECKPOINT_PATHS+=("${filtered_ckpts[@]}")
        if [ "$DEBUG" = "true" ]; then
            echo "Run ${run}: ${#filtered_ckpts[@]} checkpoints after filtering"
        fi
    done
}

if [ "${AGGREGATE_ONLY:-false}" = "true" ]; then
    if [ ! -d "$RESULTS_DIR" ]; then
        echo "Results dir not found: $RESULTS_DIR" >&2
        exit 1
    fi

    mkdir -p "$(dirname "$AGGREGATED_JSON")"

    "$PYTHON_BIN" - "$RESULTS_DIR" "$AGGREGATED_JSON" "$TRAINING_OUTPUTS_ROOT" <<'PY'
import json
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
output_path = Path(sys.argv[2])
training_outputs_root = Path(sys.argv[3])

results = []
output_path = output_path.resolve()
json_files = sorted(
    path
    for path in results_dir.rglob("*.json")
    if path.is_file() and path.resolve() != output_path
)

for output_json in json_files:
    data = json.loads(output_json.read_text(encoding="utf-8"))
    run_name = output_json.parent.name
    ckpt_name = output_json.stem
    data["run_name"] = run_name
    if ckpt_name:
        data["checkpoint_name"] = ckpt_name
        data["checkpoint"] = str(training_outputs_root / run_name / ckpt_name)
    data["output_json"] = str(output_json)
    results.append(data)

output_path.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")
print(f"Wrote {len(results)} results to {output_path}.")
PY
    exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    collect_checkpoints

    if [ "${#CHECKPOINT_PATHS[@]}" -eq 0 ]; then
        echo "No checkpoints found under $TRAINING_OUTPUTS_ROOT for latest $LATEST_COUNT runs." >&2
        exit 1
    fi

    mkdir -p logs

    array_spec="0-$(( ${#CHECKPOINT_PATHS[@]} - 1 ))"
    if [ -n "${ARRAY_SPEC:-}" ]; then
        array_spec="$ARRAY_SPEC"
    fi
    echo "Submitting SLURM array with range ${array_spec} for ${#CHECKPOINT_PATHS[@]} checkpoints."
    if [ "$DEBUG" = "true" ]; then
        printf '%s\n' "${CHECKPOINT_PATHS[@]}"
    fi

    unset RESULTS_DIR AGGREGATED_JSON
    array_job_id=$(sbatch --parsable --array="${array_spec}" \
        --export=ALL,LATEST_COUNT="$LATEST_COUNT" \
        "$0" "$LATEST_COUNT")
    array_run_id="vanilla_latest_${LATEST_COUNT}_${array_job_id}"
    array_results_dir="evaluation_results/${array_run_id}"
    array_aggregated_json="evaluation_results/${array_run_id}_combined.json"

    if [ "$AUTO_AGGREGATE" = "true" ]; then
        sbatch --array=0-0 --dependency=afterany:${array_job_id} \
            --export=ALL,ARRAY_JOB_ID="$array_job_id",AGGREGATE_ONLY=true,LATEST_COUNT="$LATEST_COUNT" \
            "$0" "$LATEST_COUNT"
    fi

    echo "Array job id: $array_job_id"
    echo "Run id: $array_run_id"
    echo "Results dir: $array_results_dir"
    echo "Aggregation output: $array_aggregated_json"
    exit 0
fi

if [ "$AUTO_AGGREGATE" = "true" ] && [ "${AGGREGATE_ONLY:-false}" != "true" ] && [ "${SLURM_ARRAY_TASK_ID:-}" = "0" ] && [ -z "${AGGREGATION_SUBMITTED:-}" ]; then
    sbatch --array=0-0 --dependency=afterany:${SLURM_ARRAY_JOB_ID} \
        --export=ALL,ARRAY_JOB_ID="$SLURM_ARRAY_JOB_ID",AGGREGATE_ONLY=true,LATEST_COUNT="$LATEST_COUNT",AGGREGATION_SUBMITTED=true \
        "$0" "$LATEST_COUNT"
fi

collect_checkpoints
if [ "${#CHECKPOINT_PATHS[@]}" -eq 0 ]; then
    echo "No checkpoints found under $TRAINING_OUTPUTS_ROOT for latest $LATEST_COUNT runs." >&2
    exit 1
fi

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$INDEX" -ge "${#CHECKPOINT_PATHS[@]}" ]; then
    echo "Array index $INDEX has no matching checkpoint (found ${#CHECKPOINT_PATHS[@]} checkpoints); exiting."
    exit 0
fi

ADAPTER_PATH="${CHECKPOINT_PATHS[$INDEX]}"
RUN_NAME="$(basename "$(dirname "$ADAPTER_PATH")")"
CKPT_NAME="$(basename "$ADAPTER_PATH")"
TASK_RESULTS_DIR="$RESULTS_DIR/${RUN_NAME}"
OUTPUT_JSON="$TASK_RESULTS_DIR/${CKPT_NAME}.json"

echo "=========================================="
echo "Vanilla inference latest-array task"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Checkpoint: ${ADAPTER_PATH}"
echo "Results dir: ${RESULTS_DIR}"
echo "Output json: ${OUTPUT_JSON}"
echo "=========================================="

mkdir -p logs "$TASK_RESULTS_DIR"

echo "Activating virtual environment..."
source "$(dirname "$PYTHON_BIN")/activate"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

which python
python --version
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

cd "$REPO_ROOT"

ARGS=(
    "--model-path" "$MODEL_PATH"
    "--attn-implementation" "$ATTN_IMPLEMENTATION"
    "--image-path" "$IMAGE_PATH"
    "--image-aspect-ratio" "$IMAGE_ASPECT_RATIO_FALLBACK"
    "--image-grid-pinpoints" "$IMAGE_GRID_PINPOINTS_FALLBACK"
    "--prompt" "$PROMPT"
    "--max-new-tokens" "$MAX_NEW_TOKENS"
    "--temperature" "$TEMPERATURE"
    "--top-p" "$TOP_P"
    "--num-beams" "$NUM_BEAMS"
    "--adapter-path" "$ADAPTER_PATH"
    "--save-output" "$OUTPUT_JSON"
)

if [ -n "$MODEL_BASE" ]; then
    ARGS+=("--model-base" "$MODEL_BASE")
fi
if [ -n "$CONV_TEMPLATE" ]; then
    ARGS+=("--conv-template" "$CONV_TEMPLATE")
fi
if [ "$QUANTIZATION" = "4bit" ]; then
    ARGS+=("--load-4bit")
elif [ "$QUANTIZATION" = "8bit" ]; then
    ARGS+=("--load-8bit")
fi
if [ "$DO_SAMPLE" = "true" ]; then
    ARGS+=("--do-sample")
fi
if [ "$DISABLE_OPTIMIZATIONS" = "true" ]; then
    ARGS+=("--disable-optimizations")
fi
if [ "$CHAT_MODE" = "true" ]; then
    ARGS+=("--chat")
fi
if [ "$NO_STREAM" = "true" ]; then
    ARGS+=("--no-stream")
fi
if [ "$DETECT_GAZE_BBOXES" = "true" ]; then
    ARGS+=("--run-gdino")
fi

echo "Executing:"
printf '%s ' "$PYTHON_BIN" "generate_vanilla_inference.py"
printf '%q ' "${ARGS[@]}"
echo

"$PYTHON_BIN" generate_vanilla_inference.py "${ARGS[@]}"
status=$?

if [ $status -eq 0 ]; then
    "$PYTHON_BIN" - "$OUTPUT_JSON" > "$TASK_RESULTS_DIR/${CKPT_NAME}.txt" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
prompt = data.get("prompt", "")
response = data.get("response", "")
print(f"prompt: {prompt}")
print("response:")
print(response)
PY
else
    echo "Checkpoint ${CKPT_NAME} failed with status $status" > "$TASK_RESULTS_DIR/${CKPT_NAME}.txt"
fi

echo "Finished with exit code $status"
exit $status
