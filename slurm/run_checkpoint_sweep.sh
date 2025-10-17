#!/bin/bash
#
# Checkpoint Sweep Script - Submits parallel SLURM jobs for each checkpoint
#
# Usage: ./run_checkpoint_sweep.sh [OPTIONS]
#
# This script will:
# 1. Discover checkpoints inside a training run directory
# 2. Submit a separate SLURM job for each checkpoint (deterministic decoding)
# 3. Collect results into a single JSON file when all jobs complete
#

set -e

PROJECT_ROOT="/galitylab/students/alonmardi/projects/LLaVA-NeXT"

# Default configuration
DEFAULT_CHECKPOINT_PARENT="$PROJECT_ROOT/training_outputs"
DEFAULT_IMAGE="$PROJECT_ROOT/baseline_images/39740.png"
DEFAULT_MODEL="lmms-lab/llava-onevision-qwen2-7b-ov-chat"
DEFAULT_MAX_TOKENS=512
DEFAULT_TOP_P=0.9
DEFAULT_NUM_BEAMS=1
DEFAULT_DO_SAMPLE="false"
DEFAULT_TEMPERATURE=0.9
DEFAULT_TIME="00:10:00"
DEFAULT_MEM="20G"
DEFAULT_GDINO_MODEL_ID="IDEA-Research/grounding-dino-base"
DEFAULT_GDINO_BOX_THRESHOLD=0.25
DEFAULT_GDINO_TEXT_THRESHOLD=0.2
DEFAULT_GDINO_DEVICE="auto"
DEFAULT_RUN_GDINO="true"
IMAGE_ASPECT_RATIO_FALLBACK="${IMAGE_ASPECT_RATIO_FALLBACK:-anyres_max_4}"
IMAGE_GRID_PINPOINTS_FALLBACK="${IMAGE_GRID_PINPOINTS_FALLBACK:-(1x1),...,(2x2)}"  # "(1x1),...,(6x6)"

# Initialize variables
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
CHECKPOINT_PARENT="${CHECKPOINT_PARENT:-$DEFAULT_CHECKPOINT_PARENT}"
IMAGE_PATH="${IMAGE_PATH:-$DEFAULT_IMAGE}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-$DEFAULT_MAX_TOKENS}"
TOP_P="${TOP_P:-$DEFAULT_TOP_P}"
NUM_BEAMS="${NUM_BEAMS:-$DEFAULT_NUM_BEAMS}"
DO_SAMPLE="${DO_SAMPLE:-$DEFAULT_DO_SAMPLE}"
TEMPERATURE="${TEMPERATURE:-$DEFAULT_TEMPERATURE}"
JOB_TIME="${JOB_TIME:-$DEFAULT_TIME}"
JOB_MEM="${JOB_MEM:-$DEFAULT_MEM}"
MODEL_BASE="${MODEL_BASE:-}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
QUANTIZATION="${QUANTIZATION:-}"
CONV_TEMPLATE="${CONV_TEMPLATE:-}"
RUN_GDINO="${RUN_GDINO:-$DEFAULT_RUN_GDINO}"
GDINO_MODEL_ID="${GDINO_MODEL_ID:-$DEFAULT_GDINO_MODEL_ID}"
GDINO_BOX_THRESHOLD="${GDINO_BOX_THRESHOLD:-$DEFAULT_GDINO_BOX_THRESHOLD}"
GDINO_TEXT_THRESHOLD="${GDINO_TEXT_THRESHOLD:-$DEFAULT_GDINO_TEXT_THRESHOLD}"
GDINO_DEVICE="${GDINO_DEVICE:-$DEFAULT_GDINO_DEVICE}"
LIMIT_CHECKPOINTS="${LIMIT_CHECKPOINTS:-0}"

# Default prompt
if [ -z "$PROMPT" ]; then
    read -r -d '' PROMPT <<'EOPROMPT' || true
You are an expert vision assistant.
Step 1 - Caption
• Provide one concise sentence that broadly describes the entire scene.
• Begin the line with: Caption:
Step 2 - Foreground people & gaze
1. Detect every person whose height is at least 5% of the image (foreground).
2. List them from left to right and number sequentially starting at 1.
For each person output exactly one line in this format:
Person {N}: {short description}, looking at {target | outside the frame | uncertain}
3. If there are more then 1 person, describe their social interactions.
Output format (no extra lines, no prose other than what is specified):
-------------------------------------------------
Caption: {your one-sentence scene description}
Person 1: {short description}, looking at ...
Person 2: {short description}, looking at ...
...
Social interactions: {describe any social interactions briefly}
-------------------------------------------------
Additional rules
• Keep the phrase "looking at" unchanged.
• {short description} must be 6 words or fewer (e.g., "man in red jacket").
• If no foreground person is detected, write exactly: No foreground people detected.
• If gaze cannot be determined, use "uncertain".
• Do not output your reasoning or any extra text.
EOPROMPT
fi

# Print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Submit parallel checkpoint sweep jobs to SLURM.

Checkpoint Options:
    -c, --checkpoint-dir PATH   Directory containing checkpoint-* folders (default: interactive)
    --checkpoint-parent PATH    Parent directory to list training runs (default: $DEFAULT_CHECKPOINT_PARENT)
    --limit INT                 Limit to the latest INT checkpoints (0 = all)

Model Options:
    -m, --model PATH            Model path (default: $DEFAULT_MODEL)
    --model-base PATH           Base model path for LoRA
    --attn-impl NAME            Attention implementation (default: sdpa)
    --quantize [4bit|8bit]      Enable quantization

Inference Options:
    -i, --image PATH            Image path (default: $DEFAULT_IMAGE)
    -p, --prompt TEXT           Custom prompt
    --max-tokens INT            Max new tokens (default: $DEFAULT_MAX_TOKENS)
    --top-p FLOAT               Top-p value (default: $DEFAULT_TOP_P)
    --num-beams INT             Number of beams (default: $DEFAULT_NUM_BEAMS)
    --temperature FLOAT         Sampling temperature (default: $DEFAULT_TEMPERATURE)
    --do-sample BOOL            Enable sampling (default: $DEFAULT_DO_SAMPLE)

SLURM Options:
    --time TIME                 Job time limit (default: $DEFAULT_TIME)
    --mem SIZE                  Memory per job (default: $DEFAULT_MEM)

Other:
    -h, --help                  Show this help message
    --interactive               Interactively select checkpoint directory (default when unspecified)

Examples:
    # Interactive selection of latest run under training_outputs
    $0

    # Specify checkpoint directory explicitly
    $0 --checkpoint-dir training_outputs/llava-20251007_011312

    # Limit to the newest 5 checkpoints
    $0 --limit 5
EOF
    exit 0
}

make_absolute() {
    local path_input="$1"
    if [ -z "$path_input" ]; then
        echo ""
        return
    fi
    if [[ "$path_input" = /* ]]; then
        echo "$path_input"
    else
        echo "$PROJECT_ROOT/$path_input"
    fi
}

# Interactive checkpoint directory selection
select_checkpoint_dir() {
    local parent_dir="$1"
    echo "=========================================="
    echo "Available recent training runs:"
    echo "  (showing up to 10 under $parent_dir)"

    mapfile -t run_dirs < <(ls -dt "$parent_dir"/* 2>/dev/null | head -10)

    if [ ${#run_dirs[@]} -eq 0 ]; then
        echo "No training runs found under: $parent_dir"
        return 1
    fi

    for i in "${!run_dirs[@]}"; do
        echo "  [$i] ${run_dirs[$i]}"
    done

    local default_selection=0
    echo "=========================================="
    echo "Default: [${default_selection}] ${run_dirs[$default_selection]}"
    read -p "Enter run number, full path, or press Enter for default: " user_input

    if [ -z "$user_input" ]; then
        CHECKPOINT_DIR="${run_dirs[$default_selection]}"
        echo "Selected: $CHECKPOINT_DIR"
        return 0
    fi

    if [[ "$user_input" =~ ^[0-9]+$ ]] && [ "$user_input" -lt "${#run_dirs[@]}" ]; then
        CHECKPOINT_DIR="${run_dirs[$user_input]}"
        echo "Selected: $CHECKPOINT_DIR"
        return 0
    fi

    CHECKPOINT_DIR="$user_input"
    echo "Selected: $CHECKPOINT_DIR"
    return 0
}

# Parse arguments
INTERACTIVE=true
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --checkpoint-parent) CHECKPOINT_PARENT="$2"; shift 2 ;;
        --limit) LIMIT_CHECKPOINTS="$2"; shift 2 ;;
        -m|--model) MODEL_PATH="$2"; shift 2 ;;
        --model-base) MODEL_BASE="$2"; shift 2 ;;
        --attn-impl) ATTN_IMPLEMENTATION="$2"; shift 2 ;;
        --quantize) QUANTIZATION="$2"; shift 2 ;;
        -i|--image) IMAGE_PATH="$2"; shift 2 ;;
        -p|--prompt) PROMPT="$2"; shift 2 ;;
        --max-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --num-beams) NUM_BEAMS="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --do-sample) DO_SAMPLE="$2"; shift 2 ;;
        --time) JOB_TIME="$2"; shift 2 ;;
        --mem) JOB_MEM="$2"; shift 2 ;;
        --interactive) INTERACTIVE=true; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

CHECKPOINT_PARENT=$(make_absolute "$CHECKPOINT_PARENT")
CHECKPOINT_DIR=$(make_absolute "$CHECKPOINT_DIR")

# Interactive checkpoint selection if requested or unspecified
if [ -z "$CHECKPOINT_DIR" ] && [ "$INTERACTIVE" = true ]; then
    if ! select_checkpoint_dir "$CHECKPOINT_PARENT"; then
        echo "Failed to select checkpoint directory."
        exit 1
    fi
    CHECKPOINT_DIR=$(make_absolute "$CHECKPOINT_DIR")
fi

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Error: --checkpoint-dir is required (or use --interactive)."
    exit 1
fi

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: checkpoint directory does not exist: $CHECKPOINT_DIR"
    exit 1
fi

# Build list of checkpoints
checkpoint_list=$(python - "$CHECKPOINT_DIR" "$LIMIT_CHECKPOINTS" <<'PY'
import json
import os
import sys

root = sys.argv[1]
limit = int(sys.argv[2])

if not os.path.isdir(root):
    print("[]")
    sys.exit(0)

entries = []
for name in os.listdir(root):
    if not name.startswith("checkpoint-"):
        continue
    full_path = os.path.join(root, name)
    if not os.path.isdir(full_path):
        continue
    suffix = name.split("-", 1)[-1]
    step = None
    if suffix.isdigit():
        step = int(suffix)
    else:
        digits = "".join(ch for ch in suffix if ch.isdigit())
        if digits:
            try:
                step = int(digits)
            except ValueError:
                step = None
    entries.append({"step": step, "name": name, "path": full_path})

entries.sort(key=lambda item: (item["step"] is None, item["step"] if item["step"] is not None else 10**15, item["name"]))

if limit > 0:
    entries = entries[-limit:]

print(json.dumps(entries))
PY
)
checkpoint_count=$(python - "$checkpoint_list" <<'PY'
import json, sys
entries = json.loads(sys.argv[1])
print(len(entries))
PY
)

if [ "$checkpoint_count" -eq 0 ]; then
    echo "No checkpoints found under: $CHECKPOINT_DIR"
    exit 1
fi

mapfile -t checkpoint_steps < <(python - "$checkpoint_list" <<'PY'
import json
import sys

entries = json.loads(sys.argv[1])
for item in entries:
    step = item.get("step")
    print(step if isinstance(step, int) else "")
PY
)

mapfile -t checkpoint_names < <(python - "$checkpoint_list" <<'PY'
import json
import sys

entries = json.loads(sys.argv[1])
for item in entries:
    print(item["name"])
PY
)

mapfile -t checkpoint_paths < <(python - "$checkpoint_list" <<'PY'
import json
import sys

entries = json.loads(sys.argv[1])
for item in entries:
    print(item["path"])
PY
)

total_checkpoints=${#checkpoint_paths[@]}

run_name=$(basename "$CHECKPOINT_DIR")
dir_suffix="${run_name}_checkpoint_sweep"

OUTPUT_DIR="evaluation_results/eval_${dir_suffix}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_ROOT/logs"

echo "=========================================="
echo "Checkpoint Sweep Configuration"
echo "=========================================="
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Total checkpoints: $total_checkpoints"
if [ "$LIMIT_CHECKPOINTS" -gt 0 ]; then
    echo "Limit applied: newest $LIMIT_CHECKPOINTS checkpoints"
fi
echo ""
echo "Model: $MODEL_PATH"
echo "Image: $IMAGE_PATH"
echo "Max tokens: $MAX_NEW_TOKENS"
echo "Top-p: $TOP_P"
echo "Do sample: $DO_SAMPLE"
echo "Temperature: $TEMPERATURE"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Job time limit: $JOB_TIME"
echo "Memory per job: $JOB_MEM"
echo "=========================================="

if [ "$RUN_GDINO" = "true" ]; then
    RUN_GDINO_BOOL=true
else
    RUN_GDINO_BOOL=false
fi

cat > "$OUTPUT_DIR/metadata.json" <<EOF
{
  "model_path": "$MODEL_PATH",
  "adapter_path": "",
  "checkpoint_dir": "$CHECKPOINT_DIR",
  "image_path": "$IMAGE_PATH",
  "prompt": $(echo "$PROMPT" | python -c "import sys, json; print(json.dumps(sys.stdin.read()))"),
  "max_new_tokens": $MAX_NEW_TOKENS,
  "top_p": $TOP_P,
  "num_beams": $NUM_BEAMS,
  "do_sample": $DO_SAMPLE,
  "temperature": $TEMPERATURE,
  "timestamp": "$(date -Iseconds)",
  "total_jobs": $total_checkpoints,
  "gdino": {
    "enabled": $RUN_GDINO_BOOL,
    "model_id": "$GDINO_MODEL_ID",
    "box_threshold": $GDINO_BOX_THRESHOLD,
    "text_threshold": $GDINO_TEXT_THRESHOLD,
    "device": "$GDINO_DEVICE"
  }
}
EOF

read -p "Submit $total_checkpoints parallel jobs? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ -n $REPLY ]]; then
    echo "Submission cancelled."
    exit 0
fi

job_ids=()
echo ""
echo "Submitting jobs..."

for idx in "${!checkpoint_paths[@]}"; do
    checkpoint_path="${checkpoint_paths[$idx]}"
    checkpoint_name="${checkpoint_names[$idx]}"
    output_file="$OUTPUT_DIR/output_checkpoint_${checkpoint_name}.json"
    job_script="$OUTPUT_DIR/job_checkpoint_${checkpoint_name}.sh"

    sanitized_name=$(echo "ckpt_${checkpoint_name}" | tr -c 'A-Za-z0-9_' '_')
    sanitized_name=${sanitized_name:0:35}

    cat > "$job_script" <<EOJOB
#!/bin/bash
#SBATCH --job-name=${sanitized_name}
#SBATCH --output=$PROJECT_ROOT/logs/${sanitized_name}_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/${sanitized_name}_%j.err
#SBATCH --time=$JOB_TIME
#SBATCH --partition=gpu-tad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=$JOB_MEM
#SBATCH --nodes=1

OUTPUT_JSON="$output_file"

echo "Checkpoint: $checkpoint_path"
echo "Output: \$OUTPUT_JSON"

source /galitylab/students/alonmardi/llava/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PROJECT_ROOT:\$PYTHONPATH"

cd "$PROJECT_ROOT"

read -r -d '' PROMPT <<'PROMPTEOF'
$PROMPT
PROMPTEOF

ARGS=(
    "--model-path" "$MODEL_PATH"
    "--attn-implementation" "$ATTN_IMPLEMENTATION"
    "--image-path" "$IMAGE_PATH"
    "--prompt" "\$PROMPT"
    "--max-new-tokens" "$MAX_NEW_TOKENS"
    "--temperature" "$TEMPERATURE"
    "--top-p" "$TOP_P"
    "--num-beams" "$NUM_BEAMS"
    "--save-output" "\$OUTPUT_JSON"
    "--image-aspect-ratio" "$IMAGE_ASPECT_RATIO_FALLBACK"
    "--image-grid-pinpoints" "$IMAGE_GRID_PINPOINTS_FALLBACK"
)

RUN_GDINO="$RUN_GDINO"
GDINO_MODEL_ID="$GDINO_MODEL_ID"
GDINO_BOX_THRESHOLD="$GDINO_BOX_THRESHOLD"
GDINO_TEXT_THRESHOLD="$GDINO_TEXT_THRESHOLD"
GDINO_DEVICE="$GDINO_DEVICE"

if [ "\$RUN_GDINO" = "true" ]; then
    ARGS+=("--run-gdino")
    ARGS+=("--gdino-model-id" "\$GDINO_MODEL_ID")
    ARGS+=("--gdino-box-threshold" "\$GDINO_BOX_THRESHOLD")
    ARGS+=("--gdino-text-threshold" "\$GDINO_TEXT_THRESHOLD")
    ARGS+=("--gdino-device" "\$GDINO_DEVICE")
fi

EOJOB

    if [ -n "$MODEL_BASE" ]; then
        cat >> "$job_script" <<EOJOB
ARGS+=("--model-base" "$MODEL_BASE")
EOJOB
    fi

    cat >> "$job_script" <<EOJOB
ARGS+=("--adapter-path" "$checkpoint_path")
EOJOB

    if [ -n "$CONV_TEMPLATE" ]; then
        cat >> "$job_script" <<EOJOB
ARGS+=("--conv-template" "$CONV_TEMPLATE")
EOJOB
    fi

    if [ "$QUANTIZATION" = "4bit" ]; then
        cat >> "$job_script" <<EOJOB
ARGS+=("--load-4bit")
EOJOB
    elif [ "$QUANTIZATION" = "8bit" ]; then
        cat >> "$job_script" <<EOJOB
ARGS+=("--load-8bit")
EOJOB
    fi

    if [ "$DO_SAMPLE" = "true" ] || [ "$DO_SAMPLE" = "True" ]; then
        cat >> "$job_script" <<EOJOB
ARGS+=("--do-sample")
EOJOB
    fi

    cat >> "$job_script" <<'EOJOB'

python generate_vanilla_inference.py "${ARGS[@]}"

exit_code=$?
if [ $exit_code -eq 0 ] && [ -f "$OUTPUT_JSON" ]; then
    echo "✓ Success"
else
    echo "✗ Failed (exit code: $exit_code)"
fi
exit $exit_code
EOJOB

    chmod +x "$job_script"

    job_id=$(sbatch --parsable "$job_script")
    job_ids+=($job_id)
    echo "  Checkpoint $checkpoint_name: Job $job_id"
done

echo ""
echo "=========================================="
echo "Submitted $total_checkpoints jobs"
echo "Job IDs: ${job_ids[*]}"
echo "=========================================="

collection_script="$OUTPUT_DIR/collect_results.sh"

cat > "$collection_script" <<'EOCOLLECT'
#!/usr/bin/env bash
set -e

RESULTS_JSON="$OUTPUT_DIR/checkpoint_sweep_results.json"

python <<'EOPY'
import glob
import json
import os
import sys

output_dir = os.environ["OUTPUT_DIR"]
results_json = os.environ["RESULTS_JSON"]
metadata_path = os.path.join(output_dir, "metadata.json")

if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

with open(metadata_path, "r") as f:
    metadata = json.load(f)

checkpoint_dir = metadata.get("checkpoint_dir")

results = []
output_files = sorted(glob.glob(os.path.join(output_dir, "output_checkpoint_*.json")))

for output_file in output_files:
    ckpt_name = os.path.basename(output_file).replace("output_checkpoint_", "").replace(".json", "")
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name) if checkpoint_dir else ckpt_name

    step = None
    if ckpt_name.startswith("checkpoint-"):
        suffix = ckpt_name.split("-", 1)[-1]
        if suffix.isdigit():
            step = int(suffix)
        else:
            digits = "".join(ch for ch in suffix if ch.isdigit())
            if digits:
                try:
                    step = int(digits)
                except ValueError:
                    step = None

    try:
        with open(output_file, "r") as f:
            data = json.load(f)

        results.append({
            "checkpoint_name": ckpt_name,
            "checkpoint_path": ckpt_path,
            "step": step,
            "output_file": output_file,
            "prompt": data.get("prompt", ""),
            "response": data.get("response", ""),
            "gdino": data.get("gdino"),
            "status": "success"
        })
    except Exception as exc:
        results.append({
            "checkpoint_name": ckpt_name,
            "checkpoint_path": ckpt_path,
            "step": step,
            "output_file": output_file,
            "error": str(exc),
            "gdino": None,
            "status": "error"
        })

results.sort(key=lambda item: (item["step"] if item["step"] is not None else float("inf"), item["checkpoint_name"]))

output_data = {
    "metadata": metadata,
    "results": results
}

with open(results_json, "w") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

deleted_json = 0
for output_file in output_files:
    try:
        os.remove(output_file)
        deleted_json += 1
    except OSError as exc:
        print(f"Warning: failed to remove {output_file}: {exc}", file=sys.stderr)
if deleted_json:
    print(f"Removed {deleted_json} intermediate result file(s).")

script_files = sorted(glob.glob(os.path.join(output_dir, "job_checkpoint_*.sh")))
deleted_scripts = 0
for script_file in script_files:
    try:
        os.remove(script_file)
        deleted_scripts += 1
    except OSError as exc:
        print(f"Warning: failed to remove {script_file}: {exc}", file=sys.stderr)
if deleted_scripts:
    print(f"Removed {deleted_scripts} job script(s).")

print(f"Results collected: {results_json}")
successful = len([r for r in results if r["status"] == "success"])
total = metadata.get("total_jobs", len(results))
print(f"Successful: {successful}/{total}")
EOPY

EOCOLLECT

chmod +x "$collection_script"

echo ""
echo "✅ All jobs submitted successfully!"
echo ""
echo "📊 Monitor progress:"
echo "  squeue -u \$USER | grep ckpt_"
echo "  watch -n 5 'squeue -u \$USER | grep ckpt_'"
echo ""
echo "📁 Output directory: $OUTPUT_DIR"
echo "📁 Job IDs: ${job_ids[*]}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏳ Waiting for all jobs to complete..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

while squeue -u $USER | grep -q "ckpt_"; do
    running=$(squeue -u $USER -t RUNNING | grep "ckpt_" | wc -l)
    pending=$(squeue -u $USER -t PENDING | grep "ckpt_" | wc -l)
    running=${running:-0}
    pending=${pending:-0}
    completed=$((total_checkpoints - running - pending))

    echo "$(date '+%H:%M:%S') - Progress: $completed/$total_checkpoints completed | $running running | $pending pending"
    sleep 15
done

echo ""
echo "✅ All jobs completed!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📥 Collecting results..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

OUTPUT_DIR="$OUTPUT_DIR" RESULTS_JSON="$OUTPUT_DIR/checkpoint_sweep_results.json" "$collection_script"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Checkpoint sweep complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📂 Results location:"
echo "   $OUTPUT_DIR/checkpoint_sweep_results.json"
echo ""
echo "🔍 View detailed responses:"
echo "   python analyze_temperature_sweep.py $OUTPUT_DIR/checkpoint_sweep_results.json --show-all-responses"
echo "(Note: analyze_temperature_sweep.py expects temperature fields; response-only analysis still works.)"
echo ""
