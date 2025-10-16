#!/bin/bash
#
# Temperature Sweep Script - Submits parallel SLURM jobs for each temperature
#
# Usage: ./run_temperature_sweep.sh [OPTIONS]
#
# This script will:
# 1. Generate a list of temperature values
# 2. Submit a separate SLURM job for each temperature
# 3. Collect results into a single JSON file when all jobs complete
#

set -e

# Default configuration
DEFAULT_TEMP_START=0.7
DEFAULT_TEMP_END=1.3
DEFAULT_TEMP_STEP=0.1
DEFAULT_IMAGE="/galitylab/students/alonmardi/projects/LLaVA-NeXT/baseline_images/39740.png"
DEFAULT_MODEL="lmms-lab/llava-onevision-qwen2-7b-ov-chat"
DEFAULT_MAX_TOKENS=512
DEFAULT_TOP_P=0.9
DEFAULT_NUM_BEAMS=1
DEFAULT_DO_SAMPLE="true"
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
TEMP_START="${TEMP_START:-$DEFAULT_TEMP_START}"
TEMP_END="${TEMP_END:-$DEFAULT_TEMP_END}"
TEMP_STEP="${TEMP_STEP:-$DEFAULT_TEMP_STEP}"
IMAGE_PATH="${IMAGE_PATH:-$DEFAULT_IMAGE}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-$DEFAULT_MAX_TOKENS}"
TOP_P="${TOP_P:-$DEFAULT_TOP_P}"
NUM_BEAMS="${NUM_BEAMS:-$DEFAULT_NUM_BEAMS}"
DO_SAMPLE="${DO_SAMPLE:-$DEFAULT_DO_SAMPLE}"
JOB_TIME="${JOB_TIME:-$DEFAULT_TIME}"
JOB_MEM="${JOB_MEM:-$DEFAULT_MEM}"
MODEL_BASE="${MODEL_BASE:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
QUANTIZATION="${QUANTIZATION:-}"
CONV_TEMPLATE="${CONV_TEMPLATE:-}"
RUN_GDINO="${RUN_GDINO:-$DEFAULT_RUN_GDINO}"
GDINO_MODEL_ID="${GDINO_MODEL_ID:-$DEFAULT_GDINO_MODEL_ID}"
GDINO_BOX_THRESHOLD="${GDINO_BOX_THRESHOLD:-$DEFAULT_GDINO_BOX_THRESHOLD}"
GDINO_TEXT_THRESHOLD="${GDINO_TEXT_THRESHOLD:-$DEFAULT_GDINO_TEXT_THRESHOLD}"
GDINO_DEVICE="${GDINO_DEVICE:-$DEFAULT_GDINO_DEVICE}"

# Default prompt
if [ -z "$PROMPT" ]; then
    read -r -d '' PROMPT <<'EOF' || true
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
EOF
fi

# Print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Submit parallel temperature sweep jobs to SLURM.

Temperature Options:
    -s, --temp-start FLOAT    Starting temperature (default: $DEFAULT_TEMP_START)
    -e, --temp-end FLOAT      Ending temperature (default: $DEFAULT_TEMP_END)
    -t, --temp-step FLOAT     Temperature step size (default: $DEFAULT_TEMP_STEP)

Model Options:
    -m, --model PATH          Model path (default: $DEFAULT_MODEL)
    -a, --adapter PATH        Adapter checkpoint path
    --model-base PATH         Base model path for LoRA
    --attn-impl NAME          Attention implementation (default: sdpa)
    --quantize [4bit|8bit]    Enable quantization

Inference Options:
    -i, --image PATH          Image path (default: $DEFAULT_IMAGE)
    -p, --prompt TEXT         Custom prompt
    --max-tokens INT          Max new tokens (default: $DEFAULT_MAX_TOKENS)
    --top-p FLOAT             Top-p value (default: $DEFAULT_TOP_P)
    --num-beams INT           Number of beams (default: $DEFAULT_NUM_BEAMS)
    --do-sample BOOL          Enable sampling (default: $DEFAULT_DO_SAMPLE)

SLURM Options:
    --time TIME               Job time limit (default: $DEFAULT_TIME)
    --mem SIZE                Memory per job (default: $DEFAULT_MEM)

Other:
    -h, --help                Show this help message
    --interactive             Interactively select adapter

Examples:
    # Basic usage (0.1 to 1.0, step 0.1)
    $0

    # Custom range
    $0 -s 0.0 -e 2.0 -t 0.2

    # With specific checkpoint
    $0 -a training_outputs/llava-20251007_011312/checkpoint-10000

    # With custom image
    $0 -i baseline_images/11865.jpg

    # Interactive adapter selection
    $0 --interactive
EOF
    exit 0
}

# Interactive adapter selection
select_adapter_interactive() {
    echo "=========================================="
    echo "Available recent checkpoints:"
    
    mapfile -t checkpoints < <(ls -dt /galitylab/students/alonmardi/projects/LLaVA-NeXT/training_outputs/llava-*/checkpoint-* 2>/dev/null | head -10)
    
    if [ ${#checkpoints[@]} -eq 0 ]; then
        echo "No checkpoints found."
        ADAPTER_PATH=""
        return
    fi
    
    for i in "${!checkpoints[@]}"; do
        echo "  [$i] ${checkpoints[$i]}"
    done
    
    echo "  [-1] No adapter (base model only)"
    echo "=========================================="
    echo "Default: [0] ${checkpoints[0]}"
    read -p "Enter checkpoint number, -1 for no adapter, full path, or press Enter for default: " user_input
    
    if [ -z "$user_input" ]; then
        ADAPTER_PATH="${checkpoints[0]}"
        echo "Selected: $ADAPTER_PATH"
    elif [ "$user_input" = "-1" ]; then
        ADAPTER_PATH=""
        echo "Selected: No adapter (base model only)"
    elif [[ "$user_input" =~ ^[0-9]+$ ]] && [ "$user_input" -lt "${#checkpoints[@]}" ]; then
        ADAPTER_PATH="${checkpoints[$user_input]}"
        echo "Selected: $ADAPTER_PATH"
    else
        ADAPTER_PATH="$user_input"
        echo "Selected: $ADAPTER_PATH"
    fi
}
# Parse arguments
INTERACTIVE=true
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--temp-start) TEMP_START="$2"; shift 2 ;;
        -e|--temp-end) TEMP_END="$2"; shift 2 ;;
        -t|--temp-step) TEMP_STEP="$2"; shift 2 ;;
        -m|--model) MODEL_PATH="$2"; shift 2 ;;
        -a|--adapter) ADAPTER_PATH="$2"; shift 2 ;;
        --model-base) MODEL_BASE="$2"; shift 2 ;;
        --attn-impl) ATTN_IMPLEMENTATION="$2"; shift 2 ;;
        --quantize) QUANTIZATION="$2"; shift 2 ;;
        -i|--image) IMAGE_PATH="$2"; shift 2 ;;
        -p|--prompt) PROMPT="$2"; shift 2 ;;
        --max-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --num-beams) NUM_BEAMS="$2"; shift 2 ;;
        --do-sample) DO_SAMPLE="$2"; shift 2 ;;
        --time) JOB_TIME="$2"; shift 2 ;;
        --mem) JOB_MEM="$2"; shift 2 ;;
        --interactive) INTERACTIVE=true; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# Interactive adapter selection if requested
if [ "$INTERACTIVE" = true ] && [ -z "$ADAPTER_PATH" ]; then
    select_adapter_interactive
fi

# Determine output directory suffix based on adapter selection
if [ -n "$ADAPTER_PATH" ]; then
    checkpoint_name=$(basename "$ADAPTER_PATH")
    parent_dir=$(dirname "$ADAPTER_PATH")
    training_run=$(basename "$parent_dir")

    if [ -z "$training_run" ] || [ "$training_run" = "." ]; then
        training_run="adapter"
    fi

    if [ -z "$checkpoint_name" ] || [ "$checkpoint_name" = "." ]; then
        checkpoint_name="checkpoint"
    fi

    dir_suffix="${training_run}_${checkpoint_name}"
else
    dir_suffix="base_model"
fi

# Change to project directory
cd "$(dirname "$0")"
PROJECT_ROOT="/galitylab/students/alonmardi/projects/LLaVA-NeXT"
cd "$PROJECT_ROOT"

# Create output directory
OUTPUT_DIR="evaluation_results/eval_${dir_suffix}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Generate temperature values
temperatures=$(python -c "
start, end, step = float('$TEMP_START'), float('$TEMP_END'), float('$TEMP_STEP')
temps = []
temp = start
while temp <= end + 1e-9:
    temps.append(f'{temp:.2f}')
    temp += step
print(' '.join(temps))
")

temp_array=($temperatures)
total_temps=${#temp_array[@]}

# Print configuration
echo "=========================================="
echo "Temperature Sweep Configuration"
echo "=========================================="
echo "Temperature range: $TEMP_START to $TEMP_END (step: $TEMP_STEP)"
echo "Number of temperatures: $total_temps"
echo "Values: ${temp_array[*]}"
echo ""
echo "Model: $MODEL_PATH"
[ -n "$ADAPTER_PATH" ] && echo "Adapter: $ADAPTER_PATH"
echo "Image: $IMAGE_PATH"
echo "Max tokens: $MAX_NEW_TOKENS"
echo "Top-p: $TOP_P"
echo "Do sample: $DO_SAMPLE"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Job time limit: $JOB_TIME"
echo "Memory per job: $JOB_MEM"
echo "=========================================="

# Save metadata
if [ "$RUN_GDINO" = "true" ]; then
    RUN_GDINO_BOOL=true
else
    RUN_GDINO_BOOL=false
fi

cat > "$OUTPUT_DIR/metadata.json" <<EOF
{
  "model_path": "$MODEL_PATH",
  "adapter_path": "$ADAPTER_PATH",
  "image_path": "$IMAGE_PATH",
  "prompt": $(echo "$PROMPT" | python -c "import sys, json; print(json.dumps(sys.stdin.read()))"),
  "temp_start": $TEMP_START,
  "temp_end": $TEMP_END,
  "temp_step": $TEMP_STEP,
  "max_new_tokens": $MAX_NEW_TOKENS,
  "top_p": $TOP_P,
  "num_beams": $NUM_BEAMS,
  "do_sample": $DO_SAMPLE,
  "timestamp": "$(date -Iseconds)",
  "total_jobs": $total_temps,
  "gdino": {
    "enabled": $RUN_GDINO_BOOL,
    "model_id": "$GDINO_MODEL_ID",
    "box_threshold": $GDINO_BOX_THRESHOLD,
    "text_threshold": $GDINO_TEXT_THRESHOLD,
    "device": "$GDINO_DEVICE"
  }
}
EOF

# Confirm submission
read -p "Submit $total_temps parallel jobs? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "Submission cancelled."
    exit 0
fi

# Submit jobs for each temperature
job_ids=()
echo ""
echo "Submitting jobs..."

for temperature in "${temp_array[@]}"; do
    output_file="$OUTPUT_DIR/output_temp_${temperature}.json"
    
    # Create SLURM script for this temperature
    job_script="$OUTPUT_DIR/job_temp_${temperature}.sh"
    
    # Write the script with proper variable expansion
    # We need to be careful: expand some vars now, but keep others for runtime
    cat > "$job_script" <<EOJOB
#!/bin/bash
#SBATCH --job-name=temp_${temperature}
#SBATCH --output=logs/temp_${temperature}_%j.out
#SBATCH --error=logs/temp_${temperature}_%j.err
#SBATCH --time=$JOB_TIME
#SBATCH --partition=gpu-tad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=$JOB_MEM
#SBATCH --nodes=1

echo "Temperature: $temperature"
echo "Output: $output_file"

source /galitylab/students/alonmardi/llava/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PROJECT_ROOT:\$PYTHONPATH"

cd $PROJECT_ROOT

# Set PROMPT variable - must preserve exact content including newlines
read -r -d '' PROMPT <<'PROMPTEOF'
$PROMPT
PROMPTEOF

# Build command arguments array (same approach as vanilla script)
ARGS=(
    "--model-path" "$MODEL_PATH"
    "--attn-implementation" "$ATTN_IMPLEMENTATION"
    "--image-path" "$IMAGE_PATH"
    "--prompt" "\$PROMPT"
    "--max-new-tokens" "$MAX_NEW_TOKENS"
    "--temperature" "$temperature"
    "--top-p" "$TOP_P"
    "--num-beams" "$NUM_BEAMS"
    "--save-output" "$output_file"
    "--image-aspect-ratio" "$IMAGE_ASPECT_RATIO_FALLBACK"
    "--image-grid-pinpoints" "$IMAGE_GRID_PINPOINTS_FALLBACK"
)

RUN_GDINO="$RUN_GDINO"
GDINO_MODEL_ID="$GDINO_MODEL_ID"
GDINO_BOX_THRESHOLD="$GDINO_BOX_THRESHOLD"
GDINO_TEXT_THRESHOLD="$GDINO_TEXT_THRESHOLD"
GDINO_DEVICE="$GDINO_DEVICE"

if [ "$RUN_GDINO" = "true" ]; then
    ARGS+=("--run-gdino")
    ARGS+=("--gdino-model-id" "$GDINO_MODEL_ID")
    ARGS+=("--gdino-box-threshold" "$GDINO_BOX_THRESHOLD")
    ARGS+=("--gdino-text-threshold" "$GDINO_TEXT_THRESHOLD")
    ARGS+=("--gdino-device" "$GDINO_DEVICE")
fi

# Add optional model arguments
EOJOB

    # Add conditional arguments based on configuration
    if [ -n "$MODEL_BASE" ]; then
        cat >> "$job_script" <<EOJOB
ARGS+=("--model-base" "$MODEL_BASE")
EOJOB
    fi

    if [ -n "$ADAPTER_PATH" ]; then
        cat >> "$job_script" <<EOJOB
ARGS+=("--adapter-path" "$ADAPTER_PATH")
EOJOB
    fi

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

    if [ "$DO_SAMPLE" = "true" ]; then
        cat >> "$job_script" <<EOJOB
ARGS+=("--do-sample")
EOJOB
    fi

    # Finish the script
    cat >> "$job_script" <<'EOJOB'

# Run the script (same as vanilla script)
python generate_vanilla_inference.py "${ARGS[@]}"

exit_code=$?
if [ $exit_code -eq 0 ] && [ -f "$output_file" ]; then
    echo "✓ Success"
else
    echo "✗ Failed (exit code: $exit_code)"
fi
exit $exit_code
EOJOB

    # Note: We need to fix the output_file reference in the final check
    sed -i "s|\$output_file|$output_file|g" "$job_script"
    
    # Submit the job
    job_id=$(sbatch --parsable "$job_script")
    job_ids+=($job_id)
    echo "  Temperature $temperature: Job $job_id"
done

echo ""
echo "=========================================="
echo "Submitted $total_temps jobs"
echo "Job IDs: ${job_ids[*]}"
echo "=========================================="

# Save job IDs
echo "${job_ids[*]}" > "$OUTPUT_DIR/job_ids.txt"

# Create collection script
collection_script="$OUTPUT_DIR/collect_results.sh"
cat > "$collection_script" <<'EOCOLLECT'
#!/bin/bash
# Collect results from parallel temperature sweep jobs

OUTPUT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_JSON="$OUTPUT_DIR/temperature_sweep_results.json"
METADATA_JSON="$OUTPUT_DIR/metadata.json"

echo "Collecting results from: $OUTPUT_DIR"

# Read metadata
if [ ! -f "$METADATA_JSON" ]; then
    echo "Error: metadata.json not found"
    exit 1
fi

# Use Python to properly format JSON with preserved newlines
# Pass OUTPUT_DIR as environment variable
export OUTPUT_DIR
python << 'EOPY'
import json
import glob
import os
import sys

output_dir = os.environ.get('OUTPUT_DIR')
if not output_dir:
    print("Error: OUTPUT_DIR not set", file=sys.stderr)
    sys.exit(1)

results_json = os.path.join(output_dir, "temperature_sweep_results.json")
metadata_json = os.path.join(output_dir, "metadata.json")

# Load metadata
try:
    with open(metadata_json, 'r') as f:
        metadata = json.load(f)
except Exception as e:
    print(f"Error loading metadata: {e}", file=sys.stderr)
    sys.exit(1)

# Collect results
results = []
output_files = sorted(glob.glob(os.path.join(output_dir, "output_temp_*.json")))

for output_file in output_files:
    temp = os.path.basename(output_file).replace("output_temp_", "").replace(".json", "")
    
    try:
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        results.append({
            "temperature": float(temp),
            "output_file": output_file,
            "prompt": data.get("prompt", ""),
            "response": data.get("response", ""),
            "gdino": data.get("gdino"),
            "status": "success"
        })
    except Exception as e:
        results.append({
            "temperature": float(temp),
            "output_file": output_file,
            "error": str(e),
            "gdino": None,
            "status": "error"
        })

# Sort by temperature
results.sort(key=lambda x: x["temperature"])

# Write combined results with pretty formatting and preserved newlines
output_data = {
    "metadata": metadata,
    "results": results
}

with open(results_json, 'w') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

deleted_json = 0
for output_file in output_files:
    try:
        os.remove(output_file)
        deleted_json += 1
    except OSError as e:
        print(f"Warning: failed to remove {output_file}: {e}", file=sys.stderr)
if deleted_json:
    print(f"Removed {deleted_json} intermediate result file(s).")

script_files = sorted(glob.glob(os.path.join(output_dir, "job_temp_*.sh")))
deleted_scripts = 0
for script_file in script_files:
    try:
        os.remove(script_file)
        deleted_scripts += 1
    except OSError as e:
        print(f"Warning: failed to remove {script_file}: {e}", file=sys.stderr)
if deleted_scripts:
    print(f"Removed {deleted_scripts} job script(s).")

print(f"Results collected: {results_json}")
successful = len([r for r in results if r['status'] == 'success'])
total = metadata.get('total_jobs', len(results))
print(f"Successful: {successful}/{total}")
EOPY

# Analyze results
if command -v python &> /dev/null; then
    echo ""
    echo "Running analysis..."
    SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
    python "$SCRIPT_DIR/analyze_temperature_sweep.py" "$RESULTS_JSON"
fi
EOCOLLECT

chmod +x "$collection_script"

echo ""
echo "✅ All jobs submitted successfully!"
echo ""
echo "📊 Monitor progress:"
echo "  squeue -u \$USER | grep temp_"
echo "  watch -n 5 'squeue -u \$USER | grep temp_'"
echo ""
echo "📁 Output directory: $OUTPUT_DIR"
echo "📁 Job IDs: ${job_ids[*]}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏳ Waiting for all jobs to complete..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Wait for all temperature jobs to complete
while squeue -u $USER | grep -q "temp_"; do
    # Count running and pending jobs
    running=$(squeue -u $USER -t RUNNING | grep "temp_" | wc -l)
    pending=$(squeue -u $USER -t PENDING | grep "temp_" | wc -l)
    running=${running:-0}
    pending=${pending:-0}
    completed=$((total_temps - running - pending))
    
    echo "$(date '+%H:%M:%S') - Progress: $completed/$total_temps completed | $running running | $pending pending"
    sleep 15
done

echo ""
echo "✅ All jobs completed!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📥 Collecting results..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run the collection script
$collection_script

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Temperature sweep complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📂 Results location:"
echo "   $OUTPUT_DIR/temperature_sweep_results.json"
echo ""
echo "🔍 View detailed results:"
echo "   python analyze_temperature_sweep.py $OUTPUT_DIR/temperature_sweep_results.json --show-all-responses"
echo ""
