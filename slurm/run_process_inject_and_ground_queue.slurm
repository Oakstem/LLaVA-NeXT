#!/bin/bash
#SBATCH --job-name=process_inject_queue
#SBATCH --output=logs/process_inject_queue_%j.out
#SBATCH --error=logs/process_inject_queue_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --nodes=1

# Repository root used for relative paths
PROJECT_ROOT="/galitylab/students/alonmardi/projects/LLaVA-NeXT"

# Queue processor configuration
QUEUE_CSV="${QUEUE_CSV:-gazefollow/data/test2_combined_source_description_results_run_20260206.csv}"
# Annotation files:
# - Train: gazefollow/data/train_annotations_release.csv
# - Test:  gazefollow/data/test_annotations_release.csv
ANNOTATION_FILE="${ANNOTATION_FILE:-gazefollow/data/test_annotations_release.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-}"
THRESHOLD="${THRESHOLD:-0.2}"
LIMIT="${LIMIT:-}"
START_INDEX="${START_INDEX:-0}"
DATA_ROOT="${DATA_ROOT:-/galitylab/students/alonmardi/gazefollow}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
MASK_DIR="${MASK_DIR:-}"
LOG_FILE="${LOG_FILE:-}"
SKIP_EXISTING="${SKIP_EXISTING:-false}"
USE_BODY_BBOX="${USE_BODY_BBOX:-false}"

# Create logs directory relative to the project root
mkdir -p "$PROJECT_ROOT/logs"

echo "=========================================="
echo "Inject & Ground Queue Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""
echo "Queue Settings:"
echo "  Queue CSV: $QUEUE_CSV"
echo "  Annotation File: $ANNOTATION_FILE"
echo "  Output CSV: ${OUTPUT_CSV:-<in-place>}"
echo "  Threshold: $THRESHOLD"
echo "  Limit: ${LIMIT:-<no limit>}"
echo "  Start Index: $START_INDEX"
echo "  Data Root: $DATA_ROOT"
echo "  Output Dir: ${OUTPUT_DIR:-<auto timestamp>}"
echo "  Mask Dir: ${MASK_DIR:-<pipeline default>}"
echo "  Log File: ${LOG_FILE:-<auto>}"
echo "  Skip Existing Rows: $SKIP_EXISTING"
echo "  Use Body BBox: $USE_BODY_BBOX"
echo "=========================================="

# Activate virtual environment
echo "Activating virtual environment..."
source /galitylab/students/alonmardi/.ground-uv/bin/activate

# Verify activation
which python
python --version

# Machine information
echo "Machine IP Address:"
hostname -I

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Change to the project directory
cd "$PROJECT_ROOT"

# Ensure logs directory exists relative to project root
mkdir -p logs

# Validate essential paths
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Data root not found at $DATA_ROOT"
    exit 1
fi

if [ -n "$MASK_DIR" ] && [ ! -d "$MASK_DIR" ]; then
    echo "ERROR: Mask directory not found at $MASK_DIR"
    exit 1
fi

if [ -n "$OUTPUT_CSV" ]; then
    mkdir -p "$(dirname "$OUTPUT_CSV")"
fi

if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

if [ -n "$LOG_FILE" ]; then
    mkdir -p "$(dirname "$LOG_FILE")"
fi

# Environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Build CLI arguments
ARGS=(
    "--threshold" "$THRESHOLD"
    "--data-root" "$DATA_ROOT"
    "--start-index" "$START_INDEX"
    "--annotation-file" "$ANNOTATION_FILE"
)

if [ -n "$QUEUE_CSV" ]; then
    ARGS+=("--queue-csv" "$QUEUE_CSV")
fi

if [ -n "$OUTPUT_CSV" ]; then
    ARGS+=("--output-csv" "$OUTPUT_CSV")
fi

if [ -n "$OUTPUT_DIR" ]; then
    ARGS+=("--output-dir" "$OUTPUT_DIR")
fi

if [ -n "$LIMIT" ]; then
    ARGS+=("--limit" "$LIMIT")
fi

if [ -n "$MASK_DIR" ]; then
    ARGS+=("--mask-dir" "$MASK_DIR")
fi

if [ -n "$LOG_FILE" ]; then
    ARGS+=("--log-file" "$LOG_FILE")
fi

if [ "${SKIP_EXISTING,,}" = "true" ]; then
    ARGS+=("--skip-existing")
fi

if [ "${USE_BODY_BBOX,,}" = "false" ]; then
    ARGS+=("--no-use-body-bbox")
fi

echo "Executing command:"
echo "python gazefollow/data_prep/process_inject_and_ground_queue.py ${ARGS[*]}"
echo "=========================================="

python gazefollow/data_prep/process_inject_and_ground_queue.py "${ARGS[@]}"
exit_code=$?

echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code
