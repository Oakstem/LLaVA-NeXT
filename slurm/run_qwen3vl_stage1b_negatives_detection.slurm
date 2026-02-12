#!/bin/bash
#SBATCH --job-name=qwen3vl_stage1b_neg
#SBATCH --output=logs/qwen3vl_stage1b_neg_%j.out
#SBATCH --error=logs/qwen3vl_stage1b_neg_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --nodes=1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/galitylab/students/alonmardi/projects/LLaVA-NeXT}"
if [ ! -d "$PROJECT_ROOT" ]; then
    PROJECT_ROOT="$(pwd)"
fi

PYTHON_BIN="${PYTHON_BIN:-/galitylab/students/alonmardi/.ground-uv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="/home/alonz/.ground-uv/bin/python"
fi
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python environment not found in /galitylab/... or /home/alonz/..."
    exit 1
fi

DATASET_JSON="${DATASET_JSON:-training_datasets/20251102/train_dataset_localization_results_normalized_filtered_conversations.json}"
IMAGES_DIR="${IMAGES_DIR:-/galitylab/students/alonmardi/gazefollow}"
OUTPUT_CSV="${OUTPUT_CSV:-training_datasets/stage1b_qwen3vl_negatives_$(date +%Y%m%d_%H%M%S).csv}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-4B-Instruct}"
QUERY="${QUERY:-Detect all visible people and objects in the image. Return ONLY a JSON array where each element is {\"label\": string, \"bbox\": [x1,y1,x2,y2], \"score\": number}. Use bbox coordinates normalized to [0,1000].}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
LIMIT="${LIMIT:-}"
START_INDEX="${START_INDEX:-0}"
REMOVE_POLICY="${REMOVE_POLICY:-smallest}"
RESUME="${RESUME:-true}"

mkdir -p logs
mkdir -p "$(dirname "$OUTPUT_CSV")"

source "$(dirname "$PYTHON_BIN")/activate"

cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
fi

if [ ! -f "$DATASET_JSON" ]; then
    echo "Dataset JSON not found: $DATASET_JSON"
    exit 1
fi
if [ ! -d "$IMAGES_DIR" ]; then
    echo "Images directory not found: $IMAGES_DIR"
    exit 1
fi
if [ ! -f "gazefollow/data_prep/build_qwen3vl_negatives_obj_detection.py" ]; then
    echo "Script not found: gazefollow/data_prep/build_qwen3vl_negatives_obj_detection.py"
    exit 1
fi

ARGS=(
    "--dataset-json" "$DATASET_JSON"
    "--images-dir" "$IMAGES_DIR"
    "--output-csv" "$OUTPUT_CSV"
    "--model-id" "$MODEL_ID"
    "--query" "$QUERY"
    "--device-map" "$DEVICE_MAP"
    "--max-new-tokens" "$MAX_NEW_TOKENS"
    "--temperature" "$TEMPERATURE"
    "--start-index" "$START_INDEX"
    "--remove-policy" "$REMOVE_POLICY"
)

if [ -n "$LIMIT" ]; then
    ARGS+=("--limit" "$LIMIT")
fi

if [ "$RESUME" = "true" ]; then
    ARGS+=("--resume")
else
    ARGS+=("--no-resume")
fi

echo "=========================================="
echo "Qwen3VL Stage-1b Negatives CSV Job"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "Project Root: $PROJECT_ROOT"
echo "Python: $PYTHON_BIN"
echo "Dataset: $DATASET_JSON"
echo "Images Dir: $IMAGES_DIR"
echo "Output CSV: $OUTPUT_CSV"
echo "Model: $MODEL_ID"
echo "Device Map: $DEVICE_MAP"
echo "Max Tokens: $MAX_NEW_TOKENS"
echo "Temperature: $TEMPERATURE"
echo "Start Index: $START_INDEX"
echo "Limit: ${LIMIT:-none}"
echo "Remove Policy: $REMOVE_POLICY"
echo "Resume: $RESUME"
echo "=========================================="
echo "$PYTHON_BIN gazefollow/data_prep/build_qwen3vl_negatives_obj_detection.py ${ARGS[*]}"

start_time=$(date +%s)
"$PYTHON_BIN" gazefollow/data_prep/build_qwen3vl_negatives_obj_detection.py "${ARGS[@]}"
exit_code=$?
end_time=$(date +%s)

echo "Completed at: $(date)"
echo "Exit code: $exit_code"
echo "Runtime: $((end_time - start_time)) seconds"
if [ -f "$OUTPUT_CSV" ]; then
    echo "CSV rows: $(wc -l < "$OUTPUT_CSV")"
fi

exit $exit_code
