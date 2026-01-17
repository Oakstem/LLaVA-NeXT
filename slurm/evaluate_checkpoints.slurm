#!/bin/bash
#SBATCH --job-name=llava_eval_ckpts
#SBATCH --output=logs/llava_eval_ckpts_%j.out
#SBATCH --error=logs/llava_eval_ckpts_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=cpu

mkdir -p logs

echo "Starting checkpoint evaluation at $(date)"

VENV_ACTIVATE="/galitylab/students/alonmardi/llava/bin/activate"
if [[ -f "$VENV_ACTIVATE" ]]; then
  echo "Activating virtual environment..."
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
else
  echo "Warning: expected virtual environment not found at $VENV_ACTIVATE"
fi

PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-/galitylab/students/alonmardi/llava/bin/python}
if [[ ! -x "$PYTHON_EXECUTABLE" ]]; then
  echo "Error: Python executable not found or not executable at $PYTHON_EXECUTABLE" >&2
  exit 1
fi

which python
python --version

echo "Machine IP Address:"
hostname -I || hostname

PROJECT_ROOT="/galitylab/students/alonmardi/projects/LLaVA-NeXT"
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export ACCELERATE_DISABLE_NUMA=1

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/galitylab/students/alonmardi/projects/LLaVA-NeXT/training_outputs/selected}"
DATASET_JSON="${DATASET_JSON:-/galitylab/students/alonmardi/projects/LLaVA-NeXT/training_datasets/sgl_conversation_data_20250928_172112/val.json}"
IMAGES_DIR="${IMAGES_DIR:-/galitylab/students/alonmardi/gazefollow}"
OUTPUT_ROOT="${OUTPUT_ROOT:-evaluation_results/multi_checkpoint}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

SUMMARY_SUFFIX="$(date +%Y%m%d_%H%M%S)"
SUMMARY_FILE="$OUTPUT_ROOT/summary_${SUMMARY_SUFFIX}.json"

mkdir -p "$OUTPUT_ROOT"

echo "Using checkpoints directory: $CHECKPOINTS_DIR"
echo "Using dataset json: $DATASET_JSON"
echo "Using images directory: $IMAGES_DIR"
echo "Output root: $OUTPUT_ROOT"

declare -a CMD
CMD=("$PYTHON_EXECUTABLE" "evaluate_checkpoints.py"
     "--checkpoints-dir" "$CHECKPOINTS_DIR"
     "--dataset-json" "$DATASET_JSON"
     "--images-dir" "$IMAGES_DIR"
     "--output-root" "$OUTPUT_ROOT"
     "--summary-file" "$SUMMARY_FILE"
     "--python-executable" "$PYTHON_EXECUTABLE"
)

if [[ -n "$MODEL_BASE" ]]; then
  CMD+=("--model-base" "$MODEL_BASE")
fi
if [[ -n "$ADAPTER_PATH" ]]; then
  CMD+=("--adapter-path" "$ADAPTER_PATH")
fi
if [[ -n "$ATTN_IMPLEMENTATION" ]]; then
  CMD+=("--attn-implementation" "$ATTN_IMPLEMENTATION")
fi
if [[ "$LOAD_4BIT" == "1" || "$LOAD_4BIT" == "true" ]]; then
  CMD+=("--load-4bit")
fi
if [[ "$LOAD_8BIT" == "1" || "$LOAD_8BIT" == "true" ]]; then
  CMD+=("--load-8bit")
fi
if [[ "$SAVE_PREDICTIONS" == "1" || "$SAVE_PREDICTIONS" == "true" ]]; then
  CMD+=("--save-predictions")
fi
if [[ "$SKIP_EXISTING" == "1" || "$SKIP_EXISTING" == "true" ]]; then
  CMD+=("--skip-existing")
fi
if [[ "$CONTINUE_ON_ERROR" == "1" || "$CONTINUE_ON_ERROR" == "true" ]]; then
  CMD+=("--continue-on-error")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARRAY=($EXTRA_ARGS)
  CMD+=("${EXTRA_ARRAY[@]}")
fi

echo "Running command: ${CMD[*]}"
"${CMD[@]}"
STATUS=$?

echo "Checkpoint evaluation finished with status $STATUS at $(date)"
exit $STATUS
