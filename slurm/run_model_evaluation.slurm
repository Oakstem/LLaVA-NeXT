#!/bin/bash
#SBATCH --job-name=model_evaluation
#SBATCH --output=logs/model_evaluation_%j.out
#SBATCH --error=logs/model_evaluation_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32G
#SBATCH --nodes=1


# Configuration variables - modify these as needed
MODEL_PATH="${MODEL_PATH:-lmms-lab/llava-onevision-qwen2-7b-ov-chat}"
MODEL_BASE="${MODEL_BASE:-}"
ADAPTER_PATH="${ADAPTER_PATH:-/galitylab/students/alonmardi/projects/LLaVA-NeXT/training_outputs/llava-20251009_234841/checkpoint-6000}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
QUANTIZATION="${QUANTIZATION:-}"  # Options: 4bit, 8bit, none

# Dataset parameters
DATASET_JSON="${DATASET_JSON:-/galitylab/students/alonmardi/projects/LLaVA-NeXT/training_datasets/20251007_165818_sgl_conversation_data_20251007_171523/val.json}"
IMAGES_DIR="${IMAGES_DIR:-/galitylab/students/alonmardi/gazefollow}"
LIMIT="${LIMIT:-}"  # Set to a number to limit samples for testing (e.g., 100)

# Generation parameters
CONV_TEMPLATE="${CONV_TEMPLATE:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.9}"
NUM_BEAMS="${NUM_BEAMS:-1}"
DO_SAMPLE="${DO_SAMPLE:-false}"  # Set to true to enable sampling
DISABLE_OPTIMIZATIONS="${DISABLE_OPTIMIZATIONS:-false}"

# Output parameters
OUTPUT_DIR="${OUTPUT_DIR:-./evaluation_results/eval_$(date +%Y%m%d_%H%M%S)}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-true}"
BATCH_SIZE="${BATCH_SIZE:-1}"
VERBOSE="${VERBOSE:-true}"
SAFE_MODE="${SAFE_MODE:-true}"  # Enable safe mode by default for CUDA stability
WANDB="${WANDB:-true}"  # Set to true to enable wandb logging

# Create necessary directories
mkdir -p logs
mkdir -p "$(dirname "$OUTPUT_DIR")"

# Print job information
echo "=========================================="
echo "Model Evaluation Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""
echo "Model Configuration:"
echo "  Model Path: $MODEL_PATH"
echo "  Model Base: $MODEL_BASE"
echo "  Adapter Path: $ADAPTER_PATH"
echo "  Attention Implementation: $ATTN_IMPLEMENTATION"
echo "  Quantization: $QUANTIZATION"
echo ""
echo "Dataset Configuration:"
echo "  Dataset JSON: $DATASET_JSON"
echo "  Images Directory: $IMAGES_DIR"
echo "  Sample Limit: $LIMIT"
echo ""
echo "Generation Parameters:"
echo "  Conversation Template: $CONV_TEMPLATE"
echo "  Max New Tokens: $MAX_NEW_TOKENS"
echo "  Temperature: $TEMPERATURE"
echo "  Top-p: $TOP_P"
echo "  Num Beams: $NUM_BEAMS"
echo "  Do Sample: $DO_SAMPLE"
echo ""
echo "Output Configuration:"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Save Predictions: $SAVE_PREDICTIONS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Verbose: $VERBOSE"
echo "  Safe Mode: $SAFE_MODE"
echo "  Log to Wandb: $WANDB"
echo "=========================================="

# Validate required files exist
echo "Validating input files..."
if [ ! -f "$DATASET_JSON" ]; then
    echo "ERROR: Dataset JSON file not found: $DATASET_JSON"
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "ERROR: Images directory not found: $IMAGES_DIR"
    exit 1
fi

echo "Input validation passed."
echo ""

# Load any required modules (uncomment and modify as needed)
# module load cuda/11.8
# module load python/3.10

# Activate virtual environment
echo "Activating virtual environment..."
source /galitylab/students/alonmardi/llava/bin/activate

# Verify activation
echo "Python environment:"
which python
python --version
echo ""

# Print machine information
echo "Machine Information:"
echo "IP Address: $(hostname -I)"
echo ""

# Verify GPU availability
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/galitylab/students/alonmardi/projects/LLaVA-NeXT:$PYTHONPATH"
# Enable CUDA debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Change to the project directory
cd /galitylab/students/alonmardi/projects/LLaVA-NeXT

# Check if evaluation script exists
if [ ! -f "evaluate_model.py" ]; then
    echo "ERROR: evaluate_model.py not found in current directory"
    echo "Current directory: $(pwd)"
    echo "Available files:"
    ls -la *.py | head -10
    exit 1
fi

# Build command arguments
ARGS=(
    "--model-path" "$MODEL_PATH"
    "--dataset-json" "$DATASET_JSON"
    "--images-dir" "$IMAGES_DIR"
    "--output-dir" "$OUTPUT_DIR"
    "--attn-implementation" "$ATTN_IMPLEMENTATION"
    "--max-new-tokens" "$MAX_NEW_TOKENS"
    "--temperature" "$TEMPERATURE"
    "--top-p" "$TOP_P"
    "--num-beams" "$NUM_BEAMS"
    "--batch-size" "$BATCH_SIZE"
)

# Add optional model arguments
if [ -n "$MODEL_BASE" ]; then
    ARGS+=("--model-base" "$MODEL_BASE")
fi

if [ -n "$ADAPTER_PATH" ]; then
    ARGS+=("--adapter-path" "$ADAPTER_PATH")
fi

if [ -n "$CONV_TEMPLATE" ]; then
    ARGS+=("--conv-template" "$CONV_TEMPLATE")
fi

# Add quantization options
if [ "$QUANTIZATION" = "4bit" ]; then
    ARGS+=("--load-4bit")
elif [ "$QUANTIZATION" = "8bit" ]; then
    ARGS+=("--load-8bit")
fi

# Add sampling option
if [ "$DO_SAMPLE" = "true" ]; then
    ARGS+=("--do-sample")
fi

# Add optimization option
if [ "$DISABLE_OPTIMIZATIONS" = "true" ]; then
    ARGS+=("--disable-optimizations")
fi

# Add save predictions option
if [ "$SAVE_PREDICTIONS" = "true" ]; then
    ARGS+=("--save-predictions")
fi

# Add verbose option
if [ "$VERBOSE" = "true" ]; then
    ARGS+=("--verbose")
fi

# Add safe mode option
if [ "$SAFE_MODE" = "true" ]; then
    ARGS+=("--safe-mode")
fi

# Add limit option for testing
if [ -n "$LIMIT" ]; then
    ARGS+=("--limit" "$LIMIT")
fi

if [ "$WANDB" = "true" ]; then
    ARGS+=("--log-to-wandb")
fi

# Print the command that will be executed
echo "=========================================="
echo "Executing evaluation command:"
echo "python evaluate_model.py ${ARGS[*]}"
echo "=========================================="
echo ""

# Record start time
start_time=$(date +%s)

# Run the evaluation script
echo "Starting model evaluation..."
python evaluate_model.py "${ARGS[@]}"

exit_code=$?
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "=========================================="
echo "Evaluation Results Summary"
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $exit_code"
echo "Total runtime: ${duration} seconds ($(($duration / 60)) minutes)"

if [ $exit_code -eq 0 ]; then
    echo "✓ Evaluation completed successfully!"
    
    # Display results if available
    if [ -f "$OUTPUT_DIR/metrics.json" ]; then
        echo ""
        echo "Key Metrics:"
        echo "----------------------------------------"
        # Extract and display key metrics using python
        python -c "
import json
import sys
try:
    with open('$OUTPUT_DIR/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    key_metrics = [
        'success_rate', 'exact_match', 'total_samples', 
        'successful_predictions', 'failed_samples',
        'avg_prediction_length', 'evaluation_time_seconds'
    ]
    
    for metric in key_metrics:
        if metric in metrics:
            value = metrics[metric]
            if isinstance(value, float):
                if metric.endswith('_rate') or metric == 'exact_match':
                    print(f'{metric}: {value:.2%}')
                else:
                    print(f'{metric}: {value:.2f}')
            else:
                print(f'{metric}: {value}')
except Exception as e:
    print(f'Could not parse metrics: {e}')
"
    fi
    
    echo ""
    echo "Output files:"
    if [ -d "$OUTPUT_DIR" ]; then
        ls -la "$OUTPUT_DIR/"
    fi
    
    echo ""
    echo "To view detailed results:"
    echo "  cat $OUTPUT_DIR/metrics.json"
    if [ "$SAVE_PREDICTIONS" = "true" ]; then
        echo "  head $OUTPUT_DIR/predictions.json"
    fi
    
else
    echo "✗ Evaluation failed with exit code: $exit_code"
    echo ""
    echo "Check the error logs for details:"
    echo "  tail -50 logs/model_evaluation_$SLURM_JOB_ID.err"
fi

echo "=========================================="

exit $exit_code