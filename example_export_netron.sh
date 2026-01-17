#!/bin/bash
#SBATCH --job-name=export_model_netron
#SBATCH --output=logs/export_model_%j.out
#SBATCH --error=logs/export_model_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu

# SLURM script for exporting LLaVA models (base + LoRA) to Netron-compatible formats
# 
# Usage:
#   sbatch example_export_netron.sh
# 
# Or customize with environment variables:
#   CHECKPOINT_DIR=/path/to/checkpoint sbatch example_export_netron.sh

mkdir -p logs

echo "Starting model export job at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Virtual environment setup
VENV_ACTIVATE="/galitylab/students/alonmardi/llava/bin/activate"
if [[ -f "$VENV_ACTIVATE" ]]; then
  echo "Activating virtual environment..."
  source "$VENV_ACTIVATE"
else
  echo "Warning: expected virtual environment not found at $VENV_ACTIVATE"
fi

PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-/galitylab/students/alonmardi/llava/bin/python}
if [[ ! -x "$PYTHON_EXECUTABLE" ]]; then
  echo "Error: Python executable not found or not executable at $PYTHON_EXECUTABLE" >&2
  exit 1
fi

echo "Python version:"
$PYTHON_EXECUTABLE --version

echo "Machine IP Address:"
hostname -I || hostname

# Navigate to project root
PROJECT_ROOT="/galitylab/students/alonmardi/projects/LLaVA-NeXT"
cd "$PROJECT_ROOT" || exit 1
echo "Working directory: $(pwd)"

# Configuration - Override with environment variables if needed
BASE_MODEL="${BASE_MODEL:-lmms-lab/llava-onevision-qwen2-7b-ov-chat}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/galitylab/students/alonmardi/projects/LLaVA-NeXT/training_outputs/llava-20251006_011418/checkpoint-6000}"
OUTPUT_DIR="${OUTPUT_DIR:-./exports}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Configuration:"
echo "=========================================="
echo "Base Model: $BASE_MODEL"
echo "LoRA Checkpoint: $CHECKPOINT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Export tasks - Comment out any you don't need

# Example 1: Export full merged model (base + LoRA) to PyTorch format
echo ""
echo "=========================================="
echo "Task 1: Exporting full merged model to ONNX format"
echo "=========================================="
$PYTHON_EXECUTABLE export_model_for_netron.py \
    --base-model "$BASE_MODEL" \
    --lora-checkpoint "$CHECKPOINT_DIR" \
    --output-path "$OUTPUT_DIR/merged_model.onnx" \
    --format onnx
EXPORT_STATUS=$?
if [ $EXPORT_STATUS -eq 0 ]; then
    echo "✓ Task 1 completed successfully"
else
    echo "✗ Task 1 failed with exit code $EXPORT_STATUS"
fi

# Example 2: Export vision tower only to ONNX (good for visualization)
echo ""
echo "=========================================="
echo "Task 2: Exporting vision tower to ONNX"
echo "=========================================="
$PYTHON_EXECUTABLE export_model_for_netron.py \
    --base-model "$BASE_MODEL" \
    --lora-checkpoint "$CHECKPOINT_DIR" \
    --output-path "$OUTPUT_DIR/vision_tower.onnx" \
    --format onnx \
    --component vision_tower
EXPORT_STATUS=$?
if [ $EXPORT_STATUS -eq 0 ]; then
    echo "✓ Task 2 completed successfully"
else
    echo "✗ Task 2 failed with exit code $EXPORT_STATUS"
fi

# Example 3: Export projector to ONNX
echo ""
echo "=========================================="
echo "Task 3: Exporting projector to ONNX"
echo "=========================================="
$PYTHON_EXECUTABLE export_model_for_netron.py \
    --base-model "$BASE_MODEL" \
    --lora-checkpoint "$CHECKPOINT_DIR" \
    --output-path "$OUTPUT_DIR/projector.onnx" \
    --format onnx \
    --component projector
EXPORT_STATUS=$?
if [ $EXPORT_STATUS -eq 0 ]; then
    echo "✓ Task 3 completed successfully"
else
    echo "✗ Task 3 failed with exit code $EXPORT_STATUS"
fi

# Example 4: Export vision resampler to ONNX
echo ""
echo "=========================================="
echo "Task 4: Exporting vision resampler to ONNX"
echo "=========================================="
$PYTHON_EXECUTABLE export_model_for_netron.py \
    --base-model "$BASE_MODEL" \
    --lora-checkpoint "$CHECKPOINT_DIR" \
    --output-path "$OUTPUT_DIR/vision_resampler.onnx" \
    --format onnx \
    --component vision_resampler
EXPORT_STATUS=$?
if [ $EXPORT_STATUS -eq 0 ]; then
    echo "✓ Task 4 completed successfully"
else
    echo "✗ Task 4 failed with exit code $EXPORT_STATUS"
fi

# Summary
echo ""
echo "=========================================="
echo "Export job completed at $(date)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Output Directory: $OUTPUT_DIR"
echo ""
echo "Exported files:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "To view in Netron:"
echo "  - Online: Visit https://netron.app and drag-drop the files"
echo "  - Desktop: Install with 'pip install netron' and run 'netron <file>'"
echo ""
echo "=========================================="
