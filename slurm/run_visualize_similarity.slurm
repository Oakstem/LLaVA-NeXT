#!/bin/bash
#SBATCH --job-name=viz_similarity
#SBATCH --output=logs/viz_similarity_%j.out
#SBATCH --error=logs/viz_similarity_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-tad-pool
#SBATCH --qos=owner
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --nodes=1

# Configuration (override via --export when submitting)
TEXT_EMB_PATH="${TEXT_EMB_PATH:-embeddings/text/text_embeddings_20251026_215051.pt}"
IMAGE_EMB_PATH="${IMAGE_EMB_PATH:-embeddings/image/image_embeddings_20251026_215052.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-similarity_outputs}"
CMAP="${CMAP:-magma}"
ALPHA="${ALPHA:-0.45}"
INCLUDE_VISION_SOURCES="${INCLUDE_VISION_SOURCES:-mm_projected_features,vision_features, vision_hidden_states}"
SKIP_EXISTING="${SKIP_EXISTING:-false}"
MIN_PERCENTILE="${MIN_PERCENTILE:-99.8}"

mkdir -p logs

echo "=========================================="
echo "Similarity Visualization Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Text Embeddings: ${TEXT_EMB_PATH:-<missing>}"
echo "Image Embeddings: ${IMAGE_EMB_PATH:-<missing>}"
echo "Output Dir: $OUTPUT_DIR"
echo "Colormap: $CMAP"
echo "Alpha: $ALPHA"
echo "Vision Sources: $INCLUDE_VISION_SOURCES"
echo "Min Percentile: $MIN_PERCENTILE"
echo "Skip Existing: $SKIP_EXISTING"
echo "=========================================="

if [ -z "$TEXT_EMB_PATH" ] || [ -z "$IMAGE_EMB_PATH" ]; then
  echo "ERROR: TEXT_EMB_PATH and IMAGE_EMB_PATH must be provided."
  exit 1
fi

echo "Activating virtual environment..."
source /galitylab/students/alonmardi/llava/bin/activate

which python
python --version

export PYTHONPATH="/galitylab/students/alonmardi/projects/LLaVA-NeXT:$PYTHONPATH"
cd /galitylab/students/alonmardi/projects/LLaVA-NeXT

ARGS=(
  "--text-embeddings" "$TEXT_EMB_PATH"
  "--image-embeddings" "$IMAGE_EMB_PATH"
  "--output-dir" "$OUTPUT_DIR"
  "--cmap" "$CMAP"
  "--alpha" "$ALPHA"
  "--include-vision-sources" "$INCLUDE_VISION_SOURCES"
  "--min-percentile" "$MIN_PERCENTILE"
)

if [ "$SKIP_EXISTING" = "true" ]; then
  ARGS+=("--skip-existing")
fi

echo "Running command:"
echo "python visualize_text_image_similarity.py ${ARGS[*]}"

python visualize_text_image_similarity.py "${ARGS[@]}"

STATUS=$?
echo "Job finished with status: $STATUS"
echo "End Time: $(date)"
