#!/bin/bash
#SBATCH --job-name=qwen3vl_grounding
#SBATCH --output=logs/qwen3vl_grounding_%j.out
#SBATCH --error=logs/qwen3vl_grounding_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-tad-pool
#SBATCH --qos=owner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1

# Configuration variables - adjust via environment overrides when submitting
IMAGE_PATH="${IMAGE_PATH:-gazefollow/auto_phrase_grounding/test.jpg}"
QUERY="${QUERY:-Detect all mugs. Return JSON: {\"objects\":[{\"label\":\"mug\",\"bbox\":[x1,y1,x2,y2],\"confidence\":float}]} Coordinates must be absolute pixel ints, origin at top-left. No extra text.}"
OUTPUT_JSON="${OUTPUT_JSON:-grounding_output.json}"

mkdir -p logs

echo "=========================================="
echo "Qwen3-VL Grounding Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Image Path: $IMAGE_PATH"
echo "Query: $QUERY"
echo "Output JSON: $OUTPUT_JSON"
echo "=========================================="

echo "Activating virtual environment..."
source /galitylab/students/alonmardi/llava/bin/activate

which python
python --version

echo "Machine IP Address:"
hostname -I

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/galitylab/students/alonmardi/projects/LLaVA-NeXT:$PYTHONPATH"

cd /galitylab/students/alonmardi/projects/LLaVA-NeXT

ARGS=(
    "--image-path" "$IMAGE_PATH"
    "--query" "$QUERY"
)

if [ -n "$OUTPUT_JSON" ]; then
    ARGS+=("--output-json" "$OUTPUT_JSON")
fi

echo "Running Qwen3-VL grounding..."
echo "Command: python gazefollow/auto_phrase_grounding/qwen3vl_grounding.py ${ARGS[*]}"

python gazefollow/auto_phrase_grounding/qwen3vl_grounding.py "${ARGS[@]}"

STATUS=$?
echo "Job finished with status: $STATUS"
echo "End Time: $(date)"
