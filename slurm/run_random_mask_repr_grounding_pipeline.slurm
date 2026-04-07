#!/bin/bash
#SBATCH --job-name=random_mask_repr_pipeline
#SBATCH --output=logs/random_mask_repr_pipeline_%j.out
#SBATCH --error=logs/random_mask_repr_pipeline_%j.err
#SBATCH --time=05:00:00
#SBATCH --partition=power-general-shared-pool
#SBATCH --qos=public
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --nodes=1

set -euo pipefail

PROJECT_ROOT="/galitylab/students/alonmardi/projects/LLaVA-NeXT"
CSV_PATH="${CSV_PATH:-$PROJECT_ROOT/region_masks/random_samples/selected_masks.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-/galitylab/students/alonmardi/gazefollow/train}"
PIPELINE_TAG="${PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/results/steered_generation/random_mask_repr_grounding_${PIPELINE_TAG}}"
REPR_OUTPUT_ROOT="${REPR_OUTPUT_ROOT:-$RUN_ROOT/repr_sweeps}"
GROUNDING_OUTPUT_ROOT="${GROUNDING_OUTPUT_ROOT:-$RUN_ROOT/review_grounding}"
REPR_SCRIPT="${REPR_SCRIPT:-$PROJECT_ROOT/slurm/run_repr_layer_sweep.slurm}"
GROUNDING_SCRIPT="${GROUNDING_SCRIPT:-$PROJECT_ROOT/slurm/run_review_sweep_grounding.slurm}"
MAX_ITEMS="${MAX_ITEMS:-}"

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$RUN_ROOT" "$REPR_OUTPUT_ROOT" "$GROUNDING_OUTPUT_ROOT"

resolve_path() {
    local raw_path="$1"
    if [[ "$raw_path" = /* ]]; then
        printf '%s\n' "$raw_path"
    else
        printf '%s\n' "$PROJECT_ROOT/$raw_path"
    fi
}

strip_cr() {
    printf '%s' "${1%$'\r'}"
}

if [ ! -f "$CSV_PATH" ]; then
    echo "ERROR: CSV file not found at $CSV_PATH"
    exit 1
fi

if [ ! -f "$REPR_SCRIPT" ]; then
    echo "ERROR: Representation sweep script not found at $REPR_SCRIPT"
    exit 1
fi

if [ ! -f "$GROUNDING_SCRIPT" ]; then
    echo "ERROR: Grounding script not found at $GROUNDING_SCRIPT"
    exit 1
fi

echo "=========================================="
echo "Random Mask Repr/Grounding Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "CSV Path: $CSV_PATH"
echo "Image Root: $IMAGE_ROOT"
echo "Run Root: $RUN_ROOT"
echo "Repr Output Root: $REPR_OUTPUT_ROOT"
echo "Grounding Output Root: $GROUNDING_OUTPUT_ROOT"
echo "Repr Script: $REPR_SCRIPT"
echo "Grounding Script: $GROUNDING_SCRIPT"
echo "Max Items: ${MAX_ITEMS:-<all>}"
echo "=========================================="

declare -a sample_ids=()
declare -a image_paths=()
declare -a mask_paths=()

row_count=0
while IFS=, read -r raw_image_rel raw_mask_rel; do
    image_rel_path="$(strip_cr "$raw_image_rel")"
    mask_rel_path="$(strip_cr "$raw_mask_rel")"
    if [ -z "$image_rel_path" ] || [ -z "$mask_rel_path" ]; then
        continue
    fi

    image_path="$IMAGE_ROOT/$image_rel_path"
    mask_path="$(resolve_path "$mask_rel_path")"
    image_stem="$(basename "${image_rel_path%.*}")"
    image_parent="$(basename "$(dirname "$image_rel_path")")"
    sample_id="${image_parent}_${image_stem}"

    if [ ! -f "$image_path" ]; then
        echo "ERROR: Image not found at $image_path"
        exit 1
    fi

    if [ ! -f "$mask_path" ]; then
        echo "ERROR: Mask not found at $mask_path"
        exit 1
    fi

    sample_ids+=("$sample_id")
    image_paths+=("$image_path")
    mask_paths+=("$mask_path")

    row_count=$((row_count + 1))
    if [ -n "$MAX_ITEMS" ] && [ "$row_count" -ge "$MAX_ITEMS" ]; then
        break
    fi
done < <(tail -n +2 "$CSV_PATH")

if [ "$row_count" -eq 0 ]; then
    echo "ERROR: No rows were loaded from $CSV_PATH"
    exit 1
fi

manifest_path="$RUN_ROOT/submitted_jobs.csv"
printf 'sample_id,image_path,mask_path,repr_output_dir,repr_job_id,grounding_output_dir,grounding_job_id\n' > "$manifest_path"

for idx in "${!sample_ids[@]}"; do
    sample_id="${sample_ids[$idx]}"
    image_path="${image_paths[$idx]}"
    mask_path="${mask_paths[$idx]}"
    repr_output_dir="$REPR_OUTPUT_ROOT/$sample_id"
    grounding_output_dir="$GROUNDING_OUTPUT_ROOT/$sample_id"

    mkdir -p "$repr_output_dir" "$grounding_output_dir"

    repr_submission="$(
        sbatch --parsable \
            --job-name="repr_${sample_id}" \
            --export=ALL,IMAGE_PATH="$image_path",MASK_PATH="$mask_path",OUTPUT_DIR="$repr_output_dir" \
            "$REPR_SCRIPT"
    )"
    repr_job_id="${repr_submission%%;*}"

    grounding_submission="$(
        sbatch --parsable \
            --dependency="afterok:${repr_job_id}" \
            --job-name="ground_${sample_id}" \
            --export=ALL,RESULTS_DIR="$repr_output_dir",OUTPUT_DIR="$grounding_output_dir" \
            "$GROUNDING_SCRIPT"
    )"
    grounding_job_id="${grounding_submission%%;*}"

    printf '%s,%s,%s,%s,%s,%s,%s\n' \
        "$sample_id" \
        "$image_path" \
        "$mask_path" \
        "$repr_output_dir" \
        "$repr_job_id" \
        "$grounding_output_dir" \
        "$grounding_job_id" >> "$manifest_path"

    echo "Submitted $sample_id"
    echo "  repr job:      $repr_job_id"
    echo "  grounding job: $grounding_job_id"
done

echo "=========================================="
echo "Submitted $row_count image pipelines"
echo "Manifest: $manifest_path"
echo "End Time: $(date)"
echo "=========================================="
