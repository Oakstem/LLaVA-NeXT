import json
import uuid
import os
import hashlib # Added for SHA256 hashing
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import sys
# set project dir in path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from gazefollow.generation_utils import fix_wsl_paths

# Define file paths
# Workspace base: /mnt/d/Projects/LLaVA-NeXT
# Data base: /mnt/d/Projects/data/gazefollow/


COMBINED_CSV_PATH = r"gazefollow/data/combined_source_extract_patchscope_valid_20260204.csv"
# COMBINED_CSV_PATH = r"/mnt/d/Projects/data/gazefollow/results/valid_runs/combined_ppl_desc_results.csv"
COMBINED_CSV_PATH = fix_wsl_paths(COMBINED_CSV_PATH)
OUTSIDE_FRAME_TARGET_DESCRIPTION = "something or someone outside the frame"
DEFAULT_MIN_IOU_THRESHOLD_SOURCE = 0.2
DEFAULT_MAX_L2_THRESHOLD_SOURCE = 0.16
DEFAULT_MIN_IOU_THRESHOLD_TARGET = 0.0
DEFAULT_MAX_L2_THRESHOLD_TARGET = 0.12

test_set = 'test' in COMBINED_CSV_PATH.lower()
# Create output directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(COMBINED_CSV_PATH).parent / f"sgl_conversations_files"
output_dir.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE_PATH = output_dir / f"{timestamp}_sgl_conversation_data.json"

# Base path for the image field in the output JSON
IMAGE_BASE_PREFIX = "train/" if not test_set else "test2/"  # Using escaped backslashes for JSON string

# Configuration for periodic saving
SAVE_INTERVAL = 30000  # Save every N processed items
TEMP_SAVE_PREFIX = "sgl_conversation_data_temp"

def _build_error_frame(df, l2_col, iou_col, mask=None):
    errors = df[[l2_col, iou_col]].copy()
    if mask is not None:
        errors = errors[mask]
    errors = errors.dropna()
    errors = errors[np.isfinite(errors[l2_col]) & np.isfinite(errors[iou_col])]
    return errors


def _source_error_filter_mask_patchscope(df):
    mask = pd.Series([0] * len(df), index=df.index, dtype=bool)
    if "source_description" in df.columns and "steered_source_description" in df.columns:
        mask |= df["source_description"] == df["steered_source_description"]
    if "patchscope_source_description" in df.columns:
        mask |= df["patchscope_source_description"].apply(lambda x: isinstance(x, str))
    return mask


def _source_error_filter_mask(df, eligibility="metrics"):
    if eligibility == "patchscope":
        return _source_error_filter_mask_patchscope(df)
    has_metrics = all(
        col in df.columns
        for col in (
            "source_grounding_normalized_l2_error",
            "source_grounding_bbox_iou",
        )
    )
    if not has_metrics:
        return _source_error_filter_mask_patchscope(df)
    l2 = df["source_grounding_normalized_l2_error"]
    iou = df["source_grounding_bbox_iou"]
    mask = l2.notna() & iou.notna()
    mask &= np.isfinite(l2) & np.isfinite(iou)
    return mask


def _compute_histogram(values, bins):
    counts, edges = np.histogram(values, bins=bins)
    return counts, edges


def _print_histogram(name, values, bins):
    counts, edges = _compute_histogram(values, bins)
    print(f"\n{name} histogram (n={len(values)}, bins={bins}):")
    for idx, count in enumerate(counts):
        left = edges[idx]
        right = edges[idx + 1]
        print(f"  {left:.4f} - {right:.4f}: {count}")


def _percentile_threshold(values, tail_percentile, upper_tail=True):
    q = 1.0 - tail_percentile / 100.0 if upper_tail else tail_percentile / 100.0
    return float(np.quantile(values, q))


def _knee_threshold(values, prefer_upper=True):
    if len(values) < 3:
        if not len(values):
            return float("nan")
        return float(np.max(values) if prefer_upper else np.min(values))
    sorted_vals = np.sort(values)
    min_val = float(sorted_vals[0])
    max_val = float(sorted_vals[-1])
    if max_val - min_val == 0:
        return max_val
    x = np.linspace(0.0, 1.0, len(sorted_vals))
    y = (sorted_vals - min_val) / (max_val - min_val)
    diff = y - x
    idx_max = int(np.argmax(diff))
    idx_min = int(np.argmin(diff))
    if prefer_upper:
        knee_idx = idx_max if sorted_vals[idx_max] >= sorted_vals[idx_min] else idx_min
    else:
        knee_idx = idx_max if sorted_vals[idx_max] <= sorted_vals[idx_min] else idx_min
    return float(sorted_vals[knee_idx])


def _summarize_threshold_impact(label, errors, l2_col, iou_col, l2_threshold, iou_threshold):
    total = len(errors)
    if total == 0:
        print(f"{label}: no valid samples for threshold preview.")
        return
    kept_mask = (errors[l2_col] <= l2_threshold) & (errors[iou_col] >= iou_threshold)
    kept = errors[kept_mask]
    dropped = total - len(kept)
    dropped_pct = 100.0 * dropped / total
    kept_l2_mean = kept[l2_col].mean() if len(kept) else float("nan")
    kept_iou_mean = kept[iou_col].mean() if len(kept) else float("nan")
    print(
        f"{label}: total={total}, dropped={dropped} ({dropped_pct:.2f}%), "
        f"kept_mean_l2={kept_l2_mean:.4f}, kept_mean_iou={kept_iou_mean:.4f}"
    )


def _prompt_tail_percentile(default_tail_percentile):
    while True:
        value = input(
            f"Enter tail percentile cutoff (0-50, current {default_tail_percentile}%): "
        ).strip()
        if not value:
            return default_tail_percentile
        try:
            parsed = float(value)
        except ValueError:
            print("Please enter a numeric percentile.")
            continue
        if 0.0 < parsed < 50.0:
            return parsed
        print("Percentile must be between 0 and 50.")


def _select_thresholds_interactively(
    combined_df,
    tail_percentile,
    histogram_bins,
    threshold_method="percentile",
    source_eligibility="metrics",
):
    has_source_metrics = all(
        col in combined_df.columns
        for col in (
            "source_grounding_normalized_l2_error",
            "source_grounding_bbox_iou",
        )
    )
    if not has_source_metrics:
        print("Source grounding metrics not found; using default thresholds.")
        return (
            DEFAULT_MIN_IOU_THRESHOLD_SOURCE,
            DEFAULT_MAX_L2_THRESHOLD_SOURCE,
            DEFAULT_MIN_IOU_THRESHOLD_TARGET,
            DEFAULT_MAX_L2_THRESHOLD_TARGET,
        )
    fallback_mask = pd.Series([1] * len(combined_df), index=combined_df.index)
    source_mask = _source_error_filter_mask(combined_df, source_eligibility)
    source_errors = _build_error_frame(
        combined_df,
        "source_grounding_normalized_l2_error",
        "source_grounding_bbox_iou",
        mask=source_mask,
    )
    has_target_metrics = all(
        col in combined_df.columns
        for col in (
            "target_grounding_normalized_l2_error",
            "target_grounding_bbox_iou",
        )
    )
    target_errors = None
    if has_target_metrics:
        target_mask = combined_df.get("in_or_out", fallback_mask) == 1
        target_errors = _build_error_frame(
            combined_df,
            "target_grounding_normalized_l2_error",
            "target_grounding_bbox_iou",
            mask=target_mask,
        )

    if source_errors.empty or (has_target_metrics and target_errors is not None and target_errors.empty):
        print("Not enough error samples to auto-select thresholds; using defaults.")
        return (
            DEFAULT_MIN_IOU_THRESHOLD_SOURCE,
            DEFAULT_MAX_L2_THRESHOLD_SOURCE,
            DEFAULT_MIN_IOU_THRESHOLD_TARGET,
            DEFAULT_MAX_L2_THRESHOLD_TARGET,
        )

    _print_histogram(
        "Source L2 error",
        source_errors["source_grounding_normalized_l2_error"].values,
        histogram_bins,
    )
    _print_histogram(
        "Source IOU error",
        source_errors["source_grounding_bbox_iou"].values,
        histogram_bins,
    )
    if has_target_metrics and target_errors is not None:
        _print_histogram(
            "Target L2 error (in-frame)",
            target_errors["target_grounding_normalized_l2_error"].values,
            histogram_bins,
        )
        _print_histogram(
            "Target IOU error (in-frame)",
            target_errors["target_grounding_bbox_iou"].values,
            histogram_bins,
        )
    else:
        print("\nTarget grounding metrics not found; skipping target histograms.")

    current_tail = tail_percentile
    current_method = threshold_method
    while True:
        if current_method == "knee":
            source_l2_threshold = _knee_threshold(
                source_errors["source_grounding_normalized_l2_error"].values
            )
            source_iou_threshold = 1.0 - _knee_threshold(
                1.0 - source_errors["source_grounding_bbox_iou"].values
            )
            if has_target_metrics and target_errors is not None:
                target_l2_threshold = _knee_threshold(
                    target_errors["target_grounding_normalized_l2_error"].values
                )
                target_iou_threshold = 1.0 - _knee_threshold(
                    1.0 - target_errors["target_grounding_bbox_iou"].values
                )
            else:
                target_l2_threshold = DEFAULT_MAX_L2_THRESHOLD_TARGET
                target_iou_threshold = DEFAULT_MIN_IOU_THRESHOLD_TARGET
        else:
            source_l2_threshold = _percentile_threshold(
                source_errors["source_grounding_normalized_l2_error"].values,
                current_tail,
                upper_tail=True,
            )
            source_iou_threshold = _percentile_threshold(
                source_errors["source_grounding_bbox_iou"].values,
                current_tail,
                upper_tail=False,
            )
            if has_target_metrics and target_errors is not None:
                target_l2_threshold = _percentile_threshold(
                    target_errors["target_grounding_normalized_l2_error"].values,
                    current_tail,
                    upper_tail=True,
                )
                target_iou_threshold = _percentile_threshold(
                    target_errors["target_grounding_bbox_iou"].values,
                    current_tail,
                    upper_tail=False,
                )
            else:
                target_l2_threshold = DEFAULT_MAX_L2_THRESHOLD_TARGET
                target_iou_threshold = DEFAULT_MIN_IOU_THRESHOLD_TARGET

        print("\n--- Auto-threshold Preview ---")
        if current_method == "knee":
            print("Method: knee detection")
        else:
            print(f"Method: percentile (tail={current_tail}%)")
        print(
            f"Source thresholds: max_l2 <= {source_l2_threshold:.4f}, "
            f"min_iou >= {source_iou_threshold:.4f}"
        )
        _summarize_threshold_impact(
            "Source preview",
            source_errors,
            "source_grounding_normalized_l2_error",
            "source_grounding_bbox_iou",
            source_l2_threshold,
            source_iou_threshold,
        )
        if has_target_metrics and target_errors is not None:
            print(
                f"Target thresholds: max_l2 <= {target_l2_threshold:.4f}, "
                f"min_iou >= {target_iou_threshold:.4f}"
            )
            _summarize_threshold_impact(
                "Target preview (in-frame)",
                target_errors,
                "target_grounding_normalized_l2_error",
                "target_grounding_bbox_iou",
                target_l2_threshold,
                target_iou_threshold,
            )
        else:
            print("Target thresholds: skipping preview (metrics unavailable).")

        accept = input("Accept these thresholds? (y/n): ").strip().lower()
        if accept == "y":
            return (
                source_iou_threshold,
                source_l2_threshold,
                target_iou_threshold,
                target_l2_threshold,
            )
        current_tail = _prompt_tail_percentile(current_tail)
        current_method = "percentile"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create SGL conversations with optional auto-thresholding."
    )
    parser.add_argument(
        "--auto-threshold",
        action="store_true",
        help="Auto-select L2/IOU thresholds from error histograms.",
    )
    parser.add_argument(
        "--tail-percentile",
        type=float,
        default=2.0,
        help="Tail percentile to drop for L2 (upper) and IOU (lower) errors.",
    )
    parser.add_argument(
        "--threshold-method",
        choices=("percentile", "knee"),
        default="percentile",
        help="Method to select thresholds when auto-thresholding.",
    )
    parser.add_argument(
        "--source-eligibility",
        choices=("metrics", "patchscope"),
        default="metrics",
        help="Which rows are eligible for source error filtering.",
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=40,
        help="Number of bins to use for histograms when auto-thresholding.",
    )
    return parser.parse_args()

def load_json_data(file_path):
    """Loads data from a JSON file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        return None


def save_progress(output_data, processed_count, is_final=False):
    """Save progress to a temporary or final file."""
    if is_final:
        save_path = OUTPUT_FILE_PATH
    else:
        save_path = Path(OUTPUT_FILE_PATH).parent / f"{TEMP_SAVE_PREFIX}_{processed_count}.json"
    
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        if is_final:
            print(f"Successfully created final conversational data at: {os.path.abspath(save_path)}")
        else:
            print(f"Progress saved: {processed_count} items at {save_path}")
        return True
    except (IOError, Exception) as e:
        print(f"Error saving to {save_path}: {e}")
        return False


def cleanup_temp_files():
    """Remove temporary save files."""
    temp_pattern = f"{TEMP_SAVE_PREFIX}_*.json"
    temp_dir = Path(OUTPUT_FILE_PATH).parent
    
    for temp_file in temp_dir.glob(temp_pattern):
        try:
            temp_file.unlink()
            # print(f"Cleaned up temporary file: {temp_file}")
        except Exception as e:
            print(f"Warning: Could not remove temporary file {temp_file}: {e}")


def find_latest_temp_save():
    """Find the latest temporary save file to resume from."""
    temp_dir = Path(OUTPUT_FILE_PATH).parent
    temp_pattern = f"{TEMP_SAVE_PREFIX}_*.json"
    temp_files = list(temp_dir.glob(temp_pattern))
    
    if not temp_files:
        return None, 0
    
    # Extract the count from filename and find the latest
    latest_file = None
    latest_count = 0
    
    for temp_file in temp_files:
        try:
            # Extract count from filename like "sgl_conversation_data_temp_1000.json"
            count_str = temp_file.stem.split('_')[-1]
            count = int(count_str)
            if count > latest_count:
                latest_count = count
                latest_file = temp_file
        except (ValueError, IndexError):
            continue
    
    return latest_file, latest_count

def create_conversational_data(
    min_iou_threshold_source=DEFAULT_MIN_IOU_THRESHOLD_SOURCE,
    max_l2_threshold_source=DEFAULT_MAX_L2_THRESHOLD_SOURCE,
    min_iou_threshold_target=DEFAULT_MIN_IOU_THRESHOLD_TARGET,
    max_l2_threshold_target=DEFAULT_MAX_L2_THRESHOLD_TARGET,
    auto_threshold=False,
    tail_percentile=2.0,
    threshold_method="percentile",
    source_eligibility="metrics",
    histogram_bins=40,
):
    """
    Generates conversational data from subject and target files
    and saves it to a single JSON file.
    """
    combined_df = pd.read_csv(COMBINED_CSV_PATH, index_col=0)

    if combined_df is None:
        print("Failed to load input data. Exiting.")
        return
    has_target_metrics = all(
        col in combined_df.columns
        for col in (
            "target_grounding_normalized_l2_error",
            "target_grounding_bbox_iou",
        )
    )
    has_source_metrics = all(
        col in combined_df.columns
        for col in (
            "source_grounding_normalized_l2_error",
            "source_grounding_bbox_iou",
        )
    )
    if has_source_metrics:
        source_filter_mask = _source_error_filter_mask(combined_df, source_eligibility)
    else:
        source_filter_mask = pd.Series(
            [False] * len(combined_df), index=combined_df.index, dtype=bool
        )
    source_filter_eligible = int(source_filter_mask.sum())

    if auto_threshold:
        (
            min_iou_threshold_source,
            max_l2_threshold_source,
            min_iou_threshold_target,
            max_l2_threshold_target,
        ) = _select_thresholds_interactively(
            combined_df,
            tail_percentile,
            histogram_bins,
            threshold_method,
            source_eligibility,
        )

    print(
        "Using thresholds: "
        f"source_max_l2={max_l2_threshold_source:.4f}, "
        f"source_min_iou={min_iou_threshold_source:.4f}, "
        f"target_max_l2={max_l2_threshold_target:.4f}, "
        f"target_min_iou={min_iou_threshold_target:.4f}"
    )
    print(f"Source eligibility mode: {source_eligibility}")
    print(f"Source error filter eligible rows: {source_filter_eligible}")
    if not has_source_metrics:
        print("Source grounding metrics not found; source error filtering will be skipped.")
    if not has_target_metrics:
        print("Target grounding metrics not found; target error filtering will be skipped.")

    # Check for existing temp files to resume from
    latest_temp_file, resumed_count = find_latest_temp_save()
    output_data = []
    
    if latest_temp_file:
        print(f"Found temporary save file: {latest_temp_file}")
        resume_choice = input(f"Resume from {resumed_count} processed items? (y/n): ").lower().strip()
        
        if resume_choice == 'y':
            try:
                with open(latest_temp_file, 'r', encoding='utf-8') as f:
                    output_data = json.load(f)
                print(f"Resumed from {len(output_data)} items")
            except Exception as e:
                print(f"Error loading temp file, starting fresh: {e}")
                output_data = []
                resumed_count = 0
        else:
            resumed_count = 0

    # Iterate through the image paths found in subjects_data
    # Assumes that most images in subjects_data will also be in targets_data
    processed_image_paths = len(output_data)  # Start with already processed count
    skipped_due_to_missing_target = 0
    skipped_due_to_missing_data = 0
    skipped_due_to_target_format = 0
    skipped_due_to_large_source_error = 0
    skipped_due_to_large_target_error = 0

    # Create progress bar
    total_items = len(combined_df)
    pbar = tqdm(combined_df.iterrows(), 
                total=total_items, 
                desc="Processing conversations",
                initial=resumed_count)

    # Skip already processed items if resuming
    items_to_skip = resumed_count
    current_index = 0

    for image_key, row in pbar:
        current_index += 1
        
        # Skip items if resuming from a checkpoint
        if current_index <= items_to_skip:
            continue
        
        eligible_for_source_filter = bool(source_filter_mask.iloc[current_index - 1])
        image_key = str(image_key).zfill(8)  # Ensure image_key is zero-padded to 8 digits
        # image_key is like "train/00000041/00041904.jpg"
        if eligible_for_source_filter:
            if row['source_grounding_bbox_iou'] < min_iou_threshold_source or row['source_grounding_normalized_l2_error'] > max_l2_threshold_source:
                skipped_due_to_large_source_error += 1
                continue

        # subject_entry is like {"caption": "hairdresser", ...}
        patchscope_subject_entry = row.get("patchscope_source_description", None)
        if patchscope_subject_entry is not None and isinstance(patchscope_subject_entry, str):
            # use the patchscope extracted subject description if available
            subject_entry = patchscope_subject_entry
        else:
            subject_entry = row["source_description"]       # prefer the person description extracted from the baseline run with no steering [more elaborate]
        # fallback to the steered description if the baseline one is missing or NaN
        if subject_entry is None or pd.isna(subject_entry): 
            subject_entry = row["steered_source_description"]

        if pd.isna(subject_entry) or not isinstance(subject_entry, str):
            # print(f"Warning: Unexpected format for subject_entry in image_key '{image_key}'. Skipping.")
            skipped_due_to_missing_data += 1
            continue

        patchscope_target_entry = row.get("patchscope_target_description", None)
        if patchscope_target_entry is not None and isinstance(patchscope_target_entry, str):
            if has_target_metrics:
                if row['target_grounding_normalized_l2_error'] < max_l2_threshold_target and row['target_grounding_bbox_iou'] > min_iou_threshold_target:
                    # use the patchscope extracted target description if available
                    target_entry = patchscope_target_entry
                elif row['in_or_out'] == 1:
                    skipped_due_to_large_target_error += 1
                    continue
            else:
                target_entry = patchscope_target_entry
        else:
            target_entry = row["steered_target_description"]    # prefer the steered target description
        # fallback to the original target description if the steered one is missing or NaN
        if pd.isna(target_entry) or not isinstance(target_entry, str):
            if pd.isna(row["target_description"]) or not isinstance(row["target_description"], str):
                skipped_due_to_missing_target += 1
                continue
            else:
                target_entry = row["target_description"]
        
        if row['in_or_out'] == 0:
            target_entry = OUTSIDE_FRAME_TARGET_DESCRIPTION
        if not isinstance(target_entry, str):
            print(f"Warning: Unexpected format for target_entry in image_key '{image_key}'. Skipping.")
            skipped_due_to_target_format += 1
            continue

        # --- Data Extraction ---
        # For subjects_data: image_key maps directly to the subject string.
        # e.g., subjects_data["train/00000041/00041904.jpg"] is "hairdresser"
        subject_text = subject_entry.strip()
        target_text = target_entry.strip()

        # strip of any symbols such as quotes, periods, etc.
        if isinstance(subject_text, str):
            subject_text = subject_text.strip().strip('\"\'.,;:!?')
        if isinstance(target_text, str):
            target_text = target_text.strip().strip('\"\'.,;:!?')


        if not all([subject_text, target_text]):
            # print(f"Warning: Missing caption or gaze target description for '{image_key}'. Skipping.")
            # if not subject_text: print(f"  Missing 'caption' in subject_entry for {image_key}")
            # if not target_text: print(f"  Missing 'gaze target description' in target_entry for {image_key}")
            skipped_due_to_missing_data +=1
            continue
        # --- End Data Extraction ---

        # --- Process target_text (lowercase and add "the" if needed) ---
        processed_target = target_text.lower()
        
        # Define prefixes that make adding "the " redundant or grammatically incorrect
        # Common articles and possessive pronouns
        prefixes_to_avoid_adding_the_before = (
            "the ", "a ", "an ", 
            "my ", "your ", "his ", "her ", "its ", "our ", "their "
            # Consider adding other determiners if necessary e.g. "some ", "any ", "this ", "that ", etc.
        )
        
        should_add_the_prefix = True
        for prefix in prefixes_to_avoid_adding_the_before:
            if processed_target.startswith(prefix):
                should_add_the_prefix = False
                break
        
        if should_add_the_prefix:
            processed_target = "the " + processed_target
        # --- End Process target_text ---

        
        # Construct the image path for the output JSON (respecting user's change to forward slashes)
        # input image_key: "train/00000041/00041904.jpg"
        # output full_image_path: "gazefollow\\train\\00000041\\00041904.jpg"
        
        # The image_key from the JSON is "train/folder/file.jpg".
        # We want to replace the first "train/" with IMAGE_BASE_PREFIX.
        # IMAGE_BASE_PREFIX is "gazefollow\\train\\"
        
        # Corrected image path logic:
        # The image_key is like "train/00000041/00041904.jpg"
        # We need to get the part after "train/", which is "00000041/00041904.jpg"
        # And then replace its slashes.
        
        if image_key.startswith("train/") or image_key.startswith("test2/"):
            image_suffix_from_key = image_key[len("train/"):] # "00000041/00041904.jpg"
            # image_path_corrected_slashes = image_suffix_from_key.replace("/", "\\\\")
            full_image_path = f"{IMAGE_BASE_PREFIX}{image_suffix_from_key}"
        else:
            # Fallback or warning if image_key doesn't start with "train/" as expected
            # print(f"Warning: Image key '{image_key}' does not start with 'train/'. Using raw key with prefix.")
            folder_name = int(image_key) // 1000
            folder_name_str = str(folder_name).zfill(8)
            full_image_path = Path(IMAGE_BASE_PREFIX) / folder_name_str / image_key


        # Extract num_people from the row
        num_people = row.get("num_people", None)
        if pd.isna(num_people):
            num_people = None
        elif isinstance(num_people, (int, float)):
            num_people = int(num_people)
        
        conversation_item = {
            "id": image_key,  # using image_key as ID for easier traceability
            "image": str(full_image_path),
            "num_people": num_people,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\nDescribe where the {subject_text} is looking at"
                },
                {
                    "from": "gpt",
                    "value": f"the {subject_text} is looking at {processed_target} "
                }
            ]
        }
        output_data.append(conversation_item)
        processed_image_paths += 1
        
        # Update progress bar with current stats
        pbar.set_postfix({
            'processed': processed_image_paths,
            'skipped': skipped_due_to_missing_data,
            'total_data': len(output_data)
        })
        
        # Periodic saving
        if processed_image_paths % SAVE_INTERVAL == 0:
            save_progress(output_data, processed_image_paths, is_final=False)

    pbar.close()

    if not output_data:
        print("No data was processed successfully. Output file will not be created.")
        return
    
    print(f"--- Processing Summary ---")
    print(f"Total image keys in dataframe: {len(combined_df)}")
    print(f"Successfully processed items: {processed_image_paths}")
    print(f"Skipped (target key not in target file): {skipped_due_to_missing_target}")
    print(f"Skipped (missing caption/target description): {skipped_due_to_missing_data}")
    print(f"Skipped (target value was simple string or unexpected format): {skipped_due_to_target_format}")
    print(f"Skipped (large source grounding error): {skipped_due_to_large_source_error}")
    print(f"Skipped (large target grounding error): {skipped_due_to_large_target_error}")
    print(f"Source error filter eligible rows: {source_filter_eligible}")

    # Final save
    if save_progress(output_data, processed_image_paths, is_final=True):
        # Clean up temporary files on successful final save
        cleanup_temp_files()


if __name__ == "__main__":
    args = parse_args()
    create_conversational_data(
        auto_threshold=args.auto_threshold,
        tail_percentile=args.tail_percentile,
        threshold_method=args.threshold_method,
        source_eligibility=args.source_eligibility,
        histogram_bins=args.histogram_bins,
    )
