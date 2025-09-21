import json
import uuid
import os
import hashlib # Added for SHA256 hashing
from pathlib import Path
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from generation_utils import fix_wsl_paths

# Define file paths
# Workspace base: /mnt/d/Projects/LLaVA-NeXT
# Data base: /mnt/d/Projects/data/gazefollow/


COMBINED_CSV_PATH = r"D:\Projects\data\gazefollow\results\valid_runs\combined_description_results.csv"
COMBINED_CSV_PATH = fix_wsl_paths(COMBINED_CSV_PATH)
OUTPUT_FILE_PATH = Path(COMBINED_CSV_PATH).parent / "sgl_conversation_data.json"
# Base path for the image field in the output JSON
IMAGE_BASE_PREFIX = "train/" # Using escaped backslashes for JSON string

# Configuration for periodic saving
SAVE_INTERVAL = 1000  # Save every N processed items
TEMP_SAVE_PREFIX = "sgl_conversation_data_temp"

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

def create_conversational_data():
    """
    Generates conversational data from subject and target files
    and saves it to a single JSON file.
    """
    combined_df = pd.read_csv(COMBINED_CSV_PATH, index_col=0)

    if combined_df is None:
        print("Failed to load input data. Exiting.")
        return

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
            
        image_key = str(image_key).zfill(8)  # Ensure image_key is zero-padded to 8 digits
        # image_key is like "train/00000041/00041904.jpg"
        # subject_entry is like {"caption": "hairdresser", ...}
        subject_entry = row["source_description"]       # prefer the person description extracted from the baseline run with no steering [more elaborate]
        # fallback to the steered description if the baseline one is missing or NaN
        if subject_entry is None or pd.isna(subject_entry): 
            subject_entry = row["steered_source_description"]


        target_entry = row["steered_target_description"]

        if not isinstance(target_entry, str):
            print(f"Warning: Unexpected format for target_entry in image_key '{image_key}'. Skipping.")
            skipped_due_to_target_format += 1
            continue

        # --- Data Extraction ---
        # For subjects_data: image_key maps directly to the subject string.
        # e.g., subjects_data["train/00000041/00041904.jpg"] is "hairdresser"
        subject_text = subject_entry
        target_text = target_entry

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
        
        if image_key.startswith("train/"):
            image_suffix_from_key = image_key[len("train/"):] # "00000041/00041904.jpg"
            # image_path_corrected_slashes = image_suffix_from_key.replace("/", "\\\\")
            full_image_path = f"{IMAGE_BASE_PREFIX}{image_suffix_from_key}"
        else:
            # Fallback or warning if image_key doesn't start with "train/" as expected
            # print(f"Warning: Image key '{image_key}' does not start with 'train/'. Using raw key with prefix.")
            folder_name = int(image_key) // 1000
            folder_name_str = str(folder_name).zfill(8)
            full_image_path = Path(IMAGE_BASE_PREFIX) / folder_name_str / image_key


        conversation_item = {
            "id": image_key,  # using image_key as ID for easier traceability
            "image": str(full_image_path),
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

    # Final save
    if save_progress(output_data, processed_image_paths, is_final=True):
        # Clean up temporary files on successful final save
        cleanup_temp_files()


if __name__ == "__main__":
    create_conversational_data() 