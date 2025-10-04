#!/usr/bin/env python3
"""
Script to fix concurrent duplicated words and drop samples with numbered person mentions
from JSON conversation data. Example fixes: "the The" -> "the" and removal of entries
mentioning "person1", "person 1", or "man1".
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DISALLOWED_MENTION_PATTERN = re.compile(r"\b(?:person|man)[\s_#-]*\d+\b", re.IGNORECASE)

def count_people_in_sample(sample: Any) -> int:
    """
    Count the number of people in a sample by checking the 'num_people' field.
    
    Args:
        sample: The sample dictionary to check
        
    Returns:
        Number of people in the sample, or 0 if not found
    """
    if isinstance(sample, dict) and 'num_people' in sample:
        try:
            return int(sample['num_people'])
        except (ValueError, TypeError):
            logger.debug(f"Invalid num_people value: {sample['num_people']}")
            return 0
    return 0

def fix_duplicated_words(text: str) -> str:
    """
    Fix concurrent duplicated words in text.
    
    Args:
        text: Input text that may contain duplicated words
        
    Returns:
        Text with duplicated words fixed
    """
    if not text or not isinstance(text, str):
        return text
    
    # Simple approach: find repeated words separated by whitespace
    # Pattern matches: word + one or more spaces + same word (case insensitive)
    pattern = r'\b(\w+)(\s+)\1\b'
    
    # Keep applying the pattern until no more duplicates are found
    prev_text = text
    max_iterations = 10  # Prevent infinite loops
    iterations = 0
    
    while iterations < max_iterations:
        new_text = re.sub(pattern, r'\1', prev_text, flags=re.IGNORECASE)
        if new_text == prev_text:
            break
        prev_text = new_text
        iterations += 1
    
    return prev_text

def contains_disallowed_mentions(obj: Any) -> bool:
    """Recursively check if the object contains disallowed person mentions."""
    if isinstance(obj, str):
        return bool(DISALLOWED_MENTION_PATTERN.search(obj))
    if isinstance(obj, dict):
        return any(contains_disallowed_mentions(value) for value in obj.values())
    if isinstance(obj, list):
        return any(contains_disallowed_mentions(item) for item in obj)
    return False


def fix_conversation_data(data: List[Any], min_people: int = 0) -> tuple[List[Any], int, int, int]:
    """
    Fix duplicated words in conversation data and remove samples with disallowed mentions.
    
    Args:
        data: List of conversation samples
        min_people: Minimum number of people required in a sample (0 means no filtering)

    Returns:
        Tuple of (fixed_data, count_of_fixes, count_of_removed_samples, count_of_people_filtered)
    """
    fixed_data = []
    total_fixes = 0
    removed_count = 0
    people_filtered_count = 0

    for sample in data:
        # Handle the case where sample might be a string instead of dict
        if isinstance(sample, str):
            if contains_disallowed_mentions(sample):
                removed_count += 1
                logger.debug(f"Removing string sample due to disallowed mention: {sample[:100]}...")
                continue
            logger.warning(f"Skipping string sample: {sample[:100]}...")
            fixed_data.append(sample)
            continue

        if contains_disallowed_mentions(sample):
            removed_count += 1
            logger.debug("Removing sample due to disallowed mentions")
            continue

        # Filter by minimum number of people if specified
        if min_people > 0:
            num_people = count_people_in_sample(sample)
            if num_people < min_people:
                people_filtered_count += 1
                logger.debug(f"Removing sample with {num_people} people (minimum required: {min_people})")
                continue

        if not isinstance(sample, dict):
            logger.warning("Skipping non-dict sample without disallowed mentions")
            fixed_data.append(sample)
            continue

        fixed_sample = sample.copy()
        
        if 'conversations' in sample:
            fixed_conversations = []
            for conv in sample['conversations']:
                fixed_conv = conv.copy()
                if 'value' in conv and isinstance(conv['value'], str):
                    original_value = conv['value']
                    fixed_value = fix_duplicated_words(original_value)
                    fixed_conv['value'] = fixed_value
                    
                    if original_value != fixed_value:
                        total_fixes += 1
                        logger.debug(f"Fixed: '{original_value}' -> '{fixed_value}'")
                
                fixed_conversations.append(fixed_conv)
            fixed_sample['conversations'] = fixed_conversations
        
        fixed_data.append(fixed_sample)
    
    return fixed_data, total_fixes, removed_count, people_filtered_count

def process_json_file(file_path: Path, backup: bool = True, min_people: int = 0) -> tuple[int, int, int]:
    """
    Process a single JSON file to fix duplicated words and remove disallowed mentions.
    
    Args:
        file_path: Path to the JSON file
        backup: Whether to create a backup of the original file
        min_people: Minimum number of people required in a sample (0 means no filtering)
        
    Returns:
        Tuple of (number of fixes made, number of samples removed, number of people-filtered samples)
    """
    logger.info(f"Processing file: {file_path}")
    
    # Load the JSON data
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return 0, 0
    
    # Fix duplicated words
    fixed_data, num_fixes, num_removed, num_people_filtered = fix_conversation_data(data, min_people)

    if num_fixes > 0 or num_removed > 0 or num_people_filtered > 0:
        # Create backup if requested
        if backup:
            backup_path = file_path.with_suffix(f'{file_path.suffix}.backup')
            logger.info(f"Creating backup: {backup_path}")
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        # Save the fixed data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(fixed_data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Fixed {num_fixes} duplicated word instances, removed {num_removed} samples "
            f"(disallowed mentions), and filtered {num_people_filtered} samples (people count) in {file_path}"
        )
    else:
        logger.info(f"No duplicated words, disallowed mentions, or people filtering needed in {file_path}")
    
    # Log final count
    logger.info(f"Final sample count in {file_path.name}: {len(fixed_data)} samples")

    return num_fixes, num_removed, num_people_filtered

def main():
    parser = argparse.ArgumentParser(description='Fix duplicated words in JSON conversation data')
    parser.add_argument('path', nargs='?', 
                       default='/galitylab/students/alonmardi/projects/LLaVA-NeXT/training_datasets/sgl_conversation_data_20250928_172112',
                       help='Path to directory containing JSON files or specific JSON file')
    parser.add_argument('--no-backup', action='store_true', 
                       help='Do not create backup files')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be fixed without making changes')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--min-people', type=int, default=0,
                       help='Minimum number of people required in a sample (0 means no filtering)')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    path = Path(args.path)
    
    # Determine files to process
    json_files = []
    if path.is_file() and path.suffix == '.json':
        json_files = [path]
    elif path.is_dir():
        json_files = list(path.glob('*.json'))
    else:
        logger.error(f"Invalid path: {path}")
        return 1
    
    if not json_files:
        logger.error("No JSON files found to process")
        return 1
    
    total_fixes = 0
    total_removed = 0
    total_people_filtered = 0
    
    for json_file in json_files:
        if args.dry_run:
            # For dry run, just count potential fixes
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            _, num_fixes, num_removed, num_people_filtered = fix_conversation_data(data, args.min_people)
            logger.info(
                f"[DRY RUN] Would fix {num_fixes} duplicated word instances, remove {num_removed} samples "
                f"(disallowed mentions), and filter {num_people_filtered} samples (people count) in {json_file}"
            )
            total_fixes += num_fixes
            total_removed += num_removed
            total_people_filtered += num_people_filtered
        else:
            # Actually process the file
            num_fixes, num_removed, num_people_filtered = process_json_file(
                json_file, backup=not args.no_backup, min_people=args.min_people
            )
            total_fixes += num_fixes
            total_removed += num_removed
            total_people_filtered += num_people_filtered

    # Calculate total final sample count
    total_final_samples = 0
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            total_final_samples += len(data)
        except Exception as e:
            logger.warning(f"Could not count samples in {json_file}: {e}")
    
    logger.info(
        f"{'[DRY RUN] Would fix' if args.dry_run else 'Fixed'} {total_fixes} duplicated word instances, "
        f"removed {total_removed} samples (disallowed mentions), and filtered {total_people_filtered} "
        f"samples (people count) across {len(json_files)} files"
    )
    logger.info(f"Total final sample count across all files: {total_final_samples} samples")

    return 0

if __name__ == '__main__':
    exit(main())
