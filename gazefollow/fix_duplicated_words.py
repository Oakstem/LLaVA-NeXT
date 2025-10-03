#!/usr/bin/env python3
"""
Script to fix concurrent duplicated words in JSON conversation data.
Example fixes: "the The" -> "the", "is is" -> "is", "man man" -> "man"
"""

import json
import re
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def fix_conversation_data(data: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
    """
    Fix duplicated words in conversation data.
    
    Args:
        data: List of conversation samples
        
    Returns:
        Tuple of (fixed_data, count_of_fixes)
    """
    fixed_data = []
    total_fixes = 0
    
    for sample in data:
        # Handle the case where sample might be a string instead of dict
        if isinstance(sample, str):
            logger.warning(f"Skipping string sample: {sample[:100]}...")
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
    
    return fixed_data, total_fixes

def process_json_file(file_path: Path, backup: bool = True) -> int:
    """
    Process a single JSON file to fix duplicated words.
    
    Args:
        file_path: Path to the JSON file
        backup: Whether to create a backup of the original file
        
    Returns:
        Number of fixes made
    """
    logger.info(f"Processing file: {file_path}")
    
    # Load the JSON data
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return 0
    
    # Fix duplicated words
    fixed_data, num_fixes = fix_conversation_data(data)
    
    if num_fixes > 0:
        # Create backup if requested
        if backup:
            backup_path = file_path.with_suffix(f'{file_path.suffix}.backup')
            logger.info(f"Creating backup: {backup_path}")
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Save the fixed data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(fixed_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Fixed {num_fixes} duplicated word instances in {file_path}")
    else:
        logger.info(f"No duplicated words found in {file_path}")
    
    return num_fixes

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
    
    for json_file in json_files:
        if args.dry_run:
            # For dry run, just count potential fixes
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            _, num_fixes = fix_conversation_data(data)
            logger.info(f"[DRY RUN] Would fix {num_fixes} duplicated word instances in {json_file}")
            total_fixes += num_fixes
        else:
            # Actually process the file
            num_fixes = process_json_file(json_file, backup=not args.no_backup)
            total_fixes += num_fixes
    
    logger.info(f"{'[DRY RUN] Would fix' if args.dry_run else 'Fixed'} {total_fixes} duplicated word instances across {len(json_files)} files")
    
    return 0

if __name__ == '__main__':
    exit(main())