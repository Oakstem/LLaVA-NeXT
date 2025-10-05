#!/usr/bin/env python3
"""
Dataset splitting script for LLaVA-NeXT training data.
This script splits a JSON dataset into train and validation splits and saves them in a structured directory format.
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    """
    Load dataset from JSON or JSONL file.
    
    Args:
        data_path: Path to the dataset file
        
    Returns:
        List of data samples
    """
    print(f"Loading dataset from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    data = []
    
    if data_path.endswith('.jsonl'):
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    elif data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported file format. Only .json and .jsonl are supported. Got: {data_path}")
    
    print(f"Loaded {len(data)} samples")
    return data


def split_dataset(data: List[Dict[str, Any]], 
                 train_ratio: float = 0.8, 
                 val_ratio: float = 0.2,
                 seed: int = 42) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split dataset into train and validation sets.
    
    Args:
        data: List of data samples
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data)
    """
    if abs(train_ratio + val_ratio - 1.0) > 1e-6:
        raise ValueError(f"Train ratio ({train_ratio}) + Val ratio ({val_ratio}) must equal 1.0")
    
    random.seed(seed)
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    total_samples = len(data_copy)
    train_size = int(total_samples * train_ratio)
    
    train_data = data_copy[:train_size]
    val_data = data_copy[train_size:]
    
    print(f"Split dataset: {len(train_data)} train samples, {len(val_data)} validation samples")
    return train_data, val_data


def create_output_structure(output_dir: str, dataset_name: str) -> tuple[str, str]:
    """
    Create output directory structure for split datasets.
    
    Structure:
    output_dir/
    ├── dataset_name/
    │   ├── train.json
    │   └── val.json
    
    Args:
        output_dir: Base output directory
        dataset_name: Name of the dataset
        
    Returns:
        Tuple of (train_path, val_path)
    """
    dataset_dir = Path(output_dir) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = dataset_dir / "train.json"
    val_path = dataset_dir / "val.json"
    
    return str(train_path), str(val_path)


def save_split_data(train_data: List[Dict[str, Any]], 
                   val_data: List[Dict[str, Any]],
                   train_path: str, 
                   val_path: str) -> None:
    """
    Save train and validation data to JSON files.
    
    Args:
        train_data: Training data samples
        val_data: Validation data samples
        train_path: Path to save training data
        val_path: Path to save validation data
    """
    print(f"Saving training data to {train_path}")
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saving validation data to {val_path}")
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)


def create_dataset_config(output_dir: str, 
                         dataset_name: str, 
                         train_samples: int, 
                         val_samples: int,
                         original_path: str) -> None:
    """
    Create a configuration file with dataset information.
    
    Args:
        output_dir: Base output directory
        dataset_name: Name of the dataset
        train_samples: Number of training samples
        val_samples: Number of validation samples
        original_path: Path to original dataset
    """
    config = {
        "dataset_name": dataset_name,
        "original_path": original_path,
        "splits": {
            "train": {
                "path": f"{dataset_name}/train.json",
                "samples": train_samples
            },
            "val": {
                "path": f"{dataset_name}/val.json", 
                "samples": val_samples
            }
        },
        "total_samples": train_samples + val_samples
    }
    
    config_path = Path(output_dir) / dataset_name / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset configuration saved to {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Split LLaVA-NeXT dataset into train/val splits")
    parser.add_argument("--input_path", type=str, required=True,
                       help="Path to input dataset (JSON or JSONL)")
    parser.add_argument("--output_dir", type=str, default="./datasets/splits",
                       help="Output directory for split datasets")
    parser.add_argument("--dataset_name", type=str, default=None,
                       help="Name for the dataset (defaults to input filename + timestamp)")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                       help="Ratio of data for training (default: 0.9)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                       help="Ratio of data for validation (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio - 1.0) > 1e-6:
        raise ValueError(f"Train ratio ({args.train_ratio}) + Val ratio ({args.val_ratio}) must equal 1.0")
    
    # Set dataset name if not provided
    if args.dataset_name is None:
        # Use timestamp as default dataset name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_stem = Path(args.input_path).stem
        args.dataset_name = f"{input_stem}_{timestamp}"
    
    print(f"Dataset splitting configuration:")
    print(f"  Input: {args.input_path}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Dataset name: {args.dataset_name}")
    print(f"  Train ratio: {args.train_ratio}")
    print(f"  Validation ratio: {args.val_ratio}")
    print(f"  Random seed: {args.seed}")
    print()
    
    try:
        # Load dataset
        data = load_dataset(args.input_path)
        
        # Split dataset
        train_data, val_data = split_dataset(
            data, 
            train_ratio=args.train_ratio, 
            val_ratio=args.val_ratio, 
            seed=args.seed
        )
        
        # Create output structure
        train_path, val_path = create_output_structure(args.output_dir, args.dataset_name)
        
        # Save split data
        save_split_data(train_data, val_data, train_path, val_path)
        
        # Create configuration file
        create_dataset_config(
            args.output_dir, 
            args.dataset_name, 
            len(train_data), 
            len(val_data),
            args.input_path
        )
        
        print(f"\nDataset splitting completed successfully!")
        print(f"Training data: {train_path} ({len(train_data)} samples)")
        print(f"Validation data: {val_path} ({len(val_data)} samples)")
        
    except Exception as e:
        print(f"Error during dataset splitting: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())