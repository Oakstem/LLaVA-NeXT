#!/usr/bin/env python3
"""
Export LLaVA model (base + LoRA adapters) to ONNX or PyTorch format for Netron visualization.

Usage:
    # Export full merged model to PyTorch
    ~/llava/bin/python export_model_for_netron.py \
        --base-model lmms-lab/llava-onevision-qwen2-7b-ov-chat \
        --lora-checkpoint ./training_outputs/checkpoint-1000 \
        --output-path ./exports/merged_model.pt \
        --format pt

    # Export specific component to ONNX
    ~/llava/bin/python export_model_for_netron.py \
        --base-model lmms-lab/llava-onevision-qwen2-7b-ov-chat \
        --lora-checkpoint ./training_outputs/checkpoint-1000 \
        --output-path ./exports/vision_tower.onnx \
        --format onnx \
        --component vision_tower

    # Export without LoRA (base model only)
    ~/llava/bin/python export_model_for_netron.py \
        --base-model lmms-lab/llava-onevision-qwen2-7b-ov-chat \
        --output-path ./exports/base_model.pt \
        --format pt
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import torch
from peft import PeftModel

from llava.model.builder import load_pretrained_model


def merge_lora_weights(model, lora_checkpoint_path):
    """
    Merge LoRA weights into the base model.
    
    Args:
        model: Base LLaVA model
        lora_checkpoint_path: Path to LoRA checkpoint directory
        
    Returns:
        Merged model with LoRA weights integrated
    """
    print(f"\nMerging LoRA weights from: {lora_checkpoint_path}")
    
    # Load LoRA weights using PEFT
    model = PeftModel.from_pretrained(model, lora_checkpoint_path)
    
    # Merge LoRA weights into base model
    print("Merging LoRA adapters into base model...")
    model = model.merge_and_unload()
    
    print("✓ LoRA weights merged successfully")
    return model


def export_to_pytorch(model, output_path, component=None):
    """
    Export model to PyTorch .pt format (state dict).
    
    Args:
        model: Model to export
        output_path: Path to save the .pt file
        component: Optional component to export ('vision_tower', 'projector', 'language_model')
    """
    print(f"\nExporting to PyTorch format: {output_path}")
    
    if component:
        print(f"Extracting component: {component}")
        if component == "vision_tower":
            model_to_save = model.get_vision_tower()
        elif component == "projector":
            model_to_save = model.get_model().mm_projector
        elif component == "vision_resampler":
            model_to_save = model.get_model().vision_resampler
        elif component == "language_model":
            model_to_save = model.get_model()
        else:
            raise ValueError(f"Unknown component: {component}")
    else:
        model_to_save = model
    
    # Save state dict
    state_dict = model_to_save.state_dict()
    torch.save(state_dict, output_path)
    
    # Print summary
    total_params = sum(p.numel() for p in state_dict.values())
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"✓ Model exported successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"\nYou can now visualize this model in Netron:")
    print(f"  - Online: https://netron.app (drag and drop the file)")
    print(f"  - Desktop: netron {output_path}")


def export_to_onnx(model, output_path, component=None):
    """
    Export model to ONNX format.
    
    Args:
        model: Model to export
        output_path: Path to save the .onnx file
        component: Optional component to export
    """
    print(f"\nExporting to ONNX format: {output_path}")
    
    model.eval()
    
    if component:
        print(f"Extracting component: {component}")
        if component == "vision_tower":
            module = model.get_vision_tower()
            # Create dummy image input
            dummy_input = torch.randn(1, 3, 384, 384)
            input_names = ["pixel_values"]
            output_names = ["image_features"]
            dynamic_axes = {
                "pixel_values": {0: "batch_size"},
                "image_features": {0: "batch_size"}
            }
            
        elif component == "projector":
            module = model.get_model().mm_projector
            # Dummy features from vision tower (adjust dimensions as needed)
            dummy_input = torch.randn(1, 729, 1152)  # Example: [batch, num_patches, hidden_dim]
            input_names = ["vision_features"]
            output_names = ["projected_features"]
            dynamic_axes = {
                "vision_features": {0: "batch_size", 1: "num_patches"},
                "projected_features": {0: "batch_size", 1: "num_patches"}
            }
            
        elif component == "vision_resampler":
            module = model.get_model().vision_resampler
            # Dummy features
            dummy_input = torch.randn(1, 729, 1152)
            input_names = ["vision_features"]
            output_names = ["resampled_features"]
            dynamic_axes = {
                "vision_features": {0: "batch_size"},
                "resampled_features": {0: "batch_size"}
            }
            
        else:
            raise ValueError(f"Component '{component}' not supported for ONNX export. "
                           f"Use 'vision_tower', 'projector', or 'vision_resampler'")
    else:
        # Export full model (language model part)
        print("Exporting full model - this may take a while for large models...")
        print("Note: For large models like 7B+, ONNX export may be very slow or fail.")
        print("Consider exporting specific components instead with --component flag.")
        
        module = model
        batch_size = 1
        seq_length = 128
        
        dummy_input_ids = torch.randint(0, 32000, (batch_size, seq_length))
        dummy_attention_mask = torch.ones((batch_size, seq_length))
        dummy_input = (dummy_input_ids, dummy_attention_mask)
        
        input_names = ["input_ids", "attention_mask"]
        output_names = ["logits"]
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        }
    
    # Export to ONNX
    print("Converting to ONNX (this may take a few minutes)...")
    torch.onnx.export(
        module,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
        export_params=True,
    )
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Model exported successfully")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"\nYou can now visualize this model in Netron:")
    print(f"  - Online: https://netron.app (drag and drop the file)")
    print(f"  - Desktop: netron {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export LLaVA model (base + LoRA) to ONNX or PyTorch format for Netron",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path or HuggingFace ID of base model (e.g., lmms-lab/llava-onevision-qwen2-7b-ov-chat)"
    )
    
    parser.add_argument(
        "--lora-checkpoint",
        type=str,
        default=None,
        help="Path to LoRA checkpoint directory (optional, if not provided will export base model only)"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path for exported model (.pt or .onnx)"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["pt", "onnx"],
        default="pt",
        help="Export format: 'pt' for PyTorch state dict, 'onnx' for ONNX (default: pt)"
    )
    
    parser.add_argument(
        "--component",
        type=str,
        choices=["vision_tower", "projector", "vision_resampler", "language_model", None],
        default=None,
        help="Export specific component only (optional). For ONNX, only vision_tower, projector, and vision_resampler are supported."
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="llava_qwen",
        help="Model architecture name (default: llava_qwen)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to load model on (default: cuda if available, else cpu)"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate output path extension
    expected_ext = f".{args.format}"
    if not args.output_path.endswith(expected_ext):
        print(f"Warning: Output path should end with {expected_ext}, adding it automatically")
        args.output_path += expected_ext
    
    print("=" * 80)
    print("LLaVA Model Export for Netron Visualization")
    print("=" * 80)
    print(f"\nBase model: {args.base_model}")
    if args.lora_checkpoint:
        print(f"LoRA checkpoint: {args.lora_checkpoint}")
    else:
        print("LoRA checkpoint: None (exporting base model only)")
    print(f"Output path: {args.output_path}")
    print(f"Export format: {args.format.upper()}")
    if args.component:
        print(f"Component: {args.component}")
    else:
        print("Component: Full model")
    print(f"Device: {args.device}")
    
    # Load base model
    print(f"\n{'=' * 80}")
    print("Loading base model...")
    print("=" * 80)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            args.base_model,
            None,
            args.model_name,
            device_map=args.device if args.device == "auto" else None,
            torch_dtype="bfloat16",
        )
    
    if args.device != "auto":
        model = model.to(args.device)
    
    print(f"✓ Base model loaded: {model.__class__.__name__}")
    
    # Merge LoRA weights if provided
    if args.lora_checkpoint:
        if not os.path.exists(args.lora_checkpoint):
            print(f"\nError: LoRA checkpoint not found: {args.lora_checkpoint}")
            sys.exit(1)
        
        model = merge_lora_weights(model, args.lora_checkpoint)
    
    # Move to CPU for export (ONNX export works better on CPU)
    if args.format == "onnx":
        print("\nMoving model to CPU for ONNX export...")
        model = model.cpu()
    
    # Export based on format
    print(f"\n{'=' * 80}")
    print(f"Exporting to {args.format.upper()}")
    print("=" * 80)
    
    if args.format == "pt":
        export_to_pytorch(model, args.output_path, args.component)
    elif args.format == "onnx":
        export_to_onnx(model, args.output_path, args.component)
    
    print(f"\n{'=' * 80}")
    print("Export complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
