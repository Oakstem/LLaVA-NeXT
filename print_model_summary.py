#!/usr/bin/env python3
"""
Print comprehensive model summary using torchinfo.
Loads the model the same way train.py does for consistency.
"""

import warnings
import torch
import transformers
from torchinfo import summary
from llava.model.builder import load_pretrained_model
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """Model configuration arguments matching train.py"""
    model_name_or_path: Optional[str] = field(default="lmms-lab/llava-onevision-qwen2-7b-ov-chat")
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_vision_select_layer: Optional[int] = field(default=-2)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_spatial_pool_mode: Optional[str] = field(default=None)
    mm_resampler_type: Optional[str] = field(default=None)


@dataclass
class TrainingArguments:
    """Training configuration arguments matching train.py"""
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2")
    local_rank: int = field(default=-1)


def print_model_summary(model, tokenizer, model_name="LLaVA Model"):
    """Print comprehensive model summary using torchinfo"""
    print("=" * 80)
    print(f"MODEL SUMMARY: {model_name}")
    print("=" * 80)
    
    # Print basic model info
    print(f"\nModel Class: {model.__class__.__name__}")
    print(f"Model Config: {model.config}")
    
    # Calculate parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print("\n" + "=" * 80)
    print("PARAMETER SUMMARY")
    print("=" * 80)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    # Print detailed module structure with parameter info
    print("\n" + "=" * 80)
    print("MODULE STRUCTURE")
    print("=" * 80)
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            num_params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if num_params > 0:
                status = "TRAINABLE" if trainable > 0 else "FROZEN"
                print(f"{name:80s} | {module.__class__.__name__:30s} | {num_params:>12,} params | {status}")
    
    # Print parameters by name with gradient status
    print("\n" + "=" * 80)
    print("PARAMETERS WITH GRADIENT STATUS")
    print("=" * 80)
    for name, param in model.named_parameters():
        grad_status = "✓ TRAINABLE" if param.requires_grad else "✗ FROZEN"
        print(f"{name:80s} | {str(param.shape):30s} | {param.numel():>12,} | {grad_status}")
    
    # Try to print torchinfo summary with forward pass
    print("\n" + "=" * 80)
    print("TORCHINFO DETAILED SUMMARY WITH FORWARD PASS")
    print("=" * 80)
    try:
        # Create dummy inputs for forward pass
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        
        print("\nCreating dummy inputs for forward pass...")
        
        # Create dummy input_ids (batch_size=1, sequence_length=128)
        batch_size = 1
        seq_length = 128
        input_ids = torch.randint(
            low=0, 
            high=tokenizer.vocab_size, 
            size=(batch_size, seq_length),
            device=device,
            dtype=torch.long
        )
        
        # Create attention mask (all ones)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        
        # Create dummy pixel_values for vision input (batch_size, num_channels, height, width)
        # SigLIP expects 384x384 images with 3 channels
        pixel_values = torch.randn(
            batch_size, 3, 384, 384,
            device=device,
            dtype=dtype
        )
        
        # Create image_sizes (for dynamic resolution)
        image_sizes = torch.tensor([[384, 384]], device=device, dtype=torch.long)
        
        print(f"  input_ids shape: {input_ids.shape}")
        print(f"  attention_mask shape: {attention_mask.shape}")
        print(f"  pixel_values shape: {pixel_values.shape}")
        print(f"  image_sizes shape: {image_sizes.shape}")
        
        print("\nRunning torchinfo summary with forward pass...")
        
        # Run torchinfo with actual inputs
        summary(
            model,
            input_data={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_sizes": image_sizes,
            },
            depth=4,
            col_names=["input_size", "output_size", "num_params", "params_percent", "trainable"],
            row_settings=["var_names"],
            verbose=1,
        )
        
        print("\n✓ Forward pass successful!")
        
    except Exception as e:
        print(f"\n✗ Could not generate torchinfo summary with forward pass: {e}")
        print("Falling back to parameter summary without forward pass...\n")
        
        try:
            summary(
                model,
                depth=4,
                col_names=["num_params", "params_percent", "trainable"],
                row_settings=["var_names"],
                verbose=1,
            )
        except Exception as e2:
            print(f"Could not generate parameter summary: {e2}")
    
    print("\n" + "=" * 80)


def main():
    """Main function to load and print model summary"""
    print("Starting model summary generation...")
    print("=" * 80)
    
    # Configuration
    model_args = ModelArguments()
    training_args = TrainingArguments()
    
    pretrained = model_args.model_name_or_path
    model_name = "llava_qwen"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_map = "auto"
    
    print(f"\nLoading model from: {pretrained}")
    print(f"Model name: {model_name}")
    print(f"Device: {device}")
    print(f"Vision tower: {model_args.vision_tower}")
    print(f"Attention implementation: {training_args.attn_implementation}")
    
    # Load model args
    llava_model_args = {
        "multimodal": True,
    }
    
    print("\nLoading pretrained model...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            pretrained, 
            None, 
            model_name, 
            device_map=device_map,
            torch_dtype="bfloat16" if training_args.bf16 else "float16" if training_args.fp16 else "float32",
            attn_implementation=training_args.attn_implementation,
            **llava_model_args
        )
    
    print(f"Model loaded successfully!")
    print(f"Max length: {max_length}")
    
    # Set vision tower if specified
    if model_args.vision_tower is not None:
        print(f"\nSetting vision tower to: {model_args.vision_tower}")
        model.config.mm_vision_tower = model_args.vision_tower
        
        # Initialize vision tower if needed
        if hasattr(model, 'get_vision_tower'):
            vision_tower = model.get_vision_tower()
            if vision_tower is not None:
                print(f"Vision tower already initialized: {vision_tower.__class__.__name__}")
            else:
                print("Initializing vision tower...")
                model.get_model().initialize_vision_modules(model_args=model_args)
                vision_tower = model.get_vision_tower()
                vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16)
                print(f"Vision tower initialized: {vision_tower.__class__.__name__}")
    
    # Print comprehensive summary
    print_model_summary(model, tokenizer, model_name=pretrained)
    
    print("\n" + "=" * 80)
    print("Model summary generation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
