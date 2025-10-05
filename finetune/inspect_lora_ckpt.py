import sys
import os
# Add project root to Python path to allow imports of local qwen2 module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from peft import PeftModel, PeftConfig
import torch
from yaml import warnings
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

def inspect_loaded_lora(checkpoint_path, base_model_path=None, attn_implementation="flash_attention_2", load_model=True, **kwargs):
    """Load LoRA adapter and inspect its configuration
    
    Args:
        checkpoint_path: Path to the LoRA checkpoint
        base_model_path: Path to the base model (optional)
        attn_implementation: Attention implementation to use
        load_model: If True, actually load the model. If False, just inspect config.
        **kwargs: Additional arguments
    """
    try:
        # Load PEFT config
        config = PeftConfig.from_pretrained(checkpoint_path)
        
        print("PEFT Configuration:")
        print(f"PEFT type: {config.peft_type}")
        print(f"Task type: {config.task_type}")
        print(f"Target modules: {config.target_modules}")
        print(f"LoRA rank: {config.r}")
        print(f"LoRA alpha: {config.lora_alpha}")
        print(f"LoRA dropout: {config.lora_dropout}")
        print(f"Base model: {config.base_model_name_or_path}")
        
        if base_model_path and load_model:
            print("\n" + "="*50)
            print("Loading model (this may take a while)...")
            print("="*50 + "\n")
            
            # Create offload directory
            offload_dir = os.path.join(os.path.dirname(checkpoint_path), "offload_cache")
            os.makedirs(offload_dir, exist_ok=True)
            
            custom_config = {'attn_layer_ind': -1}
            llava_model_args = {"multimodal": True}
            model_name = get_model_name_from_path(base_model_path) or "llava_qwen"
            
            # Load base model and adapter with offload support
            tokenizer, model, image_processor, max_length = load_pretrained_model(
                base_model_path,
                None,
                model_name,
                load_8bit=False,
                load_4bit=False,
                device_map="auto",
                attn_implementation=attn_implementation,
                overwrite_config=custom_config,
                offload_folder=offload_dir,
                **llava_model_args
            )

            print("Loading LoRA adapter weights...")
            model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)

            print(f"\nAdapter modules in loaded model:")
            adapter_count = 0
            for name, module in model.named_modules():
                if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                    print(f"  {name}")
                    adapter_count += 1
            
            print(f"\nTotal adapter modules: {adapter_count}")
        elif base_model_path and not load_model:
            print("\nSkipping model loading. Set load_model=True to load the full model.")
        
    except Exception as e:
        print(f"Error loading PEFT config: {e}")

# Usage
checkpoint_dir = "/mnt/d/Projects/LLaVA-NeXT/training_outputs/new/evaluated/saved_checkpoints/first_run/checkpoint-10000"
# fix for wsl paths
# from gazefollow.attn_utils import fix_wsl_paths
# checkpoint_dir = fix_wsl_paths(checkpoint_dir)

# First, just inspect the config without loading the model
print("=" * 70)
print("INSPECTING LORA CONFIGURATION (without loading model)")
print("=" * 70)
inspect_loaded_lora(checkpoint_dir, base_model_path="lmms-lab/llava-onevision-qwen2-7b-ov-chat", load_model=True)

# Uncomment below to actually load the model (requires significant memory and time)
# print("\n" + "=" * 70)
# print("LOADING MODEL WITH LORA ADAPTER")
# print("=" * 70)
# inspect_loaded_lora(checkpoint_dir, base_model_path="lmms-lab/llava-onevision-qwen2-7b-ov-chat", load_model=True)