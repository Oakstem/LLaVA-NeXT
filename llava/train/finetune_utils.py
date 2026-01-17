#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import sys
from llava.utils import rank0_print


def check_deepspeed_compatibility():
    """
    Check if DeepSpeed is being used by examining command line arguments.
    
    Returns:
        bool: True if DeepSpeed is detected, False otherwise.
    """
    is_deepspeed_zero3 = False
    if '--deepspeed' in sys.argv:
        rank0_print("DeepSpeed detected in command line arguments. Disabling low_cpu_mem_usage and device_map for compatibility.")
        is_deepspeed_zero3 = True
    return is_deepspeed_zero3


def get_model_loading_kwargs(device_map="auto", is_deepspeed_zero3=None, **extra_kwargs):
    """
    Get model loading kwargs with proper DeepSpeed compatibility handling.
    
    Args:
        device_map (str): Device map for model loading. Defaults to "auto".
        is_deepspeed_zero3 (bool, optional): Whether DeepSpeed Zero-3 is being used.
            If None, will auto-detect by checking command line arguments.
        **extra_kwargs: Additional keyword arguments to include in model loading.
    
    Returns:
        dict: Dictionary of kwargs suitable for model loading.
    """
    if is_deepspeed_zero3 is None:
        is_deepspeed_zero3 = check_deepspeed_compatibility()
    
    model_kwargs = extra_kwargs.copy()
    
    if is_deepspeed_zero3:
        # Remove incompatible parameters for DeepSpeed Zero-3
        model_kwargs.pop("device_map", None)
    else:
        # Include low_cpu_mem_usage and device_map for non-DeepSpeed cases
        model_kwargs["low_cpu_mem_usage"] = True
        model_kwargs["device_map"] = device_map
    
    return model_kwargs