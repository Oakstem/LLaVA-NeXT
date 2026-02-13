# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import ast
import copy
import os
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List, Any, Tuple
from PIL import Image, ImageFile
from packaging import version
import numpy as np
import warnings
import time
import random
import yaml
import math
import re
import torch
import wandb

import transformers
import tokenizers
import deepspeed

from transformers import AutoConfig
from torch.utils.data import Dataset
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, tokenizer_image_token
from llava.utils import rank0_print, process_video_with_pyav, process_video_with_decord
from llava.model.builder import load_pretrained_model
from typing import Dict, Optional, Sequence, List, Any, Tuple
from gazefollow.roi_contrastive_utils import (
    build_focus_phrase_token_ids,
    build_roi_candidate_metadata,
    build_roi_gaze_metadata,
    load_roi_candidate_lookup,
)


def wandb_log(training_args, metrics_dict, step=None):
    """Safely log metrics to wandb if available and initialized."""
    if not (training_args.report_to and "wandb" in training_args.report_to):
        return
    
    if wandb.run is not None:
        if step is not None:
            metrics_dict["step"] = step
        wandb.log(metrics_dict)


def wandb_summary(training_args, summary_dict):
    """Safely update wandb summary if available and initialized."""
    if not (training_args.report_to and "wandb" in training_args.report_to):
        return

    if training_args.local_rank not in (0, -1):
        return

    if wandb.run is not None:
        wandb.run.summary.update(summary_dict)


# Import custom evaluation functions
try:
    from evaluate_model import (
        evaluate_dataset_for_training,
        determine_template,
    )
    CUSTOM_EVAL_AVAILABLE = True
except ImportError:
    CUSTOM_EVAL_AVAILABLE = False
    print("Warning: evaluate_model not available for custom evaluation")

torch.multiprocessing.set_sharing_strategy("file_system")

ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"})

    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    )
    # deciding which part of the multimodal model to tune, will overwrite other previous settings

    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.0)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)

    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    s2: Optional[bool] = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)


    mm_newline_position: Optional[str] = field(default="grid")
    delay_load: Optional[bool] = field(default=True)
    add_faster_video: Optional[bool] = field(default=False)
    faster_token_stride: Optional[int] = field(default=10)



@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    # Split dataset parameters
    use_split_dataset: bool = field(default=False, metadata={"help": "Whether to use pre-split train/val datasets"})
    split_dataset_dir: Optional[str] = field(default=None, metadata={"help": "Directory containing split datasets (with train.json and val.json)"})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "Name of the dataset when using split datasets"})
    
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)
    image_processor: Optional[Any] = field(default=None, metadata={"help": "Image processor for processing images"})
    
    # Evaluation parameters
    eval_split_ratio: float = field(default=0.0, metadata={"help": "Ratio of data to use for evaluation (e.g., 0.2 for 20%)"})
    enable_evaluation: bool = field(default=True, metadata={"help": "Whether to enable evaluation during training"})
    eval_data_path: Optional[str] = field(default=None, metadata={"help": "Optional separate evaluation data path. If not provided, will split from training data."})
    roi_candidates_csv: Optional[str] = field(default=None, metadata={"help": "Optional Stage-1b ROI candidates CSV path."})
    roi_max_positives: int = field(default=1, metadata={"help": "Maximum positive ROI candidates per sample (GT-gaze positive only)."})
    roi_max_negatives: int = field(default=8, metadata={"help": "Maximum negative ROI candidates per sample."})
    roi_positive_radius_ratio: float = field(default=0.08, metadata={"help": "Radius ratio around GT gaze point used to build the positive ROI box."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_full_finetune: bool = field(default=False, metadata={"help": "If true, train mm_projector base weights directly even when LoRA is enabled."})
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=False)
    gradient_checkpointing_kwargs: Optional[dict] = field(default_factory=lambda: {"use_reentrant": False})
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})
    bf16: bool = field(default=True, metadata={"help": "Whether to use bf16 training."})
    fp16: bool = field(default=False, metadata={"help": "Whether to use fp16 training."})
    
    # Evaluation parameters
    eval_steps: Optional[int] = field(default=5000, metadata={"help": "Number of training steps between evaluations. If None, uses evaluation_strategy."})
    evaluation_strategy: str = field(default="steps", metadata={"help": "Evaluation strategy: 'no', 'steps', 'epoch'."})
    eval_accumulation_steps: Optional[int] = field(default=None, metadata={"help": "Number of predictions steps to accumulate before moving tensors to CPU."})
    
    # Custom evaluation parameters (from evaluate_model.py)
    use_custom_eval: bool = field(default=False, metadata={"help": "Use custom evaluation from evaluate_model.py"})
    eval_max_new_tokens: int = field(default=128, metadata={"help": "Max new tokens for evaluation generation"})
    eval_limit: Optional[int] = field(default=5, metadata={"help": "Limit number of samples for custom evaluation (None for no limit)"})
    no_loss: bool = field(default=False, metadata={"help": "Disable loss calculation in evaluation"})
    focus_loss_after_looking: bool = field(default=True, metadata={"help": "Focus loss on tokens after 'looking at' phrase"})
    focus_loss_phrase: str = field(default="looking at", metadata={"help": "Phrase to focus loss calculation"})
    focus_loss_threshold: float = field(default=5.0, metadata={"help": "Maximum loss when focus phrase not found"})
    roi_contrastive_enable: bool = field(default=False, metadata={"help": "Enable ROI-text contrastive loss during training."})
    roi_contrastive_weight: float = field(default=0.1, metadata={"help": "Base weight for ROI contrastive loss."})
    roi_contrastive_temperature: float = field(default=0.07, metadata={"help": "Temperature for ROI contrastive InfoNCE."})
    roi_contrastive_dim: int = field(default=512, metadata={"help": "Shared embedding dimension for ROI contrastive heads."})
    roi_contrastive_phrase: str = field(default="looking at", metadata={"help": "Phrase used to locate the target span for ROI contrastive pooling."})
    roi_contrastive_warmup_ratio: float = field(default=0.1, metadata={"help": "Warmup ratio for ROI contrastive loss weight based on max_steps."})
    roi_contrastive_radius_ratio: float = field(default=0.08, metadata={"help": "ROI radius as a fraction of vision patch-grid width."})
    roi_contrastive_oof_enable: bool = field(default=False, metadata={"help": "Enable outside-of-frame (OOF) text negative in ROI contrastive loss."})
    roi_contrastive_oof_text: str = field(
        default="looking at someone or something outside the frame",
        metadata={"help": "Fixed OOF negative phrase used in ROI contrastive loss."},
    )
    roi_contrastive_oof_weight: float = field(default=0.5, metadata={"help": "Weight of the OOF auxiliary term inside ROI contrastive loss."})
    roi_pos_embed_dim: int = field(default=64, metadata={"help": "Hidden dimension for ROI position MLP."})


# @dataclass
# class EvaluationArguments:
#     eval_num_processes: int = field(default=1)
#     task_names: str = field(default=None)
#     model: str = field(default="llava")
#     model_args: Optional[str] = field(default=None)
#     num_fewshot: Optional[int] = field(default=None)
#     batch_size: int = field(default=1)
#     device: Optional[str] = field(default=None)
#     limit: Optional[int] = field(default=None)
#     check_integrity: Optional[bool] = field(default=False)
#     show_task_to_terminal: Optional[bool] = field(default=False)
#     log_samples: Optional[bool] = field(default=True)
#     gen_kwargs: Optional[str] = field(default="")
#     log_samples_suffix: Optional[str] = field(default="")
#     output_path: Optional[str] = field(default="./logs/")


def _infer_zero_stage(param) -> Optional[int]:
    """Best-effort extraction of the ZeRO stage from a parameter."""
    stage_like = getattr(param, "ds_zero_stage", None)
    if stage_like is not None:
        try:
            return int(stage_like)
        except (TypeError, ValueError):
            pass

    for attr_name in ("ds_zero_config",):
        zero_cfg = getattr(param, attr_name, None)
        if zero_cfg is not None:
            if isinstance(zero_cfg, dict):
                stage_like = zero_cfg.get("stage")
            else:
                stage_like = getattr(zero_cfg, "stage", None)
            if stage_like is not None:
                try:
                    return int(stage_like)
                except (TypeError, ValueError):
                    pass

    ds_config = getattr(param, "ds_config", None)
    if ds_config is not None:
        for attr_name in ("zero_config", "zero_optimization"):
            zero_cfg = getattr(ds_config, attr_name, None)
            if zero_cfg is None:
                continue
            if isinstance(zero_cfg, dict):
                stage_like = zero_cfg.get("stage")
            else:
                stage_like = getattr(zero_cfg, "stage", None)
            if stage_like is not None:
                try:
                    return int(stage_like)
                except (TypeError, ValueError):
                    continue
    return None


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    zero_stage = _infer_zero_stage(param)
    should_gather = hasattr(param, "ds_id") and (zero_stage is None or zero_stage == 3)
    if zero_stage in (1, 2):
        should_gather = False

    warn_unavailable = hasattr(param, "ds_status") and param.ds_status == ZeroParamStatus.NOT_AVAILABLE

    if should_gather:
        if warn_unavailable and not ignore_status:
            logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return



PREDEFINED_TUNABLE_PARTS = {
    "mm_mlp_adapter",
    "mm_vision_resampler",
    "mm_vision_tower",
    "mm_projector",
    "mm_language_model",
    "lm_head",
}


def _split_tunable_specs(parts: Optional[str]) -> List[str]:
    if not parts:
        return []
    tokens: List[str] = []
    current: List[str] = []
    depth_brace = depth_bracket = depth_paren = 0
    for char in parts:
        if char == ',' and depth_brace == depth_bracket == depth_paren == 0:
            token = ''.join(current).strip()
            if token:
                tokens.append(token)
            current = []
            continue
        if char == '{':
            depth_brace += 1
        elif char == '}':
            depth_brace = max(depth_brace - 1, 0)
        elif char == '[':
            depth_bracket += 1
        elif char == ']':
            depth_bracket = max(depth_bracket - 1, 0)
        elif char == '(':
            depth_paren += 1
        elif char == ')':
            depth_paren = max(depth_paren - 1, 0)
        current.append(char)
    token = ''.join(current).strip()
    if token:
        tokens.append(token)
    return tokens


def _expand_index_selector(selector: str, length: int) -> List[int]:
    selector = selector.strip()
    if ':' not in selector:
        index = int(selector)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(f"Index {selector} out of bounds for length {length}")
        return [index]

    parts = selector.split(':')
    if len(parts) > 3:
        raise ValueError(f"Unsupported slice selector '{selector}'")
    while len(parts) < 3:
        parts.append('')
    start_str, stop_str, step_str = parts
    start = int(start_str) if start_str else None
    stop = int(stop_str) if stop_str else None
    step = int(step_str) if step_str else None

    full_range = list(range(length))
    indices = full_range[slice(start, stop, step)]
    if not isinstance(indices, list):
        indices = list(indices)
    if not indices:
        raise ValueError(f"Slice '{selector}' resolved to empty selection for length {length}")
    return [int(i) for i in indices]


def _expand_module_spec(model: torch.nn.Module, spec: str, module_map: Dict[str, torch.nn.Module]) -> List[str]:
    pending = [spec.strip()]
    expanded: List[str] = []

    while pending:
        current = pending.pop()
        if not current:
            continue

        brace_match = re.search(r"\{([^{}]+)\}", current)
        if brace_match:
            choices = [option.strip() for option in brace_match.group(1).split(',') if option.strip()]
            prefix = current[: brace_match.start()]
            suffix = current[brace_match.end() :]
            if not choices:
                rank0_print(f"Warning: Empty brace expansion in spec '{spec}'")
            for choice in choices:
                pending.append(f"{prefix}{choice}{suffix}")
            continue

        bracket_match = re.search(r"\[([^\[\]]+)\]", current)
        if bracket_match:
            prefix = current[: bracket_match.start()]
            selector = bracket_match.group(1)
            suffix = current[bracket_match.end() :]

            base_path = prefix.rstrip('.')
            suffix = suffix.lstrip('.')

            container = module_map.get(base_path)
            length = None
            available_indices = None
            if container is not None and hasattr(container, '__len__'):
                try:
                    length = len(container)
                    available_indices = set(range(length))
                except TypeError:
                    length = None

            if length is None:
                prefix_key = f"{base_path}." if base_path else ""
                child_indices = set()
                for name in module_map.keys():
                    if not prefix_key and name == "":
                        continue
                    if not name.startswith(prefix_key):
                        continue
                    remainder = name[len(prefix_key):]
                    if not remainder:
                        continue
                    child = remainder.split('.', 1)[0]
                    if child.isdigit():
                        child_indices.add(int(child))
                if not child_indices:
                    rank0_print(f"Warning: Unable to resolve module path '{base_path}' in spec '{spec}'")
                    continue
                length = max(child_indices) + 1
                available_indices = child_indices

            try:
                indices = _expand_index_selector(selector, length)
            except Exception as exc:
                rank0_print(f"Warning: Unable to resolve selector '{selector}' in spec '{spec}': {exc}")
                continue

            valid_indices = []
            for index in indices:
                if index not in available_indices:
                    rank0_print(f"Warning: Index {index} not available under '{base_path}' in spec '{spec}'")
                    continue
                valid_indices.append(index)

            if not valid_indices:
                rank0_print(f"Warning: Selector '{selector}' produced no valid indices for '{base_path}' in spec '{spec}'")
                continue

            for index in valid_indices:
                parts = []
                if base_path:
                    parts.append(base_path)
                parts.append(str(index))
                if suffix:
                    parts.append(suffix)
                pending.append('.'.join(parts))
            continue

        expanded.append(current)

    return sorted(set(expanded))


def resolve_custom_tunable_modules(model: torch.nn.Module, tunable_parts: Sequence[str]) -> List[str]:
    custom_specs = [part for part in tunable_parts if part not in PREDEFINED_TUNABLE_PARTS]
    if not custom_specs:
        return []
    module_map = dict(model.named_modules())
    module_map.setdefault('', model)
    resolved: List[str] = []
    for spec in custom_specs:
        resolved.extend(_expand_module_spec(model, spec, module_map))
    return sorted(set(resolved))


def find_all_linear_names(model, mm_tunable_parts=None, exclude_mm_projector: bool = False):
    cls = torch.nn.Linear
    lora_module_names = set()

    tunable_parts_list = _split_tunable_specs(mm_tunable_parts)

    if not tunable_parts_list:
        print("No mm_tunable_parts specified, will tune all linear layers in the model.")
        for name, module in model.named_modules():
            if isinstance(module, cls):
                if "mm_projector" in name:
                    if exclude_mm_projector:
                        continue
                    lora_module_names.add(name)
                else:
                    names = name.split('.')
                    lora_module_names.add(names[-1])
        print(f"mm_tunable_parts: {mm_tunable_parts}")
        print(f"Found these linear module names for LoRA: {lora_module_names}")
        return list(lora_module_names)

    custom_module_targets = resolve_custom_tunable_modules(model, tunable_parts_list)
    substring_filters = [part for part in tunable_parts_list if part in PREDEFINED_TUNABLE_PARTS]

    for name, module in model.named_modules():
        if not isinstance(module, cls):
            continue

        matched = False
        for target in custom_module_targets:
            if target == name or name.startswith(f"{target}."):
                lora_module_names.add(name)
                matched = True
                break
        if matched:
            continue

        if substring_filters and not any(filter_key in name for filter_key in substring_filters):
            continue

        if not substring_filters and custom_module_targets:
            continue

        if "mm_projector" in name:
            if exclude_mm_projector:
                continue
            lora_module_names.add(name)
        else:
            names = name.split('.')
            lora_module_names.add(names[-1])

    print(f"mm_tunable_parts: {mm_tunable_parts}")
    print(f"Found these linear module names for LoRA: {lora_module_names}")
    return list(lora_module_names)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if hasattr(trainer.args, "tune_mm_mlp_adapter") and trainer.args.tune_mm_mlp_adapter:
        check_only_save_mm_adapter_tunnable = True
    # only has mm_mlp_adapter and mm_vision_resampler in the tuneable parts
    elif hasattr(trainer.args, "mm_tunable_parts") and (len(trainer.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in trainer.args.mm_tunable_parts or "mm_vision_resampler" in trainer.args.mm_tunable_parts)):
        check_only_save_mm_adapter_tunnable = True
    else:
        check_only_save_mm_adapter_tunnable = False

    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()
    rank0_print(f"Only save projectors: {check_only_save_mm_adapter_tunnable}")
    if check_only_save_mm_adapter_tunnable:
        # Only save Adapter
        keys_to_match = ["mm_projector", "vision_resampler"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        non_lora_weight_to_save = get_peft_state_non_lora_maybe_zero_3(trainer.model.named_parameters())
        for key in list(non_lora_weight_to_save.keys()):
            if key in weight_to_save:
                del non_lora_weight_to_save[key]
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                if weight_to_save:
                    torch.save(weight_to_save, os.path.join(mm_projector_folder, f"{current_folder}.bin"))
                if non_lora_weight_to_save:
                    non_lora_folder = os.path.join(parent_folder, "non_lora_trainables")
                    os.makedirs(non_lora_folder, exist_ok=True)
                    torch.save(non_lora_weight_to_save, os.path.join(non_lora_folder, f"{current_folder}.bin"))
            else:
                if weight_to_save:
                    torch.save(weight_to_save, os.path.join(output_dir, "mm_projector.bin"))
                if non_lora_weight_to_save:
                    torch.save(non_lora_weight_to_save, os.path.join(output_dir, "non_lora_trainables.bin"))
        return

    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            # TODO maybe this should be changed for interleaved data?
            # if DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
            # only check for num_im=1
            num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            if num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

            # For videoInstruct-100k noisy_data. TODO: Ask Yuanhan to clean the data instead of leaving the noise code here.
            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

    return sources


def preprocess_llama_2(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_gemma(sources: List[List[Dict[str, str]]], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv: conversation_lib.Conversation = conversation_lib.default_conversation.copy()
    roles: Dict[str, str] = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations: List[str] = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source: List[Dict[str, str]] = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role: str = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids: torch.Tensor = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids: torch.Tensor = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets: torch.Tensor = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA

    # Mask target
    sep: str = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len: int = int(target.ne(tokenizer.pad_token_id).sum())

        rounds: List[str] = conversation.split(conv.sep)
        re_rounds = []
        for conv_idx in range(0, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))

        cur_len = 1  # Ignore <bos>
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep  # Re-append sep because split on this
            # Now "".join(parts)==rou

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1  # Ignore <bos>
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # Ignore <bos>
            else:
                round_len = len(tokenizer(rou).input_ids) - 1  # Ignore <bos>
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  # Ignore <bos>

            round_len += 2  # sep: <end_of_turn>\n takes 2 tokens
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"warning: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    im_start, im_end = tokenizer.additional_special_tokens_ids
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]
    nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_v1(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))  # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, "legacy", False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f"(#turns={len(re_rounds)} ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "gemma":
        return preprocess_gemma(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama_v3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def load_split_dataset(data_args: DataArguments, split: str = "train") -> List[Dict[str, Any]]:
    """
    Load dataset from split dataset directory structure.
    
    Args:
        data_args: DataArguments containing split dataset configuration
        split: Either "train" or "val"
        
    Returns:
        List of data samples
    """
    if not data_args.use_split_dataset:
        raise ValueError("use_split_dataset must be True to load split datasets")
    
    if data_args.split_dataset_dir is None:
        raise ValueError("split_dataset_dir must be specified when using split datasets")
    
    if data_args.dataset_name is None:
        raise ValueError("dataset_name must be specified when using split datasets")
    
    # Check if config file exists and load it for validation
    config_path = os.path.join(data_args.split_dataset_dir, data_args.dataset_name, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            rank0_print(f"Loading split dataset: {config['dataset_name']}")
            rank0_print(f"Total samples in dataset: {config['total_samples']}")
    
    # Load the split file
    split_file = f"{split}.json"
    split_path = os.path.join(data_args.split_dataset_dir, data_args.dataset_name, split_file)
    
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")
    
    rank0_print(f"Loading {split} split from {split_path}")
    
    with open(split_path, 'r') as f:
        data = json.load(f)
    
    rank0_print(f"Loaded {len(data)} samples from {split} split")
    return data


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments, split_indices: Optional[List[int]] = None):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.list_data_dict = []
        self.split_indices = split_indices

        # Handle multiple JSON files specified in the data_path
        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            data_args.dataset_paths = []
            for file_name in file_names:
                data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                with open(full_path, "r") as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
                data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    sampling_number = None

                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            data_args.dataset_paths = [data_path]
            rank0_print(f"Loading {data_path}")
            with open(data_path, "r") as file:
                cur_data_dict = json.load(file)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                random.shuffle(cur_data_dict)
                rank0_print(f"Shuffled {len(cur_data_dict)} samples")
                self.list_data_dict.extend(cur_data_dict)

        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.roi_candidate_lookup: Dict[str, Dict[str, List[List[float]]]] = {}
        self.roi_max_positives = max(0, int(getattr(data_args, "roi_max_positives", 0) or 0))
        self.roi_max_negatives = max(0, int(getattr(data_args, "roi_max_negatives", 0) or 0))
        self.roi_positive_radius_ratio = float(getattr(data_args, "roi_positive_radius_ratio", 0.08) or 0.08)
        self.roi_candidate_slots = self.roi_max_positives + self.roi_max_negatives
        roi_candidates_csv = getattr(data_args, "roi_candidates_csv", None)
        if roi_candidates_csv and self.roi_candidate_slots > 0:
            self.roi_candidate_lookup = load_roi_candidate_lookup(Path(roi_candidates_csv), logger=rank0_print)
            rank0_print(
                f"Loaded ROI candidate lookup from {roi_candidates_csv} "
                f"(keys={len(self.roi_candidate_lookup)}, slots={self.roi_candidate_slots})"
            )

        # Apply split indices if provided
        if self.split_indices is not None:
            original_data = self.list_data_dict
            self.list_data_dict = [original_data[i] for i in self.split_indices]
            rank0_print(f"Applied split: using {len(self.list_data_dict)} samples from the split")

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            if "image" in sample or "video" in sample or self.data_args.early_mix_text:
                length_list.append(cur_len)
            else:
                length_list.append(-cur_len)
        return length_list

    def process_image(self, image_file, overwrite_image_aspect_ratio=None):
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        # print(f"\n\nInspecting the image path, folder = {image_folder}, image={image_file}\n\n")
        if not image_file.endswith(".jpg"):
            image_file = image_file + ".jpg"
        try:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        except Exception as exn:
            print(f"Failed to open image {image_file}. Exception:", exn)
            raise exn

        image_size = image.size
        image_aspect_ratio = self.data_args.image_aspect_ratio
        if overwrite_image_aspect_ratio is not None:
            image_aspect_ratio = overwrite_image_aspect_ratio
        if image_aspect_ratio == "highres":
            image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            image, _, _, _ = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "crop_split":
            image = process_highres_image_crop_split(image, self.data_args)
        elif image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, "image"

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            if type(image_file) is list:
                image = [self.process_image(f) for f in image_file]
                # Handling multi images
                # overwrite to process with simple pad 
                if len(image_file) > 1:
                    image = [self.process_image(f, "pad") for f in image_file]
                    image = [[im[0], im[1], "image"] for im in image]
            else:
                image = [self.process_image(image_file)]
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)

        elif "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.data_args.video_folder
            video_file = os.path.join(video_folder, video_file)
            suffix = video_file.split(".")[-1]
            if not os.path.exists(video_file):
                print("File {} not exist!".format(video_file))

            try:
                if "shareVideoGPTV" in video_file:
                    frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
                    frame_files.sort()  # Ensure the frames are sorted if they are named sequentially

                    # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
                    if self.data_args.force_sample:
                        num_frames_to_sample = self.data_args.frames_upbound
                    else:
                        num_frames_to_sample = 10

                    avg_fps = 2
                    
                    total_frames = len(frame_files)
                    sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)


                    frame_time = [i/2 for i in sampled_indices]
                    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

                    video_time = total_frames / avg_fps

                    # Read and store the sampled frames
                    video = []
                    for idx in sampled_indices:
                        frame_path = frame_files[idx]
                        try:
                            with Image.open(frame_path) as img:
                                frame = img.convert("RGB")
                                video.append(frame)
                        except IOError:
                            print(f"Failed to read frame at path: {frame_path}")
                else:
                    video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_file, self.data_args)

                processor = self.data_args.image_processor
                image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
                if self.data_args.add_time_instruction:
                    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
                    sources[0]["conversations"][0]["value"] = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'
                image = [(image, video[0].size, "video")]
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
                # print(sources)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Failed to read video file: {video_file}")
                return self._get_item(i + 1)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        has_image = ("image" in self.list_data_dict[i]) or ("video" in self.list_data_dict[i])
        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)

        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
        else:
            prompt = None

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif "video" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = [
                (torch.zeros(1, 3, crop_size["height"], crop_size["width"]), (crop_size["width"], crop_size["height"]), "text"),
            ]
        # prompt exist in the data
        if prompt is not None:
            data_dict["prompt"] = prompt

        data_dict["id"] = self.list_data_dict[i].get("id", i)
        roi_gaze_xy, roi_gaze_valid = build_roi_gaze_metadata(self.list_data_dict[i])
        data_dict["roi_gaze_xy"] = roi_gaze_xy
        data_dict["roi_gaze_valid"] = roi_gaze_valid
        if self.roi_candidate_slots > 0 and self.roi_candidate_lookup:
            roi_boxes, roi_is_positive, roi_valid = build_roi_candidate_metadata(
                sample_dict=self.list_data_dict[i],
                roi_candidate_lookup=self.roi_candidate_lookup,
                roi_max_positives=self.roi_max_positives,
                roi_max_negatives=self.roi_max_negatives,
                roi_candidate_slots=self.roi_candidate_slots,
                roi_positive_radius_ratio=self.roi_positive_radius_ratio,
            )
            data_dict["roi_candidate_boxes"] = roi_boxes
            data_dict["roi_candidate_is_positive"] = roi_is_positive
            data_dict["roi_candidate_valid"] = roi_valid

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # input_ids, labels, ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "id"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
        # batch = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), ids=ids)

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            # if all(x is not None and x.shape == images[0].shape for x in images):
                # Image: (N, P, C, H, W)
                # Video: (N, F, C, H, W)
            #     batch["images"] = torch.stack(images)
            # else:
            batch["images"] = images

        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]
        if "roi_gaze_xy" in instances[0]:
            batch["roi_gaze_xy"] = torch.stack([instance["roi_gaze_xy"] for instance in instances], dim=0)
            batch["roi_gaze_valid"] = torch.stack([instance["roi_gaze_valid"] for instance in instances], dim=0).bool()
        if "roi_candidate_boxes" in instances[0]:
            batch["roi_candidate_boxes"] = torch.stack([instance["roi_candidate_boxes"] for instance in instances], dim=0)
            batch["roi_candidate_is_positive"] = torch.stack([instance["roi_candidate_is_positive"] for instance in instances], dim=0).bool()
            batch["roi_candidate_valid"] = torch.stack([instance["roi_candidate_valid"] for instance in instances], dim=0).bool()

        return batch


def create_train_eval_splits(data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments, eval_split_ratio: float = 0.2, seed: int = 42):
    """Create train and eval datasets with the specified split ratio."""
    # First create a full dataset to get all the data
    full_dataset = LazySupervisedDataset(data_path, tokenizer, data_args)
    total_samples = len(full_dataset.list_data_dict)
    
    # Create indices for splitting
    indices = list(range(total_samples))
    random.seed(seed)
    random.shuffle(indices)
    
    # Split indices
    eval_size = int(total_samples * eval_split_ratio)
    eval_indices = indices[:eval_size]
    train_indices = indices[eval_size:]
    
    rank0_print(f"Split dataset: {len(train_indices)} train samples, {len(eval_indices)} eval samples")
    
    # Create split datasets by reusing the loaded data and directly applying splits
    train_dataset = LazySupervisedDataset.__new__(LazySupervisedDataset)
    train_dataset.tokenizer = tokenizer
    train_dataset.data_args = data_args
    train_dataset.split_indices = train_indices
    train_dataset.list_data_dict = [full_dataset.list_data_dict[i] for i in train_indices]
    train_dataset.roi_candidate_lookup = getattr(full_dataset, "roi_candidate_lookup", {})
    train_dataset.roi_max_positives = getattr(full_dataset, "roi_max_positives", 0)
    train_dataset.roi_max_negatives = getattr(full_dataset, "roi_max_negatives", 0)
    train_dataset.roi_positive_radius_ratio = getattr(full_dataset, "roi_positive_radius_ratio", 0.08)
    train_dataset.roi_candidate_slots = getattr(full_dataset, "roi_candidate_slots", 0)
    rank0_print(f"Applied split: using {len(train_dataset.list_data_dict)} samples for training")
    
    eval_dataset = LazySupervisedDataset.__new__(LazySupervisedDataset)
    eval_dataset.tokenizer = tokenizer
    eval_dataset.data_args = data_args
    eval_dataset.split_indices = eval_indices
    eval_dataset.list_data_dict = [full_dataset.list_data_dict[i] for i in eval_indices]
    eval_dataset.roi_candidate_lookup = getattr(full_dataset, "roi_candidate_lookup", {})
    eval_dataset.roi_max_positives = getattr(full_dataset, "roi_max_positives", 0)
    eval_dataset.roi_max_negatives = getattr(full_dataset, "roi_max_negatives", 0)
    eval_dataset.roi_positive_radius_ratio = getattr(full_dataset, "roi_positive_radius_ratio", 0.08)
    eval_dataset.roi_candidate_slots = getattr(full_dataset, "roi_candidate_slots", 0)
    rank0_print(f"Applied split: using {len(eval_dataset.list_data_dict)} samples for evaluation")
    
    return train_dataset, eval_dataset


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    # Check if evaluation is enabled
    if data_args.enable_evaluation:
        if data_args.eval_data_path is not None:
            # Use separate eval dataset
            train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
            eval_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.eval_data_path, data_args=data_args)
            rank0_print(f"Using separate eval dataset: {len(train_dataset)} train, {len(eval_dataset)} eval")
        else:
            # Split the training data
            train_dataset, eval_dataset = create_train_eval_splits(
                data_path=data_args.data_path,
                tokenizer=tokenizer,
                data_args=data_args,
                eval_split_ratio=data_args.eval_split_ratio
            )
        return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
    else:
        # No evaluation - use all data for training
        train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
        return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def print_model_structure(model, show_params=True, show_gradients=True):
    """Print comprehensive model structure with parameter counts and gradient status."""
    rank0_print("=" * 80)
    rank0_print("MODEL STRUCTURE")
    rank0_print("=" * 80)
    
    total_params = 0
    trainable_params = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            param_count = module.weight.numel()
            total_params += param_count
            is_trainable = module.weight.requires_grad
            if is_trainable:
                trainable_params += param_count
            
            if show_params:
                rank0_print(f"{name:60} | {str(type(module).__name__):20} | {param_count:>10,} params | {'✓' if is_trainable else '✗'} trainable")
    
    if show_gradients:
        rank0_print("\n" + "=" * 60)
        rank0_print("PARAMETER GRADIENT STATUS")
        rank0_print("=" * 60)
        
        for name, param in model.named_parameters():
            status = "✓ TRAINABLE" if param.requires_grad else "✗ FROZEN"
            rank0_print(f"{name:60} | {status}")
    
    rank0_print("\n" + "=" * 60)
    rank0_print("SUMMARY")
    rank0_print("=" * 60)
    rank0_print(f"Total parameters: {total_params:,}")
    rank0_print(f"Trainable parameters: {trainable_params:,}")
    rank0_print(f"Frozen parameters: {total_params - trainable_params:,}")
    rank0_print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
    rank0_print("=" * 80)


def get_model(model_args, training_args, bnb_model_from_pretrained_args):
    assert training_args.attn_implementation
    if training_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError("The 'sdpa' attention implementation requires torch version 2.1.2 or higher.")

    customized_kwargs = dict()
    customized_kwargs.update(bnb_model_from_pretrained_args)
    cfg_pretrained = None

    overwrite_config = {}
    if any(
        [
            model_args.rope_scaling_factor is not None,
            model_args.rope_scaling_type is not None,
            model_args.mm_spatial_pool_stride is not None,
            model_args.mm_spatial_pool_out_channels is not None,
            model_args.mm_spatial_pool_mode is not None,
            model_args.mm_resampler_type is not None,
        ]
    ):
        cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)

    if model_args.use_pos_skipping is not None and model_args.pos_skipping_range is not None:
        overwrite_config["use_pos_skipping"] = model_args.use_pos_skipping
        overwrite_config["pos_skipping_range"] = model_args.pos_skipping_range

    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        overwrite_config["rope_scaling"] = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }
        if training_args.model_max_length is None:
            training_args.model_max_length = cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
            overwrite_config["max_sequence_length"] = training_args.model_max_length
        assert training_args.model_max_length == int(cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor), print(
            f"model_max_length: {training_args.model_max_length}, max_position_embeddings: {cfg_pretrained.max_position_embeddings}, rope_scaling_factor: {model_args.rope_scaling_factor}"
        )
        # overwrite_config["max_sequence_length"] = model_args.max_sequence_length
        # overwrite_config["tokenizer_model_max_length"] = model_args.tokenizer_model_max_length

    if model_args.mm_spatial_pool_stride is not None and model_args.mm_spatial_pool_out_channels is not None and model_args.mm_spatial_pool_mode is not None and model_args.mm_resampler_type is not None:
        overwrite_config["mm_resampler_type"] = model_args.mm_resampler_type
        overwrite_config["mm_spatial_pool_stride"] = model_args.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_out_channels"] = model_args.mm_spatial_pool_out_channels
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if model_args.mm_spatial_pool_mode is not None:
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if overwrite_config:
        assert cfg_pretrained is not None, "cfg_pretrained is None"

        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)

        customized_kwargs["config"] = cfg_pretrained

    if model_args.model_class_name is not None:
        actual_model_class_name = f"{model_args.model_class_name}ForCausalLM"
        model_class = getattr(transformers, actual_model_class_name)
        rank0_print(f"Using model class {model_class} from {model_args.model_class_name}")
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
            **customized_kwargs,
        )
    elif model_args.vision_tower is not None:
        if "mixtral" in model_args.model_name_or_path.lower():
            model = LlavaMixtralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
            from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

            deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
        elif "mistral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
            model = LlavaMistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        elif (
            "wizardlm-2" in model_args.model_name_or_path.lower()
            or "vicuna" in model_args.model_name_or_path.lower()
            or "llama" in model_args.model_name_or_path.lower()
            or "yi" in model_args.model_name_or_path.lower()
            or "nous-hermes" in model_args.model_name_or_path.lower()
            and "wizard-2" in model_args.model_name_or_path.lower()
        ):
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        elif "qwen" in model_args.model_name_or_path.lower():
            if "moe" in model_args.model_name_or_path.lower() or "A14B" in model_args.model_name_or_path:
                model = LlavaQwenMoeForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    **customized_kwargs,
                )
                from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

                deepspeed.utils.set_z3_leaf_modules(model, [Qwen2MoeSparseMoeBlock])
            else:
                model = LlavaQwenForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    **customized_kwargs,
                )
        elif "gemma" in model_args.model_name_or_path.lower():
            model = LlavaGemmaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        else:
            raise ValueError(f"Unknown model class {model_args}")
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
            **customized_kwargs,
        )
    return model


def train(attn_implementation=None):
    global local_rank
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    # Initialize overall timing
    overall_start = time.time()

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    parsed_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    model_args, data_args, training_args = parsed_args[:3]
    
    rank0_print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    rank0_print("Starting training setup...")
    setup_start = time.time()
    wandb_notes_env = os.getenv("WANDB_NOTES", "").strip()
    if wandb_notes_env:
        rank0_print(f"WANDB_NOTES: {wandb_notes_env}")
    

    if training_args.verbose_logging:
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")
        # rank0_print(f"evaluation_args = {vars(evaluation_args)}\n\n")

    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    # Load pretrained LLaVA model (already includes mm_mlp_adapter)
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov-chat"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    llava_model_args = {
        "multimodal": True,
        "torch_dtype": "bfloat16" if training_args.bf16 else "float16" if training_args.fp16 else "float32",
        # "attn_implementation": "sdpa",
    }
    
    # Time model loading
    model_load_start = time.time()
    rank0_print("Starting model loading...")
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module")
        tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)
    
    model_load_time = time.time() - model_load_start
    rank0_print(f"Model loading completed in {model_load_time:.2f} seconds")
    
    # Log to wandb if available and initialized
    wandb_log(training_args, {"timing/model_load_seconds": model_load_time})
    
    # Store the image processor from the pretrained model
    if image_processor is not None:
        data_args.image_processor = image_processor
        data_args.is_multimodal = True
        rank0_print("Image processor loaded from pretrained model")
    
    rank0_print(f"Model Class: {model.__class__.__name__}")
    rank0_print(f"Prompt version: {model_args.version}")

    model.config.use_cache = False
    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        model.config.rope_scaling = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        # Configure gradient checkpointing with use_reentrant=False
        # if hasattr(model, 'gradient_checkpointing_enable'):
        #     model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(
                model,
                mm_tunable_parts=model_args.mm_tunable_parts,
                exclude_mm_projector=training_args.mm_projector_full_finetune,
            ),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        # if training_args.bits == 16:
        #     if training_args.bf16:
        #         model.to(torch.bfloat16)
        #     if training_args.fp16:
        #         model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    # if "mistral" in model_args.model_name_or_path.lower() or "mixtral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
    #     tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="left")
    # elif "qwen" in model_args.model_name_or_path.lower():
    #     tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="right")
    # elif (
    #     "wizardlm-2" in model_args.model_name_or_path.lower()
    #     or "vicuna" in model_args.model_name_or_path.lower()
    #     or "llama" in model_args.model_name_or_path.lower()
    #     or "yi" in model_args.model_name_or_path.lower()
    #     or "nous-hermes" in model_args.model_name_or_path.lower()
    #     and "wizard-2" in model_args.model_name_or_path.lower()
    # ):
    #     tokenizer = transformers.AutoTokenizer.from_pretrained(
    #         model_args.model_name_or_path,
    #         cache_dir=training_args.cache_dir,
    #         model_max_length=training_args.model_max_length,
    #         padding_side="right",
    #         use_fast=False,
    #     )

    rank0_print(f"Prompt version: {model_args.version}")
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    roi_phrase_token_ids = build_focus_phrase_token_ids(tokenizer, training_args.roi_contrastive_phrase)
    roi_oof_token_ids = tokenizer.encode(training_args.roi_contrastive_oof_text, add_special_tokens=False)
    roi_oof_enabled = bool(training_args.roi_contrastive_oof_enable)
    if roi_oof_enabled and not roi_oof_token_ids:
        rank0_print(
            "[ROI contrastive warning] --roi_contrastive_oof_enable is set but "
            f"--roi_contrastive_oof_text produced empty tokenization: {training_args.roi_contrastive_oof_text!r}. "
            "Disabling OOF auxiliary loss."
        )
        roi_oof_enabled = False
    warmup_steps = 0
    if training_args.max_steps and training_args.max_steps > 0:
        warmup_steps = int(training_args.max_steps * training_args.roi_contrastive_warmup_ratio)
    model.config.roi_contrastive_enable = training_args.roi_contrastive_enable
    model.config.roi_contrastive_weight = training_args.roi_contrastive_weight
    model.config.roi_contrastive_temperature = training_args.roi_contrastive_temperature
    model.config.roi_contrastive_dim = training_args.roi_contrastive_dim
    model.config.roi_contrastive_phrase = training_args.roi_contrastive_phrase
    model.config.roi_contrastive_phrase_token_ids = roi_phrase_token_ids
    model.config.roi_contrastive_warmup_steps = warmup_steps
    model.config.roi_contrastive_radius_ratio = training_args.roi_contrastive_radius_ratio
    model.config.roi_contrastive_oof_enable = roi_oof_enabled
    model.config.roi_contrastive_oof_text = training_args.roi_contrastive_oof_text
    model.config.roi_contrastive_oof_weight = training_args.roi_contrastive_oof_weight
    model.config.roi_contrastive_oof_token_ids = roi_oof_token_ids
    model.config.roi_pos_embed_dim = training_args.roi_pos_embed_dim
    rank0_print(
        "ROI contrastive config: "
        f"enabled={training_args.roi_contrastive_enable}, "
        f"weight={training_args.roi_contrastive_weight}, "
        f"temp={training_args.roi_contrastive_temperature}, "
        f"dim={training_args.roi_contrastive_dim}, "
        f"oof_enabled={roi_oof_enabled}, "
        f"oof_weight={training_args.roi_contrastive_oof_weight}, "
        f"oof_tokens={len(roi_oof_token_ids)}, "
        f"pos_dim={training_args.roi_pos_embed_dim}, "
        f"warmup_steps={warmup_steps}, "
        f"phrase_candidates={len(roi_phrase_token_ids)}"
    )
    if data_args.roi_candidates_csv:
        rank0_print(
            "ROI candidate CSV enabled: "
            f"path={data_args.roi_candidates_csv}, "
            f"max_pos={data_args.roi_max_positives}, max_neg={data_args.roi_max_negatives}, "
            f"pos_source=gt_gaze_radius({data_args.roi_positive_radius_ratio})"
        )
    if (
        training_args.roi_contrastive_enable
        and int(training_args.per_device_train_batch_size) < 2
        and not roi_oof_enabled
    ):
        rank0_print(
            "[ROI contrastive warning] per_device_train_batch_size < 2. "
            "Current ROI loss uses in-batch negatives only, so pair_count stays <2 and "
            "roi_contrastive/* metrics will remain at zero."
        )
    
    # Set vision tower parameter from model args if provided, otherwise use pretrained
    if model_args.vision_tower is None:
        # Use the vision tower from the pretrained model
        model_args.vision_tower = getattr(model.config, 'mm_vision_tower', 'google/siglip-so400m-patch14-384')
    
    rank0_print(f"Using vision tower: {model_args.vision_tower}")
    
    # Only initialize vision modules if they're not already present or if we need to update them
    vision_init_start = time.time()
    
    if hasattr(model, 'get_vision_tower') and model.get_vision_tower() is not None:
        rank0_print("Vision modules already initialized from pretrained model")
        vision_tower = model.get_vision_tower()
    else:
        rank0_print("Initializing vision modules")
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
        vision_tower = model.get_vision_tower()
    
    vision_init_time = time.time() - vision_init_start
    rank0_print(f"Vision initialization completed in {vision_init_time:.2f} seconds")
    
    # Log to wandb if available and initialized
    wandb_log(training_args, {"timing/vision_init_seconds": vision_init_time})
    
    # Ensure vision tower is moved to the correct device and dtype
    # vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16)

    # Set image processor and multimodal flag regardless of initialization path
    if vision_tower.image_processor is not None:
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True
        rank0_print("Image processor set from vision tower")
    elif data_args.image_processor is None:
        rank0_print("WARNING: No image processor found, this may cause issues with image processing")
    
    # Verify image processor is properly set
    if data_args.image_processor is None:
        raise ValueError("Image processor is None - this will cause errors during data loading")

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    if data_args.image_grid_pinpoints is not None:
        if isinstance(data_args.image_grid_pinpoints, str) and "x" in data_args.image_grid_pinpoints:
            try:
                patch_size = data_args.image_processor.size[0]
            except Exception as e:
                patch_size = data_args.image_processor.size["shortest_edge"]

            assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
            # Use regex to extract the range from the input string
            matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
            range_start = tuple(map(int, matches[0]))
            range_end = tuple(map(int, matches[-1]))
            # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
            grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
            # Multiply all elements by patch_size
            data_args.image_grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
        elif isinstance(data_args.image_grid_pinpoints, str):
            data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)

    model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
    model.config.image_crop_resolution = data_args.image_crop_resolution
    model.config.image_split_resolution = data_args.image_split_resolution
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_newline_position = model_args.mm_newline_position
    model.config.add_faster_video = model_args.add_faster_video
    model.config.faster_token_stride = model_args.faster_token_stride
    model.config.add_time_instruction = data_args.add_time_instruction
    model.config.force_sample = data_args.force_sample
    model.config.mm_spatial_pool_stride = model_args.mm_spatial_pool_stride 

    ### Deciding train which part of the model
    if model_args.mm_tunable_parts is None:  # traditional way of deciding which part to train
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        model.config.tune_mm_vision_resampler = training_args.tune_mm_vision_resampler = model_args.tune_mm_vision_resampler
        if model_args.tune_mm_mlp_adapter or model_args.tune_mm_vision_resampler:
            model.requires_grad_(False)
        if model_args.tune_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
        if model_args.tune_mm_vision_resampler:
            for p in model.get_model().vision_resampler.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        model.config.freeze_mm_vision_resampler = training_args.freeze_mm_vision_resampler
        if training_args.freeze_mm_vision_resampler:
            for p in model.get_model().vision_resampler.parameters():
                p.requires_grad = False

        model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
        if model_args.unfreeze_mm_vision_tower:
            vision_tower.requires_grad_(True)
        else:
            vision_tower.requires_grad_(False)


    else:
        rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}")
        tunable_parts = _split_tunable_specs(model_args.mm_tunable_parts)
        sanitized_parts = ','.join(tunable_parts)
        model.config.mm_tunable_parts = training_args.mm_tunable_parts = sanitized_parts or model_args.mm_tunable_parts
        # custom_module_targets = resolve_custom_tunable_modules(model, tunable_parts)
        # model.config.resolved_custom_tunable_modules = custom_module_targets
        custom_module_targets = True
        # Set the entire model to not require gradients by default
        model.requires_grad_(False)
        vision_tower.requires_grad_(False)
        model.get_model().mm_projector.requires_grad_(False)
        model.get_model().vision_resampler.requires_grad_(False)
        if "mm_mlp_adapter" in tunable_parts:
            if training_args.lora_enable:
                for name, param in model.named_parameters():
                    if "mlp" in name and "lora_" in name:
                        param.requires_grad_(True)
            else:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
        if "mm_vision_resampler" in tunable_parts:
            if training_args.lora_enable:
                for name, param in model.named_parameters():
                    if "vision_resampler" in name and "lora_" in name:
                        param.requires_grad_(True)
            else:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = True
        if "mm_vision_tower" in tunable_parts:
            if training_args.lora_enable:
                for name, param in model.named_parameters():
                    if "vision_tower" in name and "lora_" in name:
                        param.requires_grad_(True)
            else:
                for name, param in model.named_parameters():
                    if "vision_tower" in name:
                        param.requires_grad_(True)
        if "mm_projector" in tunable_parts:
            print("Enabling mm_projector parameters")
            if training_args.mm_projector_full_finetune:
                rank0_print("mm_projector_full_finetune enabled: training base projector weights without LoRA adapters")
                if hasattr(model.get_model(), "mm_projector"):
                    model.get_model().mm_projector.requires_grad_(True)
                for name, param in model.named_parameters():
                    if "mm_projector" not in name:
                        continue
                    if "lora_" in name:
                        param.requires_grad_(False)
                    else:
                        param.requires_grad_(True)
            elif training_args.lora_enable:
                for name, param in model.named_parameters():
                    if "mm_projector" in name and "lora_" in name:
                        param.requires_grad_(True)
            else:
                for name, param in model.named_parameters():
                    if "mm_projector" in name:
                        param.requires_grad_(True)
        if "lm_head" in tunable_parts:
            print("Enabling lm_head parameters")
            if training_args.lora_enable:
                for name, param in model.named_parameters():
                    if "lm_head" in name and "lora_" in name:
                        param.requires_grad_(True)
            else:
                for name, param in model.named_parameters():
                    if "lm_head" in name:
                        param.requires_grad_(True)
        if "mm_language_model" in tunable_parts:
            if training_args.lora_enable:
                rank0_print("Language model training enabled via LoRA adapters (base model parameters remain frozen)")
                language_model_params = []
                for name, param in model.named_parameters():
                    if "lora_" in name and "model.model.layers" in name:
                        param.requires_grad_(True)
                        language_model_params.append(name)
            else:
                language_model_params = []
                for name, param in model.named_parameters():
                    if "vision_tower" not in name and "mm_projector" not in name and "vision_resampler" not in name:
                        param.requires_grad_(True)
                        language_model_params.append(name)

            rank0_print(f"Language model parameters set to require gradients ({len(language_model_params)} parameters):")
            for param_name in language_model_params:
                rank0_print(f"  {param_name}")

        if custom_module_targets:
            rank0_print(f"Enabling custom tunable modules: {custom_module_targets}")
            custom_params = set()
            if training_args.lora_enable:
                for name, param in model.named_parameters():
                    if "lora_" in name:
                        param.requires_grad_(True)
                        custom_params.add(name)
            # else:
            #     for name, param in model.named_parameters():
            #         if any(name.startswith(target) for target in custom_module_targets):
            #             param.requires_grad_(True)
            #             custom_params.add(name)
            rank0_print(f"Custom module parameters set to require gradients ({len(custom_params)} parameters):")
            for param_name in sorted(custom_params):
                rank0_print(f"  {param_name}")
    if training_args.roi_contrastive_enable:
        roi_head_params = []
        for name, param in model.named_parameters():
            if (
                "roi_text_projector" in name
                or "roi_vision_projector" in name
                or "roi_position_mlp" in name
            ):
                param.requires_grad_(True)
                roi_head_params.append(name)
        rank0_print(f"Enabled ROI contrastive head parameters ({len(roi_head_params)}):")
        for param_name in sorted(roi_head_params):
            rank0_print(f"  {param_name}")

    total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"Total parameters: {total_params:,}")
    rank0_print(f"Trainable parameters: {trainable_params:,}")
    
    # Add comprehensive model structure print
    print_model_structure(model, show_params=True, show_gradients=True)
    
    # Clear any cached memory after model setup
    torch.cuda.empty_cache()
    
    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
    model.config.mm_projector_full_finetune = training_args.mm_projector_full_finetune
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    # model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)        # todo: test if this really required

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # Time data module creation
    data_module_start = time.time()
    rank0_print("Creating data module...")
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    data_module_time = time.time() - data_module_start
    rank0_print(f"Data module creation completed in {data_module_time:.2f} seconds")
    
    # Log to wandb if available and initialized
    wandb_log(training_args, {"timing/data_module_creation_seconds": data_module_time})
    
    # Determine conversation template for evaluation
    conv_template = "qwen_1_5"
    
    # Configure evaluation settings before creating trainer
    if data_args.enable_evaluation and data_module["eval_dataset"] is not None:
        # Set evaluation strategy if not already set
        if training_args.evaluation_strategy == "no":
            training_args.evaluation_strategy = "steps"
        
        # Set eval_steps if provided
        if training_args.eval_steps is None and training_args.evaluation_strategy == "steps":
            training_args.eval_steps = 500  # Default to every 500 steps
        
        rank0_print(f"Evaluation enabled: strategy={training_args.evaluation_strategy}, eval_steps={training_args.eval_steps}")
        rank0_print(f"Eval dataset size: {len(data_module['eval_dataset'])}")
    else:
        # Explicitly disable evaluation when no eval dataset is available
        training_args.evaluation_strategy = "no"
        training_args.eval_steps = None
        rank0_print("Evaluation disabled")
    
    # Time trainer creation
    trainer_creation_start = time.time()
    rank0_print("Creating trainer...")
    
    trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    
    trainer_creation_time = time.time() - trainer_creation_start
    setup_time = time.time() - setup_start
    
    rank0_print(f"Trainer creation completed in {trainer_creation_time:.2f} seconds")
    rank0_print(f"Total setup time: {setup_time:.2f} seconds")
    
    # Log to wandb if available and initialized
    wandb_log(training_args, {
        "timing/trainer_creation_seconds": trainer_creation_time,
        "timing/total_setup_seconds": setup_time
    })
    wandb_summary(training_args, {
        "slurm_job_id": os.getenv("SLURM_JOB_ID", "unknown"),
        **({"wandb_notes": wandb_notes_env} if wandb_notes_env else {}),
    })

    # Time actual training
    training_start = time.time()
    rank0_print("Starting training...")
    
    # Check for existing checkpoints and resume from the latest one
    checkpoint_dirs = sorted(pathlib.Path(training_args.output_dir).glob("checkpoint-*"), 
                            key=lambda x: int(x.name.split("-")[-1]))
    print(f"Checkpoint dirs: {checkpoint_dirs}")
    if checkpoint_dirs:
        latest_checkpoint = str(checkpoint_dirs[-1])
        rank0_print(f"Found {len(checkpoint_dirs)} checkpoint(s) in {training_args.output_dir}")
        rank0_print(f"Resuming training from: {latest_checkpoint}")
        
        # Verify checkpoint contains trainer_state.json
        trainer_state_file = pathlib.Path(latest_checkpoint) / "trainer_state.json"
        if trainer_state_file.exists():
            rank0_print(f"✓ Trainer state found: {trainer_state_file}")
        else:
            rank0_print(f"⚠ Warning: No trainer_state.json found in {latest_checkpoint}")
        
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        rank0_print("No checkpoints found, starting fresh training")
        trainer.train()

    if (
        wandb_notes_env
        and training_args.report_to
        and "wandb" in training_args.report_to
        and training_args.local_rank in (0, -1)
        and wandb.run is not None
    ):
        wandb.run.notes = wandb_notes_env
        wandb_summary(training_args, {"wandb_notes": wandb_notes_env})
    
    training_time = time.time() - training_start
    total_time = time.time() - overall_start
    
    rank0_print(f"Training completed in {training_time:.2f} seconds")
    rank0_print(f"Total execution time: {total_time:.2f} seconds")
    
    # Log to wandb if available and initialized
    wandb_log(training_args, {
        "timing/total_training_seconds": training_time,
        "timing/total_execution_seconds": total_time
    })
        
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            if hasattr(model, "config"):
                model.config.save_pretrained(training_args.output_dir)
            if hasattr(model, "generation_config"):
                model.generation_config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    rank0_print(f"Model saved to {training_args.output_dir}")
    rank0_print(f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    train()
