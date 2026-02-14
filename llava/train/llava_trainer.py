import os
import wandb
import inspect
import torch
import torch.nn as nn
import datetime
import json
import pathlib
import time
from collections import OrderedDict

from PIL import Image, ImageDraw

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, GradientAccumulationPlugin
from torch.utils.data import Dataset, Sampler, DataLoader

from trl.trainer import DPOTrainer
from trl.trainer.utils import DPODataCollatorWithPadding

from transformers import Trainer
from transformers.trainer import is_sagemaker_mp_enabled, get_parameter_names, has_length, ALL_LAYERNORM_LAYERS, logger, is_accelerate_available, is_datasets_available, GradientAccumulationPlugin
from transformers.trainer_utils import seed_worker
from transformers.trainer_pt_utils import get_length_grouped_indices as get_length_grouped_indices_hf
from transformers.trainer_pt_utils import AcceleratorConfig

# Import sagemaker functions if available
try:
    from transformers.trainer_pt_utils import smp_forward_backward
except ImportError:
    smp_forward_backward = None

# Import apex if available
try:
    from apex import amp
except ImportError:
    amp = None
from typing import Any, List, Optional, Dict
from datetime import timedelta

# Import custom evaluation functions
try:
    from evaluate_model import (
        evaluate_dataset_for_training,
        determine_template,
    )
    CUSTOM_EVAL_AVAILABLE = True
except ImportError:
    CUSTOM_EVAL_AVAILABLE = False

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches, InitProcessGroupKwargs

if is_datasets_available():
    import datasets

from llava import conversation as conversation_lib
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle
from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
from llava.utils import rank0_print


def safe_wandb_log(args, metrics_dict, step=None):
    """Safely log metrics to wandb if available and initialized."""
    if not (args.report_to and "wandb" in args.report_to):
        return
    
    try:
        if wandb.run is not None:
            if step is not None:
                metrics_dict["step"] = step
            wandb.log(metrics_dict)
    except Exception as e:
        rank0_print(f"Warning: Could not log to wandb: {e}")


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only: bool = True):
    """Collect non-LoRA parameters (optionally only trainable ones) from possibly ZeRO sharded models."""

    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_variable_length_grouped_indices(lengths, batch_size, world_size, megabatch_mult=8, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    megabatch_size = world_size * batch_size * megabatch_mult
    megabatches = [sorted_indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: indices[i], reverse=True) for megabatch in megabatches]
    shuffled_indices = [i for megabatch in megabatches for i in megabatch]
    world_batch_size = world_size * batch_size
    batches = [shuffled_indices[i : i + world_batch_size] for i in range(0, len(lengths), world_batch_size)]
    batch_indices = torch.randperm(len(batches), generator=generator)
    batches = [batches[i] for i in batch_indices]

    return [i for batch in batches for i in batch]


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=None):
    indices = get_length_grouped_indices_hf(lengths, batch_size * world_size, generator=generator)

    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    batch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in batch_indices]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_modality_length_grouped_indices_auto(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices_auto_single(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices_auto_single(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    # FIXME: Hard code to avoid last batch mixed with different modalities
    # if len(additional_batch) > 0:
    #     megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        variable_length: bool = False,
        group_by_modality: bool = False,
        group_by_modality_auto: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.variable_length = variable_length
        self.group_by_modality = group_by_modality
        self.group_by_modality_auto = group_by_modality_auto

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.variable_length:
            assert not self.group_by_modality, "Variable length grouping is not supported with modality grouping."
            indices = get_variable_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            if self.group_by_modality:
                indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            elif self.group_by_modality_auto:
                indices = get_modality_length_grouped_indices_auto(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            else:
                indices = get_length_grouped_indices_auto_single(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize timing counters
        self.step_times = {
            "data_loading": [],
            "forward_pass": [],
            "backward_pass": [],
            "optimizer_step": [],
            "total_step": []
        }
        self.last_log_step = 0
        self.sequence_stats = {
            "total_tokens": [],
            "target_tokens": [],
            "truncations": []
        }
        self.sequence_stats_max_length = 0
        self.sequence_stats_cap = 4096
        self.sequence_stats_window = 512
        self.roi_contrastive_stats = {
            "nce_loss": [],
            "top1": [],
            "pairs": [],
            "lambda": [],
            "oof_loss": [],
            "oof_top1": [],
            "oof_rows": [],
        }
        self.roi_contrastive_stats_window = max(1, int(getattr(self.args, "roi_contrastive_metrics_window", 100)))
        self.roi_contrastive_stats_cap = max(2048, self.roi_contrastive_stats_window * 4)
        self.roi_contrastive_preview_samples = max(1, int(getattr(self.args, "roi_contrastive_preview_samples", 5)))
        self.roi_contrastive_local_preview_rows: List[Dict[str, object]] = []
        self._roi_overlay_batch: Optional[Dict[str, Any]] = None
        self._sanity_table = None
        self._sanity_consecutive_failures = 0
        self._sanity_failure_limit = getattr(self.args, "sanity_check_failures_to_stop", 5)

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Override evaluate method to use custom evaluation from evaluate_model.py
        when use_custom_eval is enabled.
        """
        # Check if custom evaluation is enabled
        if hasattr(self.args, 'use_custom_eval') and self.args.use_custom_eval and CUSTOM_EVAL_AVAILABLE:
            rank0_print("\n" + "="*60)
            rank0_print("Running Custom Evaluation (Overridden)")
            rank0_print("="*60)
            
            # Pick dataset
            eval_ds = eval_dataset if eval_dataset is not None else self.eval_dataset
            
            if eval_ds is None:
                rank0_print("No eval dataset available")
                return {}
            
            # Get required components
            try:
                # Get image processor from model or data_args
                if hasattr(self.model, 'get_vision_tower') and self.model.get_vision_tower() is not None:
                    image_processor = self.model.get_vision_tower().image_processor
                elif hasattr(self, 'data_args') and hasattr(self.data_args, 'image_processor'):
                    image_processor = self.data_args.image_processor
                else:
                    rank0_print("Warning: No image processor found, falling back to standard evaluation")
                    return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
                
                # Determine conversation template
                model_name = getattr(self.model.config, '_name_or_path', 'llava_qwen')
                if hasattr(self, 'model_args') and hasattr(self.model_args, 'version'):
                    version = self.model_args.version
                else:
                    version = getattr(self.model.config, 'version', 'qwen_1_5')
                
                conv_template = determine_template(model_name, version)
                
                # Run custom evaluation
                custom_metrics = evaluate_dataset_for_training(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    image_processor=image_processor,
                    eval_dataset=eval_ds,
                    conv_template=conv_template,
                    max_new_tokens=getattr(self.args, 'eval_max_new_tokens', 128),
                    focus_loss_after_looking=getattr(self.args, 'focus_loss_after_looking', False),
                    focus_loss_phrase=getattr(self.args, 'focus_loss_phrase', 'looking at'),
                    focus_loss_threshold=getattr(self.args, 'focus_loss_threshold', 5.0),
                    no_loss=getattr(self.args, 'no_loss', False),
                    verbose=getattr(self.args, 'verbose_logging', False),
                    limit=getattr(self.args, 'eval_limit', None),
                )
                
                # Add metric prefix and log
                metrics = OrderedDict()
                if custom_metrics:
                    for key, value in custom_metrics.items():
                        # Add prefix to metrics
                        prefixed_key = f"{metric_key_prefix}_{key}" if not key.startswith(metric_key_prefix) else key
                        metrics[prefixed_key] = value
                    
                    # Log metrics
                    rank0_print("\nCustom Evaluation Metrics:")
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            rank0_print(f"  {key}: {value:.4f}")
                        else:
                            rank0_print(f"  {key}: {value}")
                    
                    # Save to file
                    output_dir = pathlib.Path(self.args.output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    metrics_file = output_dir / f"custom_eval_step_{self.state.global_step}.json"
                    with open(metrics_file, 'w') as f:
                        json.dump(dict(metrics), f, indent=2)
                    rank0_print(f"\nCustom metrics saved to: {metrics_file}")
                    rank0_print("="*60 + "\n")
                    
                    # Log to trainer
                    self.log(metrics)
                    self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
                    
                return metrics
                
            except Exception as e:
                rank0_print(f"Error in custom evaluation: {e}")
                rank0_print("Falling back to standard evaluation")
                import traceback
                traceback.print_exc()
                return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        else:
            # Use standard HF evaluation
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def create_accelerator_and_postprocess(self):
        grad_acc_kwargs = {"num_steps": self.args.gradient_accumulation_steps}
        grad_acc_kwargs["sync_with_dataloader"] = False
        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        rank0_print("Setting NCCL timeout to INF to avoid running errors.")
    
        grad_acc_kwargs = {"num_steps": self.args.gradient_accumulation_steps}
        grad_acc_kwargs["sync_with_dataloader"] = False
        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        rank0_print("Setting NCCL timeout to INF to avoid running errors.")

        # Build accelerator parameters based on what's supported in this version
        accelerator_params = {
            "deepspeed_plugin": self.args.deepspeed_plugin,
            "gradient_accumulation_plugin": gradient_accumulation_plugin,
            "kwargs_handlers": [accelerator_kwargs]
        }
        
        # Check which parameters are supported in this accelerate version
        accelerator_signature = inspect.signature(Accelerator.__init__).parameters
        
        # Add parameters conditionally based on what's available
        if "dispatch_batches" in accelerator_signature:
            accelerator_params["dispatch_batches"] = getattr(self.args, "dispatch_batches", None)
        
        if "split_batches" in accelerator_signature:
            accelerator_params["split_batches"] = getattr(self.args, "split_batches", False)
        
        # In newer versions, gradient_accumulation_steps is passed directly instead of through plugin
        if "gradient_accumulation_steps" in accelerator_signature:
            accelerator_params["gradient_accumulation_steps"] = self.args.gradient_accumulation_steps
            # Remove the plugin if we're using the direct parameter
            if "gradient_accumulation_plugin" in accelerator_params:
                del accelerator_params["gradient_accumulation_plugin"]
            
            # Create accelerator object
        self.accelerator = Accelerator(**accelerator_params)
        # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
        self.gather_function = self.accelerator.gather_for_metrics

        # deepspeed and accelerate flags covering both trainer args and accelerate launcher
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

        # post accelerator creation setup
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get("limit_all_gathers", fsdp_plugin.limit_all_gathers)
            if is_accelerate_available("0.23.0"):
                fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get("activation_checkpointing", fsdp_plugin.activation_checkpointing)
                if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                    raise ValueError("The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg " "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic " "when using FSDP.")

        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self.propagate_args_to_deepspeed()

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_length:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
            )
        elif self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                group_by_modality=True,
            )
        elif self.args.group_by_modality_length_auto:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                group_by_modality_auto=True,
            )
        elif self.args.group_by_varlen:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                # self.args.train_batch_size, # TODO: seems that we should have gradient_accumulation_steps
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                variable_length=True,
            )
        else:
            return super()._get_train_sampler()

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`] with timing instrumentation.

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        dataloader_creation_start = time.time()
        rank0_print("Creating train dataloader...")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_num_workers * 2 if self.args.dataloader_num_workers != 0 else None

        dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

        dataloader_creation_time = time.time() - dataloader_creation_start
        rank0_print(f"Train dataloader created in {dataloader_creation_time:.2f} seconds")
        
        # Log to wandb if available and initialized
        safe_wandb_log(self.args, {"timing/dataloader_creation_seconds": dataloader_creation_time})

        return dataloader
    
    def training_step(self, model, inputs):
        """Override training step with detailed timing."""
        step_start = time.time()
        
        model.train()

        self._cache_roi_overlay_batch(inputs)
        if isinstance(inputs, dict):
            inputs.pop("sample_ids", None)
            inputs.pop("image_files", None)
        
        # Time input preparation (data movement to GPU, etc.)
        input_prep_start = time.time()
        inputs = self._prepare_inputs(inputs)
        input_prep_time = time.time() - input_prep_start

        if is_sagemaker_mp_enabled() and smp_forward_backward:
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        # Forward pass timing
        forward_start = time.time()
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        forward_time = time.time() - forward_start

        self._collect_sequence_length_metrics(model)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # Backward pass timing  
        backward_start = time.time()
        if self.use_apex and amp:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        backward_time = time.time() - backward_start
        
        total_step_time = time.time() - step_start
        
        # Store timing data
        self.step_times["data_loading"].append(input_prep_time)  # Input prep is part of data loading overhead
        self.step_times["forward_pass"].append(forward_time)
        self.step_times["backward_pass"].append(backward_time)
        self.step_times["total_step"].append(total_step_time)
        
        # Debug: Print timing details for first few steps
        if self.state.global_step <= 3:
            rank0_print(f"[DEBUG] Step {self.state.global_step}: input_prep={input_prep_time:.4f}s, "
                       f"forward={forward_time:.4f}s, backward={backward_time:.4f}s, total={total_step_time:.4f}s")
            rank0_print(f"[DEBUG] data_loading list has {len(self.step_times['data_loading'])} entries, "
                       f"last value: {self.step_times['data_loading'][-1]:.4f}s")

        # Log timing every 50 steps to avoid overhead
        if self.state.global_step % 50 == 0 and self.state.global_step > self.last_log_step:
            self._log_step_timings()
            self.log_memory_usage()
            self._log_sequence_length_stats()
            self.last_log_step = self.state.global_step

        return loss.detach() / self.args.gradient_accumulation_steps

    def _cache_roi_overlay_batch(self, inputs):
        self._roi_overlay_batch = None
        if not isinstance(inputs, dict):
            return

        image_files = inputs.get("image_files")
        roi_candidate_boxes = inputs.get("roi_candidate_boxes")
        roi_candidate_is_positive = inputs.get("roi_candidate_is_positive")
        roi_candidate_valid = inputs.get("roi_candidate_valid")
        if not isinstance(image_files, list) or not torch.is_tensor(roi_candidate_boxes):
            return
        if not torch.is_tensor(roi_candidate_is_positive) or not torch.is_tensor(roi_candidate_valid):
            return

        sample_ids = inputs.get("sample_ids")
        roi_gaze_xy = inputs.get("roi_gaze_xy")
        roi_gaze_valid = inputs.get("roi_gaze_valid")
        self._roi_overlay_batch = {
            "image_files": [str(x) if x is not None else "" for x in image_files],
            "sample_ids": [str(x) for x in sample_ids] if isinstance(sample_ids, list) else [],
            "roi_candidate_boxes": roi_candidate_boxes.detach().cpu(),
            "roi_candidate_is_positive": roi_candidate_is_positive.detach().cpu().bool(),
            "roi_candidate_valid": roi_candidate_valid.detach().cpu().bool(),
            "roi_gaze_xy": roi_gaze_xy.detach().cpu() if torch.is_tensor(roi_gaze_xy) else None,
            "roi_gaze_valid": roi_gaze_valid.detach().cpu().bool() if torch.is_tensor(roi_gaze_valid) else None,
        }

    def _collect_sequence_length_metrics(self, model):
        try:
            unwrapped_model = self.accelerator.unwrap_model(model)
        except Exception:
            unwrapped_model = model
        self._collect_roi_contrastive_metrics(unwrapped_model)

        if not hasattr(unwrapped_model, "pop_sequence_length_stats"):
            return

        stats = unwrapped_model.pop_sequence_length_stats()
        if not stats:
            return

        total_tokens = stats.get("total_tokens")
        if total_tokens is None:
            return
        total_tokens = total_tokens.detach()

        target_tokens = stats.get("target_tokens")
        if target_tokens is not None:
            target_tokens = target_tokens.detach()

        trunc_flags = stats.get("hit_max_length")
        if trunc_flags is not None:
            trunc_flags = trunc_flags.detach()

        max_length = stats.get("max_length", 0) or 0

        gathered_total = self.accelerator.gather(total_tokens)
        gathered_target = self.accelerator.gather(target_tokens) if target_tokens is not None else None
        gathered_trunc = self.accelerator.gather(trunc_flags.long()) if trunc_flags is not None else None

        if self.is_world_process_zero():
            self.sequence_stats["total_tokens"].extend(gathered_total.long().cpu().tolist())
            if gathered_target is not None:
                self.sequence_stats["target_tokens"].extend(gathered_target.long().cpu().tolist())
            else:
                self.sequence_stats["target_tokens"].extend([0] * gathered_total.numel())
            if gathered_trunc is not None:
                self.sequence_stats["truncations"].extend(gathered_trunc.long().cpu().tolist())
            else:
                self.sequence_stats["truncations"].extend([0] * gathered_total.numel())
            self.sequence_stats_max_length = max(self.sequence_stats_max_length, max_length)
            self._trim_sequence_stats()

    def _trim_sequence_stats(self):
        for key in ("total_tokens", "target_tokens", "truncations"):
            if len(self.sequence_stats[key]) > self.sequence_stats_cap:
                self.sequence_stats[key] = self.sequence_stats[key][-self.sequence_stats_cap:]

    def _collect_roi_contrastive_metrics(self, unwrapped_model):
        if not hasattr(unwrapped_model, "pop_roi_contrastive_stats"):
            return

        stats = unwrapped_model.pop_roi_contrastive_stats()
        if not stats:
            return

        nce_loss = stats.get("nce_loss")
        top1 = stats.get("top1")
        pair_count = stats.get("pairs")
        lambda_val = stats.get("lambda")
        if nce_loss is None or top1 is None or pair_count is None or lambda_val is None:
            return

        gathered_nce = self.accelerator.gather(nce_loss.detach().float().reshape(1))
        gathered_top1 = self.accelerator.gather(top1.detach().float().reshape(1))
        gathered_pairs = self.accelerator.gather(pair_count.detach().long().reshape(1))
        gathered_lambda = self.accelerator.gather(lambda_val.detach().float().reshape(1))

        oof_loss = stats.get("oof_loss")
        oof_top1 = stats.get("oof_top1")
        oof_rows = stats.get("oof_rows")
        preview_row_indices = stats.get("preview_row_indices")
        preview_pred_candidate_slots = stats.get("preview_pred_candidate_slots")
        preview_candidate_slot_scores = stats.get("preview_candidate_slot_scores")
        preview_oof_scores = stats.get("preview_oof_scores")
        gathered_oof_loss = None
        gathered_oof_top1 = None
        gathered_oof_rows = None
        if oof_loss is not None and oof_top1 is not None and oof_rows is not None:
            gathered_oof_loss = self.accelerator.gather(oof_loss.detach().float().reshape(1))
            gathered_oof_top1 = self.accelerator.gather(oof_top1.detach().float().reshape(1))
            gathered_oof_rows = self.accelerator.gather(oof_rows.detach().long().reshape(1))

        if self.is_world_process_zero():
            self.roi_contrastive_stats["nce_loss"].append(float(gathered_nce.mean().item()))
            self.roi_contrastive_stats["top1"].append(float(gathered_top1.mean().item()))
            self.roi_contrastive_stats["pairs"].append(float(gathered_pairs.float().mean().item()))
            self.roi_contrastive_stats["lambda"].append(float(gathered_lambda.mean().item()))
            self.roi_contrastive_local_preview_rows = []
            if (
                preview_row_indices is not None
                and preview_pred_candidate_slots is not None
            ):
                local_limit = min(
                    int(preview_row_indices.shape[0]),
                    int(preview_pred_candidate_slots.shape[0]),
                )
                for local_idx in range(local_limit):
                    candidate_scores: List[float] = []
                    if torch.is_tensor(preview_candidate_slot_scores) and local_idx < preview_candidate_slot_scores.shape[0]:
                        slot_scores_row = preview_candidate_slot_scores[local_idx]
                        candidate_scores = [float(score) for score in slot_scores_row.tolist()]
                    oof_scores: List[float] = []
                    if torch.is_tensor(preview_oof_scores) and local_idx < preview_oof_scores.shape[0]:
                        oof_scores_row = preview_oof_scores[local_idx]
                        oof_scores = [float(score) for score in oof_scores_row.tolist()]
                    self.roi_contrastive_local_preview_rows.append(
                        {
                            "row_idx": int(preview_row_indices[local_idx].item()),
                            "pred_candidate_slot": int(preview_pred_candidate_slots[local_idx].item()),
                            "candidate_slot_scores": candidate_scores,
                            "oof_scores": oof_scores,
                        }
                    )
            if gathered_oof_loss is not None and gathered_oof_top1 is not None and gathered_oof_rows is not None:
                self.roi_contrastive_stats["oof_loss"].append(float(gathered_oof_loss.mean().item()))
                self.roi_contrastive_stats["oof_top1"].append(float(gathered_oof_top1.mean().item()))
                self.roi_contrastive_stats["oof_rows"].append(float(gathered_oof_rows.float().mean().item()))
            self._trim_roi_contrastive_stats()

    def _trim_roi_contrastive_stats(self):
        for key in ("nce_loss", "top1", "pairs", "lambda", "oof_loss", "oof_top1", "oof_rows"):
            if len(self.roi_contrastive_stats[key]) > self.roi_contrastive_stats_cap:
                self.roi_contrastive_stats[key] = self.roi_contrastive_stats[key][-self.roi_contrastive_stats_cap:]

    def _log_sequence_length_stats(self):
        if not self.is_world_process_zero():
            return
        if not self.sequence_stats["total_tokens"]:
            return

        window = min(len(self.sequence_stats["total_tokens"]), self.sequence_stats_window)
        total_tensor = torch.tensor(self.sequence_stats["total_tokens"][-window:], dtype=torch.float32)
        target_tensor = torch.tensor(self.sequence_stats["target_tokens"][-window:], dtype=torch.float32)
        context_tensor = total_tensor - target_tensor
        trunc_tensor = torch.tensor(self.sequence_stats["truncations"][-window:], dtype=torch.float32)

        full_mean = total_tensor.mean().item()
        full_max = total_tensor.max().item()
        full_p95 = torch.quantile(total_tensor, 0.95).item() if total_tensor.numel() > 1 else full_max
        target_mean = target_tensor.mean().item() if target_tensor.numel() > 0 else 0.0
        target_max = target_tensor.max().item() if target_tensor.numel() > 0 else 0.0
        context_mean = context_tensor.mean().item() if context_tensor.numel() > 0 else 0.0
        context_max = context_tensor.max().item() if context_tensor.numel() > 0 else 0.0
        trunc_rate = trunc_tensor.mean().item() if trunc_tensor.numel() > 0 else 0.0

        rank0_print(
            f"Sequence lengths (last {window} samples) - mean: {full_mean:.1f}, p95: {full_p95:.1f}, max: {int(full_max)}, "
            f"target mean: {target_mean:.1f}, context mean: {context_mean:.1f}, truncation: {trunc_rate * 100:.1f}%"
        )

        metrics = {
            "sequence/full_tokens_mean": full_mean,
            "sequence/full_tokens_max": full_max,
            "sequence/full_tokens_p95": full_p95,
            "sequence/context_tokens_mean": context_mean,
            "sequence/context_tokens_max": context_max,
            "sequence/target_tokens_mean": target_mean,
            "sequence/target_tokens_max": target_max,
            "sequence/truncation_rate": trunc_rate,
        }
        if self.sequence_stats_max_length:
            metrics["sequence/max_config_length"] = self.sequence_stats_max_length

        safe_wandb_log(self.args, metrics, step=self.state.global_step)


    
    def _log_step_timings(self):
        """Log average timing for the last batch of steps."""
        if not self.step_times["total_step"]:
            return
            
        # Calculate averages for the last 10 steps
        recent_steps = min(10, len(self.step_times["total_step"]))
        
        avg_forward = sum(self.step_times["forward_pass"][-recent_steps:]) / recent_steps
        avg_backward = sum(self.step_times["backward_pass"][-recent_steps:]) / recent_steps  
        avg_total = sum(self.step_times["total_step"][-recent_steps:]) / recent_steps
        
        # Calculate data loading average if we have data
        avg_data_loading = 0
        if self.step_times["data_loading"]:
            recent_data_steps = min(recent_steps, len(self.step_times["data_loading"]))
            if recent_data_steps > 0:
                recent_data_values = self.step_times["data_loading"][-recent_data_steps:]
                avg_data_loading = sum(recent_data_values) / recent_data_steps
                
                # Debug first few logging calls
                if self.state.global_step <= 30:
                    rank0_print(f"[DEBUG] Data loading timing - recent {recent_data_steps} steps: "
                               f"{[f'{v:.4f}' for v in recent_data_values]}, avg={avg_data_loading:.4f}s")
        
        # Log to console
        rank0_print(f"Step {self.state.global_step} timing - "
                   f"Data: {avg_data_loading:.3f}s, Forward: {avg_forward:.3f}s, "
                   f"Backward: {avg_backward:.3f}s, Total: {avg_total:.3f}s")
        
        # Log to wandb if available and initialized
        metrics = {
            "timing/avg_forward_pass_seconds": avg_forward,
            "timing/avg_backward_pass_seconds": avg_backward,
            "timing/avg_total_step_seconds": avg_total,
            "timing/forward_backward_ratio": avg_forward / (avg_backward + 1e-8)
        }
        
        # Only add data loading if we have meaningful data
        if avg_data_loading > 0:
            metrics["timing/avg_data_loading_seconds"] = avg_data_loading

        if self.roi_contrastive_stats["nce_loss"]:
            recent_roi = min(self.roi_contrastive_stats_window, len(self.roi_contrastive_stats["nce_loss"]))
            roi_nce = sum(self.roi_contrastive_stats["nce_loss"][-recent_roi:]) / recent_roi
            roi_top1 = sum(self.roi_contrastive_stats["top1"][-recent_roi:]) / recent_roi
            roi_pairs = sum(self.roi_contrastive_stats["pairs"][-recent_roi:]) / recent_roi
            roi_lambda = sum(self.roi_contrastive_stats["lambda"][-recent_roi:]) / recent_roi
            metrics["roi_contrastive/nce_loss"] = roi_nce
            metrics["roi_contrastive/top1"] = roi_top1
            metrics["roi_contrastive/pairs"] = roi_pairs
            metrics["roi_contrastive/lambda"] = roi_lambda
            rank0_print(
                f"ROI contrastive - NCE: {roi_nce:.4f}, Top1: {roi_top1:.3f}, "
                f"Pairs: {roi_pairs:.2f}, Lambda: {roi_lambda:.4f} "
                f"(window={recent_roi})"
            )
            if self.roi_contrastive_stats["oof_loss"]:
                recent_oof = min(self.roi_contrastive_stats_window, len(self.roi_contrastive_stats["oof_loss"]))
                roi_oof_loss = sum(self.roi_contrastive_stats["oof_loss"][-recent_oof:]) / recent_oof
                roi_oof_top1 = sum(self.roi_contrastive_stats["oof_top1"][-recent_oof:]) / recent_oof
                roi_oof_rows = sum(self.roi_contrastive_stats["oof_rows"][-recent_oof:]) / recent_oof
                metrics["roi_contrastive/oof_loss"] = roi_oof_loss
                metrics["roi_contrastive/oof_top1"] = roi_oof_top1
                metrics["roi_contrastive/oof_rows"] = roi_oof_rows
                rank0_print(
                    f"ROI contrastive OOF - Loss: {roi_oof_loss:.4f}, "
                    f"Top1: {roi_oof_top1:.3f}, Rows: {roi_oof_rows:.2f} "
                    f"(window={recent_oof})"
                )
            overlay_images = self._build_roi_candidate_overlays()
            if overlay_images:
                metrics["roi_contrastive/preview_overlays"] = overlay_images
            
        safe_wandb_log(self.args, metrics, step=self.state.global_step)
        
        # Keep only recent timing data to avoid memory buildup
        max_history = 100
        for key in self.step_times:
            if len(self.step_times[key]) > max_history:
                self.step_times[key] = self.step_times[key][-max_history:]

    @staticmethod
    def _norm_box_to_pixels(box: torch.Tensor, image_w: int, image_h: int):
        x1 = int(max(0, min(image_w - 1, round(float(box[0].item()) * image_w))))
        y1 = int(max(0, min(image_h - 1, round(float(box[1].item()) * image_h))))
        x2 = int(max(0, min(image_w - 1, round(float(box[2].item()) * image_w))))
        y2 = int(max(0, min(image_h - 1, round(float(box[3].item()) * image_h))))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return x1, y1, x2, y2

    @staticmethod
    def _draw_labeled_norm_box(
        draw: ImageDraw.ImageDraw,
        box: torch.Tensor,
        image_w: int,
        image_h: int,
        color,
        width: int,
        label: str,
    ):
        x1, y1, x2, y2 = LLaVATrainer._norm_box_to_pixels(box, image_w, image_h)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        text = label.strip()
        if not text:
            return
        text_x = x1 + 2
        text_y = max(0, y1 - 14)
        if hasattr(draw, "textbbox"):
            text_left, text_top, text_right, text_bottom = draw.textbbox((text_x, text_y), text)
            draw.rectangle([text_left - 1, text_top - 1, text_right + 1, text_bottom + 1], fill=(0, 0, 0))
        draw.text((text_x, text_y), text, fill=color)

    def _resolve_overlay_image_path(self, image_file: str) -> Optional[pathlib.Path]:
        if not image_file:
            return None
        candidate_paths = []
        image_path = pathlib.Path(image_file)
        if image_path.is_absolute():
            candidate_paths.append(image_path)
        else:
            image_root = getattr(getattr(self.model, "config", None), "train_image_folder", None)
            if image_root:
                candidate_paths.append(pathlib.Path(str(image_root)) / image_path)
            candidate_paths.append(image_path)

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
        for candidate in candidate_paths:
            if candidate.suffix:
                if candidate.is_file():
                    return candidate
                continue
            for ext in exts:
                with_ext = candidate.with_suffix(ext)
                if with_ext.is_file():
                    return with_ext
        return None

    def _build_roi_candidate_overlays(self) -> List[wandb.Image]:
        if not self.is_world_process_zero():
            return []
        if not self.roi_contrastive_local_preview_rows or not self._roi_overlay_batch:
            return []
        batch = self._roi_overlay_batch
        image_files = batch.get("image_files", [])
        boxes = batch.get("roi_candidate_boxes")
        positives = batch.get("roi_candidate_is_positive")
        valids = batch.get("roi_candidate_valid")
        if not isinstance(image_files, list) or not torch.is_tensor(boxes):
            return []
        if not torch.is_tensor(positives) or not torch.is_tensor(valids):
            return []

        sample_ids = batch.get("sample_ids", [])
        gaze_xy = batch.get("roi_gaze_xy")
        gaze_valid = batch.get("roi_gaze_valid")
        max_overlays = min(self.roi_contrastive_preview_samples, len(self.roi_contrastive_local_preview_rows))
        overlays: List[wandb.Image] = []
        for preview_idx in range(max_overlays):
            preview = self.roi_contrastive_local_preview_rows[preview_idx]
            row_idx = int(preview.get("row_idx", -1))
            pred_slot = int(preview.get("pred_candidate_slot", -1))
            if row_idx < 0 or row_idx >= len(image_files) or row_idx >= boxes.shape[0]:
                continue
            image_path = self._resolve_overlay_image_path(image_files[row_idx])
            if image_path is None:
                continue

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                continue
            draw = ImageDraw.Draw(image)
            image_w, image_h = image.size

            row_boxes = boxes[row_idx]
            row_pos = positives[row_idx]
            row_valid = valids[row_idx]
            for cand_idx in range(int(row_boxes.shape[0])):
                if not bool(row_valid[cand_idx].item()):
                    continue
                is_gt = bool(row_pos[cand_idx].item())
                is_pred = pred_slot >= 0 and cand_idx == pred_slot
                if is_pred:
                    color = (255, 64, 64)
                    line_width = 4
                elif is_gt:
                    color = (0, 255, 80)
                    line_width = 3
                else:
                    color = (255, 214, 10)
                    line_width = 2
                label_parts = [f"cand_{cand_idx}"]
                slot_scores = preview.get("candidate_slot_scores", [])
                if cand_idx < len(slot_scores):
                    score_val = float(slot_scores[cand_idx])
                    if score_val == score_val and score_val not in (float("inf"), float("-inf")):
                        label_parts.append(f"s={score_val:.3f}")
                if is_gt:
                    label_parts.append("GT")
                if is_pred:
                    label_parts.append("pred")
                self._draw_labeled_norm_box(
                    draw,
                    row_boxes[cand_idx],
                    image_w,
                    image_h,
                    color=color,
                    width=line_width,
                    label="|".join(label_parts),
                )

            if (
                torch.is_tensor(gaze_xy)
                and torch.is_tensor(gaze_valid)
                and row_idx < gaze_xy.shape[0]
                and row_idx < gaze_valid.shape[0]
                and bool(gaze_valid[row_idx].item())
            ):
                gx = int(max(0, min(image_w - 1, round(float(gaze_xy[row_idx][0].item()) * image_w))))
                gy = int(max(0, min(image_h - 1, round(float(gaze_xy[row_idx][1].item()) * image_h))))
                r = 6
                draw.ellipse([gx - r, gy - r, gx + r, gy + r], outline=(0, 255, 255), width=3)

            sample_id = sample_ids[row_idx] if row_idx < len(sample_ids) else str(row_idx)
            oof_parts: List[str] = []
            configured_oof_labels = list(getattr(getattr(self.model, "config", None), "roi_contrastive_oof_texts", []) or [])
            oof_scores = preview.get("oof_scores", [])
            for oof_idx, raw_score in enumerate(oof_scores):
                score_val = float(raw_score)
                if score_val != score_val or score_val in (float("inf"), float("-inf")):
                    continue
                label = configured_oof_labels[oof_idx] if oof_idx < len(configured_oof_labels) else f"oof_{oof_idx}"
                oof_parts.append(f"{label}:{score_val:.2f}")
            if oof_parts:
                overlay_text = "OOF " + " | ".join(oof_parts)
                text_x = 8
                text_y = 8
                if hasattr(draw, "textbbox"):
                    left, top, right, bottom = draw.textbbox((text_x, text_y), overlay_text)
                    draw.rectangle([left - 2, top - 2, right + 2, bottom + 2], fill=(0, 0, 0))
                draw.text((text_x, text_y), overlay_text, fill=(255, 255, 255))
            caption = (
                f"step={self.state.global_step} sample={sample_id} row={row_idx} "
                f"pred_slot={pred_slot}"
            )
            overlays.append(wandb.Image(image, caption=caption))
        return overlays
    
    def optimizer_step(self, optimizer):
        """Override optimizer step with timing."""
        optimizer_start = time.time()
        
        # Call the parent optimizer step
        super().optimizer_step(optimizer)
        
        optimizer_time = time.time() - optimizer_start
        self.step_times["optimizer_step"].append(optimizer_time)
        
        # Log optimizer timing every 10 steps
        if self.state.global_step % 10 == 0:
            recent_steps = min(10, len(self.step_times["optimizer_step"]))
            avg_optimizer = sum(self.step_times["optimizer_step"][-recent_steps:]) / recent_steps
            
            safe_wandb_log(self.args, {
                "timing/avg_optimizer_step_seconds": avg_optimizer
            }, step=self.state.global_step)
    
    def log_memory_usage(self):
        """Log GPU memory usage if available."""
        if torch.cuda.is_available() and self.state.global_step % 50 == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            
            rank0_print(f"Step {self.state.global_step} - GPU Memory: "
                       f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
            safe_wandb_log(self.args, {
                "memory/gpu_allocated_gb": allocated,
                "memory/gpu_reserved_gb": reserved
            }, step=self.state.global_step)
    
    def log_training_summary(self):
        """Log a comprehensive training timing summary."""
        if not any(self.step_times.values()):
            return
            
        rank0_print("\n" + "="*60)
        rank0_print("TRAINING TIMING SUMMARY")
        rank0_print("="*60)
        
        total_steps = len(self.step_times["total_step"])
        if total_steps > 0:
            # Calculate averages
            avg_forward = sum(self.step_times["forward_pass"]) / len(self.step_times["forward_pass"]) if self.step_times["forward_pass"] else 0
            avg_backward = sum(self.step_times["backward_pass"]) / len(self.step_times["backward_pass"]) if self.step_times["backward_pass"] else 0
            avg_optimizer = sum(self.step_times["optimizer_step"]) / len(self.step_times["optimizer_step"]) if self.step_times["optimizer_step"] else 0
            avg_data_loading = sum(self.step_times["data_loading"]) / len(self.step_times["data_loading"]) if self.step_times["data_loading"] else 0
            avg_total = sum(self.step_times["total_step"]) / len(self.step_times["total_step"])
            
            rank0_print(f"Total training steps: {total_steps}")
            rank0_print(f"Average per step:")
            rank0_print(f"  - Data loading: {avg_data_loading:.3f}s ({avg_data_loading/avg_total*100:.1f}%)")
            rank0_print(f"  - Forward pass: {avg_forward:.3f}s ({avg_forward/avg_total*100:.1f}%)")
            rank0_print(f"  - Backward pass: {avg_backward:.3f}s ({avg_backward/avg_total*100:.1f}%)")
            rank0_print(f"  - Optimizer step: {avg_optimizer:.3f}s ({avg_optimizer/avg_total*100:.1f}%)")
            rank0_print(f"  - Total step: {avg_total:.3f}s")
            rank0_print(f"  - Steps per second: {1.0/avg_total:.2f}")
            
            # Calculate bottleneck
            max_component = max([
                ("Data loading", avg_data_loading),
                ("Forward pass", avg_forward), 
                ("Backward pass", avg_backward),
                ("Optimizer step", avg_optimizer)
            ], key=lambda x: x[1])
            
            rank0_print(f"\nBottleneck: {max_component[0]} ({max_component[1]:.3f}s)")
            
            # Log final summary to wandb
            safe_wandb_log(self.args, {
                "summary/avg_data_loading_seconds": avg_data_loading,
                "summary/avg_forward_pass_seconds": avg_forward,
                "summary/avg_backward_pass_seconds": avg_backward,
                "summary/avg_optimizer_step_seconds": avg_optimizer,
                "summary/avg_total_step_seconds": avg_total,
                "summary/steps_per_second": 1.0/avg_total,
                "summary/bottleneck_component": max_component[0],
                "summary/bottleneck_time": max_component[1],
                "summary/total_training_steps": total_steps
            })
        if self.sequence_stats["total_tokens"] and self.is_world_process_zero():
            total_tensor = torch.tensor(self.sequence_stats["total_tokens"], dtype=torch.float32)
            target_tensor = torch.tensor(self.sequence_stats["target_tokens"], dtype=torch.float32)
            context_tensor = total_tensor - target_tensor
            trunc_tensor = torch.tensor(self.sequence_stats["truncations"], dtype=torch.float32) if self.sequence_stats["truncations"] else torch.tensor([], dtype=torch.float32)

            full_mean = total_tensor.mean().item()
            full_max = total_tensor.max().item()
            full_p95 = torch.quantile(total_tensor, 0.95).item() if total_tensor.numel() > 1 else full_max
            target_mean = target_tensor.mean().item() if target_tensor.numel() > 0 else 0.0
            target_max = target_tensor.max().item() if target_tensor.numel() > 0 else 0.0
            context_mean = context_tensor.mean().item() if context_tensor.numel() > 0 else 0.0
            context_max = context_tensor.max().item() if context_tensor.numel() > 0 else 0.0
            trunc_rate = trunc_tensor.mean().item() if trunc_tensor.numel() > 0 else 0.0

            rank0_print("\nSequence length summary:")
            rank0_print(f"  - Mean full length: {full_mean:.1f}")
            rank0_print(f"  - 95th percentile full length: {full_p95:.1f}")
            rank0_print(f"  - Max full length: {int(full_max)}")
            rank0_print(f"  - Mean context length: {context_mean:.1f}")
            rank0_print(f"  - Mean target length: {target_mean:.1f}")
            rank0_print(f"  - Truncation rate: {trunc_rate * 100:.2f}% (max config {self.sequence_stats_max_length})")

            safe_wandb_log(self.args, {
                "summary/sequence_full_mean": full_mean,
                "summary/sequence_full_p95": full_p95,
                "summary/sequence_full_max": full_max,
                "summary/sequence_context_mean": context_mean,
                "summary/sequence_context_max": context_max,
                "summary/sequence_target_mean": target_mean,
                "summary/sequence_target_max": target_max,
                "summary/sequence_truncation_rate": trunc_rate,
            }, step=self.state.global_step)

        
        rank0_print("="*60 + "\n")
    
    def train(self, *args, **kwargs):
        """Override train method to add timing summary at the end."""
        try:
            result = super().train(*args, **kwargs)
            self.log_training_summary()
            return result
        except Exception as e:
            self.log_training_summary()
            raise e
    
    def _save_checkpoint(self, model, trial, metrics=None):
        """Override checkpoint saving with timing."""
        checkpoint_start = time.time()
        rank0_print("Starting checkpoint save...")
        
        # Call parent save checkpoint method
        super()._save_checkpoint(model, trial, metrics)
        
        checkpoint_time = time.time() - checkpoint_start
        rank0_print(f"Checkpoint saved in {checkpoint_time:.2f} seconds")
        
        # Log to wandb if available and initialized
        safe_wandb_log(self.args, {
            "timing/checkpoint_save_seconds": checkpoint_time
        }, step=self.state.global_step)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            if self.args.mm_projector_lr is not None:
                lr_mapper["mm_projector"] = self.args.mm_projector_lr
            if self.args.mm_vision_tower_lr is not None:
                lr_mapper["vision_tower"] = self.args.mm_vision_tower_lr
            if len(lr_mapper) > 0:
                special_lr_parameters = [name for name, _ in opt_model.named_parameters() if any(module_keyword in name for module_keyword in lr_mapper)]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                for module_keyword, lr in lr_mapper.items():
                    module_parameters = [name for name, _ in opt_model.named_parameters() if module_keyword in name]
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _run_checkpoint_sanity_check(self) -> bool:
        should_stop = False
        if self.is_world_process_zero():
            image_processor = None
            if hasattr(self.model, "get_vision_tower") and self.model.get_vision_tower() is not None:
                image_processor = self.model.get_vision_tower().image_processor

            if image_processor is None:
                rank0_print("Sanity check skipped: image processor not available.")
            else:
                image_path = pathlib.Path(__file__).resolve().parents[2] / "baseline_images" / "39740.png"
                if not image_path.is_file():
                    rank0_print(f"Sanity check skipped: image not found at {image_path}.")
                else:
                    prompt_text = "First analyze where each person is looking, then infer the social interaction between them."
                    conv = conversation_lib.default_conversation.copy()
                    conv.tokenizer = self.tokenizer
                    if getattr(self.model.config, "mm_use_im_start_end", False):
                        user_content = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{prompt_text}"
                    else:
                        user_content = f"{DEFAULT_IMAGE_TOKEN}\n{prompt_text}"
                    conv.append_message(conv.roles[0], user_content)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()

                    image = Image.open(image_path).convert("RGB")
                    image_sizes = [image.size]
                    image_tensor = process_images([image], image_processor, self.model.config)
                    if isinstance(image_tensor, tuple):
                        image_tensor = image_tensor[0]
                    if isinstance(image_tensor, list):
                        image_tensor = image_tensor[0] if image_tensor else None

                    if image_tensor is None:
                        rank0_print("Sanity check skipped: failed to process image.")
                    else:
                        if isinstance(image_tensor, torch.Tensor) and image_tensor.ndim == 3:
                            image_tensor = image_tensor.unsqueeze(0)

                        param = next(self.model.parameters())
                        device = param.device
                        image_tensor = image_tensor.to(device=device, dtype=param.dtype)
                        input_ids = tokenizer_image_token(
                            prompt,
                            self.tokenizer,
                            IMAGE_TOKEN_INDEX,
                            return_tensors="pt",
                        ).unsqueeze(0).to(device)

                        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

                        was_training = self.model.training
                        self.model.eval()
                        with torch.inference_mode():
                            output_ids = self.model.generate(
                                inputs=input_ids,
                                images=image_tensor,
                                image_sizes=image_sizes,
                                do_sample=False,
                                max_new_tokens=256,
                                use_cache=True
                            )
                        if isinstance(output_ids, tuple):
                            output_ids = output_ids[0]
                        if was_training:
                            self.model.train()

                        output_text = self.tokenizer.decode(
                            output_ids[0, :],
                            skip_special_tokens=True,
                        ).strip()
                        output_token_count = len(self.tokenizer.encode(output_text, add_special_tokens=False))
                        passed_min_tokens = output_token_count >= 50
                        checkpoint_name = f"checkpoint-{self.state.global_step}"
                        
                        rank0_print(f"Sanity check for step {self.state.global_step} resulted text: {output_text}")
                        rank0_print(f"Sanity check output token count: {output_token_count} tokens")

                        safe_wandb_log(self.args, {
                            "sanity/checkpoint_prompt": prompt_text,
                            "sanity/checkpoint_output": output_text,
                            "sanity/checkpoint_output_tokens": output_token_count,
                        }, step=self.state.global_step)

                        if wandb.run is not None:
                            if self._sanity_table is None:
                                rank0_print("Initializing sanity check wandb table.")
                                self._sanity_table = wandb.Table(columns=[
                                    "step",
                                    "checkpoint",
                                    "prompt",
                                    "output",
                                    "output_tokens",
                                    "passed_min_tokens",
                                ], log_mode="MUTABLE")
                            rank0_print(f"Adding sanity check entry to wandb table for step {self.state.global_step}.")
                            self._sanity_table.add_data(
                                int(self.state.global_step),
                                checkpoint_name,
                                prompt_text,
                                output_text,
                                int(output_token_count),
                                bool(passed_min_tokens),
                            )
                            safe_wandb_log(self.args, {
                                "sanity/checkpoint_table": self._sanity_table
                            }, step=self.state.global_step)

                        if not passed_min_tokens:
                            self._sanity_consecutive_failures += 1
                            rank0_print(
                                "Sanity check failed: output had "
                                f"{output_token_count} tokens (<50). "
                                "Consecutive failures: "
                                f"{self._sanity_consecutive_failures}/{self._sanity_failure_limit}."
                            )
                            if self._sanity_consecutive_failures >= self._sanity_failure_limit:
                                should_stop = True
                        else:
                            if self._sanity_consecutive_failures:
                                rank0_print(
                                    "Sanity check passed: resetting consecutive failure count "
                                    f"(was {self._sanity_consecutive_failures})."
                                )
                            self._sanity_consecutive_failures = 0

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            stop_tensor = torch.tensor(int(should_stop), device=self.args.device)
            torch.distributed.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())

        return should_stop

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False) or (
            hasattr(self.args, "mm_tunable_parts") and (len(self.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in self.args.mm_tunable_parts or "mm_vision_resampler" in self.args.mm_tunable_parts))
        ):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ["mm_projector", "vision_resampler"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)
            non_lora_weight_to_save = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters())
            for key in list(non_lora_weight_to_save.keys()):
                if key in weight_to_save:
                    del non_lora_weight_to_save[key]

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                if weight_to_save:
                    torch.save(weight_to_save, os.path.join(output_dir, "mm_projector.bin"))
                if non_lora_weight_to_save:
                    torch.save(non_lora_weight_to_save, os.path.join(output_dir, "non_lora_trainables.bin"))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

            if getattr(self.args, "lora_enable", False):
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)

                non_lora_weight_to_save = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters())
                if non_lora_weight_to_save:
                    os.makedirs(output_dir, exist_ok=True)
                    torch.save(non_lora_weight_to_save, os.path.join(output_dir, "non_lora_trainables.bin"))
                    projector_weights = {k: v for k, v in non_lora_weight_to_save.items() if "mm_projector" in k or "vision_resampler" in k}
                    if projector_weights:
                        torch.save(projector_weights, os.path.join(output_dir, "mm_projector.bin"))

        if self._run_checkpoint_sanity_check():
            rank0_print("Stopping training: sanity check output too short.")
            self.control.should_training_stop = True

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model)

        target_model = model if model is not None else self.model
        if target_model is None:
            return

        non_lora_path = os.path.join(resume_from_checkpoint, "non_lora_trainables.bin")
        if os.path.isfile(non_lora_path):
            state_dict = torch.load(non_lora_path, map_location="cpu")
            if state_dict:
                load_result = target_model.load_state_dict(state_dict, strict=False)
                self._issue_warnings_after_load(load_result)

    def _move_model_to_device(self, model: nn.Module, device: torch.device) -> None:
        """
        Move model to device, handling meta tensors properly.
        """
        try:
            # Check if any parameters are meta tensors
            has_meta_params = any(param.is_meta for param in model.parameters())
            
            if has_meta_params:
                # Use to_empty() for meta tensors
                model = model.to_empty(device=device)
            else:
                # Use standard to() for regular tensors
                model = model.to(device)
        except (RuntimeError, NotImplementedError) as e:
            if "meta tensor" in str(e).lower():
                # Fallback to to_empty() if meta tensor error occurs
                model = model.to_empty(device=device)
            else:
                raise e


class LLaVADPOTrainer(DPOTrainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False) or (
            hasattr(self.args, "mm_tunable_parts") and (len(self.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in self.args.mm_tunable_parts or "mm_vision_resampler" in self.args.mm_tunable_parts))
        ):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ["mm_projector", "vision_resampler"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)
            non_lora_weight_to_save = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters())
            for key in list(non_lora_weight_to_save.keys()):
                if key in weight_to_save:
                    del non_lora_weight_to_save[key]

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                if weight_to_save:
                    torch.save(weight_to_save, os.path.join(output_dir, "mm_projector.bin"))
                if non_lora_weight_to_save:
                    torch.save(non_lora_weight_to_save, os.path.join(output_dir, "non_lora_trainables.bin"))
        else:
            # super(LLaVADPOTrainer, self)._save_checkpoint(model, trial, metrics)
            # print(type(model))
            # from transformers.modeling_utils import unwrap_model
            # print(type(unwrap_model(model)))
            # print(unwrap_model(model).config)
            if self.args.lora_enable:
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)
                from transformers.modeling_utils import unwrap_model

                unwrapped_model = unwrap_model(model)
                non_lora_weight_to_save = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters())
                if non_lora_weight_to_save:
                    os.makedirs(output_dir, exist_ok=True)
                    torch.save(non_lora_weight_to_save, os.path.join(output_dir, "non_lora_trainables.bin"))
                    projector_weights = {k: v for k, v in non_lora_weight_to_save.items() if "mm_projector" in k or "vision_resampler" in k}
                    if projector_weights:
                        torch.save(projector_weights, os.path.join(output_dir, "mm_projector.bin"))
                self.save_my_lora_ckpt(output_dir, self.args, unwrapped_model)
            else:
                super(LLaVADPOTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(LLaVADPOTrainer, self)._save(output_dir, state_dict)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model)

        target_model = model if model is not None else self.model
        if target_model is None:
            return

        non_lora_path = os.path.join(resume_from_checkpoint, "non_lora_trainables.bin")
        if os.path.isfile(non_lora_path):
            state_dict = torch.load(non_lora_path, map_location="cpu")
            if state_dict:
                load_result = target_model.load_state_dict(state_dict, strict=False)
                self._issue_warnings_after_load(load_result)
