import os
import inspect
import torch
import torch.nn as nn
import datetime
import json
import pathlib
import time
from collections import OrderedDict

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
from typing import List, Optional, Dict
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

from llava.utils import rank0_print


def safe_wandb_log(args, metrics_dict, step=None):
    """Safely log metrics to wandb if available and initialized."""
    if not (args.report_to and "wandb" in args.report_to):
        return
    
    try:
        import wandb
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
            self.last_log_step = self.state.global_step

        return loss.detach() / self.args.gradient_accumulation_steps
    
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
            
        safe_wandb_log(self.args, metrics, step=self.state.global_step)
        
        # Keep only recent timing data to avoid memory buildup
        max_history = 100
        for key in self.step_times:
            if len(self.step_times[key]) > max_history:
                self.step_times[key] = self.step_times[key][-max_history:]
    
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

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

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

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
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
                self.save_my_lora_ckpt(output_dir, self.args, unwrapped_model)
            else:
                super(LLaVADPOTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(LLaVADPOTrainer, self)._save(output_dir, state_dict)
