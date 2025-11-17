# coding=utf-8
"""Utilities for injecting hidden states through forward hooks on Qwen2 layers."""

from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

HiddenStatePatchEntry = Tuple[int, torch.Tensor]
HiddenStatePatchConfig = Dict[int, List[HiddenStatePatchEntry]]


def _normalize_layer_index(idx: int, num_layers: int) -> int:
    normalized_idx = idx
    if normalized_idx < 0:
        normalized_idx = num_layers + normalized_idx
    if normalized_idx < 0 or normalized_idx >= num_layers:
        raise ValueError(f"Layer index {idx} is out of range for {num_layers} decoder layers.")
    return normalized_idx


def _prepare_patch_entries(
    entry: Dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[HiddenStatePatchEntry]:
    positions = entry.get("positions")
    embedding = entry.get("embedding")
    if positions is None or embedding is None:
        return []

    positions_tensor = torch.as_tensor(positions, device=device, dtype=torch.long).view(-1)
    if positions_tensor.numel() == 0:
        return []

    value = embedding.to(device=device, dtype=dtype)
    if value.dim() == 1:
        value = value.unsqueeze(0).unsqueeze(0)
    elif value.dim() == 2:
        value = value.unsqueeze(0)
    elif value.dim() != 3:
        raise ValueError(f"Unsupported embedding shape for repr injection: {tuple(value.shape)}")

    target_len = positions_tensor.numel()
    if value.size(1) == 1 and target_len > 1:
        value = value.expand(-1, target_len, -1)
    elif value.size(1) != target_len:
        raise ValueError(
            f"Representation embedding length ({value.size(1)}) does not match "
            f"number of target positions ({target_len})."
        )

    if value.size(0) == 1 and batch_size > 1:
        value = value.expand(batch_size, -1, -1)
    elif value.size(0) != batch_size:
        raise ValueError(
            f"Representation embedding batch dimension ({value.size(0)}) does not match "
            f"model batch size ({batch_size})."
        )

    value = value.contiguous()
    patches: List[HiddenStatePatchEntry] = []
    for offset, position in enumerate(positions_tensor.tolist()):
        token_value = value[:, offset, :].clone()
        if token_value.size(0) == 1:
            token_value = token_value.squeeze(0)
        patches.append((int(position), token_value))
    return patches


def build_qwen2_hidden_state_patch_config(
    repr_injection: Optional[Dict[str, Dict[str, torch.Tensor]]],
    repr_layer_idx: Optional[Union[int, Dict[str, int]]],
    num_hidden_layers: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> HiddenStatePatchConfig:
    """
    Convert repr_injection payload into per-layer patch configuration consumable by the hook helpers.
    """
    if not repr_injection or repr_layer_idx is None:
        return {}

    patch_config: HiddenStatePatchConfig = {}

    if isinstance(repr_layer_idx, dict):
        for key, idx in repr_layer_idx.items():
            if idx is None:
                continue
            normalized_idx = _normalize_layer_index(idx, num_hidden_layers)
            entry = repr_injection.get(key)
            if not entry:
                continue
            patches = _prepare_patch_entries(entry, batch_size, device, dtype)
            if patches:
                patch_config.setdefault(normalized_idx, []).extend(patches)
    else:
        normalized_idx = _normalize_layer_index(repr_layer_idx, num_hidden_layers)
        combined_patches: List[HiddenStatePatchEntry] = []
        for entry in repr_injection.values():
            if not entry:
                continue
            combined_patches.extend(_prepare_patch_entries(entry, batch_size, device, dtype))
        if combined_patches:
            patch_config[normalized_idx] = combined_patches

    return patch_config


def _assign_patch(hidden_states: torch.Tensor, position: int, value: torch.Tensor) -> None:
    seq_len = hidden_states.size(1)
    if position < 0:
        position = seq_len + position
    if position < 0 or position >= seq_len:
        return

    if value.dim() == 1:
        patch_value = value.unsqueeze(0).expand(hidden_states.size(0), -1)
    elif value.dim() == 2:
        if value.size(0) != hidden_states.size(0):
            raise ValueError(
                f"Patch tensor batch dimension ({value.size(0)}) does not match hidden states "
                f"batch size ({hidden_states.size(0)})."
            )
        patch_value = value
    else:
        raise ValueError(f"Unsupported value rank for patching hidden states: {value.dim()}")

    hidden_states[:, position, :] = patch_value


def _make_patch_fn(
    patches: List[HiddenStatePatchEntry],
    generation_mode: bool,
    patch_input: bool,
):
    def _apply(target: torch.Tensor) -> None:
        if target.dim() != 3:
            return
        if generation_mode and target.size(1) == 1:
            return
        for position, value in patches:
            _assign_patch(target, position, value)

    if patch_input:

        def _pre_hook(_module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
            if not inputs:
                return
            hidden_states = inputs[0]
            if torch.is_tensor(hidden_states):
                _apply(hidden_states)

        return _pre_hook

    def _post_hook(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], outputs):
        if isinstance(outputs, torch.Tensor):
            hidden_states = outputs
        elif isinstance(outputs, (tuple, list)) and outputs:
            hidden_states = outputs[0]
        else:
            return

        if torch.is_tensor(hidden_states):
            _apply(hidden_states)

    return _post_hook


def set_qwen2_hs_patch_hooks(
    layers: nn.ModuleList,
    hs_patch_config: HiddenStatePatchConfig,
    patch_input: bool = True,
    generation_mode: bool = False,
) -> List[RemovableHandle]:
    """
    Register forward hooks on decoder layers to patch hidden states at runtime.
    """
    hooks: List[RemovableHandle] = []
    for layer_idx, patches in hs_patch_config.items():
        if not patches:
            continue
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} is out of range for provided decoder layers.")
        hook_fn = _make_patch_fn(patches, generation_mode=generation_mode, patch_input=patch_input)
        if patch_input:
            hook = layers[layer_idx].register_forward_pre_hook(hook_fn)
        else:
            hook = layers[layer_idx].register_forward_hook(hook_fn)
        hooks.append(hook)
    return hooks


def remove_hooks(hooks: Iterable[RemovableHandle]) -> None:
    for hook in hooks:
        hook.remove()
