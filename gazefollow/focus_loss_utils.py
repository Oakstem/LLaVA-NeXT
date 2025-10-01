from typing import Any, List, Optional

import torch


FOCUS_LOSS_DEFAULT_THRESHOLD = 20.0


def prepare_focus_phrase_sequences(raw_ids: Optional[Any], device: torch.device) -> List[torch.LongTensor]:
    """Normalize focus phrase id inputs to a list of unique tensors on the target device."""
    sequences: List[torch.LongTensor] = []
    if raw_ids is None:
        return sequences

    if isinstance(raw_ids, torch.Tensor):
        if raw_ids.ndim == 1:
            sequences.append(raw_ids.to(device=device, dtype=torch.long))
        elif raw_ids.ndim == 2:
            for row in raw_ids:
                sequences.append(row.to(device=device, dtype=torch.long))
        else:
            raise ValueError("focus_loss_phrase_token_ids tensor must be 1D or 2D")
    elif isinstance(raw_ids, (list, tuple)):
        nested = raw_ids if (raw_ids and isinstance(raw_ids[0], (list, tuple, torch.Tensor))) else [raw_ids]
        for seq in nested:
            seq_tensor = torch.as_tensor(seq, dtype=torch.long, device=device)
            if seq_tensor.numel() > 0:
                sequences.append(seq_tensor)
    else:
        raise TypeError("focus_loss_phrase_token_ids must be a tensor or (list of) sequences")

    unique_sequences: List[torch.LongTensor] = []
    seen: set = set()
    for seq in sequences:
        key = tuple(seq.tolist())
        if key not in seen:
            seen.add(key)
            unique_sequences.append(seq)

    return unique_sequences


def locate_focus_start_index(sample_labels: torch.LongTensor, candidate_sequences: List[torch.LongTensor]) -> Optional[int]:
    """Return index immediately after the focus phrase, or None if the phrase is absent."""
    match_end_index: Optional[int] = None
    seq_length = sample_labels.size(0)
    for candidate in candidate_sequences:
        cand_len = candidate.size(0)
        if cand_len == 0 or seq_length < cand_len:
            continue
        for start_idx in range(seq_length - cand_len + 1):
            window = sample_labels[start_idx : start_idx + cand_len]
            if (window >= 0).all() and torch.equal(window, candidate):
                candidate_end = start_idx + cand_len
                if match_end_index is None or candidate_end > match_end_index:
                    match_end_index = candidate_end
    return match_end_index


def compute_focus_loss_after_phrase(
    shift_logits: torch.Tensor,
    shift_labels: torch.Tensor,
    full_labels: torch.Tensor,
    loss_fct: torch.nn.Module,
    candidate_sequences: List[torch.LongTensor],
    missing_value: Optional[float] = None,
) -> torch.Tensor:
    """Compute the loss using only tokens that follow the target phrase."""
    if not candidate_sequences:
        raise ValueError("focus_loss_after_phrase is enabled but no focus phrase sequences were provided")

    batch_size = shift_labels.size(0)
    seq_len_minus_one = shift_labels.size(1)
    threshold = FOCUS_LOSS_DEFAULT_THRESHOLD if missing_value is None else float(missing_value)
    missing_value_tensor = shift_logits.new_tensor(threshold)
    sample_losses: List[torch.Tensor] = []

    for batch_idx in range(batch_size):
        sample_labels = full_labels[batch_idx]
        focus_start = locate_focus_start_index(sample_labels, candidate_sequences)
        focus_mask = torch.zeros(seq_len_minus_one, dtype=torch.bool, device=shift_labels.device)

        if focus_start is not None and focus_start < sample_labels.size(0):
            shift_start_idx = max(focus_start - 1, 0)
            if shift_start_idx < seq_len_minus_one:
                focus_mask[shift_start_idx:] = True

        valid_mask = (shift_labels[batch_idx] >= 0) & focus_mask
        if valid_mask.any():
            sample_logits = shift_logits[batch_idx][valid_mask]
            sample_targets = shift_labels[batch_idx][valid_mask]
            sample_losses.append(loss_fct(sample_logits, sample_targets))
        else:
            sample_losses.append(missing_value_tensor)

    return torch.stack(sample_losses).mean()

