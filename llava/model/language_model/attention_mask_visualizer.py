from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
import torch


@dataclass
class AttentionMaskVizMetadata:
    """Metadata for serialized attention mask visualizations."""

    step_idx: int
    token_text: str
    query_index: int
    sequence_length: int
    category_ranges: Dict[str, List[List[int]]]
    top_candidates: Optional[List[Dict[str, Any]]] = None


@dataclass
class AttentionMaskFrame:
    """In-memory representation of a single attention mask snapshot."""

    step_idx: int
    token_text: str
    query_index: int
    mask: np.ndarray
    category_ranges: Dict[str, List[List[int]]]
    top_candidates: Optional[List[Dict[str, Any]]] = None


def _sanitize_token_text(token_text: str) -> str:
    filtered = "".join(ch if ch.isalnum() else "_" for ch in token_text)
    return filtered or "token"


def _ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _prepare_mask_tensor(mask: torch.Tensor) -> torch.Tensor:
    if mask is None:
        raise ValueError("Received None attention mask for visualization.")

    if mask.dim() == 4:
        mask_2d = mask[0, 0]
    elif mask.dim() == 3:
        mask_2d = mask[0]
    elif mask.dim() == 2:
        mask_2d = mask
    else:
        raise ValueError(f"Unsupported attention mask rank: {mask.dim()}")

    return mask_2d.detach().to(device="cpu", dtype=torch.float32)


def _extract_indices(
    values: Optional[Sequence[int] | torch.Tensor], seq_len: int
) -> torch.Tensor:
    if values is None:
        return torch.empty(0, dtype=torch.long)
    if isinstance(values, list):
        values = values[0] if values else None
    if isinstance(values, torch.Tensor):
        tensor = values.detach().cpu().to(dtype=torch.long)
    else:
        tensor = torch.as_tensor(list(values), dtype=torch.long)
    if tensor.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    tensor = tensor[(tensor >= 0) & (tensor < seq_len)]
    if tensor.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    return torch.unique(tensor)


def _indices_to_ranges(indices: torch.Tensor) -> List[List[int]]:
    if indices.numel() == 0:
        return []
    sorted_vals = indices.sort().values.tolist()
    ranges: List[List[int]] = []
    start = prev = sorted_vals[0]
    for val in sorted_vals[1:]:
        if val == prev + 1:
            prev = val
        else:
            ranges.append([start, prev])
            start = prev = val
    ranges.append([start, prev])
    return ranges


def _collect_category_ranges(
    seq_len: int,
    tokens_indexing: Optional[Dict],
    system_token_indices: Optional[Sequence[int] | torch.Tensor],
) -> Dict[str, List[List[int]]]:
    ranges: Dict[str, List[List[int]]] = {"system": [], "image": [], "text": []}
    text_source = tokens_indexing.get("text") if isinstance(tokens_indexing, dict) else None
    image_source = tokens_indexing.get("image") if isinstance(tokens_indexing, dict) else None
    fallback_system_source = tokens_indexing.get("system") if isinstance(tokens_indexing, dict) else None

    text_tensor = _extract_indices(text_source, seq_len)
    image_tensor = _extract_indices(image_source, seq_len)
    system_tensor = _extract_indices(system_token_indices, seq_len)
    if system_tensor.numel() == 0:
        system_tensor = _extract_indices(fallback_system_source, seq_len)

    ranges["system"] = _indices_to_ranges(system_tensor)
    ranges["image"] = _indices_to_ranges(image_tensor)

    if system_tensor.numel() > 0 and text_tensor.numel() > 0:
        mask = ~torch.isin(text_tensor, system_tensor)
        text_tensor = text_tensor[mask]
    ranges["text"] = _indices_to_ranges(text_tensor)
    return ranges


def _category_shapes_and_annotations(
    ranges: Dict[str, List[List[int]]],
) -> Tuple[List[Dict], List[Dict]]:
    colors = {"system": "#636EFA", "image": "#EF553B", "text": "#00CC96"}
    shapes = []
    annotations = []
    for category, spans in ranges.items():
        color = colors.get(category, "#FFFFFF")
        for start, end in spans:
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=start - 0.5,
                    x1=end + 0.5,
                    y0=1.02,
                    y1=1.05,
                    fillcolor=color,
                    line=dict(width=0),
                )
            )
            shapes.append(
                dict(
                    type="rect",
                    xref="paper",
                    yref="y",
                    x0=-0.08,
                    x1=-0.04,
                    y0=start - 0.5,
                    y1=end + 0.5,
                    fillcolor=color,
                    line=dict(width=0),
                )
            )
        if spans:
            annotations.append(
                dict(
                    x=1.03,
                    y=(spans[0][0] + spans[-1][1]) / 2,
                    xref="paper",
                    yref="y",
                    text=category.title(),
                    showarrow=False,
                    font=dict(color=color),
                )
            )

    return shapes, annotations


def _normalize_top_candidates(
    top_candidates: Optional[Sequence[Dict[str, Any]]],
    limit: int = 3,
) -> Optional[List[Dict[str, Any]]]:
    if not top_candidates:
        return None

    normalized: List[Dict[str, Any]] = []
    for candidate in top_candidates:
        if candidate is None:
            continue
        token_text = candidate.get("token_text") or candidate.get("text") or ""
        token_text = str(token_text).strip()
        if not token_text:
            token_id = candidate.get("token_id")
            token_text = f"<{token_id}>" if token_id is not None else "<unk>"
        probability = candidate.get("probability")
        try:
            prob_value = float(probability) if probability is not None else None
        except (TypeError, ValueError):
            prob_value = None
        normalized.append(
            {
                "token_text": token_text,
                "probability": prob_value,
                "token_id": candidate.get("token_id"),
            }
        )
        if len(normalized) >= limit:
            break
    return normalized or None


def _format_top_candidate_text(
    top_candidates: Optional[Sequence[Dict[str, Any]]],
) -> Optional[str]:
    if not top_candidates:
        return None
    parts: List[str] = []
    for candidate in top_candidates:
        token_text = candidate.get("token_text") or "<unk>"
        prob = candidate.get("probability")
        if prob is None:
            parts.append(token_text)
        else:
            parts.append(f"{token_text} ({prob:.3f})")
    if not parts:
        return None
    return "Top candidates: " + " · ".join(parts)


def _build_candidate_annotation(
    top_candidates: Optional[Sequence[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    summary = _format_top_candidate_text(top_candidates)
    if not summary:
        return None
    return dict(
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        text=summary,
        showarrow=False,
        font=dict(size=12),
    )


def _add_category_shapes(
    fig: go.Figure,
    ranges: Dict[str, List[List[int]]],
) -> None:
    shapes, annotations = _category_shapes_and_annotations(ranges)
    existing_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    existing_shapes.extend(shapes)
    fig.update_layout(shapes=existing_shapes)
    if annotations:
        existing_annotations = list(fig.layout.annotations) if fig.layout.annotations else []
        existing_annotations.extend(annotations)
        fig.update_layout(annotations=existing_annotations)


def _render_interactive_heatmap(
    mask_2d: torch.Tensor,
    step_idx: int,
    token_text: str,
    query_index: int,
    category_ranges: Dict[str, List[List[int]]],
    output_path: Path,
    *,
    top_candidates: Optional[List[Dict[str, Any]]] = None,
) -> None:
    mask_array = mask_2d.numpy()
    heatmap = go.Heatmap(
        z=mask_array,
        colorscale="Viridis",
        colorbar=dict(title="Attention weight"),
        hovertemplate="Query %{y}<br>Key %{x}<br>Weight %{z:.4f}<extra></extra>",
    )
    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title=f"Custom Attention Mask · Step {step_idx} · Token: {token_text}",
        xaxis=dict(
            title="Key token index",
            showgrid=False,
            zeroline=False,
            showspikes=True,
            spikemode="across",
        ),
        yaxis=dict(
            title="Query token index",
            showgrid=False,
            zeroline=False,
            autorange="reversed",
            showspikes=True,
            spikemode="across",
        ),
        margin=dict(l=80, r=120, t=80, b=120),
        height=800,
        width=900,
    )
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=mask_array.shape[1] - 0.5,
        y0=query_index,
        y1=query_index,
        line=dict(color="#FF4136", width=2),
    )
    _add_category_shapes(fig, category_ranges)
    candidate_annotation = _build_candidate_annotation(top_candidates)
    if candidate_annotation:
        existing_annotations = list(fig.layout.annotations) if fig.layout.annotations else []
        existing_annotations.append(candidate_annotation)
        fig.update_layout(annotations=existing_annotations)
    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)


def _write_metadata(output_path: Path, metadata: AttentionMaskVizMetadata) -> None:
    json_path = output_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata.__dict__, f, indent=2)

def capture_attention_mask_frame(
    attention_mask: torch.Tensor,
    tokens_indexing: Optional[Dict],
    step_idx: int,
    token_text: str,
    *,
    query_index: Optional[int] = None,
    system_token_indices: Optional[Sequence[int] | torch.Tensor] = None,
    top_candidates: Optional[Sequence[Dict[str, Any]]] = None,
) -> AttentionMaskFrame:
    if attention_mask is None:
        raise ValueError("attention_mask is required to capture a visualization frame.")

    token_text = token_text if token_text else "<unk>"
    mask_2d = _prepare_mask_tensor(attention_mask)
    seq_len = mask_2d.shape[0]
    query_position = seq_len - 1 if query_index is None else query_index
    query_position = min(max(query_position, 0), seq_len - 1)
    category_ranges = _collect_category_ranges(seq_len, tokens_indexing, system_token_indices)

    normalized_candidates = _normalize_top_candidates(top_candidates)

    return AttentionMaskFrame(
        step_idx=int(step_idx),
        token_text=token_text,
        query_index=query_position,
        mask=mask_2d.numpy(),
        category_ranges=category_ranges,
        top_candidates=normalized_candidates,
    )


def _build_frame_shapes(frame: AttentionMaskFrame) -> Tuple[List[Dict], List[Dict]]:
    shapes, annotations = _category_shapes_and_annotations(frame.category_ranges)
    shapes = list(shapes)
    shapes.append(
        dict(
            type="line",
            x0=-0.5,
            x1=frame.mask.shape[1] - 0.5,
            y0=frame.query_index,
            y1=frame.query_index,
            line=dict(color="#FF4136", width=2),
        )
    )
    candidate_annotation = _build_candidate_annotation(frame.top_candidates)
    if candidate_annotation:
        annotations = list(annotations)
        annotations.append(candidate_annotation)
    return shapes, annotations


def save_attention_mask_sequence(
    frames: List[AttentionMaskFrame],
    output_path: Path | str,
    *,
    title_prefix: str = "Custom Attention Mask",
) -> Optional[Path]:
    if not frames:
        return None

    sorted_frames = sorted(frames, key=lambda f: f.step_idx)
    zmin = min(float(np.min(frame.mask)) for frame in sorted_frames)
    zmax = max(float(np.max(frame.mask)) for frame in sorted_frames)

    traces = []
    frame_shapes: List[Tuple[List[Dict], List[Dict]]] = []
    for idx, frame in enumerate(sorted_frames):
        show_scale = idx == 0
        trace = go.Heatmap(
            z=frame.mask,
            colorscale="Viridis",
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Attention weight") if show_scale else None,
            showscale=show_scale,
            hovertemplate="Query %{y}<br>Key %{x}<br>Weight %{z:.4f}<extra></extra>",
            visible=idx == 0,
        )
        traces.append(trace)
        frame_shapes.append(_build_frame_shapes(frame))

    base_frame = sorted_frames[0]
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{title_prefix} · Step {base_frame.step_idx} · Token: {base_frame.token_text}",
        xaxis=dict(
            title="Key token index",
            showgrid=False,
            zeroline=False,
            showspikes=True,
            spikemode="across",
        ),
        yaxis=dict(
            title="Query token index",
            showgrid=False,
            zeroline=False,
            autorange="reversed",
            showspikes=True,
            spikemode="across",
        ),
        margin=dict(l=80, r=120, t=80, b=120),
        height=800,
        width=900,
    )
    initial_shapes, initial_annotations = frame_shapes[0]
    fig.update_layout(shapes=initial_shapes, annotations=initial_annotations)

    slider_steps = []
    trace_count = len(traces)
    for idx, frame in enumerate(sorted_frames):
        label_text = frame.token_text if frame.token_text else "<unk>"
        label = f"{frame.step_idx}: {label_text[:24]}"
        visibility = [False] * trace_count
        visibility[idx] = True
        shapes, annotations = frame_shapes[idx]
        slider_steps.append(
            {
                "method": "update",
                "label": label,
                "args": [
                    {"visible": visibility},
                    {
                        "title": f"{title_prefix} · Step {frame.step_idx} · Token: {frame.token_text}",
                        "shapes": shapes,
                        "annotations": annotations,
                    },
                ],
            }
        )
    fig.update_layout(
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Active Step: "},
                "pad": {"t": 50},
                "steps": slider_steps,
            }
        ],
    )

    output_path = _ensure_output_dir(Path(output_path).parent) / Path(output_path).name
    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)
    return output_path


def visualize_attention_mask_step(
    attention_mask: torch.Tensor,
    tokens_indexing: Optional[Dict],
    step_idx: int,
    token_text: str,
    output_dir: Path | str,
    *,
    query_index: Optional[int] = None,
    system_token_indices: Optional[Sequence[int] | torch.Tensor] = None,
    filename_prefix: str = "custom_mask",
    top_candidates: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[Path]:
    """
    Legacy helper that writes an individual HTML file for a single attention mask frame.
    """
    if attention_mask is None:
        return None

    frame = capture_attention_mask_frame(
        attention_mask=attention_mask,
        tokens_indexing=tokens_indexing,
        step_idx=step_idx,
        token_text=token_text,
        query_index=query_index,
        system_token_indices=system_token_indices,
        top_candidates=top_candidates,
    )

    safe_token = _sanitize_token_text(frame.token_text)
    filename = f"{filename_prefix}_{frame.step_idx:05d}_{safe_token}.html"
    output_path = _ensure_output_dir(Path(output_dir)) / filename

    _render_interactive_heatmap(
        mask_2d=torch.from_numpy(frame.mask),
        step_idx=frame.step_idx,
        token_text=frame.token_text,
        query_index=frame.query_index,
        category_ranges=frame.category_ranges,
        output_path=output_path,
        top_candidates=frame.top_candidates,
    )

    metadata = AttentionMaskVizMetadata(
        step_idx=frame.step_idx,
        token_text=frame.token_text,
        query_index=frame.query_index,
        sequence_length=frame.mask.shape[0],
        category_ranges=frame.category_ranges,
        top_candidates=frame.top_candidates,
    )
    _write_metadata(output_path, metadata)
    return output_path


__all__ = [
    "visualize_attention_mask_step",
    "capture_attention_mask_frame",
    "save_attention_mask_sequence",
    "AttentionMaskFrame",
]
