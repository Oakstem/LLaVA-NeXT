#!/usr/bin/env python3
"""Visualize cosine similarity between saved LLaVA text and image embeddings."""

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute similarity heatmaps between saved text embeddings and image features."
    )
    parser.add_argument("--text-embeddings", required=True, help="Path to the .pt file produced for text.")
    parser.add_argument("--image-embeddings", required=True, help="Path to the .pt file produced for image.")
    parser.add_argument(
        "--output-dir",
        default="similarity_outputs",
        help="Directory where similarity heatmaps will be written.",
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        help="Matplotlib colormap to use for heatmap overlays.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Alpha value for heatmap overlay (0=no heatmap, 1=heatmap only).",
    )
    parser.add_argument(
        "--include-vision-sources",
        default="mm_projected_features,vision_features",
        help="Comma-separated list of vision feature keys to consider.",
    )
    parser.add_argument(
        "--min-percentile",
        type=float,
        default=85.0,
        help="Percentile threshold (0-100) applied to similarity map before visualization. Use 0 to disable.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip generating outputs that already exist on disk.",
    )
    return parser.parse_args()


def sanitize_label(label: str) -> str:
    sanitized = re.sub(r"[^0-9a-zA-Z._-]+", "_", label).strip("_")
    return sanitized or "unnamed"


def to_float_tensor(value: torch.Tensor) -> torch.Tensor:
    if isinstance(value, List):
        value = torch.stack(value, dim=0)
        if value.ndim > 3:
            # find the axis with size 1 and squeeze it
            for axis in range(value.ndim):
                if value.shape[axis] == 1:
                    value = value.squeeze(dim=axis)
                    break
    if isinstance(value, torch.Tensor):
        return value.to(torch.float32)
    raise TypeError(f"Expected torch.Tensor but received {type(value)!r}.")


def extract_text_vectors(text_payload: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    vectors: Dict[str, torch.Tensor] = {}
    embeddings = text_payload.get("embeddings")
    if embeddings is not None:
        embeddings = to_float_tensor(embeddings)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("Expected embeddings tensor shaped [seq_len, hidden_size].")
        mean_vector = embeddings.mean(dim=0)
        vectors["raw_embedding_mean"] = mean_vector

    hidden_states: Iterable[torch.Tensor] = text_payload.get("hidden_states") or []
    for idx, layer in enumerate(hidden_states):
        layer_tensor = to_float_tensor(layer)
        if layer_tensor.ndim != 3 or layer_tensor.shape[1] == 0:
            continue
        last_token = layer_tensor[0, -1]
        vectors[f"layer_{idx:02d}_last_token"] = last_token
    return vectors


def gather_vision_feature_sets(
    payload: Dict[str, torch.Tensor],
    source_key: str,
    group_key: str,
) -> List[Tuple[str, torch.Tensor]]:
    if source_key not in payload:
        print(f" - No feature sets found for vision source '{source_key}', skipping.")
        return []

    source_tensor = to_float_tensor(payload[source_key])
    groups = payload.get(group_key)

    feature_sets: List[Tuple[str, torch.Tensor]] = []
    print(f"Gathering features from the sets with key '{source_key}' and group key '{group_key}':")
    if isinstance(groups, list) and groups:
        for idx, group in enumerate(groups):
            group_tensor = to_float_tensor(group)
            if group_tensor.ndim == 3:
                reduced = group_tensor.mean(dim=0)
            elif group_tensor.ndim == 2:
                reduced = group_tensor
            else:
                continue
            feature_sets.append((f"{source_key}_group{idx:02d}", reduced))
            print(f" - Added group {idx} with shape {reduced.shape}")
        return feature_sets

    if source_tensor.ndim == 3:
        for idx in range(source_tensor.shape[0]):
            feature_sets.append((f"{source_key}_view{idx:02d}", source_tensor[idx]))
            print(f" - Added view {idx} with shape {source_tensor[idx].shape}")
    elif source_tensor.ndim == 2:
        feature_sets.append((source_key, source_tensor))
        print(f" - Added full tensor with shape {source_tensor.shape}")
    else:
        print(f" - Source tensor has unsupported shape {source_tensor.shape}, skipping.")


    return feature_sets


def compute_similarity_map(text_vec: torch.Tensor, vision_tokens: torch.Tensor) -> torch.Tensor:
    normalized_text = F.normalize(text_vec, dim=0)
    normalized_tokens = F.normalize(vision_tokens, dim=-1)
    return torch.matmul(normalized_tokens, normalized_text)


def upscale_to_image(similarity: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    num_tokens = similarity.shape[0]
    grid_size = int(round(math.sqrt(num_tokens)))
    if grid_size * grid_size != num_tokens:
        raise ValueError(f"Token count {num_tokens} is not a perfect square; cannot reshape to grid.")
    as_grid = similarity.view(1, 1, grid_size, grid_size)
    resized = F.interpolate(
        as_grid,
        size=(image_size[1], image_size[0]),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0)


def overlay_heatmap(
    image: Image.Image,
    heatmap: torch.Tensor,
    cmap: str,
    alpha: float,
    min_percentile: float,
) -> Image.Image:
    heatmap_np = heatmap.numpy()

    if 0.0 < min_percentile < 100.0:
        cutoff = float(np.percentile(heatmap_np, min_percentile))
        low_value = float(heatmap_np.min())
        if cutoff > low_value:
            heatmap_np = np.where(heatmap_np >= cutoff, heatmap_np, low_value)

    min_val = float(heatmap_np.min())
    max_val = float(heatmap_np.max())
    if math.isclose(max_val - min_val, 0.0, rel_tol=1e-6, abs_tol=1e-6):
        normalized = np.zeros_like(heatmap_np, dtype=np.float32)
    else:
        normalized = (heatmap_np - min_val) / (max_val - min_val)

    colormap = cm.get_cmap(cmap)
    colored = colormap(normalized)[..., :3]  # Drop alpha channel

    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    overlay = (1.0 - alpha) * base + alpha * colored
    overlay = np.clip(overlay, 0.0, 1.0)
    return Image.fromarray((overlay * 255.0).astype(np.uint8))


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    text_payload = torch.load(args.text_embeddings, map_location="cpu")
    image_payload = torch.load(args.image_embeddings, map_location="cpu")
    
    print(f"Loaded text embeddings with keys: {list(text_payload.keys())}")
    print(f"Loaded image embeddings with keys: {list(image_payload.keys())}")

    text_vectors = extract_text_vectors(text_payload)
    if not text_vectors:
        raise RuntimeError("No text vectors available for similarity computation.")

    requested_sources = [key.strip() for key in args.include_vision_sources.split(",") if key.strip()]
    if not requested_sources:
        raise ValueError("At least one vision source must be specified via --include-vision-sources.")

    original_image_path = image_payload.get("image_path")
    if not original_image_path:
        raise RuntimeError("Image embeddings file does not contain 'image_path'.")
    original_image = Image.open(original_image_path).convert("RGB")
    original_size = tuple(image_payload.get("original_image_size", original_image.size))

    if original_image.size != original_size:
        original_image = original_image.resize(original_size, Image.BILINEAR)

    combinations = 0

    print(f"Requested vision sources: {requested_sources}")
    for vision_key in requested_sources:
        feature_sets = gather_vision_feature_sets(
            image_payload,
            source_key=vision_key,
            group_key=f"{vision_key.replace('features', 'feature')}_groups",
        )
        if not feature_sets:
            print(f" - No feature sets found for vision source '{vision_key}', skipping.")
            continue

        print(f"Processing vision source '{vision_key}' with feature sets: {[label for label, _ in feature_sets]}")

        for vision_label, vision_tensor in feature_sets:
            vision_tensor = to_float_tensor(vision_tensor)
            if vision_tensor.ndim != 2:
                print(f" - Skipping vision tensor with unsupported shape {vision_tensor.shape}")
                continue

            token_count, hidden_dim = vision_tensor.shape
            grid_dim = int(round(math.sqrt(token_count)))
            if grid_dim * grid_dim != token_count:
                print(f" - Skipping vision tensor with non-square token count {token_count}")
                continue

            text_vectors_list = list(text_vectors.items())
            last_key = text_vectors_list[-1][0] if text_vectors_list else "none"

            text_vec = text_vectors[last_key]
            text_label = last_key
            # for text_label, text_vec in text_vectors.items():
            # if text_vec.numel() == 0 or text_vec.shape[-1] != hidden_dim:
            #     print(f" - Skipping text vector '{text_label}' with shape {text_vec.shape}")
            #     continue

            similarity = compute_similarity_map(text_vec, vision_tensor)
            heatmap = upscale_to_image(similarity, original_size)
            overlay = overlay_heatmap(original_image, heatmap, args.cmap, args.alpha, args.min_percentile)

            text_name = sanitize_label(text_label)
            vision_name = sanitize_label(vision_label)
            filename = f"{vision_name}__{text_name}.png"
            destination_npy = output_dir / 'npy' / filename
            destination = output_dir / filename
            destination_npy.parent.mkdir(parents=True, exist_ok=True)
            heatmap_path = destination_npy.with_suffix(".npy")

            # if args.skip_existing and destination.exists() and heatmap_path.exists():
            #     print(f" - Skipping existing output for {filename}")
            #     continue

            # np.save(heatmap_path, heatmap.cpu().numpy())
            overlay.save(destination)
            combinations += 1

    print(f"Generated {combinations} similarity overlays in {output_dir}.")


if __name__ == "__main__":
    main()
