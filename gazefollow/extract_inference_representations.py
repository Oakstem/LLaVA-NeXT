#!/usr/bin/env python3
"""Collect vision and decoder representations for multimodal inference runs."""

import argparse
import copy
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_INDEX = -200
DEFAULT_MODEL_PATH = "lmms-lab/llava-onevision-qwen2-7b-ov-chat"
DEFAULT_ADAPTER_PATH = "training_outputs/llava-20260420_030228/checkpoint-6000"
DEFAULT_ADAPTER_PATH = None
DEFAULT_OUTPUT_DIR = "inference_representations"
DEFAULT_VIDEO = "/galitylab/students/alonmardi/Sherlock.S01E01.A.Study.in.Pink.mkv"
DEFAULT_MOVIE_NAME = "Sherlock_llava_20260420_030228_6k_ckpt"
DEFAULT_PROMPTS = [
    "Describe where each person in the image is looking and whether they are facing each other. Conclude by describing the interaction between them, if any.",
]
DEFAULT_WINDOW_DURATION_SECONDS = 3.0
DEFAULT_WINDOW_STRIDE_SECONDS = 1.5
DEFAULT_END_SECONDS = 3000
DEFAULT_LAYER_INDICES = [20, 28]


def resolve_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device"):
        return model.device  # type: ignore[return-value]
    return next(model.parameters()).device


def resolve_dtype(model: torch.nn.Module) -> torch.dtype:
    if hasattr(model, "dtype"):
        return model.dtype  # type: ignore[return-value]
    return next(model.parameters()).dtype


def fix_wsl_paths(path: Union[str, Path]) -> str:
    path = str(path)
    if path.startswith("/mnt/"):
        return path
    normalized = path.replace("\\", "/")
    if len(normalized) > 2 and normalized[1] == ":" and normalized[2] == "/":
        return f"/mnt/{normalized[0].lower()}/{normalized[3:]}"
    return normalized


def load_image(image_path: Union[str, Path]) -> Image.Image:
    image_path = str(image_path)
    if image_path.startswith("http"):
        import requests

        response = requests.get(image_path, stream=True)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    resolved = Path(image_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Image file not found at {resolved}")
    return Image.open(resolved).convert("RGB")


def default_output_path_for_image(image_path: Union[str, Path]) -> Path:
    resolved = fix_wsl_paths(image_path)
    image_name = Path(resolved.split("?")[0]).stem or "image"
    return Path(DEFAULT_OUTPUT_DIR) / f"{image_name}.pt"


def format_seconds_for_path(seconds: float) -> str:
    return f"{int(seconds)}s" if float(seconds).is_integer() else f"{seconds:g}s"


def default_output_dir_for_video(
    window_duration_seconds: float,
    movie_name: str = DEFAULT_MOVIE_NAME,
) -> Path:
    return Path(DEFAULT_OUTPUT_DIR) / f"{movie_name}_{format_seconds_for_path(window_duration_seconds)}"


def default_output_path_for_video_window(
    output_dir: Union[str, Path],
    start_frame: int,
) -> Path:
    return Path(output_dir) / f"{start_frame}.pt"


def prompt_output_dir_name(prompt_index: int) -> str:
    return f"prompt_{prompt_index + 1:02d}"


def write_prompt_run_metadata(
    output_dir: Union[str, Path],
    prompt: str,
    prompt_index: int,
    num_prompts: int,
    metadata: Dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"prompt_index: {prompt_index + 1}",
        f"num_prompts: {num_prompts}",
        f"prompt: {prompt}",
    ]
    lines.extend(f"{key}: {value}" for key, value in metadata.items())
    (output_dir / "run_metadata.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_prompts_file(prompts_path: Union[str, Path]) -> List[str]:
    path = Path(fix_wsl_paths(prompts_path))
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        prompts = json.loads(text)
        if not isinstance(prompts, list):
            raise ValueError("--prompts-file JSON must contain a list of prompt strings.")
        return [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def resolve_prompt_list(
    prompt: Optional[str],
    prompts: Optional[Sequence[str]],
    prompts_file: Optional[Union[str, Path]],
) -> List[str]:
    resolved_prompts: List[str] = []
    if prompt:
        resolved_prompts.append(prompt)
    if prompts:
        resolved_prompts.extend(prompts)
    if prompts_file:
        resolved_prompts.extend(load_prompts_file(prompts_file))

    resolved_prompts = [item.strip() for item in resolved_prompts if item and item.strip()]
    if not resolved_prompts:
        resolved_prompts = [prompt.strip() for prompt in DEFAULT_PROMPTS if prompt.strip()]
    if not resolved_prompts:
        raise ValueError("Provide at least one prompt using --prompt, --prompts, --prompts-file, or DEFAULT_PROMPTS.")
    return resolved_prompts


def parse_layer_indices(values: Optional[Sequence[str]]) -> Optional[List[int]]:
    if values is None:
        return list(DEFAULT_LAYER_INDICES) if DEFAULT_LAYER_INDICES is not None else None

    indices: List[int] = []
    for value in values:
        indices.extend(int(item) for item in str(value).replace(",", " ").split())
    return indices


def prepare_multimodal_image_tensor(
    pil_image: Image.Image,
    image_processor,
    model,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    from llava.mm_utils import process_images

    processed = process_images([pil_image.copy()], image_processor, model.config)
    image_tensor = processed[0] if isinstance(processed, tuple) else processed

    if isinstance(image_tensor, list):
        if not image_tensor:
            raise ValueError("Image processor returned an empty tensor list.")
        image_tensor = image_tensor[0]

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(
        device=resolve_device(model),
        dtype=resolve_dtype(model),
    )
    return image_tensor, pil_image.size


def load_video_window(
    video_path: Union[str, Path],
    max_frames_num: int,
    start_frame: int,
    fps: float,
    duration_seconds: float,
    vr=None,
) -> Dict[str, Any]:
    from decord import VideoReader, cpu

    if max_frames_num < 1:
        raise ValueError("max_frames_num must be positive.")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive.")

    resolved_video_path = fix_wsl_paths(video_path)
    if vr is None:
        vr = VideoReader(resolved_video_path, ctx=cpu(0))

    total_frames = len(vr)
    if total_frames < 1:
        raise ValueError(f"Video contains no frames: {resolved_video_path}")

    start_frame = max(0, min(int(start_frame), total_frames - 1))
    end_frame = min(start_frame + int(round(fps * duration_seconds)), total_frames - 1)
    frame_idx = np.linspace(start_frame, end_frame, max_frames_num, dtype=int).tolist()
    frames = vr.get_batch(frame_idx).asnumpy()
    return {
        "frames": frames,
        "frame_indices": frame_idx,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_seconds": start_frame / fps,
        "end_seconds": end_frame / fps,
        "fps": fps,
        "vr": vr,
    }


def prepare_multimodal_video_tensor(
    frames: np.ndarray,
    image_processor,
    model,
) -> Tuple[List[torch.Tensor], List[List[int]]]:
    video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    video_tensor = video_tensor.to(device=resolve_device(model), dtype=resolve_dtype(model))
    image_sizes = [[int(frame.shape[1]), int(frame.shape[0])] for frame in frames]
    return [video_tensor], image_sizes


def _as_vision_batch(image_tensor: torch.Tensor) -> torch.Tensor:
    if image_tensor.ndim == 5:
        return torch.cat([image for image in image_tensor], dim=0)
    if image_tensor.ndim == 4:
        return image_tensor
    if image_tensor.ndim == 3:
        return image_tensor.unsqueeze(0)
    raise ValueError(f"Unexpected image tensor shape for vision tower: {tuple(image_tensor.shape)}")


def _as_vision_batch_from_model_images(images: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
    if isinstance(images, list):
        return torch.cat([_as_vision_batch(image) for image in images], dim=0)
    return _as_vision_batch(images)


def _to_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu()


def _mean_pool_feature_tensor_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if tensor.ndim <= 1:
        return tensor.cpu()
    return tensor.mean(dim=tuple(range(tensor.ndim - 1))).cpu()


def _clone_tokens_indexing(tokens_indexing: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if tokens_indexing is None:
        return None

    def clone_value(value: Any) -> Any:
        if torch.is_tensor(value):
            return _to_cpu_tensor(value)
        if isinstance(value, dict):
            return {key: clone_value(val) for key, val in value.items()}
        if isinstance(value, list):
            return [clone_value(val) for val in value]
        if isinstance(value, tuple):
            return tuple(clone_value(val) for val in value)
        return value

    return {key: clone_value(value) for key, value in tokens_indexing.items()}


def _select_uniform_decoder_layers(hidden_states: Sequence[torch.Tensor], count: int = 3) -> List[int]:
    num_decoder_layers = len(hidden_states) - 1
    if num_decoder_layers < 1:
        raise RuntimeError("Decoder hidden states did not include layer outputs.")
    selected = torch.linspace(1, num_decoder_layers, steps=count).round().to(torch.long).tolist()
    return sorted(dict.fromkeys(int(idx) for idx in selected))


def _resolve_decoder_layer_indices(
    hidden_states: Sequence[torch.Tensor],
    layer_indices: Optional[Sequence[int]],
) -> List[int]:
    if layer_indices is None:
        return _select_uniform_decoder_layers(hidden_states)

    num_decoder_layers = len(hidden_states) - 1
    selected = list(dict.fromkeys(int(idx) for idx in layer_indices))
    if not selected:
        raise ValueError("At least one decoder layer index must be selected.")

    invalid = [idx for idx in selected if idx < 1 or idx > num_decoder_layers]
    if invalid:
        raise ValueError(
            "Decoder layer indices must be in hidden-state index range "
            f"1..{num_decoder_layers}; got {invalid}."
        )
    return selected


def _build_multimodal_prompt(prompt: str, tokenizer, conv_template: str) -> torch.Tensor:
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Prompt must be non-empty.")

    user_content = prompt if DEFAULT_IMAGE_TOKEN in prompt else f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.tokenizer = tokenizer
    conv.append_message(conv.roles[0], user_content)
    conv.append_message(conv.roles[1], None)
    return tokenizer_image_token(
        conv.get_prompt(),
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0)


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    filtered_logits = logits.clone()
    filtered_logits.scatter_(1, sorted_indices, sorted_logits.masked_fill(sorted_indices_to_remove, float("-inf")))
    return filtered_logits


def _sample_next_token(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[int],
) -> torch.Tensor:
    if not do_sample:
        return torch.argmax(logits, dim=-1)

    if temperature and temperature != 1.0:
        logits = logits / temperature
    if top_k is not None and top_k > 0:
        top_k_logits, top_k_indices = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
        filtered = torch.full_like(logits, float("-inf"))
        logits = filtered.scatter(1, top_k_indices, top_k_logits)
    if top_p is not None and top_p < 1.0:
        logits = _top_p_filter(logits, top_p)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)


def _collect_prepared_inference_representations(
    prompt: str,
    tokenizer,
    model,
    images: Union[torch.Tensor, List[torch.Tensor]],
    image_sizes_for_model: List[Any],
    media_metadata: Dict[str, Any],
    output_path: Union[str, Path],
    conv_template: str = "qwen_1_5",
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    modality: str = "image",
    layer_indices: Optional[Sequence[int]] = DEFAULT_LAYER_INDICES,
) -> Dict[str, Any]:
    """Run one multimodal prompt pass and save selected representations.

    The saved file is a torch-serialized dictionary. Hidden-state indices follow the
    Hugging Face convention: index 0 is the token embedding output, and indices
    1..N are decoder layer outputs.
    """
    model.eval()
    base_model = model.get_model()
    vision_tower = base_model.get_vision_tower()
    if vision_tower is None:
        raise RuntimeError("Loaded model does not expose a vision tower.")

    with torch.inference_mode():
        vision_inputs = _as_vision_batch_from_model_images(images)
        vision_encoder_output = vision_tower(vision_inputs)
        mm_projector_output = base_model.mm_projector(vision_encoder_output)

    input_ids = _build_multimodal_prompt(prompt, tokenizer, conv_template).to(resolve_device(model))
    current_input_ids = input_ids
    past_key_values = None
    generated_token_ids: List[int] = []
    selected_layer_indices: Optional[List[int]] = None
    generated_token_hidden_states: Dict[str, List[torch.Tensor]] = {}
    captured_tokens_indexing: Optional[Dict[str, Any]] = None

    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]

    for step_idx in range(max_new_tokens + 1):
        model_inputs = {
            "input_ids": current_input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if step_idx == 0:
            model_inputs.update(
                {
                    "images": images,
                    "image_sizes": image_sizes_for_model,
                    "modalities": [modality],
                    "include_image_inputs": True,
                }
            )

        with torch.inference_mode():
            outputs = model(**model_inputs)

        hidden_states = outputs.hidden_states or ()
        if not hidden_states:
            raise RuntimeError("Model did not return decoder hidden states.")
        if selected_layer_indices is None:
            selected_layer_indices = _resolve_decoder_layer_indices(hidden_states, layer_indices)
            generated_token_hidden_states = {f"layer_{idx:03d}": [] for idx in selected_layer_indices}

        if step_idx > 0:
            for layer_idx in selected_layer_indices:
                layer_key = f"layer_{layer_idx:03d}"
                layer_hidden = hidden_states[layer_idx]
                generated_token_hidden_states[layer_key].append(_to_cpu_tensor(layer_hidden[0, -1, :]))

        if step_idx == 0:
            captured_tokens_indexing = _clone_tokens_indexing(getattr(model, "tokens_indexing", None))

        if step_idx >= max_new_tokens:
            break

        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = _sample_next_token(
            next_token_logits,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        if eos_token_id is not None and int(next_token_id.item()) == int(eos_token_id):
            break

        generated_token_ids.append(int(next_token_id.item()))
        current_input_ids = next_token_id.view(1, -1)
        past_key_values = outputs.past_key_values

    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()
    selected_layer_indices = selected_layer_indices or []
    generated_layer_tensors = {
        layer_key: torch.stack(values, dim=0) if values else torch.empty(0)
        for layer_key, values in generated_token_hidden_states.items()
    }
    generated_last_token_tensors = {
        layer_key: values[-1] if values else torch.empty(0)
        for layer_key, values in generated_token_hidden_states.items()
    }
    generated_mean_pooled_tensors = {
        layer_key: tensor.mean(dim=0) if tensor.numel() else torch.empty(0)
        for layer_key, tensor in generated_layer_tensors.items()
    }

    output_path = Path(fix_wsl_paths(output_path))
    metadata = {
        **media_metadata,
        "output_path": str(output_path),
        "prompt": prompt,
        "provided_prompt": prompt,
        "conv_template": conv_template,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "modality": modality,
        "layer_indices": selected_layer_indices,
    }
    result: Dict[str, Any] = {
        "prompt": prompt,
        "metadata": metadata,
        "generation": {
            "text": generated_text,
            "input_token_ids": _to_cpu_tensor(input_ids[0]),
            "generated_token_ids": torch.tensor(generated_token_ids, dtype=torch.long),
        },
        "vision_encoder": {
            "pre_mm_projector_mean_pooled": _mean_pool_feature_tensor_to_cpu(vision_encoder_output),
            "post_mm_projector_mean_pooled": _mean_pool_feature_tensor_to_cpu(mm_projector_output),
        },
        "llm_decoder": {
            "selected_hidden_state_indices": selected_layer_indices,
            "indexing_note": "hidden_states[0] is token embeddings; selected indices are decoder layer outputs.",
            "generated_text_note": "Last-token and mean-pooled states are computed only over generated token hidden states, excluding the initial prompt/image forward pass.",
            "generated_text_last_token_hidden_states": generated_last_token_tensors,
            "generated_text_mean_pooled_hidden_states": generated_mean_pooled_tensors,
            "tokens_indexing": captured_tokens_indexing,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    return result


def collect_inference_representations(
    image_path: Union[str, Path],
    prompt: str,
    tokenizer,
    model,
    image_processor,
    output_path: Optional[Union[str, Path]] = None,
    conv_template: str = "qwen_1_5",
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    layer_indices: Optional[Sequence[int]] = DEFAULT_LAYER_INDICES,
) -> Dict[str, Any]:
    resolved_image_path = fix_wsl_paths(str(image_path))
    pil_image = load_image(resolved_image_path)
    image_tensor, image_size = prepare_multimodal_image_tensor(pil_image, image_processor, model)
    output_path = output_path or default_output_path_for_image(resolved_image_path)
    return _collect_prepared_inference_representations(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        images=image_tensor,
        image_sizes_for_model=[list(image_size)],
        media_metadata={
            "media_type": "image",
            "image_path": resolved_image_path,
            "image_size": image_size,
        },
        output_path=output_path,
        conv_template=conv_template,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        modality="image",
        layer_indices=layer_indices,
    )


def collect_movie_inference_representations(
    video_path: Union[str, Path],
    prompt: str,
    tokenizer,
    model,
    image_processor,
    output_dir: Optional[Union[str, Path]] = None,
    conv_template: str = "qwen_1_5",
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_frames_num: int = 16,
    window_duration_seconds: float = DEFAULT_WINDOW_DURATION_SECONDS,
    window_stride_seconds: Optional[float] = DEFAULT_WINDOW_STRIDE_SECONDS,
    offset_seconds: float = 0.0,
    end_seconds: Optional[float] = DEFAULT_END_SECONDS,
    limit: Optional[int] = None,
    layer_indices: Optional[Sequence[int]] = DEFAULT_LAYER_INDICES,
    movie_name: str = DEFAULT_MOVIE_NAME,
) -> List[Dict[str, Any]]:
    from decord import VideoReader, cpu

    resolved_video_path = fix_wsl_paths(video_path)
    vr = VideoReader(resolved_video_path, ctx=cpu(0))
    fps = float(vr.get_avg_fps())
    total_frames = len(vr)
    if total_frames < 1:
        raise ValueError(f"Video contains no frames: {resolved_video_path}")

    output_dir = (
        Path(fix_wsl_paths(output_dir))
        if output_dir is not None
        else default_output_dir_for_video(window_duration_seconds, movie_name)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stride_seconds = window_stride_seconds or window_duration_seconds
    if stride_seconds <= 0:
        raise ValueError("window_stride_seconds must be positive.")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided.")

    start_frame = max(0, int(round(offset_seconds * fps)))
    stop_frame = total_frames if end_seconds is None else min(total_frames, int(round(end_seconds * fps)))
    step = max(1, int(round(fps * stride_seconds)))
    results: List[Dict[str, Any]] = []
    manifest_entries: List[Dict[str, Any]] = []

    for frame_ind in range(start_frame, stop_frame, step):
        if limit is not None and len(results) >= limit:
            break
        window = load_video_window(
            video_path=resolved_video_path,
            max_frames_num=max_frames_num,
            start_frame=frame_ind,
            fps=fps,
            duration_seconds=window_duration_seconds,
            vr=vr,
        )
        images, image_sizes = prepare_multimodal_video_tensor(window["frames"], image_processor, model)
        output_path = default_output_path_for_video_window(
            output_dir,
            window["start_frame"],
        )
        result = _collect_prepared_inference_representations(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            images=images,
            image_sizes_for_model=image_sizes,
            media_metadata={
                "media_type": "video",
                "movie_name": movie_name,
                "video_path": resolved_video_path,
                "fps": fps,
                "total_frames": total_frames,
                "max_frames_num": max_frames_num,
                "sampled_frame_indices": window["frame_indices"],
                "start_frame": window["start_frame"],
                "end_frame": window["end_frame"],
                "start_seconds": window["start_seconds"],
                "end_seconds": window["end_seconds"],
                "window_duration_seconds": window_duration_seconds,
                "window_stride_seconds": stride_seconds,
                "limit": limit,
                "layer_indices": list(layer_indices) if layer_indices is not None else None,
            },
            output_path=output_path,
            conv_template=conv_template,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            modality="video",
            layer_indices=layer_indices,
        )
        results.append(result)
        manifest_entries.append(
            {
                "output_path": str(output_path),
                "prompt": prompt,
                "start_frame": window["start_frame"],
                "end_frame": window["end_frame"],
                "start_seconds": window["start_seconds"],
                "end_seconds": window["end_seconds"],
                "generated_text": result["generation"]["text"],
            }
        )
        print(
            f"Saved {window['start_seconds']:.2f}s-{window['end_seconds']:.2f}s "
            f"({window['start_frame']}-{window['end_frame']}) to {output_path}"
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_entries, indent=2), encoding="utf-8")
    return results


def collect_and_save_inference_representations(
    image_path: Union[str, Path],
    prompt: str,
    output_path: Optional[Union[str, Path]] = None,
    model_path: str = DEFAULT_MODEL_PATH,
    model_base: Optional[str] = None,
    adapter_path: Optional[str] = None,
    attn_implementation: str = "sdpa",
    load_4bit: bool = False,
    load_8bit: bool = False,
    conv_template: str = "qwen_1_5",
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    disable_optimizations: bool = False,
    layer_indices: Optional[Sequence[int]] = DEFAULT_LAYER_INDICES,
) -> Dict[str, Any]:
    if not disable_optimizations:
        from generation_utils import enable_inference_optimizations

        enable_inference_optimizations()

    from generation_utils import load_model_and_setup

    tokenizer, model, image_processor, _ = load_model_and_setup(
        model_path=model_path,
        model_base=model_base,
        adapter_path=adapter_path,
        attn_implementation=attn_implementation,
        load_4bit=load_4bit,
        load_8bit=load_8bit,
        attn_layer_ind=-1,
    )
    return collect_inference_representations(
        image_path=image_path,
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        output_path=output_path,
        conv_template=conv_template,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        layer_indices=layer_indices,
    )


def collect_and_save_movie_inference_representations(
    video_path: Union[str, Path],
    prompt: str,
    output_dir: Optional[Union[str, Path]] = None,
    model_path: str = DEFAULT_MODEL_PATH,
    model_base: Optional[str] = None,
    adapter_path: Optional[str] = None,
    attn_implementation: str = "sdpa",
    load_4bit: bool = False,
    load_8bit: bool = False,
    conv_template: str = "qwen_1_5",
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    disable_optimizations: bool = False,
    max_frames_num: int = 16,
    window_duration_seconds: float = DEFAULT_WINDOW_DURATION_SECONDS,
    window_stride_seconds: Optional[float] = DEFAULT_WINDOW_STRIDE_SECONDS,
    offset_seconds: float = 0.0,
    end_seconds: Optional[float] = DEFAULT_END_SECONDS,
    limit: Optional[int] = None,
    layer_indices: Optional[Sequence[int]] = DEFAULT_LAYER_INDICES,
    movie_name: str = DEFAULT_MOVIE_NAME,
) -> List[Dict[str, Any]]:
    if not disable_optimizations:
        from generation_utils import enable_inference_optimizations

        enable_inference_optimizations()

    from generation_utils import load_model_and_setup

    tokenizer, model, image_processor, _ = load_model_and_setup(
        model_path=model_path,
        model_base=model_base,
        adapter_path=adapter_path,
        attn_implementation=attn_implementation,
        load_4bit=load_4bit,
        load_8bit=load_8bit,
        attn_layer_ind=-1,
    )
    return collect_movie_inference_representations(
        video_path=video_path,
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        output_dir=output_dir,
        conv_template=conv_template,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_frames_num=max_frames_num,
        window_duration_seconds=window_duration_seconds,
        window_stride_seconds=window_stride_seconds,
        offset_seconds=offset_seconds,
        end_seconds=end_seconds,
        limit=limit,
        layer_indices=layer_indices,
        movie_name=movie_name,
    )


def collect_and_save_movie_prompts_inference_representations(
    video_path: Union[str, Path],
    prompts: Sequence[str],
    output_dir: Optional[Union[str, Path]] = None,
    model_path: str = DEFAULT_MODEL_PATH,
    model_base: Optional[str] = None,
    adapter_path: Optional[str] = None,
    attn_implementation: str = "sdpa",
    load_4bit: bool = False,
    load_8bit: bool = False,
    conv_template: str = "qwen_1_5",
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    disable_optimizations: bool = False,
    max_frames_num: int = 16,
    window_duration_seconds: float = DEFAULT_WINDOW_DURATION_SECONDS,
    window_stride_seconds: Optional[float] = DEFAULT_WINDOW_STRIDE_SECONDS,
    offset_seconds: float = 0.0,
    end_seconds: Optional[float] = DEFAULT_END_SECONDS,
    limit: Optional[int] = None,
    layer_indices: Optional[Sequence[int]] = DEFAULT_LAYER_INDICES,
    movie_name: str = DEFAULT_MOVIE_NAME,
) -> List[List[Dict[str, Any]]]:
    if not disable_optimizations:
        from generation_utils import enable_inference_optimizations

        enable_inference_optimizations()

    from generation_utils import load_model_and_setup

    tokenizer, model, image_processor, _ = load_model_and_setup(
        model_path=model_path,
        model_base=model_base,
        adapter_path=adapter_path,
        attn_implementation=attn_implementation,
        load_4bit=load_4bit,
        load_8bit=load_8bit,
        attn_layer_ind=-1,
    )

    base_output_dir = (
        Path(fix_wsl_paths(output_dir))
        if output_dir is not None
        else default_output_dir_for_video(window_duration_seconds, movie_name)
    )
    all_results: List[List[Dict[str, Any]]] = []
    for prompt_index, prompt in enumerate(prompts):
        prompt_output_dir = base_output_dir
        if len(prompts) > 1:
            prompt_output_dir = base_output_dir / prompt_output_dir_name(prompt_index)
        write_prompt_run_metadata(
            output_dir=prompt_output_dir,
            prompt=prompt,
            prompt_index=prompt_index,
            num_prompts=len(prompts),
            metadata={
                "video_path": fix_wsl_paths(video_path),
                "movie_name": movie_name,
                "model_path": model_path,
                "model_base": model_base,
                "adapter_path": adapter_path,
                "attn_implementation": attn_implementation,
                "load_4bit": load_4bit,
                "load_8bit": load_8bit,
                "conv_template": conv_template,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_frames_num": max_frames_num,
                "window_duration_seconds": window_duration_seconds,
                "window_stride_seconds": window_stride_seconds or window_duration_seconds,
                "offset_seconds": offset_seconds,
                "end_seconds": end_seconds,
                "limit": limit,
                "layer_indices": list(layer_indices) if layer_indices is not None else None,
            },
        )
        print(f"Running prompt {prompt_index + 1}/{len(prompts)} -> {prompt_output_dir}")
        all_results.append(
            collect_movie_inference_representations(
                video_path=video_path,
                prompt=prompt,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                output_dir=prompt_output_dir,
                conv_template=conv_template,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_frames_num=max_frames_num,
                window_duration_seconds=window_duration_seconds,
                window_stride_seconds=window_stride_seconds,
                offset_seconds=offset_seconds,
                end_seconds=end_seconds,
                limit=limit,
                layer_indices=layer_indices,
                movie_name=movie_name,
            )
        )
    return all_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract LLaVA inference representations.")
    parser.add_argument("--image-path", default=None, help="Image path.")
    parser.add_argument("--video-path", default=None, help=f"Video path. Defaults to DEFAULT_VIDEO when no image path is provided: {DEFAULT_VIDEO}")
    parser.add_argument("--prompt", default=None, help="Prompt to run with the media. Defaults to DEFAULT_PROMPTS when no prompt source is provided.")
    parser.add_argument("--prompts", nargs="+", default=None, help="Additional prompts to run. Video mode writes one directory per prompt when multiple prompts are provided.")
    parser.add_argument("--prompts-file", default=None, help="Text file with one prompt per line, or a JSON list of prompt strings.")
    parser.add_argument(
        "--output-path",
        default=None,
        help=(
            "Path to write the torch .pt dictionary. Defaults to "
            f"{DEFAULT_OUTPUT_DIR}/<image_stem>_inference_representations.pt."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for video-window .pt files. Defaults to "
            f"{DEFAULT_OUTPUT_DIR}/{DEFAULT_MOVIE_NAME}_<window_duration>s."
        ),
    )
    parser.add_argument(
        "--movie-name",
        default=DEFAULT_MOVIE_NAME,
        help="Movie name used in default video output directories and metadata.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--conv-template", default="qwen_1_5")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--disable-optimizations", action="store_true")
    parser.add_argument("--max-frames-num", type=int, default=16)
    parser.add_argument("--window-duration-seconds", type=float, default=DEFAULT_WINDOW_DURATION_SECONDS)
    parser.add_argument(
        "--window-stride-seconds",
        type=float,
        default=DEFAULT_WINDOW_STRIDE_SECONDS,
        help="Seconds between window starts. Defaults to DEFAULT_WINDOW_STRIDE_SECONDS, or --window-duration-seconds when unset.",
    )
    parser.add_argument("--offset-seconds", type=float, default=0.0)
    parser.add_argument("--end-seconds", type=float, default=DEFAULT_END_SECONDS)
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of movie-window inferences to run per prompt.")
    parser.add_argument(
        "--layer-indices",
        nargs="+",
        default=None,
        help=f"Decoder hidden-state indices to save, e.g. --layer-indices 1 14 28 or 1,14,28. Defaults to {DEFAULT_LAYER_INDICES}.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    video_path = args.video_path
    if not args.image_path and not video_path:
        video_path = DEFAULT_VIDEO

    if bool(args.image_path) == bool(video_path):
        raise ValueError("Provide exactly one of --image-path or --video-path.")

    prompts = resolve_prompt_list(args.prompt, args.prompts, args.prompts_file)
    layer_indices = parse_layer_indices(args.layer_indices)

    if video_path:
        prompt_results = collect_and_save_movie_prompts_inference_representations(
            video_path=video_path,
            prompts=prompts,
            output_dir=args.output_dir,
            model_path=args.model_path,
            model_base=args.model_base,
            adapter_path=args.adapter_path,
            attn_implementation=args.attn_implementation,
            load_4bit=args.load_4bit,
            load_8bit=args.load_8bit,
            conv_template=args.conv_template,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            disable_optimizations=args.disable_optimizations,
            max_frames_num=args.max_frames_num,
            window_duration_seconds=args.window_duration_seconds,
            window_stride_seconds=args.window_stride_seconds,
            offset_seconds=args.offset_seconds,
            end_seconds=args.end_seconds,
            limit=args.limit,
            layer_indices=layer_indices,
            movie_name=args.movie_name,
        )
        num_windows = sum(len(results) for results in prompt_results)
        if num_windows:
            output_dirs = {
                str(Path(results[0]["metadata"]["output_path"]).parent)
                for results in prompt_results
                if results
            }
            print(f"Saved {num_windows} video-window representation files across {len(output_dirs)} prompt directories.")
        else:
            print("No video windows were processed.")
        return

    if len(prompts) > 1:
        raise ValueError("Image mode supports one prompt because --output-path names a single representation file.")

    result = collect_and_save_inference_representations(
        image_path=args.image_path,
        prompt=prompts[0],
        output_path=args.output_path,
        model_path=args.model_path,
        model_base=args.model_base,
        adapter_path=args.adapter_path,
        attn_implementation=args.attn_implementation,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        conv_template=args.conv_template,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        disable_optimizations=args.disable_optimizations,
        layer_indices=layer_indices,
    )
    print(f"Saved representations to {result['metadata']['output_path']}")
    print(f"Generated text: {result['generation']['text']}")


if __name__ == "__main__":
    main(parse_args())
