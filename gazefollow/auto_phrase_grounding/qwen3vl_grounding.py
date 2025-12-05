import argparse
import inspect
import json
import re
from pathlib import Path
from typing import Any, Optional, Tuple, TYPE_CHECKING

import torch
from PIL import Image


def _ensure_torch_pytree_register() -> None:
    """Shim torch's pytree register API to ignore new kwargs the installed torch does not know."""
    pytree = getattr(getattr(torch, "utils", None), "_pytree", None)
    if pytree is None:
        return

    register_impl = getattr(pytree, "register_pytree_node", None)
    if register_impl is None and hasattr(pytree, "_register_pytree_node"):
        register_impl = pytree._register_pytree_node  # type: ignore[attr-defined]
        pytree.register_pytree_node = register_impl  # type: ignore[attr-defined]

    if register_impl is None:
        return

    try:
        signature = inspect.signature(register_impl)
    except (TypeError, ValueError):
        signature = None

    if signature and "serialized_type_name" in signature.parameters:
        return

    def _shim_register_pytree_node(*args: Any, **kwargs: Any) -> None:
        kwargs.pop("serialized_type_name", None)
        kwargs.pop("serialized_type_version", None)
        register_impl(*args, **kwargs)

    pytree.register_pytree_node = _shim_register_pytree_node  # type: ignore[attr-defined]


_ensure_torch_pytree_register()

import transformers
from transformers import AutoProcessor

try:  # transformers >= 4.44
    from transformers import Qwen3VLForConditionalGeneration  # type: ignore
except ImportError:
    Qwen3VLForConditionalGeneration = None  # type: ignore[assignment]

try:  # transformers >= 4.50 (Qwen2-VL rename)
    from transformers import Qwen2VLForConditionalGeneration  # type: ignore
except ImportError:
    Qwen2VLForConditionalGeneration = None  # type: ignore[assignment]

try:  # fallback for older wheels with trust_remote_code
    from transformers import AutoModelForVision2Seq  # type: ignore
except ImportError:
    AutoModelForVision2Seq = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    QwenModelT = PreTrainedModel
else:
    QwenModelT = Any  # runtime typing


class _TorchCompilerStub:
    @staticmethod
    def is_compiling() -> bool:
        return False


def _ensure_torch_compiler_stub() -> None:
    """Patch older torch builds missing torch.compiler.is_compiling."""
    compiler = getattr(torch, "compiler", None)
    if compiler is None:
        torch.compiler = _TorchCompilerStub()
        return
    if not hasattr(compiler, "is_compiling"):
        compiler.is_compiling = lambda: False


_ensure_torch_compiler_stub()


def parse_grounding_predictions(raw_text: str) -> list[dict[str, Any]]:
    """Extract the first JSON payload from the model response."""
    cleaned = raw_text.strip()

    if "assistant\n" in cleaned:
        cleaned = cleaned.split("assistant\n")[-1].strip()

    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.IGNORECASE | re.DOTALL)
    if fenced_match:
        cleaned = fenced_match.group(1).strip()

    decoder = json.JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
                # Fallback: search for bbox pattern with 4 numbers
            bbox_match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', cleaned)
            if bbox_match:
                bbox = [int(bbox_match.group(i)) for i in range(1, 5)]
                return [{"bbox": bbox}]
            else:
                continue

        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return parsed

    return []



def load_qwen3vl_model(
    model_id: str,
    *,
    device_map: Optional[str] = "auto",
) -> Tuple[AutoProcessor, QwenModelT]:
    """Load the Qwen3-VL processor and model."""
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    if Qwen3VLForConditionalGeneration is not None:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map=device_map or "auto",
            trust_remote_code=True,
        )
    elif Qwen2VLForConditionalGeneration is not None:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map=device_map or "auto",
            trust_remote_code=True,
        )
    elif AutoModelForVision2Seq is not None:
        try:
            model = AutoModelForVision2Seq.from_pretrained(  # type: ignore[call-arg]
                model_id,
                dtype=torch.bfloat16,
                device_map=device_map or "auto",
                trust_remote_code=True,
            )
        except ValueError as exc:
            message = str(exc)
            if "does not recognize this architecture" in message and "qwen" in message.lower():
                raise RuntimeError(
                    (
                        f"Transformers {transformers.__version__} cannot load '{model_id}' "
                        "because the `qwen3_vl` architecture is newer than this environment. "
                        "Upgrade transformers to >=4.45 (recommended 4.47+) or choose a different "
                        "GAZE_MODEL_ID such as Qwen/Qwen2-VL-7B-Instruct."
                    )
                ) from exc
            raise
    else:
        raise ImportError(
            "Your transformers installation does not expose Qwen3-VL or AutoModelForVision2Seq. "
            "Upgrade transformers (>= 4.44) or install the Qwen3-VL integration."
        )
    return processor, model


def run_qwen3vl_grounding(
    image_path: str,
    query: str,
    model_id: str,
    max_new_tokens: int = 300,
    *,
    processor: Optional[AutoProcessor] = None,
    model: Optional[QwenModelT] = None,
    device_map: Optional[str] = "auto",
    temperature: float = 0.4,
) -> list[dict[str, Any]]:
    """Run the Qwen3-VL model and return detections with pixel bbox and centers."""
    owns_model = False
    if processor is None or model is None:
        processor, model = load_qwen3vl_model(model_id, device_map=device_map)
        owns_model = True

    img = Image.open(image_path)
    img_width, img_height = img.size

    messages = [
        {"role": "system", "content": "You are a vision assistant. Output ONLY JSON."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": query},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=temperature > 0)
    pred = processor.batch_decode(out, skip_special_tokens=True)[0]
    detections = parse_grounding_predictions(pred)
    raw_response = pred[pred.find('assistant\n'):].strip("assistant\n")

    cleaned_detections: list[dict[str, Any]] = []
    for raw_detection in detections:
        detection: Any = raw_detection
        # Some generations output bare lists like [x1, y1, x2, y2]; wrap them.
        if isinstance(detection, list) and len(detection) == 4 and all(isinstance(val, (int, float)) for val in detection):
            detection = {"bbox": detection}
        if not isinstance(detection, dict):
            continue

        bbox_key = next((key for key in detection.keys() if isinstance(key, str) and "bbox" in key.lower()), None)
        if not bbox_key:
            continue

        bbox_value = detection.get(bbox_key)
        # Flatten if nested (e.g., [[x1, y1, x2, y2]])
        if isinstance(bbox_value, list) and len(bbox_value) == 1 and isinstance(bbox_value[0], list):
            bbox_value = bbox_value[0]
        if (
            not isinstance(bbox_value, (list, tuple))
            or len(bbox_value) != 4
        ):
            continue
        try:
            x1_norm, y1_norm, x2_norm, y2_norm = [float(coord) for coord in bbox_value]
        except (TypeError, ValueError):
            continue
        abs_bbox = [
            int((x1_norm / 1000.0) * img_width),
            int((y1_norm / 1000.0) * img_height),
            int((x2_norm / 1000.0) * img_width),
            int((y2_norm / 1000.0) * img_height),
        ]
        detection['bbox'] = abs_bbox
        detection["bbox_center"] = [
            (abs_bbox[0] + abs_bbox[2]) / 2.0,
            (abs_bbox[1] + abs_bbox[3]) / 2.0,
        ]
        if bbox_key != 'bbox':
            detection.pop(bbox_key, None)

        cleaned_detections.append(detection)

    return cleaned_detections, raw_response


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL grounding detection")
    parser.add_argument("--image-path", type=str, default="/mnt/d/Projects/data/gazefollow/train/00000065/00065874.jpg", help="Path to input image")
    parser.add_argument("--query", type=str, default="Locate the wooden planks, output its bbox coordinates using JSON format.", help="Detection query")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-4B-Instruct", help="Model ID")
    parser.add_argument("--max-new-tokens", type=int, default=300, help="Max tokens to generate")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save detections as JSON")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    args = parser.parse_args()

    print(f"Loading model: {args.model_id}")
    print(f"Loading image: {args.image_path}")
    processor, model = load_qwen3vl_model(args.model_id)
    detections, raw_response = run_qwen3vl_grounding(
        image_path=args.image_path,
        query=args.query,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        processor=processor,
        model=model,
    )

    print(json.dumps(detections, indent=2))
    
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(detections, f, indent=2)
        print(f"\nSaved output to: {args.output_json}")


if __name__ == "__main__":
    main()
