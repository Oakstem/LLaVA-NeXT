#!/usr/bin/env python3
"""
Run text-conditioned object detection to estimate gaze targets for described people.

Given an image and a free-form description that includes person-centric sentences
containing phrases like "looking at ...", the script extracts the gaze target
phrases, queries a GroundingDINO model for those targets, and stores the top
detected bounding box per person as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

# Patch PyTorch pytree for older torch versions used with newer Transformers
if hasattr(torch, 'utils') and hasattr(torch.utils, '_pytree'):
    _pytree = torch.utils._pytree
    if not hasattr(_pytree, 'register_pytree_node') and hasattr(_pytree, '_register_pytree_node'):
        def _compat_register_pytree_node(*args, **kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k not in {'serialized_type_name', 'serialized_context'}}
            return _pytree._register_pytree_node(*args, **kwargs)
        _pytree.register_pytree_node = _compat_register_pytree_node

from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
try:
    from transformers import infer_device
except ImportError:
    def infer_device():
        return "cuda" if torch.cuda.is_available() else "cpu"

# Disable custom C++ kernel loading when not available
os.environ.setdefault("DISABLE_TRANSFORMERS_CUSTOM_KERNELS", "1")

@dataclass
class PersonDescription:
    person_id: str
    label: str
    raw_description: str
    gaze_target: Optional[str]


GAZE_PATTERN = re.compile(
    r"\blooking(?:\s+(?:at|to|toward|towards|into))?\s+(?P<target>[^.;,]+)",
    flags=re.IGNORECASE,
)

PERSON_PATTERN = re.compile(
    r"(Person\s+\d+)\s*:\s*(.+?)(?=(?:Person\s+\d+\s*:|Social interactions?:|$))",
    flags=re.IGNORECASE | re.DOTALL,
)

ARTICLE_PATTERN = re.compile(r"^(the|a|an)\s+", flags=re.IGNORECASE)
COPULA_PATTERN = re.compile(r"\b(is|are|was|were|be|being|been)\b\s*$", flags=re.IGNORECASE)


def extract_gaze_target(description: str) -> Optional[str]:
    match = GAZE_PATTERN.search(description)
    if not match:
        return None
    return match.group("target").strip().rstrip(".")


def normalize_person_id(label: str) -> str:
    number_match = re.search(r"\d+", label)
    if number_match:
        return f"person_{number_match.group(0)}"
    return re.sub(r"\s+", "_", label.strip().lower())


def parse_person_descriptions(text: str) -> List[PersonDescription]:
    persons: List[PersonDescription] = []
    for raw_label, description in PERSON_PATTERN.findall(text):
        person_id = normalize_person_id(raw_label)
        persons.append(
            PersonDescription(
                person_id=person_id,
                label=raw_label.strip(),
                raw_description=description.strip(),
                gaze_target=extract_gaze_target(description),
            )
        )
    return persons


def strip_gaze_clause(description: str) -> str:
    if not description:
        return description
    match = GAZE_PATTERN.search(description)
    if match:
        return description[:match.start()].strip().rstrip(":;,")
    return description.strip()


def extract_gaze_metrics_from_generation(
    model_generation_entry: Dict[str, Any],
    focus_phrase: str = "looking at"
) -> Tuple[str, str, Optional[float], Optional[float]]:
    """
    Extract gaze-related metrics from a model generation entry for wandb logging.
    
    Args:
        model_generation_entry: Dictionary containing model generation results with keys:
            - "ground_truth": The ground truth text containing the gaze target
            - "gaze_detections": Dictionary of person detections with gaze information
        focus_phrase: The phrase used to locate the gaze target in ground truth (default: "looking at")
    
    Returns:
        Tuple containing:
        - ground_truth_gaze_target: Text after the focus phrase in ground truth
        - gaze_target: Predicted gaze target from first person detection
        - gaze_l2_error: L2 distance error (if available)
        - gaze_normalized_l2_error: Normalized L2 error (if available)
    """
    # Extract ground truth gaze target (text after focus phrase)
    ground_truth_text = model_generation_entry.get("ground_truth", "")
    ground_truth_gaze_target = ""
    if focus_phrase in ground_truth_text.lower():
        # Find the position after the focus phrase
        idx = ground_truth_text.lower().find(focus_phrase)
        ground_truth_gaze_target = ground_truth_text[idx + len(focus_phrase):].strip()
        # Remove any trailing punctuation
        ground_truth_gaze_target = ground_truth_gaze_target.rstrip(".;,!?")
    
    # Extract gaze_target and error metrics from first person in gaze_detections
    gaze_detections = model_generation_entry.get("gaze_detections", {})
    gaze_target = ""
    gaze_l2_error = None
    gaze_normalized_l2_error = None
    
    if gaze_detections:
        # Get the first person's data (usually there's only one)
        first_person_key = next(iter(gaze_detections.keys()), None)
        if first_person_key:
            person_data = gaze_detections[first_person_key]
            gaze_target = person_data.get("gaze_target", "")
            gaze_l2_error = person_data.get("gaze_l2_error")
            gaze_normalized_l2_error = person_data.get("gaze_normalized_l2_error")
    
    return ground_truth_gaze_target, gaze_target, gaze_l2_error, gaze_normalized_l2_error


def normalize_person_description(description: str) -> str:
    if not description:
        return description
    text = strip_gaze_clause(description)
    text = ARTICLE_PATTERN.sub("", text).strip()
    text = COPULA_PATTERN.sub("", text).strip(" ,;:")
    return text


def normalize_gaze_target_text(target: Optional[str]) -> Optional[str]:
    if target is None:
        return None
    text = ARTICLE_PATTERN.sub("", target).strip()
    text = COPULA_PATTERN.sub("", text).strip(" ,;:")
    return text


def load_image(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image




def load_grounding_dino(model_id: str, device: str) -> Tuple[AutoProcessor, AutoModelForZeroShotObjectDetection]:
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return processor, model


def detect_gaze_targets(
    image: Image.Image,
    persons: Iterable[PersonDescription],
    model_id: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
    processor: Optional[AutoProcessor] = None,
    model: Optional[AutoModelForZeroShotObjectDetection] = None,
) -> Dict[str, Dict[str, Any]]:
    if processor is None or model is None:
        processor, model = load_grounding_dino(model_id, device)

    image_height, image_width = image.height, image.width
    tokenizer = getattr(processor, "tokenizer", None)
    max_text_length = None
    if tokenizer is not None:
        max_text_length = getattr(tokenizer, "model_max_length", None)
    if not isinstance(max_text_length, int) or max_text_length <= 0:
        max_text_length = getattr(model.config, "text_encoder_model_max_length", None)
    if not isinstance(max_text_length, int) or max_text_length <= 0:
        text_encoder = getattr(model.config, "text_encoder", None)
        max_text_length = getattr(text_encoder, "model_max_length", None) if text_encoder else None
    if not isinstance(max_text_length, int) or max_text_length <= 0:
        max_text_length = 256
    max_text_length = max(1, min(int(max_text_length), 256))

    results: Dict[str, Dict[str, Any]] = {}
    for person in persons:
        person_result: Dict[str, Any] = {
            "label": person.label,
            "description": normalize_person_description(person.raw_description),
            "full_description": f"{person.label}: {person.raw_description}",
            "gaze_target": normalize_gaze_target_text(person.gaze_target),
            "coordinates": None,
            "score": None,
        }

        if not person_result["gaze_target"]:
            results[person.person_id] = person_result
            continue

        text_prompt = str(person_result["gaze_target"])
        inputs = processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_text_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor(
            [[image_height, image_width]],
            dtype=torch.float32,
            device=model.device,
        )
        detections = processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            target_sizes=target_sizes,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )[0]

        boxes = detections.get("boxes")
        scores = detections.get("scores")

        if boxes is None or boxes.numel() == 0:
            results[person.person_id] = person_result
            continue

        best_idx = int(scores.argmax().item())
        best_box = boxes[best_idx].tolist()
        best_score = float(scores[best_idx].item())

        person_result["coordinates"] = [float(coord) for coord in best_box]
        person_result["score"] = best_score
        results[person.person_id] = person_result

    return results


def run_grounding_dino_detection(
    image: Image.Image,
    text_prompt: Optional[str],
    processor: AutoProcessor,
    model: AutoModelForZeroShotObjectDetection,
    box_threshold: float,
    text_threshold: float,
) -> Tuple[Optional[List[float]], Optional[float]]:
    if not text_prompt or not text_prompt.strip():
        return None, None

    tokenizer = getattr(processor, "tokenizer", None)
    max_text_length = None
    if tokenizer is not None:
        max_text_length = getattr(tokenizer, "model_max_length", None)
    if not isinstance(max_text_length, int) or max_text_length <= 0:
        max_text_length = getattr(model.config, "text_encoder_model_max_length", None)
    if not isinstance(max_text_length, int) or max_text_length <= 0:
        text_encoder = getattr(model.config, "text_encoder", None)
        max_text_length = getattr(text_encoder, "model_max_length", None) if text_encoder else None
    if not isinstance(max_text_length, int) or max_text_length <= 0:
        max_text_length = 256
    max_text_length = max(1, min(int(max_text_length), 256))

    inputs = processor(
        images=image,
        text=text_prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_text_length,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor(
        [[image.height, image.width]],
        dtype=torch.float32,
        device=model.device,
    )
    detections = processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs.input_ids,
        target_sizes=target_sizes,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )[0]

    boxes = detections.get("boxes")
    scores = detections.get("scores")

    if boxes is None or boxes.numel() == 0:
        return None, None

    best_idx = int(scores.argmax().item())
    best_box = [float(coord) for coord in boxes[best_idx].tolist()]
    best_score = float(scores[best_idx].item())
    return best_box, best_score


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect gaze targets for each person description using GroundingDINO."
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        required=True,
        help="Path to the image file.",
    )
    description_group = parser.add_mutually_exclusive_group(required=True)
    description_group.add_argument(
        "--description-file",
        type=Path,
        help="Path to a text file containing the scene description.",
    )
    description_group.add_argument(
        "--description-text",
        type=str,
        help="Directly provide the scene description as a string.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="IDEA-Research/grounding-dino-base",
        help="Hugging Face model identifier for GroundingDINO.",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for bounding boxes.",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for text scoring.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the resulting JSON file.",
    )
    return parser


def load_description(args: argparse.Namespace) -> str:
    if args.description_text is not None:
        return args.description_text
    return args.description_file.read_text(encoding="utf-8")


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    image_path: Path = args.image_path
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    description_text = load_description(args)
    persons = parse_person_descriptions(description_text)

    if not persons:
        raise ValueError("No person descriptions found in the provided text.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = load_image(image_path)
    detections = detect_gaze_targets(
        image=image,
        persons=persons,
        model_id=args.model_id,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=device,
    )

    payload = {
        "image_path": str(image_path),
        "model_id": args.model_id,
        "results": detections,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
