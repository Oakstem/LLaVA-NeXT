import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


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
            continue

        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
        raise ValueError("Parsed payload must be a JSON object or array.")

    raise ValueError("Unable to locate a JSON payload in the model response.")



def run_qwen3vl_grounding(
    image_path: str,
    query: str,
    model_id: str,
    max_new_tokens: int = 300,
) -> list[dict[str, Any]]:
    """Run the Qwen3-VL model and return detections with pixel bbox and centers."""
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

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
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    pred = processor.batch_decode(out, skip_special_tokens=True)[0]
    detections = parse_grounding_predictions(pred)

    for detection in detections:
        bbox_key = next((key for key in detection.keys() if "bbox" in key), None)
        if not bbox_key:
            continue

        x1_norm, y1_norm, x2_norm, y2_norm = detection[bbox_key]
        abs_bbox = [
            int((x1_norm / 1000.0) * img_width),
            int((y1_norm / 1000.0) * img_height),
            int((x2_norm / 1000.0) * img_width),
            int((y2_norm / 1000.0) * img_height),
        ]
        detection[bbox_key] = abs_bbox
        detection["bbox_center"] = [
            (abs_bbox[0] + abs_bbox[2]) / 2.0,
            (abs_bbox[1] + abs_bbox[3]) / 2.0,
        ]

    return detections


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL grounding detection")
    parser.add_argument("--image-path", type=str, default="/mnt/d/Projects/data/gazefollow/train/00000024/00024026.jpg", help="Path to input image")
    parser.add_argument("--query", type=str, default="Locate the laptop screen with photo of a couple on it, output its bbox coordinates using JSON format.", help="Detection query")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-4B-Instruct", help="Model ID")
    parser.add_argument("--max-new-tokens", type=int, default=300, help="Max tokens to generate")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save detections as JSON")
    args = parser.parse_args()

    print(f"Loading model: {args.model_id}")
    print(f"Loading image: {args.image_path}")
    print("Generating grounding predictions...")
    detections = run_qwen3vl_grounding(
        image_path=args.image_path,
        query=args.query,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
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
