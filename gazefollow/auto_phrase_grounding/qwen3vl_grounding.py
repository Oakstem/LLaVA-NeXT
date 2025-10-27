from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import json
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL grounding detection")
    parser.add_argument("--image-path", type=str, default="/mnt/d/Projects/data/gazefollow/train/00000024/00024026.jpg", help="Path to input image")
    parser.add_argument("--query", type=str, default="Locate the laptop screen with photo of a couple on it, output its bbox coordinates using JSON format.", help="Detection query")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save output JSON (optional)")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-4B-Instruct", help="Model ID")
    parser.add_argument("--max-new-tokens", type=int, default=300, help="Max tokens to generate")
    args = parser.parse_args()

    print(f"Loading model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    print(f"Loading image: {args.image_path}")
    img = Image.open(args.image_path)
    img_width, img_height = img.size
    print(f"Image size: {img_width}x{img_height}")

    messages = [
        {"role": "system", "content": "You are a vision assistant. Output ONLY JSON."},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": args.query}
        ]}
    ]

    print("Generating grounding predictions...")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    pred = processor.batch_decode(out, skip_special_tokens=True)[0]
    
    # Extract only the assistant's response (after the last "assistant\n")
    if "assistant\n" in pred:
        pred = pred.split("assistant\n")[-1].strip()
    
    print(f"\nRaw output:\n{repr(pred)}\n")
    print(f"Raw output (display):\n{pred}\n")
    
    # Strip markdown code blocks if present
    if pred.startswith("```"):
        # Remove opening ```json or ``` and closing ```
        pred = pred.strip()
        if pred.startswith("```json"):
            pred = pred[7:]  # Remove ```json
        elif pred.startswith("```"):
            pred = pred[3:]  # Remove ```
        if pred.endswith("```"):
            pred = pred[:-3]  # Remove closing ```
        pred = pred.strip()
    
    parsed = json.loads(pred)
    
    # Convert Qwen's normalized coordinates (0-1000) to absolute pixels
    for detection in parsed:
        bbox_key = [key for key in detection.keys() if "bbox" in key]
            
        if bbox_key:
            bbox_key = bbox_key[0]
            # Qwen outputs [x1, y1, x2, y2] in 0-1000 range
            x1_norm, y1_norm, x2_norm, y2_norm = detection[bbox_key]
            detection[bbox_key] = [
                int((x1_norm / 1000.0) * img_width),
                int((y1_norm / 1000.0) * img_height),
                int((x2_norm / 1000.0) * img_width),
                int((y2_norm / 1000.0) * img_height)
            ]
    
    print(json.dumps(parsed, indent=2))
    
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(parsed, f, indent=2)
        print(f"\nSaved output to: {args.output_json}")


if __name__ == "__main__":
    main()
