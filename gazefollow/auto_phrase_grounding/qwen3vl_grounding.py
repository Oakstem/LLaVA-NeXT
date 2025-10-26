from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import base64
from PIL import Image
import json
import io
import argparse
from pathlib import Path


def pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL grounding detection")
    parser.add_argument("--image-path", type=str, required=True, help="Path to input image")
    parser.add_argument("--query", type=str, default="Detect all mugs. Return JSON: {\"objects\":[{\"label\":\"mug\",\"bbox\":[x1,y1,x2,y2],\"confidence\":float}]} Coordinates must be absolute pixel ints, origin at top-left. No extra text.", help="Detection query")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save output JSON (optional)")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Model ID")
    parser.add_argument("--max-new-tokens", type=int, default=300, help="Max tokens to generate")
    args = parser.parse_args()

    print(f"Loading model: {args.model_id}")
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    print(f"Loading image: {args.image_path}")
    img = Image.open(args.image_path)
    img_b64 = pil_to_b64(img)

    messages = [
        {"role": "system", "content": "You are a vision assistant. Output ONLY JSON."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64}},
            {"type": "text", "text": args.query}
        ]}
    ]

    print("Generating grounding predictions...")
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(model.device)
    out = model.generate(inputs, max_new_tokens=args.max_new_tokens)
    pred = tok.decode(out[0], skip_special_tokens=True)
    
    print(f"\nRaw output:\n{pred}\n")
    
    boxes = json.loads(pred)
    print(f"Parsed JSON:\n{json.dumps(boxes, indent=2)}")
    
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(boxes, f, indent=2)
        print(f"\nSaved output to: {args.output_json}")


if __name__ == "__main__":
    main()
