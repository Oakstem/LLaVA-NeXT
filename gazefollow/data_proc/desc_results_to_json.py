import json
import argparse
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

client = OpenAI()

schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "gaze_extraction",
        "schema": {
            "type": "object",
            "properties": {
                "person_description": {"type": "string"},
                "gaze_target": {"type": ["string", "null"]},
                "gaze_target_type": {"type": "string", "enum": ["person", "object", "unknown"]}
            },
            "required": ["person_description", "gaze_target", "gaze_target_type"],
            "additionalProperties": False
        },
        "strict": True
    }
}

system_prompt = """You specialize in extracting gaze targets from descriptions and must reply with one concise phrase. Reply with JSON only, following the schema exactly."""


def extract_gaze_info(description: str) -> dict:
    """Extract structured gaze information from model prediction text."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"<description>\n{description}\n</description>"}
    ]
    
    resp = client.chat.completions.create(
        model="gpt-5-nano",
        messages=messages,
        response_format=schema
    )
    
    return json.loads(resp.choices[0].message.content)


def process_json_file(input_path: str, output_path: str = None, limit: int = None, save_interval: int = 50):
    """Process model_generation_results.json and extract gaze info from model_prediction."""
    with open(input_path) as f:
        data = json.load(f)
    
    if limit:
        data = data[:limit]
    
    if output_path is None:
        output_path = Path(input_path).parent / "gpt_gaze_extracted.json"
    
    results = []
    for idx, item in enumerate(tqdm(data, desc="Processing predictions"), 1):
        prediction = item.get("model_prediction", "")
        if not prediction:
            continue
        
        try:
            gaze_info = extract_gaze_info(prediction)
            results.append({
                "id": item.get("id"),
                "image_path": item.get("image_path"),
                "model_prediction": prediction,
                "extracted_gaze": gaze_info
            })
        except Exception as e:
            print(f"Error processing {item.get('id')}: {e}")
            continue
        
        # Periodic save
        if idx % save_interval == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Checkpoint: saved {len(results)} results to {output_path}")
    
    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Processed {len(results)} predictions. Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract gaze info from model predictions")
    parser.add_argument("input_json", help="Path to model_generation_results.json")
    parser.add_argument("--output", "-o", help="Output JSON path (default: input_dir/gaze_extracted.json)")
    parser.add_argument("--limit", "-l", type=int, help="Limit number of entries to process")
    parser.add_argument("--save-interval", "-s", type=int, default=5, help="Save interval (default: 50)")
    args = parser.parse_args()
    
    process_json_file(args.input_json, args.output, args.limit, args.save_interval)
