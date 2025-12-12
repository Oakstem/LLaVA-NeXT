import json, argparse, time
from pathlib import Path
from openai import OpenAI

client = OpenAI()

# your structured outputs schema (unchanged)
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

SYSTEM_PROMPT = (
    "You specialize in extracting gaze targets from descriptions "
    "and must reply with one concise phrase. Reply with JSON only, following the schema exactly."
)

def build_batch_jsonl(input_json, out_jsonl):
    data = json.loads(Path(input_json).read_text())
    tasks = []
    for i, item in enumerate(data):
        pred = item.get("model_prediction") or ""
        if not pred:
            continue
        custom_id = f"{item.get('id', i)}"  # keep it stable so we can join later
        task = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",  # you can also use /v1/responses
            "body": {
                "model": "gpt-5-nano",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": pred}
                ],
                "response_format": schema,
                # optional: "temperature": 0
            },
        }
        tasks.append(task)

    with open(out_jsonl, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

def submit_batch(jsonl_path, endpoint="/v1/chat/completions"):
    # 1) upload the JSONL
    upload = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    # 2) create batch job (24h window is required)
    job = client.batches.create(
        input_file_id=upload.id,
        endpoint=endpoint,
        completion_window="24h",
    )
    return job.id

def wait_and_download(batch_id, out_results_jsonl):
    # poll until completed
    while True:
        job = client.batches.retrieve(batch_id)
        if job.status in ("completed", "failed", "cancelled", "expired"):
            break
        time.sleep(5)

    if job.status != "completed":
        raise RuntimeError(f"Batch ended with status: {job.status}")

    # download results file
    result_bytes = client.files.content(job.output_file_id).content
    Path(out_results_jsonl).write_bytes(result_bytes)

def merge_results(input_json, results_jsonl, merged_out):
    # load originals for id -> metadata
    originals = {str(x.get("id")): x for x in json.loads(Path(input_json).read_text())}

    merged = []
    with open(results_jsonl, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = str(obj["custom_id"])

            # chat.completions result path:
            content = obj["response"]["body"]["choices"][0]["message"]["content"]
            extracted = json.loads(content)  # thanks to Structured Outputs, this should be valid JSON

            orig = originals.get(cid, {})
            merged.append({
                "id": cid,
                "image_path": orig.get("image_path"),
                "model_prediction": orig.get("model_prediction"),
                "extracted_gaze": extracted,
            })

    Path(merged_out).write_text(json.dumps(merged, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input_json", help="model_generation_results.json")
    ap.add_argument("--workdir", default="batch_out/test2_baseline", help="where to write files")
    args = ap.parse_args()

    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    batch_in = str(Path(args.workdir) / "requests.jsonl")
    batch_out = str(Path(args.workdir) / "results.jsonl")
    merged = str(Path(args.workdir) / "gpt_gaze_extracted.json")

    build_batch_jsonl(args.input_json, batch_in)
    batch_id = submit_batch(batch_in, endpoint="/v1/chat/completions")
    print("batch_id:", batch_id)

    # heads-up: batch jobs complete async on OpenAI side; just poll until done
    wait_and_download(batch_id, batch_out)
    merge_results(args.input_json, batch_out, merged)
    print(f"merged → {merged}")
