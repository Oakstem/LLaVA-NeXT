"""
Demo server for single-region extraction workflow.
Replicates the PowerShell script's functionality via a web UI.

Usage:
    /home/alonz/llava/bin/python demo/server.py [--port 7860] [--host 0.0.0.0]
"""
import sys
import json
import base64
import argparse
import threading
from pathlib import Path
from datetime import datetime
from io import BytesIO

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "gazefollow"))

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, send_file

from gazefollow.generation_utils import (
    enable_inference_optimizations,
    fix_wsl_paths,
    load_model_and_setup,
    load_image,
)
from gazefollow.json_utils import make_json_safe

app = Flask(__name__, static_folder=str(PROJECT_ROOT / "demo"))

# ── Global state ──────────────────────────────────────────────────────────────
_model_state = {
    "loaded": False,
    "loading": False,
    "tokenizer": None,
    "model": None,
    "image_processor": None,
    "error": None,
}
_model_lock = threading.Lock()

DEFAULT_IMAGE_PATH = "/mnt/d/Projects/data/gazefollow/train/00000096/00096721.jpg"
MASK_DIR = PROJECT_ROOT / "region_masks"
OUTPUT_DIR = PROJECT_ROOT / "attention_output" / "single_region_extraction"

DEFAULT_PROMPT = """Complete the sentence in the following format, for example:
a tired guy in a hoodie → a guy in a gray hoodie and ripped jeans sitting on a worn wooden bench.  
a chic woman in a beige coat → a woman in a beige coat and ankle boots holding a phone.  
a techy man in a leather jacket → a man in a black leather jacket and glasses.  
a relaxed woman in a green sweater → a woman in a dark green sweater and black jeans carrying a tan shoulder bag.

The sentence: a _ → """


def _load_model_background(model_path, attn_impl, load_4bit, load_8bit, attn_layer_ind):
    """Load model in background thread."""
    with _model_lock:
        _model_state["loading"] = True
        _model_state["error"] = None

    enable_inference_optimizations()
    tokenizer, model, image_processor, _ = load_model_and_setup(
        model_path=model_path,
        attn_implementation=attn_impl,
        load_4bit=load_4bit,
        load_8bit=load_8bit,
        attn_layer_ind=attn_layer_ind,
    )

    with _model_lock:
        _model_state["tokenizer"] = tokenizer
        _model_state["model"] = model
        _model_state["image_processor"] = image_processor
        _model_state["loaded"] = True
        _model_state["loading"] = False

    print("✅ Model loaded and ready for demo.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/status")
def model_status():
    with _model_lock:
        return jsonify({
            "loaded": _model_state["loaded"],
            "loading": _model_state["loading"],
            "error": _model_state["error"],
        })


@app.route("/default-image")
def default_image():
    """Serve the default demo image."""
    path = Path(DEFAULT_IMAGE_PATH)
    if not path.exists():
        return jsonify({"error": f"Default image not found: {path}"}), 404
    return send_file(str(path), mimetype="image/jpeg")


@app.route("/run", methods=["POST"])
def run_extraction():
    """
    Run the full single-region extraction pipeline.
    Expects JSON with:
      - image_base64: base64-encoded image (or null to use default)
      - bbox: {x_min, y_min, x_max, y_max} in normalized [0,1] coords
      - prompt: optional prompt override
      - bias_strength: optional float
      - max_new_tokens: optional int
      - temperature: optional float
    """
    with _model_lock:
        if not _model_state["loaded"]:
            status = "loading" if _model_state["loading"] else "not started"
            return jsonify({"error": f"Model not ready (status: {status})"}), 503

    data = request.get_json(force=True)

    # ── Load image ────────────────────────────────────────────────────────
    image_b64 = data.get("image_base64")
    if image_b64:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        # Save temporary copy for the pipeline
        tmp_image_path = MASK_DIR / "demo_input_image.jpg"
        MASK_DIR.mkdir(parents=True, exist_ok=True)
        image.save(str(tmp_image_path))
        image_path_str = str(tmp_image_path)
    else:
        image = load_image(DEFAULT_IMAGE_PATH)
        image_path_str = DEFAULT_IMAGE_PATH

    width, height = image.size

    # ── Build mask from bbox ──────────────────────────────────────────────
    bbox = data.get("bbox", {"x_min": 0.2, "y_min": 0.2, "x_max": 0.6, "y_max": 0.8})
    x_min = int(round(bbox["x_min"] * (width - 1)))
    y_min = int(round(bbox["y_min"] * (height - 1)))
    x_max = int(round(bbox["x_max"] * (width - 1)))
    y_max = int(round(bbox["y_max"] * (height - 1)))

    x_min = max(0, min(width - 1, x_min))
    y_min = max(0, min(height - 1, y_min))
    x_max = max(x_min + 1, min(width, x_max))
    y_max = max(y_min + 1, min(height, y_max))

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y_min:y_max, x_min:x_max] = 1

    MASK_DIR.mkdir(parents=True, exist_ok=True)
    mask_path = MASK_DIR / "active_region_mask.npy"
    np.save(str(mask_path), mask)

    # ── Save metadata ─────────────────────────────────────────────────────
    meta_path = mask_path.with_suffix(".json")
    meta = {
        "image_path": image_path_str,
        "image_width": width,
        "image_height": height,
        "timestamp": datetime.now().isoformat(),
        "role": "active",
        "mask_path": str(mask_path),
        "pixel_bounds": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
        "normalized_bounds": bbox,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # ── Run extraction ────────────────────────────────────────────────────
    from gazefollow.run_single_region_extraction import (
        _load_mask,
        _prepare_generation_config,
        _prepare_attention_config,
        _prepare_guidance_config,
        _default_query_indices,
    )
    from gazefollow.extract_attention_direct_gt_masks import run_generation_with_attention

    prompt = data.get("prompt", DEFAULT_PROMPT)
    bias_strength = data.get("bias_strength", 2.5)
    max_new_tokens = data.get("max_new_tokens", 64)
    temperature = data.get("temperature", 0.1)

    # Build a namespace that mimics argparse output
    class Args:
        pass

    args = Args()
    args.bias_strength = bias_strength
    args.max_new_tokens = max_new_tokens
    args.temperature = temperature
    args.do_sample = False
    args.top_k = 50
    args.attn_layer_ind = -1
    args.repr_capture_layer = data.get("repr_capture_layer", 0)
    args.repr_inject_layer = data.get("repr_inject_layer", 0)
    args.query_indices = _default_query_indices()
    args.guidance_top_k = 5
    args.similarity_weight = 0.7
    args.probability_weight = 0.3
    args.guidance_enable_after_step = 0
    args.guidance_enable_after_keyword = "looking"
    args.save_debug_files = False

    generation_config = _prepare_generation_config(args)
    attention_config = _prepare_attention_config(args)
    guidance_config = _prepare_guidance_config(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = OUTPUT_DIR / f"demo_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    mask_temp_path = run_dir / "region_mask.npy"
    np.save(str(mask_temp_path), mask)

    common_kwargs = {
        "image_path": image_path_str,
        "mask_path": str(mask_temp_path),
        "prompt": prompt,
        "model": _model_state["model"],
        "tokenizer": _model_state["tokenizer"],
        "image_processor": _model_state["image_processor"],
        "generation_config": generation_config,
        "attention_config": attention_config,
        "bias_strength": generation_config.get("bias_strength", 0),
        "use_gaze_guidance": False,
        "guidance_config": guidance_config,
        "save_debug_files": False,
        "use_gt_gaze_csv": False,
        "save_mask_overlays": False,
        "mask_overlay_alpha": 0.4,
        "include_image_inputs": True,
        "filter_image_tokens_to_person_mask": False,
        "same_mask_for_person": True,
        "attention_mask_viz_dir": False,
    }

    # Initial step for hidden state capture
    init_dir = run_dir / "initial_single_step"
    init_dir.mkdir(parents=True, exist_ok=True)
    init_results = run_generation_with_attention(
        output_dir=str(init_dir),
        break_after_first_step=True,
        **{**common_kwargs, "prompt": ""},
    )
    prev_hidden_state = init_results.get("first_step_hidden_state")

    # Main generation
    results = run_generation_with_attention(
        output_dir=str(run_dir),
        break_after_first_step=False,
        prev_run_last_hidden_state=prev_hidden_state,
        **common_kwargs,
    )

    generated_text = results.get("generated_text", "")
    summary = {
        "generated_text": generated_text,
        "num_tokens": results.get("num_tokens"),
        "bbox_pixel": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
        "bbox_normalized": bbox,
        "image_size": {"width": width, "height": height},
        "output_dir": str(run_dir),
        "evaluation_summary": results.get("evaluation_summary"),
    }

    summary_path = run_dir / "demo_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(summary), f, indent=2)

    return jsonify(summary)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Single Region Extraction Demo Server")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--model-path", default="lmms-lab/llava-onevision-qwen2-7b-ov-chat")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--load-4bit", action="store_true", default=False)
    parser.add_argument("--load-8bit", action="store_true", default=False)
    parser.add_argument("--attn-layer-ind", type=int, default=-1)
    cli = parser.parse_args()

    # Start model loading in background
    loader_thread = threading.Thread(
        target=_load_model_background,
        args=(cli.model_path, cli.attn_implementation, cli.load_4bit, cli.load_8bit, cli.attn_layer_ind),
        daemon=True,
    )
    loader_thread.start()

    print(f"Starting demo server on {cli.host}:{cli.port}")
    print("Model is loading in the background...")
    app.run(host=cli.host, port=cli.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
