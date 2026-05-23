"""
Demo server for vanilla LLaVA inference with a LoRA adapter.

Loads the base model + adapter on startup, then serves a web UI
that lets users submit images and prompts for generation.

Usage:
    /home/alonz/llava/bin/python demo/vanilla_server.py [--port 7861]
"""
import sys
import json
import base64
import argparse
import threading
from pathlib import Path
from io import BytesIO

from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, send_file

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "gazefollow"))

from gazefollow.generation_utils import (
    enable_inference_optimizations,
    fix_wsl_paths,
    load_model_and_setup,
    load_image,
)

app = Flask(__name__, static_folder=str(PROJECT_ROOT / "demo"))

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_ADAPTER_PATH = "training_outputs/llava-20260304_225738/checkpoint-10000"
DEFAULT_IMAGE_PATH = "/mnt/d/Projects/LLaVA-NeXT/baseline_images/39740.png"

# ── Global state ──────────────────────────────────────────────────────────────
_model_state = {
    "loaded": False,
    "loading": False,
    "tokenizer": None,
    "model": None,
    "image_processor": None,
    "model_path": None,
    "adapter_path": None,
    "error": None,
}
_model_lock = threading.Lock()


def _load_model_background(model_path, adapter_path, attn_impl, load_4bit, load_8bit):
    """Load model + adapter in background thread."""
    with _model_lock:
        _model_state["loading"] = True
        _model_state["error"] = None

    enable_inference_optimizations()

    resolved_adapter = fix_wsl_paths(adapter_path) if adapter_path else None
    tokenizer, model, image_processor, _ = load_model_and_setup(
        model_path=model_path,
        attn_implementation=attn_impl,
        load_4bit=load_4bit,
        load_8bit=load_8bit,
        adapter_path=resolved_adapter,
    )

    # Apply image config defaults
    from gazefollow.generate_vanilla_inference import ensure_image_config
    ensure_image_config(model, "anyres_max_4", "(1x1),...,(2x2)")

    with _model_lock:
        _model_state["tokenizer"] = tokenizer
        _model_state["model"] = model
        _model_state["image_processor"] = image_processor
        _model_state["model_path"] = model_path
        _model_state["adapter_path"] = adapter_path
        _model_state["loaded"] = True
        _model_state["loading"] = False

    print(f"✅ Model loaded with adapter: {adapter_path}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/vanilla/")
def index():
    return send_from_directory(app.static_folder, "vanilla_inference.html")


@app.route("/vanilla/status")
def model_status():
    with _model_lock:
        return jsonify({
            "loaded": _model_state["loaded"],
            "loading": _model_state["loading"],
            "error": _model_state["error"],
        })


@app.route("/vanilla/default-image")
def default_image():
    path = Path(DEFAULT_IMAGE_PATH)
    if not path.exists():
        return jsonify({"error": f"Default image not found: {path}"}), 404
    return send_file(str(path), mimetype="image/jpeg")


@app.errorhandler(Exception)
def handle_exception(e):
    """Return JSON for any unhandled exception instead of Flask's HTML error page."""
    import traceback
    traceback.print_exc()
    return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


def _resolve_device(model):
    """Get device from model, handling PeftModel wrappers."""
    if hasattr(model, "device"):
        dev = model.device
        if str(dev) != "meta":
            return dev
    # PeftModel: walk to the base model
    base = getattr(model, "base_model", model)
    base = getattr(base, "model", base)
    if hasattr(base, "device"):
        dev = base.device
        if str(dev) != "meta":
            return dev
    # Fallback: check first parameter
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


@app.route("/vanilla/run", methods=["POST"])
def run_inference():
    """
    Run vanilla inference. Expects JSON with:
      - image_base64: base64-encoded image (or null → default)
      - prompt: text prompt
      - max_new_tokens, temperature, top_p, do_sample, num_beams
    """
    import torch
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from gazefollow.generate_vanilla_inference import determine_template

    with _model_lock:
        if not _model_state["loaded"]:
            status = "loading" if _model_state["loading"] else "not started"
            return jsonify({"error": f"Model not ready (status: {status})"}), 503

    data = request.get_json(force=True)

    # ── Image ─────────────────────────────────────────────────────────────
    image_b64 = data.get("image_base64")
    if image_b64:
        image_bytes = base64.b64decode(image_b64)
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_path_str = "(uploaded)"
    else:
        pil_image = load_image(DEFAULT_IMAGE_PATH)
        image_path_str = DEFAULT_IMAGE_PATH

    model = _model_state["model"]
    tokenizer = _model_state["tokenizer"]
    image_processor = _model_state["image_processor"]
    device = _resolve_device(model)

    processed = process_images([pil_image], image_processor, model.config)
    if isinstance(processed, tuple):
        image_tensor = processed[0]
    else:
        image_tensor = processed
    if isinstance(image_tensor, list):
        image_tensor = image_tensor[0]
    if isinstance(image_tensor, torch.Tensor) and image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device, dtype=model.dtype)
    image_size = pil_image.size

    # ── Build conversation ────────────────────────────────────────────────
    prompt_text = data.get("prompt", "").strip()
    max_new_tokens = data.get("max_new_tokens", 300)
    temperature = data.get("temperature", 0.9)
    top_p = data.get("top_p", 0.9)
    do_sample = data.get("do_sample", True)
    use_adapter = data.get("use_adapter", True)

    # Toggle LoRA adapter layers
    from peft import PeftModel
    if isinstance(model, PeftModel):
        if use_adapter:
            model.enable_adapter_layers()
        else:
            model.disable_adapter_layers()

    model_name_source = model.config._name_or_path if hasattr(model.config, "_name_or_path") else _model_state["model_path"]
    model_name = get_model_name_from_path(model_name_source)
    conv_name = determine_template(model_name, None)
    conv = conv_templates[conv_name].copy()
    conv.tokenizer = tokenizer

    user_content = f"{DEFAULT_IMAGE_TOKEN}\n{prompt_text}" if DEFAULT_IMAGE_TOKEN not in prompt_text else prompt_text
    conv.append_message(conv.roles[0], user_content)
    conv.append_message(conv.roles[1], None)
    prompt_for_tokenizer = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_for_tokenizer, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
    ).unsqueeze(0).to(device)

    gen_kwargs = {
        "inputs": input_ids,
        "images": image_tensor,
        "image_sizes": [list(image_size)],
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    # ── Generate ──────────────────────────────────────────────────────────
    with torch.inference_mode():
        raw_output = model.generate(**gen_kwargs)

    # LlavaQwen.generate() returns (sequences, image_features) tuple
    if isinstance(raw_output, (list, tuple)) and len(raw_output) == 2:
        output = raw_output[0]
    else:
        output = raw_output

    if hasattr(output, "sequences"):
        output_ids = output.sequences[0].tolist()
    elif isinstance(output, torch.Tensor):
        output_ids = (output[0] if output.ndim > 1 else output).tolist()
    else:
        output_ids = list(output)

    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    return jsonify({
        "response": response,
        "image_path": image_path_str,
        "num_tokens": len(output_ids),
        "model_path": _model_state["model_path"],
        "adapter_path": _model_state["adapter_path"] if use_adapter else "(disabled)",
    })


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vanilla Inference Demo Server")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--model-path", default="lmms-lab/llava-onevision-qwen2-7b-ov-chat")
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--load-4bit", action="store_true", default=False)
    parser.add_argument("--load-8bit", action="store_true", default=False)
    cli = parser.parse_args()

    loader_thread = threading.Thread(
        target=_load_model_background,
        args=(cli.model_path, cli.adapter_path, cli.attn_implementation, cli.load_4bit, cli.load_8bit),
        daemon=True,
    )
    loader_thread.start()

    print(f"Starting vanilla inference demo on {cli.host}:{cli.port}")
    print(f"Adapter: {cli.adapter_path}")
    print("Model is loading in the background...")
    app.run(host=cli.host, port=cli.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
