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
import re
import threading
import uuid
from pathlib import Path, PurePosixPath
from io import BytesIO

from PIL import Image
from flask import Flask, request, jsonify, redirect, send_from_directory, send_file
from werkzeug.exceptions import HTTPException

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
UPLOADED_ADAPTER_ROOT = Path("/tmp/llava_vanilla_uploaded_adapters")

# ── Global state ──────────────────────────────────────────────────────────────
_model_state = {
    "loaded": False,
    "loading": False,
    "tokenizer": None,
    "model": None,
    "image_processor": None,
    "model_path": None,
    "adapter_path": None,
    "adapters": [],
    "default_adapter_name": None,
    "error": None,
}
_model_lock = threading.Lock()
_adapter_lock = threading.Lock()


def _adapter_label(adapter_path):
    path = Path(adapter_path.rstrip("/"))
    parent = path.parent.name
    return f"{parent}/{path.name}" if parent else path.name


def _adapter_name(label, existing):
    base = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)
    base = base.strip("._-") or "adapter"
    name = base
    suffix = 2
    while name in existing:
        name = f"{base}_{suffix}"
        suffix += 1
    existing.add(name)
    return name


def _build_adapter_specs(default_adapter_path, extra_adapter_paths):
    specs = []
    seen_paths = set()
    used_names = set()

    def add(path, name=None, label_prefix=None):
        if not path or path in seen_paths:
            return
        seen_paths.add(path)
        label = _adapter_label(path)
        adapter_name = name or _adapter_name(label, used_names)
        if name:
            used_names.add(name)
        specs.append({
            "name": adapter_name,
            "label": f"{label_prefix}: {label}" if label_prefix else label,
            "path": path,
        })

    add(default_adapter_path, name="default", label_prefix="Default")
    for adapter_path in extra_adapter_paths:
        add(adapter_path)
    return specs


def _load_extra_adapter(adapter_path):
    from peft import PeftModel

    with _adapter_lock:
        model = _model_state["model"]
        if not isinstance(model, PeftModel):
            raise TypeError("Extra adapter loading requires the default adapter to be loaded first.")

        adapter_by_path = {spec["path"]: spec for spec in _model_state["adapters"]}
        if adapter_path in adapter_by_path:
            return adapter_by_path[adapter_path]

        used_names = {spec["name"] for spec in _model_state["adapters"]}
        name = _adapter_name(_adapter_label(adapter_path), used_names)
        spec = {
            "name": name,
            "label": _adapter_label(adapter_path),
            "path": adapter_path,
        }
        materialized_count = _materialize_adapter_target_modules(model, adapter_path)
        if materialized_count:
            print(f"Materialized {materialized_count} adapter target modules before loading '{name}'.")
        model.load_adapter(
            fix_wsl_paths(adapter_path),
            adapter_name=name,
            is_trainable=False,
        )
        _model_state["adapters"].append(spec)
        return spec


def _target_module_matches(config, key):
    target_modules = config.target_modules
    if isinstance(target_modules, str):
        matched = re.fullmatch(target_modules, key) is not None
    else:
        matched = any(re.match(f".*\\.{target_key}$", key) for target_key in target_modules)
        matched = matched or any(target_key == key for target_key in target_modules)

    layers_to_transform = getattr(config, "layers_to_transform", None)
    if not matched or layers_to_transform is None:
        return matched

    layers_pattern = getattr(config, "layers_pattern", None) or ["layers", "h", "block", "blocks"]
    if isinstance(layers_pattern, str):
        layers_pattern = [layers_pattern]
    for pattern in layers_pattern:
        layer_index = re.match(f".*.{pattern}\\.(\\d+)\\.*", key)
        if layer_index is None:
            continue
        layer_index = int(layer_index.group(1))
        if isinstance(layers_to_transform, int):
            return layer_index == layers_to_transform
        return layer_index in layers_to_transform
    return False


def _materialize_adapter_target_modules(model, adapter_path):
    from accelerate.hooks import remove_hook_from_module
    from peft import PeftConfig
    import torch

    config = PeftConfig.from_pretrained(fix_wsl_paths(adapter_path))
    module_by_name = dict(model.named_modules())
    materialized_ids = set()
    materialized_count = 0
    for name, module in model.named_modules():
        if not _target_module_matches(config, name):
            continue
        hooked_name, hooked_module = _nearest_hooked_ancestor(name, module_by_name)
        if hooked_module is None or id(hooked_module) in materialized_ids:
            continue
        materialized_ids.add(id(hooked_module))
        execution_device = _hook_execution_device(hooked_module._hf_hook)
        if isinstance(execution_device, int):
            execution_device = torch.device("cuda", execution_device)
        remove_hook_from_module(hooked_module, recurse=True)
        if execution_device and torch.device(execution_device).type != "meta":
            hooked_module.to(execution_device)
        materialized_count += 1
    return materialized_count


def _nearest_hooked_ancestor(name, module_by_name):
    parts = name.split(".")
    for end in range(len(parts), -1, -1):
        ancestor_name = ".".join(parts[:end])
        module = module_by_name.get(ancestor_name)
        if module is not None and hasattr(module, "_hf_hook"):
            return ancestor_name, module
    return None, None


def _hook_execution_device(hook):
    execution_device = getattr(hook, "execution_device", None)
    if execution_device is not None:
        return execution_device
    for child_hook in getattr(hook, "hooks", ()):
        execution_device = _hook_execution_device(child_hook)
        if execution_device is not None:
            return execution_device
    return None


def _safe_upload_path(filename):
    parts = [
        part for part in PurePosixPath(filename.replace("\\", "/")).parts
        if part not in {"", ".", "/"}
    ]
    if not parts or any(part == ".." for part in parts):
        return None
    return Path(*parts)


def _save_uploaded_adapter(files):
    upload_dir = UPLOADED_ADAPTER_ROOT / uuid.uuid4().hex
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for file_storage in files:
        rel_path = _safe_upload_path(file_storage.filename)
        if rel_path is None:
            continue
        target_path = upload_dir / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        file_storage.save(target_path)
        saved_files.append(target_path)

    if not saved_files:
        raise ValueError("No adapter files were uploaded.")

    config_files = [path for path in saved_files if path.name == "adapter_config.json"]
    if not config_files:
        raise ValueError("Uploaded folder does not contain adapter_config.json.")

    adapter_dir = config_files[0].parent
    return str(adapter_dir)


def _load_model_background(model_path, default_adapter_path, extra_adapter_paths, attn_impl, load_4bit, load_8bit):
    """Load model + one or more adapters in a background thread."""
    with _model_lock:
        _model_state["loading"] = True
        _model_state["error"] = None

    enable_inference_optimizations()

    adapter_specs = _build_adapter_specs(default_adapter_path, extra_adapter_paths)
    primary_adapter = adapter_specs[0]["path"] if adapter_specs else None
    resolved_adapter = fix_wsl_paths(primary_adapter) if primary_adapter else None
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

    if adapter_specs:
        from peft import PeftModel
        if not isinstance(model, PeftModel):
            raise TypeError("Expected a PeftModel after loading the default adapter.")
        for spec in adapter_specs[1:]:
            print(f"Loading additional LoRA adapter '{spec['name']}' from: {spec['path']}")
            materialized_count = _materialize_adapter_target_modules(model, spec["path"])
            if materialized_count:
                print(f"Materialized {materialized_count} adapter target modules before loading '{spec['name']}'.")
            model.load_adapter(
                fix_wsl_paths(spec["path"]),
                adapter_name=spec["name"],
                is_trainable=False,
            )
        model.set_adapter(adapter_specs[0]["name"])

    with _model_lock:
        _model_state["tokenizer"] = tokenizer
        _model_state["model"] = model
        _model_state["image_processor"] = image_processor
        _model_state["model_path"] = model_path
        _model_state["adapter_path"] = primary_adapter
        _model_state["adapters"] = adapter_specs
        _model_state["default_adapter_name"] = adapter_specs[0]["name"] if adapter_specs else None
        _model_state["loaded"] = True
        _model_state["loading"] = False

    adapter_summary = ", ".join(f"{spec['name']}={spec['path']}" for spec in adapter_specs) or "(none)"
    print(f"✅ Model loaded with adapters: {adapter_summary}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/vanilla/")
def index():
    return send_from_directory(app.static_folder, "vanilla_inference.html")


@app.route("/")
def root():
    return redirect("/vanilla/")


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/vanilla/status")
def model_status():
    with _model_lock:
        return jsonify({
            "loaded": _model_state["loaded"],
            "loading": _model_state["loading"],
            "error": _model_state["error"],
            "adapters": _model_state["adapters"],
            "default_adapter_name": _model_state["default_adapter_name"],
        })


@app.route("/vanilla/load-adapter", methods=["POST"])
def load_adapter():
    data = request.get_json(force=True)
    adapter_path = (data.get("adapter_path") or "").strip()
    if not adapter_path:
        return jsonify({"error": "adapter_path is required"}), 400

    with _model_lock:
        if not _model_state["loaded"]:
            status = "loading" if _model_state["loading"] else "not started"
            return jsonify({"error": f"Model not ready (status: {status})"}), 503

    spec = _load_extra_adapter(adapter_path)
    return jsonify({"adapter": spec, "adapters": _model_state["adapters"]})


@app.route("/vanilla/upload-adapter", methods=["POST"])
def upload_adapter():
    files = request.files.getlist("adapter_files")
    if not files:
        return jsonify({"error": "No adapter files were uploaded."}), 400

    with _model_lock:
        if not _model_state["loaded"]:
            status = "loading" if _model_state["loading"] else "not started"
            return jsonify({"error": f"Model not ready (status: {status})"}), 503

    try:
        adapter_path = _save_uploaded_adapter(files)
    except ValueError as err:
        return jsonify({"error": str(err)}), 400

    spec = _load_extra_adapter(adapter_path)
    return jsonify({"adapter": spec, "adapters": _model_state["adapters"]})


@app.route("/vanilla/default-image")
def default_image():
    path = Path(DEFAULT_IMAGE_PATH)
    if not path.exists():
        return jsonify({"error": f"Default image not found: {path}"}), 404
    return send_file(str(path), mimetype="image/jpeg")


@app.errorhandler(Exception)
def handle_exception(e):
    """Return JSON for any unhandled exception instead of Flask's HTML error page."""
    if isinstance(e, HTTPException):
        return jsonify({"error": f"{e.name}: {e.description}"}), e.code

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
    adapter_name = data.get("adapter_name") or _model_state["default_adapter_name"]
    adapter_by_name = {spec["name"]: spec for spec in _model_state["adapters"]}

    # Validate the requested LoRA adapter before generation.
    from peft import PeftModel
    is_peft_model = isinstance(model, PeftModel)
    if is_peft_model and use_adapter and adapter_name not in adapter_by_name:
        return jsonify({"error": f"Unknown adapter: {adapter_name}"}), 400

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
    with _adapter_lock, torch.inference_mode():
        if is_peft_model:
            if use_adapter:
                model.set_adapter(adapter_name)
                model.enable_adapter_layers()
            else:
                model.disable_adapter_layers()
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
        "adapter_path": adapter_by_name[adapter_name]["path"] if use_adapter else "(disabled)",
        "adapter_name": adapter_name if use_adapter else None,
    })


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import logging
    from flask import cli as flask_cli

    parser = argparse.ArgumentParser(description="Vanilla Inference Demo Server")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--model-path", default="lmms-lab/llava-onevision-qwen2-7b-ov-chat")
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--extra-adapter-path", action="append", default=[])
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--load-4bit", action="store_true", default=False)
    parser.add_argument("--load-8bit", action="store_true", default=False)
    cli = parser.parse_args()

    loader_thread = threading.Thread(
        target=_load_model_background,
        args=(cli.model_path, cli.adapter_path, cli.extra_adapter_path, cli.attn_implementation, cli.load_4bit, cli.load_8bit),
        daemon=True,
    )
    loader_thread.start()

    flask_cli.show_server_banner = lambda *args, **kwargs: None
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    print(f"Starting vanilla inference demo on {cli.host}:{cli.port}")
    print(f"Open: http://localhost:{cli.port}/vanilla/")
    print(f"Default adapter: {cli.adapter_path}")
    for adapter_path in cli.extra_adapter_path:
        print(f"Extra adapter: {adapter_path}")
    print("Model is loading in the background...")
    app.run(host=cli.host, port=cli.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
