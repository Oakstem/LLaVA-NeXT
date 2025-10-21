# Copilot instructions for LLaVA-NeXT (repo-specific)

This fork adds custom inference, evaluation, and training glue around LLaVA/LLaVA-OneVision with gaze-target analysis and SLURM workflows. Use the notes below to be productive quickly and keep changes aligned with project patterns.

## Architecture and key modules
- Core package: `llava/`
  - `llava/model/*`: multimodal backbones/builders; `builder.load_pretrained_model` is the canonical loader.
  - `llava/mm_utils.py`: image processing (`process_images`), tokenizer helpers (`tokenizer_image_token`).
  - `llava/conversation.py`: conversation templates (`conv_templates`); default to `qwen_1_5` for Qwen-style.
  - `llava/train/train.py`: training entry with DeepSpeed/LoRA, plus custom evaluation hooks into `evaluate_model.py`.
- Root utilities and entry points:
  - `generation_utils.py`: shared loading, inference optimizations, and compatibility patches for older Torch/Transformers. Prefer reusing helpers from here.
  - `generate_vanilla_inference.py`: single-image prompt inference; supports base or LoRA adapters; optional GroundingDINO post-processing.
  - `evaluate_model.py`: dataset-driven evaluation with optional "focus loss after 'looking at'" analysis, wandb logging, and iterative generation mode.
  - `detect_gaze_targets.py`: GroundingDINO-based target detection from free-form descriptions (extracts "looking at …" phrases per person).
- Work orchestration: `slurm/` contains ready-to-run jobs (training, evaluation, inference, gaze detection, temperature sweeps). Prefer adding new jobs here.

## Environment, builds, and runtime
- Install once per machine (Python 3.10 recommended):
  - `pip install -e ".[train]"` (see `pyproject.toml` for pinned deps: Torch 2.1.2, Transformers pinned commit, DeepSpeed 0.14.4, bitsandbytes 0.41.0, etc.).
  - Long lines: Black line length is 240 (`pyproject.toml`).
- The code includes safety shims/monkey patches for older Torch/Transformers in `generation_utils.py` and `generate_vanilla_inference.py`. Avoid upgrading core versions without testing those paths.
- Use `enable_inference_optimizations()` (TF32, cuDNN tweaks). Many CLIs expose `--disable-optimizations` to turn this off.

## Typical workflows (local or SLURM)
- Inference (single image):
  - `python generate_vanilla_inference.py --model-path lmms-lab/llava-onevision-qwen2-7b-ov-chat --image-path baseline_images/39740.png --prompt "Describe where each person is looking."`
  - Adapters/quantization: `--model-base`, `--adapter-path`, `--load-4bit|--load-8bit`, `--attn-implementation sdpa|flash_attention_2`.
  - Image preprocessing defaults: aspect ratio fallback and grid pinpoints are auto-resolved; override via `--image-aspect-ratio` and `--image-grid-pinpoints`.
- Evaluation:
  - `python evaluate_model.py --model-path <ckpt|hf_id> --dataset-json <file>.json --images-dir <dir> [--adapter-path ...] [--do-sample]`.
  - Custom analysis flags: `--focus-loss-after-looking`, `--focus-loss-phrase "looking at"`, `--use-iterative-generation`, `--no-loss`, `--log-to-wandb`.
- Training:
  - Entry: `python -m llava.train.train ...` or use `slurm/run_train.slurm`.
  - `llava/train/train.py` wires optional custom eval from `evaluate_model.py` via `TrainingArguments.use_custom_eval` and related `eval_*` flags. Prefer extending these hooks over bespoke evaluation code.
- SLURM jobs:
  - See `slurm/run_vanilla_inference.slurm`, `slurm/run_model_evaluation.slurm`, `slurm/run_detect_gaze_targets.slurm`, `slurm/run_temperature_sweep.sh`. Jobs standardize env activation, logging to `logs/`, and GPU sanity checks.

## Project-specific conventions and patterns
- Minimalism: write concise, focused code with minimal abstractions; only introduce utilities when they’re reused across entrypoints.
- Exceptions: do not add extra/broad exception catching—let exceptions surface unless there’s a specific, expected failure with a clear recovery path or user-facing message.
- Image + prompt formatting: include `DEFAULT_IMAGE_TOKEN` at the start of the user message if not present. Use `conv_templates[...]` to build prompts consistently.
- Always use `process_images` from `llava/mm_utils.py` to obtain tensors compatible with model config (handles anyres/vision towers).
- When adding inference features, centralize shared logic in `generation_utils.py` (so CLIs can reuse it) and respect `--disable-optimizations`.
- For LoRA integration, prefer runtime merge via `--model-base` + `--adapter-path`; training uses PEFT through `llava/train/*` utilities.
- GroundingDINO integration:
  - Standalone: `detect_gaze_targets.py --image-path ... --description-text "Person 1: … looking at …"` (or `--description-file`).
  - From inference: `generate_vanilla_inference.py --run-gdino` to parse generated descriptions and detect objects (`--gdino-*` knobs).
- Logging/metrics: use `safe_wandb_log` from `llava/train/train.py` and metrics helpers in `generation_metrics.py` when extending evaluation.
- Style: 4-space Python, explicit typing where practical, and no defensive wrappers; follow the guidance in `AGENTS.md`.

## Integration points and external dependencies
- Hugging Face Transformers (pinned commit), Accelerate, DeepSpeed, bitsandbytes, PEFT, OpenCLIP, timm; video stacks via `av`/`decord`; GroundingDINO via `transformers`' `GroundingDinoForObjectDetection`.
- Conversation and tokenizer coupling: `qwen_1_5` template is the default for Qwen-style checkpoints; use `determine_template()` from CLIs/eval.

## Useful references in this repo
- Inference examples: `generate_vanilla_inference.py` (streaming, chat mode, GDINO hooks).
- Training glue and eval hooks: `llava/train/train.py`, `llava/train/llava_trainer*.py`.
- Gaze detection pipeline: `detect_gaze_targets.py` + regex patterns for "looking at …" + GroundingDINO post-processing.
- Jobs: `slurm/*` for repeatable cluster runs.

If anything above is unclear or missing (e.g., dataset JSON schema examples for `evaluate_model.py`, or preferred training argument presets), tell me what you’re trying to do and I’ll refine this guidance.