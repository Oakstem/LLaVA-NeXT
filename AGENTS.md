# Repository Guidelines

## Project Structure & Module Organization
Core multimodal code lives in `llava/`: `model/` handles architectures and checkpoint loading, `serve/` exposes CLI and service runners, and `train/` hosts shared training utilities. Finetuning recipes and DeepSpeed configs live in `scripts/train/`. Dataset-driven experiments sit at the repo root (e.g., `beam_search.py`, `parallel_inference_simple.py`), while heavier assets land under `finetune/`, `gazefollow/`, or `training_outputs/`. Regression tests stay in `tests/`; lighter smoke scripts remain separate `test_*.py` files.

## Build, Test, and Development Commands
Provision the environment once per machine:
```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip
pip install -e ".[train]"
```

**Important**: All Python commands should use the conda environment's Python executable at `~/llava/bin/python` to ensure proper dependencies and environment isolation.
Launch the chat CLI to sanity-check checkpoints:
```bash
~/llava/bin/python -m llava.serve.cli --model-path lmms-lab/LLaVA-NeXT-Video-7B-DPO --image-file path/to/sample.jpg
```
Run the Python unit suite (mock-heavy, CPU-safe):
```bash
~/llava/bin/python -m unittest discover -s tests -p 'test_*.py'
```

## Coding Style & Naming Conventions
Follow standard Python style with 4-space indentation and `snake_case` functions. Run `black` (line length 240 as defined in `pyproject.toml`) before pushing. Prefer explicit typing on new APIs and reuse existing dataclass or helper patterns already used in `llava/model`. Name scripts with descriptive verbs (e.g., `generate_*`, `extract_*`) and keep modules import-safe.

Avoid overusing try-except blocks as they pollute the code and make it unreadable. Use exception handling judiciously for specific, expected failure cases rather than wrapping large code blocks defensively.

## Testing Guidelines
Unit coverage relies on `unittest` with extensive mocking (`tests/test_extract_attention_interactive_refactored.py`). Mirror that structure when adding suites and name files `test_<feature>.py`. GPU-dependent checks should guard on `torch.cuda.is_available()` and use minimal fixtures. When tests generate artifacts, write them under a temp directory and clean them up inside `tearDown` or context managers.

## Commit & Pull Request Guidelines
Recent history favors concise, descriptive summaries (`Refactor attention mask handling`, `Disable auto saving of similarity maps`). Use imperative or present-tense verbs and keep the subject on the changed behavior; add extended notes in the body as needed. Pull requests must link relevant issues, summarize model or evaluation impact, list commands run, and attach key logs or screenshots (attention maps, gaze overlays) when behavior changes.
