---
applyTo: "**/*.py"
description: Python standards for this repo
---

# Copilot: Python rules

## Language & style
- use the "llava" environment located at `~/llava/bin/python`
- Target Python ≥3.10. Use type hints everywhere; prefer `typing.Protocol` and `TypedDict` where appropriate.
- Use `pathlib` (not `os.path`), `logging` (not bare prints in libs), and f-strings for formatting.
- Prefer relative imports within the package; keep modules under `src/` and tests under `tests/` mirroring structure.
- Keep public APIs small; mark internal helpers with a leading underscore and avoid exporting them in `__all__`.

## Error handling (very important)
- **Do NOT wrap code in `try/except` unless it handles a specific, expected exception and adds value** (e.g., fallback, user-facing message, cleanup).
- Catch the narrowest exception class; never use bare `except:`.
- Do not silence exceptions; re-raise with context (`raise` or `raise ... from e`) and log at the appropriate level.

## Tooling
- Lint with Ruff; format with Black; type-check with mypy (strict for new modules).
- For data files or paths, prefer config-driven paths and **relative paths** within the repo.
- Provide minimal CLI entry points via `argparse` or `typer` for scripts; document usage in the module docstring.

## Libraries & patterns
- Prefer dependency injection over globals/singletons.
- Isolate I/O (network/DB/filesystem) behind thin adapters to simplify testing.
- For data validation at boundaries, use `pydantic` models or explicit `TypedDict` + validators.
