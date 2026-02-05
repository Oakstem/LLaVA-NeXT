#!/usr/bin/env python3
"""Sweep temperature values for generate_vanilla_inference.py and save responses."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def parse_model_response(stdout_text: str) -> str:
    marker = "=== Model Response ==="
    if marker not in stdout_text:
        return ""
    _, tail = stdout_text.split(marker, 1)
    lines = [line for line in tail.splitlines() if "UserWarning" not in line]
    return "\n".join(lines).strip()


def build_temperatures(start: float, end: float, step: float) -> list[float]:
    count = int(round((end - start) / step)) + 1
    return [round(start + i * step, 1) for i in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--temperature-start", type=float, default=0.)
    parser.add_argument("--temperature-end", type=float, default=0.5)
    parser.add_argument("--temperature-step", type=float, default=0.1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--python", default="/home/alonz/llava/bin/python")
    parser.add_argument(
        "--script",
        default=str(Path(__file__).with_name("generate_vanilla_inference.py")),
    )
    parser.add_argument("--output", default="gazefollow/temperature_sweep_results.json")
    args = parser.parse_args()

    temps = build_temperatures(args.temperature_start, args.temperature_end, args.temperature_step)
    results = []

    total = len(temps) * max(args.repetitions, 1)
    run_index = 0
    for temp in temps:
        for rep in range(1, max(args.repetitions, 1) + 1):
            run_index += 1
            print(
                f"[{run_index}/{total}] temperature={temp:.1f} rep={rep}",
                flush=True,
            )
            cmd = [
                args.python,
                args.script,
                "--image-path",
                args.image_path,
                "--adapter-path",
                args.adapter_path,
                "--do-sample",
                "--temperature",
                f"{temp:.1f}",
            ]
            start = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            duration = time.time() - start
            response = parse_model_response(proc.stdout)
            results.append(
                {
                    "temperature": temp,
                    "repetition": rep,
                    "response": response,
                    "returncode": proc.returncode,
                    "duration_sec": round(duration, 3),
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    pretty_path = output_path.with_suffix(".pretty.txt")
    with pretty_path.open("w", encoding="utf-8") as handle:
        for entry in results:
            handle.write(
                f"Temperature: {entry['temperature']:.1f} | Rep: {entry['repetition']}\n"
            )
            handle.write(entry["response"])
            handle.write("\n" + "-" * 60 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
