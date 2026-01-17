from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from transformers import pipeline

DEFAULT_MODEL_ID = "openai/gpt-oss-20b"
SYSTEM_PROMPT = (
    "You specialize in extracting gaze targets from descriptions and must reply with one concise phrase."
)
USER_PROMPT_TEMPLATE = (
    "Description: {description}\n"
    "Respond only with the short phrase that states where the person is looking."
)


def load_generation_results(json_path: Path) -> List[Dict]:
    """Load generation results from a JSON file."""
    with json_path.open("r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {json_path}, found {type(data).__name__}")
    return data


def extract_model_predictions(entries: Iterable[Dict]) -> List[Dict[str, str]]:
    """Return entries that contain a non-empty model prediction."""
    extracted = []
    for entry in entries:
        prediction = entry.get("model_prediction")
        if isinstance(prediction, str) and prediction.strip():
            extracted.append(
                {
                    "id": entry.get("id"),
                    "model_prediction": prediction.strip(),
                }
            )
    return extracted


def build_messages(description: str) -> List[Dict[str, str]]:
    """Construct chat-style messages for the language model."""
    cleaned_description = " ".join(description.split())
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(description=cleaned_description)},
    ]


class GptOssGazeInference:
    """Wrapper around the GPT-OSS 20B text-generation pipeline."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device_map: str = "auto",
        pipeline_kwargs: Optional[Dict] = None,
        default_generation_kwargs: Optional[Dict] = None,
    ) -> None:
        generator_kwargs = dict(pipeline_kwargs or {})
        generator_kwargs.setdefault("model", model_id)
        generator_kwargs.setdefault("tokenizer", model_id)
        generator_kwargs.setdefault("device_map", device_map)
        generator_kwargs.setdefault("torch_dtype", "auto")
        self._generator = pipeline("text-generation", **generator_kwargs)
        self._default_generation_kwargs = default_generation_kwargs or {
            "max_new_tokens": 32,
            "do_sample": False,
            "temperature": 0.2,
        }

    def generate_short_answer(
        self,
        description: str,
        *,
        generation_kwargs: Optional[Dict] = None,
    ) -> str:
        """Produce a short answer describing where the person is looking."""
        messages = build_messages(description)
        merged_kwargs = dict(self._default_generation_kwargs)
        if generation_kwargs:
            merged_kwargs.update(generation_kwargs)
        results = self._generator(messages, **merged_kwargs)
        generated = results[0].get("generated_text")

        if isinstance(generated, list):
            for message in reversed(generated):
                if message.get("role") == "assistant":
                    content = message.get("content", "").strip()
                    if content:
                        return content
        elif isinstance(generated, str):
            return generated.strip()

        return ""


def generate_answers_for_predictions(
    predictions: Iterable[Dict[str, str]],
    inference: GptOssGazeInference,
    *,
    generation_kwargs: Optional[Dict] = None,
) -> List[Dict[str, str]]:
    """Run inference over a set of model predictions."""
    outputs = []
    for prediction_entry in predictions:
        answer = inference.generate_short_answer(
            prediction_entry["model_prediction"],
            generation_kwargs=generation_kwargs,
        )
        outputs.append(
            {
                "id": prediction_entry.get("id"),
                "model_prediction": prediction_entry["model_prediction"],
                "gpt_oss_short_answer": answer,
            }
        )
    return outputs


def write_answers(answers: List[Dict[str, str]], output_path: Path) -> None:
    """Persist generated answers to disk."""
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(answers, output_file, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate short gaze answers from model predictions using GPT-OSS 20B."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("evaluation_results/eval_20251025_124116/model_generation_results.json"),
        help="Path to a model_generation_results.json file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination for the generated answers.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Model identifier to load from Hugging Face.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to the Transformers pipeline.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum number of tokens to generate for each answer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Restrict inference to the first N predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_generation_results(args.input)
    predictions = extract_model_predictions(entries)
    if args.limit is not None:
        predictions = predictions[: args.limit]

    inference = GptOssGazeInference(
        model_id=args.model_id,
        device_map=args.device_map,
        default_generation_kwargs={
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "temperature": args.temperature,
        },
    )

    answers = generate_answers_for_predictions(predictions, inference)

    if args.output:
        write_answers(answers, args.output)
    else:
        for answer in answers:
            identifier = answer.get("id")
            prefix = f"{identifier}: " if identifier else ""
            print(f"{prefix}{answer['gpt_oss_short_answer']}")


if __name__ == "__main__":
    main()
