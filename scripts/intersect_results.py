import argparse
import json
from pathlib import Path


def load_records(path: Path):
    """Return list of records and container info for reconstruction."""
    with path.open() as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload, None

    if isinstance(payload, dict):
        for key in ("results", "data"):
            section = payload.get(key)
            if isinstance(section, list):
                return section, (payload, key)

    raise ValueError(f"{path} must be a list or contain a 'results'/'data' list")


def build_payload(filtered, container):
    if container is None:
        return filtered

    payload, key = container
    output = dict(payload)
    output[key] = filtered
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Intersect result records with a target id set."
    )
    parser.add_argument("results_path", type=Path, help="Path to results json")
    parser.add_argument("target_path", type=Path, help="Path to target json")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path (defaults to results dir)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_records, results_container = load_records(args.results_path)
    target_records, _ = load_records(args.target_path)

    target_ids = {item["id"] for item in target_records if "id" in item}
    results_ids = {item["id"] for item in results_records if "id" in item}

    filtered_results = [
        item for item in results_records if item.get("id") in target_ids
    ]
    missing = target_ids - results_ids

    output_path = args.output
    if output_path is None:
        suffix = args.results_path.suffix or ".json"
        output_name = (
            f"{args.results_path.stem}_intersect_{args.target_path.stem}{suffix}"
        )
        output_path = args.results_path.with_name(output_name)

    output_payload = build_payload(filtered_results, results_container)
    output_path.write_text(json.dumps(output_payload, indent=2))

    print(f"Total results: {len(results_records)}")
    print(f"Target items: {len(target_records)}")
    print(f"Intersection saved to: {output_path}")
    print(f"Missing from results: {len(missing)}")


if __name__ == "__main__":
    main()
