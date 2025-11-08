from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Sequence


@dataclass
class BinaryPrecisionResult:
    """Precision summary for binary predictions."""

    positive_label: int
    true_positives: int
    false_positives: int
    predicted_positives: int
    actual_positives: int
    total_samples: int

    @property
    def precision(self) -> float | None:
        if self.predicted_positives == 0:
            return None
        return self.true_positives / self.predicted_positives

    def to_dict(self) -> Dict[str, int | float | None]:
        payload: Dict[str, int | float | None] = asdict(self)
        precision_value = self.precision
        payload["precision"] = precision_value if precision_value is not None else None
        return payload


def compute_binary_precision(
    predictions: Sequence[int],
    labels: Sequence[int],
    *,
    positive_label: int = 1,
) -> BinaryPrecisionResult:
    """Compute precision for binary predictions and labels."""
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")

    total_samples = len(predictions)
    true_positives = 0
    false_positives = 0
    predicted_positives = 0
    actual_positives = 0

    for predicted, label in zip(predictions, labels):
        is_predicted_positive = predicted == positive_label
        is_actual_positive = label == positive_label

        if is_actual_positive:
            actual_positives += 1

        if is_predicted_positive:
            predicted_positives += 1
            if is_actual_positive:
                true_positives += 1
            else:
                false_positives += 1

    return BinaryPrecisionResult(
        positive_label=positive_label,
        true_positives=true_positives,
        false_positives=false_positives,
        predicted_positives=predicted_positives,
        actual_positives=actual_positives,
        total_samples=total_samples,
    )


def serialize_binary_precision(result: BinaryPrecisionResult) -> Dict[str, float]:
    """Serialize binary precision result to a dictionary."""
    return result.to_dict()
