"""Small, dependency-free metrics used by the reliability report."""

from __future__ import annotations

import math
from collections.abc import Sequence


def calibration_bins(
    confidences: Sequence[float],
    outcomes: Sequence[bool | int],
    n_bins: int = 15,
) -> list[dict[str, float | int | None]]:
    """Return auditable fixed-width bin counts, means, and empirical accuracy."""
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    pairs = _validated_pairs(confidences, outcomes)
    if not pairs:
        raise ValueError("calibration bins are undefined for an empty collection")

    bin_rows: list[list[float | int]] = [[0.0, 0.0, 0] for _ in range(n_bins)]
    for confidence, outcome in pairs:
        index = min(int(confidence * n_bins), n_bins - 1)
        bin_rows[index][0] += confidence
        bin_rows[index][1] += outcome
        bin_rows[index][2] += 1

    output = []
    for index, (confidence_sum, outcome_sum, count) in enumerate(bin_rows):
        count = int(count)
        output.append(
            {
                "bin_index": index,
                "lower": index / n_bins,
                "upper": (index + 1) / n_bins,
                "count": count,
                "mean_confidence": float(confidence_sum) / count if count else None,
                "accuracy": float(outcome_sum) / count if count else None,
            }
        )
    return output


def _validated_pairs(
    confidences: Sequence[float], outcomes: Sequence[bool | int]
) -> list[tuple[float, int]]:
    if len(confidences) != len(outcomes):
        raise ValueError("confidence and outcome counts must match")

    pairs: list[tuple[float, int]] = []
    for confidence, outcome in zip(confidences, outcomes):
        score = float(confidence)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("confidence scores must be finite values in [0, 1]")
        if outcome not in (False, True, 0, 1):
            raise ValueError("outcomes must be boolean or binary values")
        pairs.append((score, int(outcome)))
    return pairs


def expected_calibration_error(
    confidences: Sequence[float],
    outcomes: Sequence[bool | int],
    n_bins: int = 15,
) -> float:
    """Return fixed-width ECE; the final bin includes confidence exactly 1."""
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    pairs = _validated_pairs(confidences, outcomes)
    if not pairs:
        raise ValueError("ECE is undefined for an empty collection")

    bin_sums = [[0.0, 0.0, 0] for _ in range(n_bins)]
    for confidence, outcome in pairs:
        index = min(int(confidence * n_bins), n_bins - 1)
        bin_sums[index][0] += confidence
        bin_sums[index][1] += outcome
        bin_sums[index][2] += 1

    total = len(pairs)
    return sum(
        (count / total) * abs((confidence_sum / count) - (outcome_sum / count))
        for confidence_sum, outcome_sum, count in bin_sums
        if count
    )


def binary_roc_auc(
    outcomes: Sequence[bool | int], confidence_scores: Sequence[float]
) -> float | None:
    """Return tie-correct binary AUROC, or ``None`` when only one class exists.

    Scores must be oriented so that larger values indicate the positive class.
    """
    pairs = _validated_pairs(confidence_scores, outcomes)
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None

    ordered = sorted((score, label) for score, label in pairs)
    positive_rank_sum = 0.0
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][0] == ordered[cursor][0]:
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label for _, label in ordered[cursor:end]
        )
        cursor = end

    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )
