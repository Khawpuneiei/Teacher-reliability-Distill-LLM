"""Math-Verify-backed answer parsing, equivalence, and correctness labels."""

from __future__ import annotations

import os
from functools import lru_cache
from dataclasses import dataclass

from .answers import AnswerSpan, extract_final_answer_span

MAX_ANSWER_CHARACTERS = 2048


@dataclass(frozen=True, slots=True)
class GradeOutcome:
    correct: bool | None
    prediction_parseable: bool
    gold_parseable: bool
    status: str
    prediction_span: AnswerSpan | None = None


def _parse_boxed(value: str):
    from math_verify import parse

    if len(value) > MAX_ANSWER_CHARACTERS:
        return None
    # A bare extracted expression is wrapped because Math-Verify's default
    # LaTeX extraction expects a recognized math environment such as \boxed.
    # Math-Verify's positive timeout uses multiprocessing on Windows and its
    # nested worker is not spawn-picklable. Windows smoke runs therefore parse
    # only bounded answer spans without that package timeout; Linux keeps it.
    timeout = 0 if os.name == "nt" else 5
    parsed = parse(
        r"\boxed{" + value + "}",
        fallback_mode="no_fallback",
        parsing_timeout=timeout,
    )
    return parsed if parsed else None


@lru_cache(maxsize=8192)
def _try_parse(value: str | None):
    if value is None or not value.strip():
        return None
    try:
        return _parse_boxed(value.strip())
    except Exception:
        return None


def answer_parseable(value: str | None) -> bool:
    """Return whether Math-Verify found a mathematical expression in an answer."""
    return _try_parse(value) is not None


def answers_equivalent(left: str | None, right: str | None) -> bool:
    """Compare two extracted answers, treating Math-Verify's relation as symmetric.

    Math-Verify intentionally gives gold-first comparisons asymmetric behavior
    for some interval/equation and solution-chain cases. Sample clustering is
    a peer comparison, so either gold-first ordering counts as agreement.
    """
    parsed_left = _try_parse(left)
    parsed_right = _try_parse(right)
    if parsed_left is None or parsed_right is None:
        return False
    from math_verify import verify

    try:
        timeout = 0 if os.name == "nt" else 5
        return bool(
            verify(parsed_left, parsed_right, timeout_seconds=timeout)
            or verify(parsed_right, parsed_left, timeout_seconds=timeout)
        )
    except Exception:
        return False


def grade_prediction(prediction_text: str, gold_text: str | None) -> GradeOutcome:
    """Grade the extracted final answer and retain prediction/gold parse coverage."""
    span = extract_final_answer_span(prediction_text)
    parsed_prediction = _try_parse(span.value) if span is not None else None
    parsed_gold = _try_parse(gold_text)

    if gold_text is None or not gold_text.strip():
        return GradeOutcome(
            None,
            parsed_prediction is not None,
            False,
            "missing_gold",
            span,
        )
    if parsed_gold is None:
        return GradeOutcome(
            None,
            parsed_prediction is not None,
            False,
            "unparseable_gold",
            span,
        )
    if span is None:
        return GradeOutcome(False, False, True, "missing_prediction_answer")
    if parsed_prediction is None:
        return GradeOutcome(False, False, True, "unparseable_prediction", span)

    from math_verify import verify

    try:
        timeout = 0 if os.name == "nt" else 5
        correct = bool(
            verify(parsed_gold, parsed_prediction, timeout_seconds=timeout)
        )
    except Exception:
        return GradeOutcome(None, True, True, "verifier_error", span)
    return GradeOutcome(correct, True, True, "graded", span)
