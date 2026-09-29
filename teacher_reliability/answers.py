"""Format-aware answer-span extraction used for grading and token alignment."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AnswerSpan:
    value: str
    start: int
    end: int
    method: str


_BOXED = re.compile(r"\\boxed\s*\{")
_ANSWER_LABEL = re.compile(
    r"(?im)^[ \t]*(?:final[ \t]+)?answer[ \t]*[:=][ \t]*(.*?)[ \t]*$"
)


def _brace_is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def _matching_brace(text: str, opening_brace: int) -> int | None:
    depth = 1
    for index in range(opening_brace + 1, len(text)):
        if text[index] not in "{}" or _brace_is_escaped(text, index):
            continue
        if text[index] == "{":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return index
    return None


def extract_final_answer_span(text: str) -> AnswerSpan | None:
    """Return the last well-formed boxed answer or explicit final-answer line.

    Start/end are offsets into the original string. An unmarked reasoning
    paragraph is not guessed as an answer because that can misalign token
    probabilities and silently grade an intermediate expression.
    """
    if not isinstance(text, str) or not text:
        return None

    boxed_spans: list[AnswerSpan] = []
    for match in _BOXED.finditer(text):
        opening_brace = match.end() - 1
        closing_brace = _matching_brace(text, opening_brace)
        if closing_brace is None:
            continue
        start, end = opening_brace + 1, closing_brace
        value = text[start:end].strip()
        if value:
            left_trim = len(text[start:end]) - len(text[start:end].lstrip())
            right_trim = len(text[start:end].rstrip())
            boxed_spans.append(
                AnswerSpan(value, start + left_trim, start + right_trim, "boxed")
            )
    if boxed_spans:
        return boxed_spans[-1]

    labelled_spans: list[AnswerSpan] = []
    for match in _ANSWER_LABEL.finditer(text):
        raw_value = match.group(1)
        value = raw_value.strip()
        if value:
            left_trim = len(raw_value) - len(raw_value.lstrip())
            labelled_spans.append(
                AnswerSpan(
                    value,
                    match.start(1) + left_trim,
                    match.start(1) + len(raw_value.rstrip()),
                    "answer_label",
                )
            )
    return labelled_spans[-1] if labelled_spans else None
