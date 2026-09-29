"""Aggregate per-token generation scores into auditable confidence signals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence

from .answers import AnswerSpan


@dataclass(frozen=True, slots=True)
class ConfidenceSummary:
    content_token_count: int
    mean_entropy_nats: float | None
    entropy_confidence: float | None
    sequence_logprob_sum: float | None
    sequence_geometric_probability: float | None
    answer_token_count: int
    answer_token_geometric_probability: float | None
    answer_alignment_status: str


def aggregate_confidence(
    token_entropies: Sequence[float],
    token_logprobs: Sequence[float],
    token_char_spans: Sequence[tuple[int, int] | None] | None,
    content_token_mask: Sequence[bool],
    answer_span: AnswerSpan | None,
    vocab_size: int,
) -> ConfidenceSummary:
    """Summarize visible completion tokens; special tokens are masked out.

    Answer-token confidence is the geometric mean probability over tokens
    whose decoded character offsets overlap the extracted final-answer span.
    Failed tokenizer round-trip alignment is represented as missing.
    """
    size = len(token_entropies)
    if len(token_logprobs) != size or len(content_token_mask) != size:
        raise ValueError("entropy, log-probability, and mask vectors need the same token count")
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size < 2:
        raise ValueError("vocab_size must be an integer greater than one")

    entropies = [float(value) for value in token_entropies]
    logprobs = [float(value) for value in token_logprobs]
    if any(not math.isfinite(value) or value < 0 for value in entropies):
        raise ValueError("token entropies must be finite and non-negative")
    if any(not math.isfinite(value) or value > 1e-7 for value in logprobs):
        raise ValueError("token log-probabilities must be finite and at most zero")
    if any(not isinstance(value, bool) for value in content_token_mask):
        raise ValueError("content_token_mask must contain booleans")

    content_indices = [i for i, is_content in enumerate(content_token_mask) if is_content]
    if content_indices:
        mean_entropy = sum(entropies[i] for i in content_indices) / len(content_indices)
        entropy_confidence = min(1.0, max(0.0, 1.0 - mean_entropy / math.log(vocab_size)))
        sequence_logprob = sum(logprobs[i] for i in content_indices)
        sequence_probability = math.exp(sequence_logprob / len(content_indices))
    else:
        mean_entropy = None
        entropy_confidence = None
        sequence_logprob = None
        sequence_probability = None

    answer_indices: list[int] = []
    if answer_span is None:
        alignment_status = "no_answer_span"
    elif token_char_spans is None or len(token_char_spans) != size:
        alignment_status = "token_alignment_unavailable"
    else:
        for index in content_indices:
            offsets = token_char_spans[index]
            if offsets is None:
                continue
            start, end = offsets
            if (
                isinstance(start, int)
                and isinstance(end, int)
                and start < answer_span.end
                and end > answer_span.start
                and end > start
            ):
                answer_indices.append(index)
        alignment_status = "aligned" if answer_indices else "no_answer_tokens"

    if answer_indices:
        answer_probability = math.exp(
            sum(logprobs[i] for i in answer_indices) / len(answer_indices)
        )
    else:
        answer_probability = None

    return ConfidenceSummary(
        content_token_count=len(content_indices),
        mean_entropy_nats=mean_entropy,
        entropy_confidence=entropy_confidence,
        sequence_logprob_sum=sequence_logprob,
        sequence_geometric_probability=sequence_probability,
        answer_token_count=len(answer_indices),
        answer_token_geometric_probability=answer_probability,
        answer_alignment_status=alignment_status,
    )
