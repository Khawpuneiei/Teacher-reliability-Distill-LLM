"""Math-equivalence-aware self-consistency summaries."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from .grading import (
    check_answer_parseability,
    check_answers_equivalence,
    grade_prediction,
)


@dataclass(frozen=True, slots=True)
class ConsistencySummary:
    sample_count: int
    sample_answer_parseable_count: int
    sample_answer_parse_unknown_count: int
    greedy_agreement_count: int
    greedy_agreement_unknown_count: int
    greedy_agreement_share: float | None
    majority_answer: str | None
    majority_vote_count: int | None
    majority_vote_share: float | None
    majority_correct: bool | None
    majority_grade_status: str
    majority_comparison_unknown_count: int


def summarize_consistency(
    greedy_answer: str | None,
    sample_answers: Sequence[str | None],
    gold_answer: str | None,
) -> ConsistencySummary:
    """Count greedy agreement and a separate modal answer over k samples.

    Mathematically equivalent parsed expressions share a vote. Samples are
    processed in stable index order; the earliest cluster wins a count tie.
    Missing answers form one abstention cluster but never match a parsed answer.
    """
    if not sample_answers:
        raise ValueError("at least one sample answer is required")

    greedy_comparisons = [
        False
        if answer is None or greedy_answer is None
        else check_answers_equivalence(greedy_answer, answer)
        for answer in sample_answers
    ]
    greedy_count = sum(result is True for result in greedy_comparisons)
    greedy_unknown_count = sum(result is None for result in greedy_comparisons)

    clusters: list[dict[str, object]] = []
    majority_comparison_unknown_count = 0
    for answer in sample_answers:
        matching_cluster = None
        for cluster in clusters:
            representative = cluster["answer"]
            if answer is None and representative is None:
                matching_cluster = cluster
                break
            if answer is not None and isinstance(representative, str):
                equivalent = check_answers_equivalence(representative, answer)
                if equivalent is True:
                    matching_cluster = cluster
                    break
                if equivalent is None:
                    majority_comparison_unknown_count += 1
        if matching_cluster is None:
            clusters.append({"answer": answer, "count": 1})
        else:
            matching_cluster["count"] = int(matching_cluster["count"]) + 1

    majority = max(clusters, key=lambda cluster: int(cluster["count"]))
    majority_answer = majority["answer"]
    majority_count = int(majority["count"])
    majority_text = (
        "" if majority_answer is None else r"\boxed{" + str(majority_answer) + "}"
    )
    if majority_comparison_unknown_count:
        majority_correct = None
        majority_grade_status = "majority_comparison_incomplete"
    else:
        majority_grade = grade_prediction(majority_text, gold_answer)
        majority_correct = majority_grade.correct
        majority_grade_status = majority_grade.status

    parseability = [check_answer_parseability(answer) for answer in sample_answers]

    count = len(sample_answers)
    return ConsistencySummary(
        sample_count=count,
        sample_answer_parseable_count=sum(result is True for result in parseability),
        sample_answer_parse_unknown_count=sum(result is None for result in parseability),
        greedy_agreement_count=greedy_count,
        greedy_agreement_unknown_count=greedy_unknown_count,
        greedy_agreement_share=(
            None if greedy_unknown_count else greedy_count / count
        ),
        majority_answer=(
            majority_answer
            if not majority_comparison_unknown_count and isinstance(majority_answer, str)
            else None
        ),
        majority_vote_count=(
            None if majority_comparison_unknown_count else majority_count
        ),
        majority_vote_share=(
            None if majority_comparison_unknown_count else majority_count / count
        ),
        majority_correct=majority_correct,
        majority_grade_status=majority_grade_status,
        majority_comparison_unknown_count=majority_comparison_unknown_count,
    )
