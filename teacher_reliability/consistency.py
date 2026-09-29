"""Math-equivalence-aware self-consistency summaries."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from .grading import answer_parseable, answers_equivalent, grade_prediction


@dataclass(frozen=True, slots=True)
class ConsistencySummary:
    sample_count: int
    sample_answer_parseable_count: int
    greedy_agreement_count: int
    greedy_agreement_share: float
    majority_answer: str | None
    majority_vote_count: int
    majority_vote_share: float
    majority_correct: bool | None
    majority_grade_status: str


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

    greedy_count = sum(
        answer is not None
        and greedy_answer is not None
        and answers_equivalent(greedy_answer, answer)
        for answer in sample_answers
    )

    clusters: list[dict[str, object]] = []
    for answer in sample_answers:
        matching_cluster = None
        for cluster in clusters:
            representative = cluster["answer"]
            if answer is None and representative is None:
                matching_cluster = cluster
                break
            if (
                answer is not None
                and isinstance(representative, str)
                and answers_equivalent(representative, answer)
            ):
                matching_cluster = cluster
                break
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
    majority_grade = grade_prediction(majority_text, gold_answer)

    count = len(sample_answers)
    return ConsistencySummary(
        sample_count=count,
        sample_answer_parseable_count=sum(answer_parseable(answer) for answer in sample_answers),
        greedy_agreement_count=greedy_count,
        greedy_agreement_share=greedy_count / count,
        majority_answer=majority_answer if isinstance(majority_answer, str) else None,
        majority_vote_count=majority_count,
        majority_vote_share=majority_count / count,
        majority_correct=majority_grade.correct,
        majority_grade_status=majority_grade.status,
    )
