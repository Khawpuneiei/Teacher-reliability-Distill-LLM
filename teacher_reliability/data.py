"""Schema checks and source adapters for the three evaluation benchmarks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .answers import extract_final_answer_span


@dataclass(frozen=True, slots=True)
class Example:
    example_id: str
    dataset: str
    source_revision: str
    source_config: str
    split: str
    source_row_id: str
    question: str
    gold_answer: str | None
    gold_solution: str | None = None
    gold_status: str = "parsed"
    subject: str | None = None
    perturbation_type: str | None = None
    seed_question_id: str | None = None

    @property
    def scorable(self) -> bool:
        return self.gold_answer is not None and self.gold_status == "parsed"


def _required_text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"dataset row requires non-empty text field {key!r}")
    return value.strip()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _source_id(dataset: str, config: str, split: str, row_id: Any) -> str:
    return f"{dataset}:{config}:{split}:{row_id}"


def _seed_question_id(seed_question: str | None) -> str | None:
    if seed_question is None:
        return None
    normalized = " ".join(seed_question.casefold().split())
    if not normalized:
        return None
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"gsm8k-test-seed:{digest}"


def normalize_example(
    dataset: str,
    config: str,
    split: str,
    row_id: Any,
    row: dict[str, Any],
    source_revision: str,
) -> Example:
    """Normalize one source row without putting any reference into its prompt."""
    if dataset not in {"gsm8k", "math", "gsm_plus"}:
        raise ValueError(f"unsupported dataset {dataset!r}")
    if not all(isinstance(value, str) and value.strip() for value in (config, split, source_revision)):
        raise ValueError("config, split, and source_revision must be non-empty strings")
    if not isinstance(row, dict):
        raise ValueError("dataset row must be a mapping")

    if dataset == "gsm8k":
        question = _required_text(row, "question")
        solution = _optional_text(row.get("answer"))
        marker = solution.rfind("####") if solution is not None else -1
        if marker >= 0:
            answer = solution[marker + len("####") :].strip() or None
            gold_status = "parsed" if answer else "missing_final_answer"
        else:
            answer = None
            gold_status = "missing_final_marker"
        return Example(
            example_id=_source_id(dataset, config, split, row_id),
            dataset=dataset,
            source_revision=source_revision,
            source_config=config,
            split=split,
            source_row_id=str(row_id),
            question=question,
            gold_answer=answer,
            gold_solution=solution,
            gold_status=gold_status,
        )

    if dataset == "math":
        question = _required_text(row, "problem")
        solution = _optional_text(row.get("solution"))
        answer_span = extract_final_answer_span(solution or "")
        answer = answer_span.value if answer_span and answer_span.method == "boxed" else None
        gold_status = "parsed" if answer else "missing_boxed_answer"
        return Example(
            example_id=_source_id(dataset, config, split, row_id),
            dataset=dataset,
            source_revision=source_revision,
            source_config=config,
            split=split,
            source_row_id=str(row_id),
            question=question,
            gold_answer=answer,
            gold_solution=solution,
            gold_status=gold_status,
            subject=config,
        )

    question = _required_text(row, "question")
    solution = _optional_text(row.get("solution"))
    raw_answer = row.get("answer")
    answer_text = _optional_text(raw_answer)
    if answer_text is not None and answer_text.casefold() in {"none", "null", "not answerable"}:
        answer_text = None
    gold_status = "parsed" if answer_text is not None else "unanswerable"
    seed_question = _optional_text(row.get("seed_question"))
    return Example(
        example_id=_source_id(dataset, config, split, row_id),
        dataset=dataset,
        source_revision=source_revision,
        source_config=config,
        split=split,
        source_row_id=str(row_id),
        question=question,
        gold_answer=answer_text,
        gold_solution=solution,
        gold_status=gold_status,
        perturbation_type=_optional_text(row.get("perturbation_type")),
        seed_question_id=_seed_question_id(seed_question),
    )
