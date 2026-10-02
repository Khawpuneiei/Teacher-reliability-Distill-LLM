"""Immutable run profiles and order-independent benchmark subset selection."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .data import Example


@dataclass(frozen=True, slots=True)
class RunProfile:
    name: str
    gsm8k_limit: int | None
    math_per_subject: int | None
    gsm_plus_limit: int | None
    gsm_plus_one_per_seed: bool
    gsm_plus_split: str
    sample_count: int
    max_new_tokens: int
    target_hours: int | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("profile name must not be empty")
        for field_name in ("gsm8k_limit", "math_per_subject", "gsm_plus_limit"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 1
            ):
                raise ValueError(f"{field_name} must be a positive integer or None")
        if not self.gsm_plus_split.strip():
            raise ValueError("gsm_plus_split must not be empty")
        if not isinstance(self.sample_count, int) or isinstance(self.sample_count, bool) or self.sample_count < 1:
            raise ValueError("sample_count must be a positive integer")
        if not isinstance(self.max_new_tokens, int) or isinstance(self.max_new_tokens, bool) or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if self.target_hours is not None and (
            not isinstance(self.target_hours, int)
            or isinstance(self.target_hours, bool)
            or self.target_hours < 1
        ):
            raise ValueError("target_hours must be a positive integer or None")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "gsm8k_limit": self.gsm8k_limit,
            "math_per_subject": self.math_per_subject,
            "gsm_plus_limit": self.gsm_plus_limit,
            "gsm_plus_one_per_seed": self.gsm_plus_one_per_seed,
            "gsm_plus_split": self.gsm_plus_split,
            "sample_count": self.sample_count,
            "max_new_tokens": self.max_new_tokens,
            "target_hours": self.target_hours,
        }


PROFILES = {
    # Nine-row pipeline check (two samples) before the mini study.
    "smoke": RunProfile(
        "smoke", 1, 1, 1, True, "test", 2, 1024
    ),
    # Mini study: full per-question protocol (8 samples, 1,024 tokens) on
    # 111 rows, sized from measured RTX 4060 batch-one rates to fit 12 hours.
    "mini_12h": RunProfile(
        "mini_12h", 40, 3, 50, True, "test", 8, 1024, target_hours=12
    ),
}


def get_profile(name: str) -> RunProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown profile {name!r}; choose from {', '.join(PROFILES)}") from exc


def _rank(example: Example, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}\0{example.example_id}".encode("utf-8")).hexdigest()
    return digest, example.example_id


def _take_ranked(rows: list[Example], limit: int | None, seed: int) -> list[Example]:
    ranked = sorted(rows, key=lambda row: _rank(row, seed))
    return ranked if limit is None else ranked[:limit]


def select_examples(
    examples: Iterable[Example], profile: RunProfile, seed: int
) -> list[Example]:
    """Select deterministic, capped rows without depending on source order.

    MATH caps are applied independently within each subject/configuration.
    When ``gsm_plus_one_per_seed`` is set, one ranked GSM-Plus row is kept per
    seed question before the overall cap is applied.
    """
    rows = list(examples)
    identifiers = [row.example_id for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate example_id in input rows")
    unsupported = sorted({row.dataset for row in rows} - {"gsm8k", "math", "gsm_plus"})
    if unsupported:
        raise ValueError(f"unsupported datasets: {', '.join(unsupported)}")

    gsm8k_rows = [row for row in rows if row.dataset == "gsm8k"]
    selected = _take_ranked(gsm8k_rows, profile.gsm8k_limit, seed)

    math_by_subject: dict[str, list[Example]] = defaultdict(list)
    for row in rows:
        if row.dataset == "math":
            math_by_subject[row.source_config].append(row)
    for subject in sorted(math_by_subject):
        selected.extend(
            _take_ranked(math_by_subject[subject], profile.math_per_subject, seed)
        )

    plus_rows = [row for row in rows if row.dataset == "gsm_plus"]
    ranked_plus = _take_ranked(plus_rows, None, seed)
    if profile.gsm_plus_one_per_seed:
        unique_seeds: set[str] = set()
        deduplicated: list[Example] = []
        for row in ranked_plus:
            group = row.seed_question_id or f"missing-seed:{row.example_id}"
            if group not in unique_seeds:
                unique_seeds.add(group)
                deduplicated.append(row)
        ranked_plus = deduplicated
    selected.extend(
        ranked_plus if profile.gsm_plus_limit is None else ranked_plus[: profile.gsm_plus_limit]
    )

    return sorted(selected, key=lambda row: row.example_id)
