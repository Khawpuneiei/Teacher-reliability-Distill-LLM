"""Pinned Hugging Face dataset resolution and benchmark loading."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .data import Example, normalize_example
from .selection import RunProfile, select_examples


DATASET_REPOSITORIES = {
    "gsm8k": "openai/gsm8k",
    "math": "EleutherAI/hendrycks_math",
    "gsm_plus": "qintongli/GSM-Plus",
}
MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40,64}$")
RevisionResolver = Callable[[str], str]
DatasetLoader = Callable[..., Iterable[Mapping[str, Any]]]


def _resolve_dataset_revision(repository: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repository)
    return info.sha


def _load_huggingface_split(
    repository: str, config: str | None, split: str, revision: str
) -> Iterable[Mapping[str, Any]]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split, "revision": revision}
    if config is not None:
        kwargs["name"] = config
    return load_dataset(repository, **kwargs)


def load_examples(
    profile: RunProfile,
    seed: int,
    *,
    dataset_revisions: Mapping[str, str] | None = None,
    revision_resolver: RevisionResolver | None = None,
    dataset_loader: DatasetLoader | None = None,
) -> tuple[list[Example], dict[str, str]]:
    """Load all requested test sources at immutable Hub commit revisions.

    The optional callables isolate the network boundary for small contract
    tests; the production defaults use ``HfApi`` and ``datasets.load_dataset``.
    """
    resolve = revision_resolver or _resolve_dataset_revision
    load = dataset_loader or _load_huggingface_split
    revisions = dict(dataset_revisions) if dataset_revisions is not None else {
        repository: resolve(repository)
        for repository in DATASET_REPOSITORIES.values()
    }
    missing = set(DATASET_REPOSITORIES.values()) - set(revisions)
    if missing:
        raise ValueError(f"missing dataset commit revisions: {', '.join(sorted(missing))}")
    for repository, revision in revisions.items():
        if not isinstance(revision, str) or not _COMMIT_SHA.fullmatch(revision):
            raise ValueError(
                f"{repository} did not resolve to a commit SHA; observed {revision!r}"
            )

    requests = [
        ("gsm8k", "main", "test"),
        *[("math", subject, "test") for subject in MATH_CONFIGS],
        ("gsm_plus", None, profile.gsm_plus_split),
    ]
    examples: list[Example] = []
    for dataset, config, split in requests:
        repository = DATASET_REPOSITORIES[dataset]
        revision = revisions[repository]
        source_config = config or "default"
        rows = load(
            repository=repository,
            config=config,
            split=split,
            revision=revision,
        )
        source_start = len(examples)
        for row_index, row in enumerate(rows):
            examples.append(
                normalize_example(
                    dataset=dataset,
                    config=source_config,
                    split=split,
                    row_id=row_index,
                    row=dict(row),
                    source_revision=revision,
                )
            )
        if len(examples) == source_start:
            raise ValueError(
                f"empty requested source: {repository}/{source_config} "
                f"split={split} revision={revision}"
            )

    return select_examples(examples, profile, seed), revisions
