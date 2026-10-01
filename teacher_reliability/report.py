"""Build calibration metrics, reliability plots, and a measured gate summary."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Sequence

from .hub_data import DATASET_REPOSITORIES
from .metrics import binary_roc_auc, calibration_bins, expected_calibration_error
from .run import (
    FULL_LOCAL_A100_ARCHIVE_SHA256,
    FULL_LOCAL_DATASET_REVISIONS,
    FULL_LOCAL_STUDY_SEED,
    FULL_STUDY_ORIGIN_DATASET_COUNTS,
    FULL_STUDY_SELECTED_DATASET_COUNTS,
    FULL_STUDY_SELECTED_ROW_COUNT,
    LOCAL_MODEL_REVISION,
    _canonical_digest,
)
from .state import load_completed_predictions
from .teacher import derive_sample_seed


GREEDY_SIGNALS = {
    "entropy_confidence": ("greedy", "entropy_confidence"),
    "answer_token_probability": ("greedy", "answer_token_geometric_probability"),
    "sequence_geometric_probability": ("greedy", "sequence_geometric_probability"),
    "self_consistency_agreement": ("self_consistency", "greedy_agreement_share"),
}
MAJORITY_SIGNAL = ("self_consistency", "majority_vote_share")
PRIMARY_SIGNALS = ("entropy_confidence", "self_consistency_agreement")
MIN_SOURCE_SCORABLE_FRACTION_FOR_RECOMMENDATION = 0.95
FULL_GREEDY_REQUIRED_FIELDS = {
    "text", "token_ids", "token_entropies_nats", "token_logprobs",
    "token_char_spans", "content_token_mask", "token_alignment_status",
    "eos_generated", "was_truncated", "termination_reason", "predicted_answer",
    "answer_method", "prediction_parseable", "gold_parseable", "correct",
    "grading_status", "content_token_count", "mean_entropy_nats",
    "entropy_confidence", "sequence_logprob_sum", "sequence_geometric_probability",
    "answer_token_count", "answer_token_geometric_probability", "answer_alignment_status",
}
FULL_SAMPLE_REQUIRED_FIELDS = {
    "index", "seed", "text", "token_ids", "answer", "answer_method",
    "eos_generated", "was_truncated", "termination_reason",
}
CRITICAL_UNKNOWN_COUNT_FIELDS = (
    "sample_answer_parse_unknown_count",
    "greedy_agreement_unknown_count",
    "majority_comparison_unknown_count",
)


def _nested(record: dict[str, Any], first: str, second: str):
    value = record.get(first)
    return value.get(second) if isinstance(value, dict) else None


def _valid_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    return score if math.isfinite(score) and 0.0 <= score <= 1.0 else None


def _full_row_issues(record: dict[str, Any], *, seed: Any) -> list[str]:
    """Validate the row structure needed to treat a full-profile checkpoint as complete."""
    example_id = record.get("example_id")
    issues: list[str] = []
    greedy = record.get("greedy")
    if not isinstance(greedy, dict):
        return ["greedy_record_missing"]
    missing_greedy = FULL_GREEDY_REQUIRED_FIELDS - greedy.keys()
    if missing_greedy:
        issues.append("greedy_fields_missing")

    token_ids = greedy.get("token_ids")
    if not isinstance(token_ids, list) or len(token_ids) > 1024 or any(
        not isinstance(token, int) or isinstance(token, bool) or not 0 <= token < 152064
        for token in token_ids
    ):
        issues.append("greedy_token_ids_invalid")
        token_count = None
    else:
        token_count = len(token_ids)
    vectors = [
        greedy.get("token_entropies_nats"),
        greedy.get("token_logprobs"),
        greedy.get("content_token_mask"),
    ]
    if token_count is None or any(
        not isinstance(vector, list) or len(vector) != token_count for vector in vectors
    ):
        issues.append("greedy_score_alignment_invalid")
    else:
        entropies, logprobs, content_mask = vectors
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or value < 0
            for value in entropies
        ) or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or value > 1e-7
            for value in logprobs
        ) or any(not isinstance(value, bool) for value in content_mask):
            issues.append("greedy_score_values_invalid")
        if greedy.get("content_token_count") != sum(content_mask) if all(
            isinstance(value, bool) for value in content_mask
        ) else False:
            issues.append("greedy_content_count_mismatch")

    spans = greedy.get("token_char_spans")
    if spans is not None and (
        token_count is None or not isinstance(spans, list) or len(spans) != token_count
        or any(
            span is not None and (
                not isinstance(span, list) or len(span) != 2
                or any(not isinstance(offset, int) or isinstance(offset, bool) for offset in span)
                or span[0] < 0 or span[1] < span[0]
            )
            for span in spans
        )
    ):
        issues.append("greedy_character_spans_invalid")
    if not isinstance(greedy.get("text"), str):
        issues.append("greedy_text_invalid")
    for field in ("eos_generated", "was_truncated"):
        if not isinstance(greedy.get(field), bool):
            issues.append("greedy_termination_flags_invalid")
            break
    for field in ("token_alignment_status", "termination_reason", "grading_status", "answer_alignment_status"):
        if not isinstance(greedy.get(field), str):
            issues.append("greedy_status_fields_invalid")
            break
    for field in ("predicted_answer", "answer_method"):
        if greedy.get(field) is not None and not isinstance(greedy.get(field), str):
            issues.append("greedy_answer_fields_invalid")
            break
    for field in ("prediction_parseable", "gold_parseable", "correct"):
        if greedy.get(field) is not None and not isinstance(greedy.get(field), bool):
            issues.append("greedy_grade_fields_invalid")
            break
    for field in ("content_token_count", "answer_token_count"):
        value = greedy.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            issues.append("greedy_token_counts_invalid")
            break
    for field in (
        "mean_entropy_nats", "entropy_confidence", "sequence_logprob_sum",
        "sequence_geometric_probability", "answer_token_geometric_probability",
    ):
        value = greedy.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or (field != "sequence_logprob_sum" and value < 0)
            or (field in {"entropy_confidence", "sequence_geometric_probability", "answer_token_geometric_probability"} and value > 1)
        ):
            issues.append("greedy_confidence_invalid")
            break

    samples = record.get("samples")
    if record.get("requested_sample_count") != 8 or not isinstance(samples, list) or len(samples) != 8:
        issues.append("sample_count_invalid")
        samples = samples if isinstance(samples, list) else []
    seen_indices: set[int] = set()
    for sample in samples:
        if not isinstance(sample, dict) or not FULL_SAMPLE_REQUIRED_FIELDS.issubset(sample):
            issues.append("sample_fields_missing")
            continue
        index = sample.get("index")
        if (
            not isinstance(index, int) or isinstance(index, bool)
            or not 0 <= index < 8 or index in seen_indices
        ):
            issues.append("sample_indices_invalid")
        else:
            seen_indices.add(index)
            if (
                not isinstance(seed, int) or isinstance(seed, bool)
                or sample.get("seed") != derive_sample_seed(seed, str(example_id), index)
            ):
                issues.append("sample_seed_invalid")
        sample_tokens = sample.get("token_ids")
        if not isinstance(sample.get("text"), str) or not isinstance(sample_tokens, list) or len(sample_tokens) > 1024 or any(
            not isinstance(token, int) or isinstance(token, bool) or not 0 <= token < 152064
            for token in sample_tokens
        ):
            issues.append("sample_generation_invalid")
        if any(not isinstance(sample.get(field), bool) for field in ("eos_generated", "was_truncated")) or not isinstance(sample.get("termination_reason"), str):
            issues.append("sample_termination_invalid")
        if sample.get("answer") is not None and not isinstance(sample.get("answer"), str):
            issues.append("sample_answer_invalid")
        if sample.get("answer_method") is not None and not isinstance(sample.get("answer_method"), str):
            issues.append("sample_answer_invalid")
    if seen_indices != set(range(8)):
        issues.append("sample_indices_incomplete")
    consistency = record.get("self_consistency")
    if not isinstance(consistency, dict) or consistency.get("sample_count") != 8:
        issues.append("self_consistency_sample_count_invalid")
    else:
        count_fields = ("sample_answer_parseable_count", "greedy_agreement_count")
        unknown_fields = (
            "sample_answer_parse_unknown_count",
            "greedy_agreement_unknown_count",
            "majority_comparison_unknown_count",
        )
        invalid_counts = any(
            not isinstance(consistency.get(field), int)
            or isinstance(consistency.get(field), bool)
            or consistency[field] < 0
            for field in count_fields
        ) or any(
            consistency.get(field) is not None
            and (
                not isinstance(consistency.get(field), int)
                or isinstance(consistency.get(field), bool)
                or consistency[field] < 0
            )
            for field in unknown_fields
        )
        if not invalid_counts:
            sample_parseable = consistency["sample_answer_parseable_count"]
            sample_unknown = consistency.get("sample_answer_parse_unknown_count")
            greedy_agreement = consistency["greedy_agreement_count"]
            greedy_unknown = consistency.get("greedy_agreement_unknown_count")
            majority_unknown = consistency.get("majority_comparison_unknown_count")
            majority_count = consistency.get("majority_vote_count")
            invalid_counts = (
                sample_parseable > 8
                or greedy_agreement > 8
                or (sample_unknown is not None and sample_unknown > 8)
                or (greedy_unknown is not None and greedy_unknown > 8)
                or (majority_unknown is not None and majority_unknown > 28)
                or (sample_unknown is not None and sample_parseable + sample_unknown > 8)
                or (greedy_unknown is not None and greedy_agreement + greedy_unknown > 8)
            )
            if majority_unknown is not None and majority_unknown > 0:
                invalid_counts = invalid_counts or any(
                    consistency.get(field) is not None
                    for field in (
                        "majority_answer", "majority_vote_count", "majority_vote_share",
                        "majority_correct",
                    )
                ) or consistency.get("majority_grading_status") != "majority_comparison_incomplete"
            else:
                invalid_counts = invalid_counts or (
                    not isinstance(majority_count, int)
                    or isinstance(majority_count, bool)
                    or not 1 <= majority_count <= 8
                    or consistency.get("majority_vote_share") is None
                    or consistency.get("majority_grading_status") == "majority_comparison_incomplete"
                )
        if invalid_counts:
            issues.append("self_consistency_counts_invalid")
        elif any(
            consistency.get(field) is not None and _valid_score(consistency.get(field)) is None
            for field in ("greedy_agreement_share", "majority_vote_share")
        ):
            issues.append("self_consistency_scores_invalid")
    return list(dict.fromkeys(issues))


def _sum_nullable_counts(records: list[dict[str, Any]], field: str) -> int | None:
    values = [_nested(row, "self_consistency", field) for row in records]
    if any(
        value is None
        or not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        for value in values
    ):
        return None
    return sum(values)


def _validate_unknown_count_availability(records: list[dict[str, Any]]) -> dict[str, Any]:
    unavailable_rows: list[str | None] = []
    unavailable_by_field = {field: 0 for field in CRITICAL_UNKNOWN_COUNT_FIELDS}
    unavailable_value_count = 0
    for row in records:
        consistency = row.get("self_consistency")
        consistency = consistency if isinstance(consistency, dict) else {}
        row_unavailable = False
        for field in CRITICAL_UNKNOWN_COUNT_FIELDS:
            value = consistency.get(field)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                unavailable_by_field[field] += 1
                unavailable_value_count += 1
                row_unavailable = True
        if row_unavailable:
            unavailable_rows.append(row.get("example_id"))
    return {
        "valid": unavailable_value_count == 0,
        "unavailable_row_count": len(unavailable_rows),
        "unavailable_value_count": unavailable_value_count,
        "unavailable_by_field": unavailable_by_field,
        "unavailable_example_ids": unavailable_rows[:10],
    }


def _valid_commit_revision(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _validate_full_run_revisions(
    manifest: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Ensure a full-study manifest and every source row use the same pinned revisions."""
    issues: set[str] = set()
    identity = manifest.get("identity")
    identity = identity if isinstance(identity, dict) else {}
    identity_model = identity.get("model")
    identity_model = identity_model if isinstance(identity_model, dict) else {}
    identity_model_revision = identity_model.get("revision")
    model_revision = manifest.get("model_revision")
    tokenizer_revision = manifest.get("tokenizer_revision")
    if (
        not _valid_commit_revision(model_revision)
        or model_revision != identity_model_revision
    ):
        issues.add("model_revision_mismatch")
    if model_revision != LOCAL_MODEL_REVISION:
        issues.add("model_revision_unpinned")
    if (
        not _valid_commit_revision(tokenizer_revision)
        or tokenizer_revision != model_revision
        or tokenizer_revision != identity_model_revision
    ):
        issues.add("tokenizer_revision_mismatch")

    dataset_revisions = manifest.get("dataset_revisions")
    identity_dataset_revisions = identity.get("datasets")
    required_repositories = set(DATASET_REPOSITORIES.values())
    if (
        not isinstance(dataset_revisions, dict)
        or set(dataset_revisions) != required_repositories
        or any(not _valid_commit_revision(value) for value in dataset_revisions.values())
        or dataset_revisions != identity_dataset_revisions
    ):
        issues.add("dataset_revisions_mismatch")
        dataset_revisions = dataset_revisions if isinstance(dataset_revisions, dict) else {}
    if dataset_revisions != FULL_LOCAL_DATASET_REVISIONS:
        issues.add("dataset_revisions_unpinned")
    if (
        manifest.get("seed") != FULL_LOCAL_STUDY_SEED
        or identity.get("base_seed") != FULL_LOCAL_STUDY_SEED
    ):
        issues.add("seed_unpinned")

    source_revision_mismatch_ids = []
    for row in records:
        source = row.get("source")
        source = source if isinstance(source, dict) else {}
        dataset = source.get("dataset")
        repository = DATASET_REPOSITORIES.get(dataset)
        expected_revision = dataset_revisions.get(repository) if repository else None
        if (
            repository is None
            or not _valid_commit_revision(expected_revision)
            or source.get("revision") != expected_revision
        ):
            source_revision_mismatch_ids.append(row.get("example_id"))
    if source_revision_mismatch_ids:
        issues.add("source_revision_mismatch")

    return {
        "valid": not issues,
        "issues": sorted(issues),
        "source_revision_mismatch_count": len(source_revision_mismatch_ids),
        "source_revision_mismatch_example_ids": source_revision_mismatch_ids[:10],
    }


def _outcome(value: Any) -> int | None:
    if value is True or value == 1:
        return 1
    if value is False or value == 0:
        return 0
    return None


def _collect_metric(
    rows: list[dict[str, Any]],
    analysis: str,
    dataset: str,
    signal: str,
    path: tuple[str, str],
    outcome_path: tuple[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scorable = [
        row for row in rows if _outcome(_nested(row, *outcome_path)) is not None
    ]
    pairs = [
        (score, _outcome(_nested(row, *outcome_path)))
        for row in scorable
        if (score := _valid_score(_nested(row, *path))) is not None
    ]
    scores = [pair[0] for pair in pairs]
    outcomes = [int(pair[1]) for pair in pairs]
    n = len(pairs)
    ece = expected_calibration_error(scores, outcomes) if n else None
    auc = binary_roc_auc(outcomes, scores) if n else None
    bins = calibration_bins(scores, outcomes) if n else []
    row = {
        "analysis": analysis,
        "dataset": dataset,
        "signal": signal,
        "n": n,
        "scorable_n": len(scorable),
        "coverage": n / len(scorable) if scorable else 0.0,
        "accuracy": sum(outcomes) / len(outcomes) if outcomes else None,
        "ece": ece,
        "auroc": auc,
    }
    bin_rows = [
        {"analysis": analysis, "dataset": dataset, "signal": signal, **item}
        for item in bins
    ]
    return row, bin_rows


def _metrics_by_execution(
    records: list[dict[str, Any]], datasets: list[str], origins: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for origin in origins:
        origin_records = [row for row in records if row.get("execution_origin") == origin]
        for dataset in datasets:
            dataset_rows = [
                row for row in origin_records
                if row.get("source", {}).get("dataset") == dataset
                and row.get("source", {}).get("gold_status") == "parsed"
                and row.get("gold_answer") is not None
            ]
            for signal, path in GREEDY_SIGNALS.items():
                metric, _ = _collect_metric(
                    dataset_rows, "greedy", dataset, signal, path, ("greedy", "correct")
                )
                rows.append({"execution_origin": origin, **metric})
            eligible = [
                row for row in dataset_rows
                if all(_valid_score(_nested(row, *GREEDY_SIGNALS[name])) is not None
                       for name in PRIMARY_SIGNALS)
            ]
            for signal in PRIMARY_SIGNALS:
                metric, _ = _collect_metric(
                    eligible, "paired_primary", dataset, signal,
                    GREEDY_SIGNALS[signal], ("greedy", "correct"),
                )
                metric["scorable_n"] = sum(
                    _outcome(_nested(row, "greedy", "correct")) is not None
                    for row in dataset_rows
                )
                metric["coverage"] = (
                    metric["n"] / metric["scorable_n"] if metric["scorable_n"] else 0.0
                )
                rows.append({"execution_origin": origin, **metric})
            metric, _ = _collect_metric(
                dataset_rows, "majority", dataset, "majority_vote_share",
                MAJORITY_SIGNAL, ("self_consistency", "majority_correct"),
            )
            rows.append({"execution_origin": origin, **metric})
    return rows


def _recommendation_coverage_gate(
    dataset_summaries: dict[str, Any], datasets: list[str]
) -> dict[str, Any]:
    by_dataset = {
        dataset: {
            "source_scorable_fraction": dataset_summaries.get(dataset, {}).get(
                "source_scorable_fraction", 0.0
            ),
            "outcome_label_coverage": dataset_summaries.get(dataset, {}).get(
                "outcome_label_coverage", 0.0
            ),
            "signal_score_coverage": dataset_summaries.get(dataset, {}).get(
                "signal_score_coverage", {}
            ),
        }
        for dataset in datasets
    }
    valid = all(
        details["source_scorable_fraction"] >= MIN_SOURCE_SCORABLE_FRACTION_FOR_RECOMMENDATION
        and details["outcome_label_coverage"] == 1.0
        for details in by_dataset.values()
    )
    return {
        "minimum_source_scorable_fraction": MIN_SOURCE_SCORABLE_FRACTION_FOR_RECOMMENDATION,
        "required_outcome_label_coverage": 1.0,
        "valid": valid,
        "by_dataset": by_dataset,
    }


def _macro_and_recommendation(
    metric_rows: list[dict[str, Any]],
    datasets: list[str],
    run_complete: bool,
    profile: str,
    dataset_summaries: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    macro: dict[str, Any] = {}
    candidates: list[tuple[float, float, str]] = []
    coverage_gate = _recommendation_coverage_gate(dataset_summaries, datasets)
    coverage_ready = coverage_gate["valid"]
    coverage_by_dataset = coverage_gate["by_dataset"]
    for signal in GREEDY_SIGNALS:
        selected = {
            row["dataset"]: row
            for row in metric_rows
            if row["analysis"] == "greedy" and row["signal"] == signal
        }
        valid = len(selected) == len(datasets) and all(
            selected.get(dataset, {}).get("auroc") is not None
            and selected.get(dataset, {}).get("ece") is not None
            for dataset in datasets
        )
        if valid:
            macro_auc = sum(float(selected[dataset]["auroc"]) for dataset in datasets) / len(datasets)
            macro_ece = sum(float(selected[dataset]["ece"]) for dataset in datasets) / len(datasets)
            minimum_auc = min(float(selected[dataset]["auroc"]) for dataset in datasets)
            macro[signal] = {
                "macro_auroc": macro_auc,
                "macro_ece": macro_ece,
                "minimum_dataset_auroc": minimum_auc,
                "all_datasets_defined": True,
                "coverage_by_dataset": coverage_by_dataset,
                "coverage_ready": coverage_ready,
                "eligible_for_recommendation": all(
                    selected[dataset]["coverage"] == 1.0 for dataset in datasets
                ) and coverage_ready,
            }
            if macro[signal]["eligible_for_recommendation"]:
                candidates.append((macro_auc, -macro_ece, signal))
        else:
            macro[signal] = {
                "macro_auroc": None,
                "macro_ece": None,
                "minimum_dataset_auroc": None,
                "all_datasets_defined": False,
                "coverage_by_dataset": coverage_by_dataset,
                "coverage_ready": coverage_ready,
                "eligible_for_recommendation": False,
            }

    if not run_complete:
        return macro, {
            "signal": None,
            "status": "insufficient",
            "reason": "The run is incomplete; no gate recommendation is made.",
        }
    if profile != "full":
        if profile in {"a100_12h", "a100_24h"}:
            return macro, {
                "signal": None,
                "status": "exploratory",
                "reason": (
                    f"The {profile} time-bounded subset is not the full benchmark; "
                    "no full-study KD-gate recommendation is made."
                ),
            }
        return macro, {
            "signal": None,
            "status": "exploratory",
            "reason": f"The {profile} profile is for setup/runtime checks, not a full-study recommendation.",
        }
    if not coverage_ready:
        details = []
        for dataset in datasets:
            coverage = coverage_by_dataset[dataset]
            source_fraction = coverage["source_scorable_fraction"]
            label_coverage = coverage["outcome_label_coverage"]
            if source_fraction < MIN_SOURCE_SCORABLE_FRACTION_FOR_RECOMMENDATION:
                details.append(
                    f"{dataset} source coverage {source_fraction:.1%} "
                    f"(minimum {MIN_SOURCE_SCORABLE_FRACTION_FOR_RECOMMENDATION:.0%})"
                )
            if label_coverage < 1.0:
                details.append(f"{dataset} label coverage {label_coverage:.1%} (required 100%)")
        return macro, {
            "signal": None,
            "status": "insufficient",
            "reason": (
                "Recommendation withheld because the outcome label coverage gate failed: "
                + "; ".join(details)
                + "."
            ),
        }
    if not candidates:
        return macro, {
            "signal": None,
            "status": "insufficient",
            "reason": "No signal has complete scorable-row coverage and defined AUROC/ECE on every requested dataset.",
        }

    best_auc, _negative_ece, best_signal = max(candidates)
    minimum_auc = macro[best_signal]["minimum_dataset_auroc"]
    if minimum_auc < 0.5:
        return macro, {
            "signal": None,
            "status": "mixed",
            "reason": (
                f"{best_signal} leads by macro AUROC ({best_auc:.3f}) but is below chance "
                f"on at least one dataset (minimum AUROC {minimum_auc:.3f})."
            ),
        }
    return macro, {
        "signal": best_signal,
        "status": "candidate",
        "reason": (
            f"{best_signal} has the highest macro AUROC ({best_auc:.3f}), "
            "among signals covering every scorable row, with no dataset below chance; "
            "macro ECE breaks AUROC ties. "
            f"The recommendation also requires at least {MIN_SOURCE_SCORABLE_FRACTION_FOR_RECOMMENDATION:.0%} source-scorable input coverage "
            "and known outcomes for every source-scorable row in every dataset. "
            "This is a candidate for a later KD ablation, not proof of student improvement."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_reliability_plots(
    run_dir: Path,
    datasets: list[str],
    records: list[dict[str, Any]],
    execution_label: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for dataset in datasets:
        rows = [row for row in records if row.get("source", {}).get("dataset") == dataset]
        figure, axes = plt.subplots(1, 5, figsize=(21, 4.2), constrained_layout=True)
        plot_signals = list(GREEDY_SIGNALS.items()) + [("majority_vote_share", MAJORITY_SIGNAL)]
        for axis, (signal, path) in zip(axes, plot_signals):
            outcome_path = (
                ("greedy", "correct")
                if signal != "majority_vote_share"
                else ("self_consistency", "majority_correct")
            )
            pairs = [
                (_valid_score(_nested(row, *path)), _outcome(_nested(row, *outcome_path)))
                for row in rows
            ]
            pairs = [(score, outcome) for score, outcome in pairs if score is not None and outcome is not None]
            if not pairs:
                axis.text(0.5, 0.5, "No scorable predictions", ha="center", va="center")
                axis.set_title(signal.replace("_", " "))
                axis.set_xlim(0, 1)
                axis.set_ylim(0, 1)
                continue
            scores = [pair[0] for pair in pairs]
            outcomes = [int(pair[1]) for pair in pairs]
            bins = calibration_bins(scores, outcomes)
            points = [item for item in bins if item["count"]]
            axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
            axis.plot(
                [item["mean_confidence"] for item in points],
                [item["accuracy"] for item in points],
                marker="o",
                linewidth=1.5,
            )
            ece = expected_calibration_error(scores, outcomes)
            axis.set_title(f"{signal.replace('_', ' ')}\nECE={ece:.3f}, n={len(scores)}")
            axis.set_xlabel("Mean confidence")
            axis.set_ylabel("Accuracy")
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.grid(alpha=0.2)
        label = f" ({execution_label})" if execution_label else ""
        figure.suptitle(f"Reliability diagrams: {dataset}{label}")
        figure.savefig(run_dir / f"reliability_{dataset}.png", dpi=150)
        plt.close(figure)


def _write_markdown(
    path: Path,
    summary: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    execution_metric_rows: list[dict[str, Any]],
) -> None:
    recommendation = summary["recommendation"]
    lines = [
        "# Teacher reliability report",
        "",
        f"- Run: `{summary['run_id']}`",
        f"- Profile: `{summary['profile']}`",
        f"- Run status: `{summary['run_status']}`",
        f"- Completed rows: {summary['completed_count']} / {summary['selected_count']}",
        f"- Failed rows: {summary['failed_count']}",
        f"- Pending rows: {summary['pending_count']}",
        f"- Recommendation status: **{recommendation['status']}**",
        f"- Candidate signal: `{recommendation['signal'] or 'none'}`",
        f"- Reason: {recommendation['reason']}",
        "",
        "The primary greedy analysis predicts greedy-answer correctness. Majority-vote correctness is a separate analysis.",
        "The paired_primary rows compare entropy and self-consistency on their shared eligible cohort; greedy rows also show per-signal coverage.",
        "Math-Verify timeout and worker-failure checks are recorded as unknown, not as false.",
        "The candidate ranking is descriptive; no confidence intervals or statistical-significance claims are produced.",
        "Smoke, pilot, and A100 time-bounded runs are exploratory and cannot justify a full-study KD gate.",
        "",
        "## Grading-check uncertainty",
        "",
        "| Dataset | Sample parse unknowns | Greedy-comparison unknowns | Majority-comparison unknowns |",
        "|---|---:|---:|---:|",
    ]
    qualification = summary.get("qualification")
    if isinstance(qualification, dict) and qualification.get("status") == "unqualified-fast-start":
        reason = qualification.get("reason", "The full profile matrix was skipped.")
        insert_at = lines.index("## Grading-check uncertainty")
        lines[insert_at:insert_at] = [
            "## Qualification",
            "",
            "Profile qualification: **unqualified-fast-start**.",
            f"Initial profile: `{qualification.get('configuration', 'unknown')}`.",
            f"Reason: {reason}",
            "No measured throughput ETA is available at launch.",
            "",
        ]

    def show_unknown_count(value: Any) -> str:
        return "unavailable" if value is None else str(value)

    for dataset, details in summary.get("by_dataset", {}).items():
        lines.append(
        f"| {dataset} | {show_unknown_count(details.get('sample_answer_parse_unknown_count'))} | "
        f"{show_unknown_count(details.get('greedy_agreement_unknown_count'))} | "
        f"{show_unknown_count(details.get('majority_comparison_unknown_count'))} |"
        )
    if summary.get("malformed_rows"):
        lines.extend(
            [
                "",
                "## Malformed full-run rows",
                "",
                f"Malformed rows: {summary.get('malformed_row_count', 0)}. A malformed row prevents the report from marking the full run complete.",
                "",
                "| Example ID | Issues |",
                "|---|---|",
            ]
        )
        for malformed in summary["malformed_rows"]:
            lines.append(
                f"| {malformed['example_id']} | {', '.join(malformed['issues'])} |"
            )
    lines.extend(
        [
            "",
            "## Outcome label coverage",
            "",
            "Candidate recommendations require at least 95% source-scorable rows and a known binary outcome for every source-scorable row in every dataset.",
            "",
            "| Dataset | Selected rows | Source-scorable rows | Source coverage | Known outcomes | Outcome coverage | Unknown outcomes |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset, details in summary.get("by_dataset", {}).items():
        lines.append(
            f"| {dataset} | {details.get('selected_rows', 0)} | "
            f"{details.get('source_scorable_rows', 0)} | "
            f"{details.get('source_scorable_fraction', 0.0):.1%} | "
            f"{details.get('scorable_rows', 0)} | "
            f"{details.get('outcome_label_coverage', 0.0):.1%} | "
            f"{details.get('unknown_outcome_label_count', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Analysis | Dataset | Signal | n | Coverage | Accuracy | ECE | AUROC |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    validation = summary.get("origin_count_validation")
    if validation is not None:
        origin_lines = [
            "## Full-study origin count validation",
            "",
            f"Validation: **{'pass' if validation['valid'] else 'failed'}**",
            "",
            "| Execution | Dataset | Expected rows | Completed rows |",
            "|---|---|---:|---:|",
        ]
        for origin, expected_counts in validation["expected"].items():
            actual_counts = validation["actual"].get(origin, {})
            for dataset in ("gsm8k", "math", "gsm_plus"):
                origin_lines.append(
                    f"| {origin} | {dataset} | {expected_counts[dataset]} | {actual_counts.get(dataset, 0)} |"
                )
        origin_lines.extend(["", "The full-study recommendation remains withheld unless this matrix matches exactly.", ""])
        insert_at = lines.index("## Metrics")
        lines[insert_at:insert_at] = origin_lines
    provenance_validation = summary.get("origin_provenance_validation")
    if provenance_validation is not None:
        provenance_lines = [
            "## Full-study origin provenance validation",
            "",
            f"Validation: **{'pass' if provenance_validation['valid'] else 'failed'}**",
            "",
            "This checks internal archive provenance consistency; it is not hardware attestation.",
        ]
        if provenance_validation["errors"]:
            provenance_lines.extend(
                ["", "Issues: " + ", ".join(f"`{error}`" for error in provenance_validation["errors"])]
            )
        provenance_lines.append("")
        insert_at = lines.index("## Metrics")
        lines[insert_at:insert_at] = provenance_lines
    revision_validation = summary.get("revision_validation")
    if revision_validation is not None:
        revision_lines = [
            "## Full-study revision validation",
            "",
            f"Validation: **{'pass' if revision_validation['valid'] else 'failed'}**",
        ]
        if revision_validation["issues"]:
            revision_lines.extend(
                ["", "Issues: " + ", ".join(f"`{issue}`" for issue in revision_validation["issues"])]
            )
        if revision_validation["source_revision_mismatch_count"]:
            revision_lines.extend(
                [
                    "",
                    "Rows with source revisions that differ from the manifest: "
                    f"{revision_validation['source_revision_mismatch_count']}.",
                ]
            )
        revision_lines.append("")
        insert_at = lines.index("## Metrics")
        lines[insert_at:insert_at] = revision_lines
    unknown_count_validation = summary.get("unknown_count_validation")
    if unknown_count_validation is not None:
        unknown_lines = [
            "## Full-study unknown-count validation",
            "",
            f"Validation: **{'pass' if unknown_count_validation['valid'] else 'failed'}**",
            "",
            "Unavailable unknown-count values: "
            f"{unknown_count_validation['unavailable_value_count']} across "
            f"{unknown_count_validation['unavailable_row_count']} rows.",
        ]
        unavailable_fields = [
            f"`{field}` ({count} rows)"
            for field, count in unknown_count_validation["unavailable_by_field"].items()
            if count
        ]
        if unavailable_fields:
            unknown_lines.extend(["", "Unavailable fields: " + ", ".join(unavailable_fields) + "."])
        unknown_lines.append("")
        insert_at = lines.index("## Metrics")
        lines[insert_at:insert_at] = unknown_lines
    for row in metric_rows:
        def show(value):
            return "—" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)
        lines.append(
            "| {analysis} | {dataset} | {signal} | {n} | {coverage} | {accuracy} | {ece} | {auroc} |".format(
                analysis=row["analysis"],
                dataset=row["dataset"],
                signal=row["signal"],
                n=row["n"],
                coverage=show(row["coverage"]),
                accuracy=show(row["accuracy"]),
                ece=show(row["ece"]),
                auroc=show(row["auroc"]),
            )
        )
    if summary.get("by_execution"):
        index = lines.index("## Metrics")
        execution_lines = [
            "## Execution provenance",
            "",
            f"Execution label: **{summary['execution_label']}**",
            "",
            "| Execution | Hardware | Rows | GSM8K | MATH | GSM-Plus |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for origin, details in sorted(summary["by_execution"].items()):
            counts = details["by_dataset"]
            execution_lines.append(
                f"| {origin} | {details['hardware']} | {details['row_count']} | "
                f"{counts.get('gsm8k', 0)} | {counts.get('math', 0)} | {counts.get('gsm_plus', 0)} |"
            )
        execution_lines.extend(["", "Metrics by source execution are also in `metrics_by_execution.csv`.", ""])
        lines[index:index] = execution_lines
    lines.extend(
        [
            "",
            "## Metrics by execution",
            "",
            "| Execution | Analysis | Dataset | Signal | n | Coverage | Accuracy | ECE | AUROC |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in execution_metric_rows:
        def show_execution(value):
            return "—" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)
        lines.append(
            "| {execution_origin} | {analysis} | {dataset} | {signal} | {n} | {coverage} | {accuracy} | {ece} | {auroc} |".format(
                execution_origin=row["execution_origin"],
                analysis=row["analysis"],
                dataset=row["dataset"],
                signal=row["signal"],
                n=row["n"],
                coverage=show_execution(row["coverage"]),
                accuracy=show_execution(row["accuracy"]),
                ece=show_execution(row["ece"]),
                auroc=show_execution(row["auroc"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "Entropy confidence is a normalized concentration proxy, not a probability of correctness. This report does not measure whether a student improves under knowledge distillation.",
            "",
            "Raw per-example generations and score vectors are in `predictions.jsonl`; this run directory is ignored by Git.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_report(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    predictions_path = run_dir / "predictions.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError("run_manifest.json is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpointed_backends = {"vllm-hybrid", "transformers-local"}
    if not predictions_path.is_file() and manifest.get("backend") not in checkpointed_backends:
        raise FileNotFoundError("predictions.jsonl is required")
    records_by_id = load_completed_predictions(predictions_path, repair_torn_tail=False)
    records = list(records_by_id.values())
    if not records and manifest.get("backend") not in checkpointed_backends:
        raise ValueError("the run contains no complete prediction records")

    selected_ids = manifest.get("selected_example_ids", [])
    full_study_scope = (
        manifest.get("profile", {}).get("name") == "full"
        and (
            manifest.get("study_scope") != "synthetic"
            or (
                len(selected_ids) == FULL_STUDY_SELECTED_ROW_COUNT
                and manifest.get("selected_example_count") == FULL_STUDY_SELECTED_ROW_COUNT
            )
        )
    )
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("run manifest contains duplicate selected IDs")
    if set(records_by_id) - set(selected_ids):
        raise ValueError("prediction checkpoints contain IDs outside the selected examples")
    failed_count = manifest.get("failed_example_count", 0)
    if manifest.get("backend") in checkpointed_backends:
        staged_failures = set()
        for path in (run_dir / "stages").glob("*.json"):
            stage = json.loads(path.read_text(encoding="utf-8"))
            example_id = stage.get("example_id")
            if stage.get("run_signature") != manifest.get("run_signature") or example_id not in selected_ids:
                raise ValueError("hybrid stage identity differs from the run manifest")
            if stage.get("failure") and example_id not in records_by_id:
                staged_failures.add(example_id)
        if not isinstance(failed_count, int) or isinstance(failed_count, bool) or failed_count < 0:
            raise ValueError("manifest failed count is invalid")
        failed_count = max(failed_count, len(staged_failures))
        pending_count = len(selected_ids) - len(records) - failed_count
    else:
        pending_count = manifest.get("pending_example_count", len(selected_ids) - len(records) - failed_count)
    if (
        not isinstance(failed_count, int) or isinstance(failed_count, bool) or failed_count < 0
        or not isinstance(pending_count, int) or isinstance(pending_count, bool) or pending_count < 0
        or failed_count + pending_count + len(records) != len(selected_ids)
    ):
        raise ValueError("manifest completed, failed, and pending counts do not match selection")
    run_complete = (
        manifest.get("status") == "complete"
        and set(records_by_id) == set(selected_ids)
        and len(records_by_id) == manifest.get("selected_example_count")
        and failed_count == 0 and pending_count == 0
    )
    malformed_rows = []
    if manifest.get("profile", {}).get("name") == "full":
        malformed_rows = [
            {"example_id": row.get("example_id"), "issues": issues}
            for row in records
            if (issues := _full_row_issues(row, seed=manifest.get("seed")))
        ]
        run_complete = run_complete and not malformed_rows
    expected_datasets = sorted(
        {identifier.split(":", 1)[0] for identifier in selected_ids}
        or {row.get("source", {}).get("dataset") for row in records}
    )
    if manifest.get("profile", {}).get("name") == "full":
        expected_datasets = sorted(set(expected_datasets) | set(DATASET_REPOSITORIES))
    represented_datasets = {row.get("source", {}).get("dataset") for row in records}
    missing_datasets = sorted(set(expected_datasets) - represented_datasets)
    run_complete = run_complete and not missing_datasets
    origin_count_validation = None
    origin_provenance_validation = None
    revision_validation = None
    unknown_count_validation = None
    if full_study_scope:
        revision_validation = _validate_full_run_revisions(manifest, records)
        run_complete = run_complete and revision_validation["valid"]
        unknown_count_validation = _validate_unknown_count_availability(records)
        expected_origin_counts = FULL_STUDY_ORIGIN_DATASET_COUNTS
        actual_origins = set(expected_origin_counts)
        actual_origins.update(
            row.get("execution_origin")
            for row in records
            if isinstance(row.get("execution_origin"), str) and row.get("execution_origin")
        )
        actual_origin_counts = {
            origin: {
                dataset: sum(
                    row.get("execution_origin") == origin
                    and row.get("source", {}).get("dataset") == dataset
                    for row in records
                )
                for dataset in ("gsm8k", "math", "gsm_plus")
            }
            for origin in sorted(actual_origins)
        }
        selected_dataset_counts = {
            dataset: sum(
                isinstance(identifier, str) and identifier.split(":", 1)[0] == dataset
                for identifier in selected_ids
            )
            for dataset in ("gsm8k", "math", "gsm_plus")
        }
        execution_hardware = manifest.get("execution_origins", {})
        a100_metadata_count = (
            execution_hardware.get("a100-import", {}).get("imported_count")
            if isinstance(execution_hardware, dict)
            and isinstance(execution_hardware.get("a100-import"), dict)
            else None
        )
        selected_counts_valid = (
            len(selected_ids) == FULL_STUDY_SELECTED_ROW_COUNT
            and manifest.get("selected_example_count") == FULL_STUDY_SELECTED_ROW_COUNT
            and selected_dataset_counts == FULL_STUDY_SELECTED_DATASET_COUNTS
        )
        origin_counts_valid = actual_origin_counts == expected_origin_counts
        imported_metadata_valid = a100_metadata_count == 110
        origin_count_validation = {
            "expected": expected_origin_counts,
            "actual": actual_origin_counts,
            "expected_selected_dataset_counts": FULL_STUDY_SELECTED_DATASET_COUNTS,
            "actual_selected_dataset_counts": selected_dataset_counts,
            "selected_counts_valid": selected_counts_valid,
            "a100_manifest_imported_count": a100_metadata_count,
            "imported_metadata_valid": imported_metadata_valid,
            "origin_counts_valid": origin_counts_valid,
            "valid": selected_counts_valid and imported_metadata_valid and origin_counts_valid,
        }
        origin_provenance_validation = _validate_origin_provenance(
            manifest, records, run_dir=run_dir
        )
        run_complete = (
            run_complete
            and origin_count_validation["valid"]
            and origin_provenance_validation["valid"]
        )
    metric_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    dataset_summaries: dict[str, Any] = {}
    for dataset in expected_datasets:
        dataset_rows = [row for row in records if row.get("source", {}).get("dataset") == dataset]
        selected_dataset_count = sum(
            identifier.startswith(dataset + ":") for identifier in selected_ids
        )
        source_scorable_rows = [
            row for row in dataset_rows
            if row.get("source", {}).get("gold_status") == "parsed"
            and row.get("gold_answer") is not None
        ]
        greedy_correct = [
            value for row in source_scorable_rows
            if (value := _outcome(_nested(row, "greedy", "correct"))) is not None
        ]
        parsed_count = sum(bool(_nested(row, "greedy", "prediction_parseable")) for row in dataset_rows)
        sample_count = sum(
            int(_nested(row, "self_consistency", "sample_count") or 0)
            for row in dataset_rows
        )
        sample_parseable_count = sum(
            int(_nested(row, "self_consistency", "sample_answer_parseable_count") or 0)
            for row in dataset_rows
        )
        sample_parse_unknown_count = _sum_nullable_counts(
            dataset_rows, "sample_answer_parse_unknown_count"
        )
        greedy_agreement_unknown_count = _sum_nullable_counts(
            dataset_rows, "greedy_agreement_unknown_count"
        )
        majority_comparison_unknown_count = _sum_nullable_counts(
            dataset_rows, "majority_comparison_unknown_count"
        )
        dataset_summaries[dataset] = {
            "total_rows": len(dataset_rows),
            "selected_rows": selected_dataset_count,
            "scorable_rows": len(greedy_correct),
            "unscorable_rows": len(dataset_rows) - len(greedy_correct),
            "source_scorable_rows": len(source_scorable_rows),
            "source_unscorable_rows": len(dataset_rows) - len(source_scorable_rows),
            "source_scorable_fraction": (
                len(source_scorable_rows) / selected_dataset_count
                if selected_dataset_count else 0.0
            ),
            "outcome_label_coverage": (
                len(greedy_correct) / len(source_scorable_rows)
                if source_scorable_rows else 0.0
            ),
            "unknown_outcome_label_count": len(source_scorable_rows) - len(greedy_correct),
            "greedy_accuracy": sum(greedy_correct) / len(greedy_correct) if greedy_correct else None,
            "prediction_parse_coverage": parsed_count / len(dataset_rows) if dataset_rows else 0.0,
            "sample_answer_parseable_count": sample_parseable_count,
            "sample_answer_parse_unknown_count": sample_parse_unknown_count,
            "sample_count": sample_count,
            "greedy_agreement_unknown_count": greedy_agreement_unknown_count,
            "majority_comparison_unknown_count": majority_comparison_unknown_count,
            "sample_answer_parse_coverage": (
                sample_parseable_count / sample_count if sample_count else 0.0
            ),
            "signal_score_coverage": {
                signal: {
                    "scored_rows": sum(
                        _valid_score(_nested(row, *path)) is not None
                        for row in source_scorable_rows
                    ),
                    "source_scorable_rows": len(source_scorable_rows),
                    "coverage": (
                        sum(
                            _valid_score(_nested(row, *path)) is not None
                            for row in source_scorable_rows
                        ) / len(source_scorable_rows)
                        if source_scorable_rows else 0.0
                    ),
                }
                for signal, path in GREEDY_SIGNALS.items()
            },
        }
        for signal, path in GREEDY_SIGNALS.items():
            row, bins = _collect_metric(
                source_scorable_rows,
                "greedy",
                dataset,
                signal,
                path,
                ("greedy", "correct"),
            )
            metric_rows.append(row)
            bin_rows.extend(bins)
        paired_rows = [
            row for row in source_scorable_rows
            if all(_valid_score(_nested(row, *GREEDY_SIGNALS[signal])) is not None
                   for signal in PRIMARY_SIGNALS)
        ]
        for signal in PRIMARY_SIGNALS:
            row, bins = _collect_metric(
                paired_rows, "paired_primary", dataset, signal,
                GREEDY_SIGNALS[signal], ("greedy", "correct"),
            )
            row["scorable_n"] = len(greedy_correct)
            row["coverage"] = row["n"] / len(greedy_correct) if greedy_correct else 0.0
            metric_rows.append(row)
            bin_rows.extend(bins)
        row, bins = _collect_metric(
            source_scorable_rows,
            "majority",
            dataset,
            "majority_vote_share",
            MAJORITY_SIGNAL,
            ("self_consistency", "majority_correct"),
        )
        metric_rows.append(row)
        bin_rows.extend(bins)

    macro, recommendation = _macro_and_recommendation(
        metric_rows,
        expected_datasets,
        run_complete,
        manifest.get("profile", {}).get("name", "unknown"),
        dataset_summaries,
    )
    if full_study_scope and unknown_count_validation and not unknown_count_validation["valid"]:
        for signal_summary in macro.values():
            signal_summary["eligible_for_recommendation"] = False
            signal_summary["eligibility_blockers"] = ["unknown_counts_unavailable"]
        recommendation = {
            "signal": None,
            "status": "insufficient",
            "reason": (
                "Full-study recommendation withheld because unknown/comparison counts are "
                f"unavailable for {unknown_count_validation['unavailable_row_count']} rows."
            ),
        }
    origins = sorted({
        origin for row in records
        if isinstance((origin := row.get("execution_origin")), str) and origin
    })
    if origin_count_validation is not None:
        origins = sorted(set(origins) | set(FULL_STUDY_ORIGIN_DATASET_COUNTS))
    execution_metric_rows = _metrics_by_execution(records, expected_datasets, origins)
    execution_hardware = manifest.get("execution_origins", {})
    by_execution = {}
    for origin in origins:
        origin_rows = [row for row in records if row.get("execution_origin") == origin]
        info = execution_hardware.get(origin, {}) if isinstance(execution_hardware, dict) else {}
        by_execution[origin] = {
            "row_count": len(origin_rows),
            "by_dataset": {
                dataset: sum(row.get("source", {}).get("dataset") == dataset for row in origin_rows)
                for dataset in expected_datasets
            },
            "hardware": info.get("gpu_name", origin) if isinstance(info, dict) else origin,
        }
    if set(origins) >= {"a100-import", "rtx4060-local"}:
        execution_label = "mixed A100 / RTX 4060"
    elif origins == ["a100-import"]:
        execution_label = "A100 import"
    elif origins == ["rtx4060-local"]:
        execution_label = "RTX 4060 local"
    else:
        execution_label = "mixed execution"
    summary = {
        "run_id": manifest.get("run_id", run_dir.name),
        "run_status": manifest.get("status", "unknown"),
        "profile": manifest.get("profile", {}).get("name", "unknown"),
        "qualification": manifest.get("qualification"),
        "run_complete": run_complete,
        "required_datasets": expected_datasets,
        "missing_datasets": missing_datasets,
        "selected_count": len(selected_ids),
        "completed_count": len(records),
        "failed_count": failed_count,
        "pending_count": pending_count,
        "malformed_row_count": len(malformed_rows),
        "malformed_rows": malformed_rows,
        "execution_label": execution_label if origins else None,
        "by_execution": by_execution,
        "origin_count_validation": origin_count_validation,
        "origin_provenance_validation": origin_provenance_validation,
        "revision_validation": revision_validation,
        "unknown_count_validation": unknown_count_validation,
        "recommendation_coverage_gate": _recommendation_coverage_gate(
            dataset_summaries, expected_datasets
        ),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "by_dataset": dataset_summaries,
        "macro": macro,
        "recommendation": recommendation,
    }
    _write_csv(
        run_dir / "metrics.csv",
        metric_rows,
        ["analysis", "dataset", "signal", "n", "scorable_n", "coverage", "accuracy", "ece", "auroc"],
    )
    _write_csv(
        run_dir / "calibration_bins.csv",
        bin_rows,
        ["analysis", "dataset", "signal", "bin_index", "lower", "upper", "count", "mean_confidence", "accuracy"],
    )
    if execution_metric_rows:
        _write_csv(
            run_dir / "metrics_by_execution.csv",
            execution_metric_rows,
            ["execution_origin", "analysis", "dataset", "signal", "n", "scorable_n", "coverage", "accuracy", "ece", "auroc"],
        )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_reliability_plots(run_dir, expected_datasets, records, summary["execution_label"])
    _write_markdown(run_dir / "report.md", summary, metric_rows, execution_metric_rows)
    return summary


def _validate_origin_provenance(
    manifest: dict[str, Any], records: list[dict[str, Any]], *, run_dir: Path
) -> dict[str, Any]:
    """Check imported-row identity against the pinned archive and its lineage."""
    errors: list[str] = []
    local_identity = manifest.get("identity")
    local_runtime = (
        local_identity.get("runtime")
        if isinstance(local_identity, dict) else None
    )
    local_hardware = manifest.get("hardware")
    execution_origins = manifest.get("execution_origins")
    local_origin = (
        execution_origins.get("rtx4060-local")
        if isinstance(execution_origins, dict) else None
    )
    if (
        not isinstance(local_identity, dict)
        or manifest.get("run_signature") != _canonical_digest(local_identity)
        or not isinstance(local_runtime, dict)
        or not isinstance(local_hardware, dict)
        or local_runtime != local_hardware
        or not isinstance(local_origin, dict)
        or local_origin.get("gpu_name") != local_runtime.get("gpu_name")
    ):
        errors.append("local_runtime_metadata_mismatch")

    imported_ids_value = manifest.get("imported_example_ids")
    imported_ids_valid = (
        isinstance(imported_ids_value, list)
        and all(isinstance(example_id, str) and example_id for example_id in imported_ids_value)
        and len(imported_ids_value) == len(set(imported_ids_value))
    )
    if not imported_ids_valid:
        errors.append("imported_example_ids_invalid")
    imported_ids = set(imported_ids_value) if imported_ids_valid else set()
    imported_rows = [row for row in records if row.get("execution_origin") == "a100-import"]
    imported_row_ids = {row.get("example_id") for row in imported_rows}
    if imported_row_ids != imported_ids:
        errors.append("imported_example_ids_mismatch")
    local_ids = {
        row.get("example_id") for row in records
        if row.get("execution_origin") == "rtx4060-local"
    }
    if local_ids & imported_ids:
        errors.append("local_row_id_overlap")

    import_provenance = manifest.get("import_provenance")
    a100_origin = (
        execution_origins.get("a100-import")
        if isinstance(execution_origins, dict)
        else None
    )
    if not isinstance(import_provenance, dict) or not isinstance(a100_origin, dict):
        errors.append("manifest_lineage_incomplete")
        import_provenance = import_provenance if isinstance(import_provenance, dict) else {}
        a100_origin = a100_origin if isinstance(a100_origin, dict) else {}

    expected_lineage = {
        "archive_sha256": import_provenance.get("archive_sha256"),
        "source_run_id": import_provenance.get("source_run_id"),
        "source_run_signature": import_provenance.get("source_run_signature"),
        "source_gpu": import_provenance.get("source_gpu"),
    }
    if expected_lineage["archive_sha256"] != FULL_LOCAL_A100_ARCHIVE_SHA256:
        errors.append("source_archive_unpinned")
    if any(not isinstance(value, str) or not value for value in expected_lineage.values()):
        errors.append("manifest_lineage_incomplete")
    origin_lineage = {
        "archive_sha256": a100_origin.get("archive_sha256"),
        "source_run_id": a100_origin.get("source_run_id"),
        "source_run_signature": a100_origin.get("source_run_signature"),
    }
    origin_gpus = [
        a100_origin[key] for key in ("gpu_name", "source_gpu") if key in a100_origin
    ]
    if (
        any(origin_lineage[key] != expected_lineage[key] for key in origin_lineage)
        or any(gpu != expected_lineage["source_gpu"] for gpu in origin_gpus)
    ):
        errors.append("manifest_lineage_conflict")

    imported_count = len(imported_ids)
    manifest_counts = [a100_origin.get("imported_count")]
    if "imported_example_count" in manifest:
        manifest_counts.append(manifest["imported_example_count"])
    if "imported_count" in import_provenance:
        manifest_counts.append(import_provenance["imported_count"])
    if any(
        not isinstance(count, int) or isinstance(count, bool) or count != imported_count
        for count in manifest_counts
    ):
        errors.append("manifest_imported_count_mismatch")

    lineage_matches = all(
        row.get("execution_provenance", {}).get(key) == value
        if isinstance(row.get("execution_provenance"), dict)
        else False
        for row in imported_rows
        for key, value in expected_lineage.items()
    )
    if not lineage_matches:
        errors.append("row_provenance_mismatch")

    archive_valid, archived_rows_valid = _validate_preserved_a100_rows(
        run_dir,
        import_provenance,
        imported_rows,
        expected_lineage,
    )
    if not archive_valid:
        errors.append("preserved_archive_invalid")
    if not archived_rows_valid:
        errors.append("archived_row_content_mismatch")

    local_receipts_valid: bool | None = None
    if archive_valid and archived_rows_valid and lineage_matches:
        local_receipts_valid = _validate_local_record_receipts(
            run_dir, manifest.get("run_signature"), records
        )
    if local_receipts_valid is False:
        errors.append("local_record_integrity_mismatch")

    return {
        "valid": not errors,
        "errors": errors,
        "expected_imported_count": imported_count,
        "actual_imported_count": len(imported_rows),
        "preserved_archive_valid": archive_valid,
        "archived_rows_valid": archived_rows_valid,
        "local_record_integrity_valid": local_receipts_valid,
        "scope": "internal_archive_provenance_consistency",
    }


def _matches_preserved_source_row(
    imported_row: Any, source_row: dict[str, Any]
) -> bool:
    if imported_row == source_row:
        return True
    if not isinstance(imported_row, dict):
        return False
    imported_consistency = imported_row.get("self_consistency")
    source_consistency = source_row.get("self_consistency")
    if not isinstance(imported_consistency, dict) or not isinstance(source_consistency, dict):
        return False
    normalized = dict(imported_row)
    normalized_consistency = dict(imported_consistency)
    for field in CRITICAL_UNKNOWN_COUNT_FIELDS:
        if field in normalized_consistency and normalized_consistency[field] is None and field not in source_consistency:
            normalized_consistency.pop(field)
    normalized["self_consistency"] = normalized_consistency
    return normalized == source_row


def _validate_preserved_a100_rows(
    run_dir: Path,
    import_provenance: dict[str, Any],
    imported_rows: list[dict[str, Any]],
    expected_lineage: dict[str, Any],
) -> tuple[bool, bool]:
    """Bind imported report rows to the preserved, checksum-verified source archive."""
    archive_path = run_dir / "provenance" / "source_run.zip"
    source_manifest_path = run_dir / "provenance" / "source_run_manifest.json"
    expected_digest = import_provenance.get("archive_sha256")
    if not isinstance(expected_digest, str) or not archive_path.is_file():
        return False, False
    digester = hashlib.sha256()
    try:
        with archive_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digester.update(block)
        if digester.hexdigest() != expected_digest:
            return False, False

        with zipfile.ZipFile(archive_path) as archive:
            members: dict[str, zipfile.ZipInfo] = {}
            roots: set[str | None] = set()
            for info in archive.infolist():
                name = info.filename
                path = PurePosixPath(name)
                if (
                    not name or "\\" in name or path.is_absolute()
                    or any(part in {".", ".."} for part in path.parts)
                ):
                    return False, False
                if info.is_dir():
                    if len(path.parts) != 1:
                        return False, False
                    roots.add(path.parts[0])
                    continue
                if len(path.parts) not in {1, 2}:
                    return False, False
                roots.add(path.parts[0] if len(path.parts) == 2 else None)
                basename = path.name
                if basename in members or basename not in {
                    "run_manifest.json", "predictions.jsonl", "runner.log", "runner.pid"
                }:
                    return False, False
                members[basename] = info
            if len(roots) != 1 or not {"run_manifest.json", "predictions.jsonl"}.issubset(members):
                return False, False

            source_manifest_bytes = archive.read(members["run_manifest.json"])
            if (
                not source_manifest_path.is_file()
                or source_manifest_path.read_bytes() != source_manifest_bytes
            ):
                return False, False
            source_manifest = json.loads(source_manifest_bytes)
            if not isinstance(source_manifest, dict) or any(
                source_manifest.get(source_field) != expected_lineage.get(lineage_field)
                for source_field, lineage_field in (
                    ("run_id", "source_run_id"),
                    ("run_signature", "source_run_signature"),
                )
            ):
                return False, False
            source_identity = source_manifest.get("identity")
            source_hardware = source_manifest.get("hardware")
            source_runtime = (
                source_identity.get("runtime")
                if isinstance(source_identity, dict) else None
            )
            if (
                not isinstance(source_identity, dict)
                or not isinstance(source_hardware, dict)
                or not isinstance(source_runtime, dict)
                or source_manifest.get("run_signature") != _canonical_digest(source_identity)
                or source_runtime != source_hardware
                or source_hardware.get("gpu_name") != expected_lineage.get("source_gpu")
            ):
                return False, False

            expected_by_id = {
                row.get("example_id"): {
                    key: value
                    for key, value in row.items()
                    if key not in {"execution_origin", "execution_provenance"}
                }
                for row in imported_rows
            }
            seen_ids: set[str] = set()
            with archive.open(members["predictions.jsonl"]) as stream:
                for line in stream:
                    if not line.endswith(b"\n"):
                        return True, False
                    try:
                        source_row = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        return True, False
                    if not isinstance(source_row, dict):
                        return True, False
                    example_id = source_row.get("example_id")
                    if (
                        not isinstance(example_id, str)
                        or example_id in seen_ids
                        or not _matches_preserved_source_row(
                            expected_by_id.get(example_id), source_row
                        )
                    ):
                        return True, False
                    seen_ids.add(example_id)
            return True, seen_ids == set(expected_by_id)
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError):
        return False, False


def _validate_local_record_receipts(
    run_dir: Path, run_signature: Any, records: list[dict[str, Any]]
) -> bool:
    local_rows = [row for row in records if row.get("execution_origin") == "rtx4060-local"]
    if not local_rows:
        return True
    if not isinstance(run_signature, str) or not run_signature:
        return not local_rows
    def receipt_matches(row: dict[str, Any]) -> bool:
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            return False
        key = hashlib.sha256(example_id.encode("utf-8")).hexdigest()
        receipt_path = run_dir / "stages" / "integrity" / f"{key}.json"
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return not (
            not isinstance(receipt, dict)
            or receipt.get("schema_version") != 1
            or receipt.get("run_signature") != run_signature
            or receipt.get("example_id") != example_id
            or receipt.get("record_sha256") != _canonical_digest(row)
        )

    worker_count = min(32, len(local_rows))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return all(executor.map(receipt_matches, local_rows))


def _console_safe(message: str, stream: Any) -> str:
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return message
    return message.encode(encoding, errors="backslashreplace").decode(encoding)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        summary = run_report(args.run_dir)
    except Exception as exc:
        print(_console_safe(f"Report failed: {exc}", sys.stderr), file=sys.stderr)
        return 1
    recommendation = summary["recommendation"]
    print(
        _console_safe(
            f"Report written to {args.run_dir.resolve()}; "
            f"recommendation: {recommendation['signal'] or recommendation['status']}",
            sys.stdout,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
