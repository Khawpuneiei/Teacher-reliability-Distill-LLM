"""Build calibration metrics, reliability plots, and a measured gate summary."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .metrics import binary_roc_auc, calibration_bins, expected_calibration_error
from .state import load_completed_predictions


GREEDY_SIGNALS = {
    "entropy_confidence": ("greedy", "entropy_confidence"),
    "answer_token_probability": ("greedy", "answer_token_geometric_probability"),
    "sequence_geometric_probability": ("greedy", "sequence_geometric_probability"),
    "self_consistency_agreement": ("self_consistency", "greedy_agreement_share"),
}
MAJORITY_SIGNAL = ("self_consistency", "majority_vote_share")
PRIMARY_SIGNALS = ("entropy_confidence", "self_consistency_agreement")
MIN_SOURCE_SCORABLE_FRACTION_FOR_RECOMMENDATION = 0.95
def _nested(record: dict[str, Any], first: str, second: str):
    value = record.get(first)
    return value.get(second) if isinstance(value, dict) else None


def _valid_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    return score if math.isfinite(score) and 0.0 <= score <= 1.0 else None


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
    if profile == "mini_12h":
        return macro, {
            "signal": None,
            "status": "exploratory",
            "reason": (
                "The mini_12h subset (111 questions) is too small to justify a KD gate; "
                "metrics are descriptive only."
            ),
        }
    return macro, {
        "signal": None,
        "status": "exploratory",
        "reason": f"The {profile} profile is a setup/runtime check, not a gate recommendation.",
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
        "Smoke and mini_12h runs are exploratory and cannot justify a KD gate.",
        "",
        "## Grading-check uncertainty",
        "",
        "| Dataset | Sample parse unknowns | Greedy-comparison unknowns | Majority-comparison unknowns |",
        "|---|---:|---:|---:|",
    ]

    def show_unknown_count(value: Any) -> str:
        return "unavailable" if value is None else str(value)

    for dataset, details in summary.get("by_dataset", {}).items():
        lines.append(
        f"| {dataset} | {show_unknown_count(details.get('sample_answer_parse_unknown_count'))} | "
        f"{show_unknown_count(details.get('greedy_agreement_unknown_count'))} | "
        f"{show_unknown_count(details.get('majority_comparison_unknown_count'))} |"
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
    checkpointed_backends = {"transformers-local"}
    if not predictions_path.is_file() and manifest.get("backend") not in checkpointed_backends:
        raise FileNotFoundError("predictions.jsonl is required")
    records_by_id = load_completed_predictions(predictions_path, repair_torn_tail=False)
    records = list(records_by_id.values())
    if not records and manifest.get("backend") not in checkpointed_backends:
        raise ValueError("the run contains no complete prediction records")

    selected_ids = manifest.get("selected_example_ids", [])
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
                raise ValueError("stage identity differs from the run manifest")
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
    expected_datasets = sorted(
        {identifier.split(":", 1)[0] for identifier in selected_ids}
        or {row.get("source", {}).get("dataset") for row in records}
    )
    represented_datasets = {row.get("source", {}).get("dataset") for row in records}
    missing_datasets = sorted(set(expected_datasets) - represented_datasets)
    run_complete = run_complete and not missing_datasets
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
    origins = sorted({
        origin for row in records
        if isinstance((origin := row.get("execution_origin")), str) and origin
    })
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
    if origins == ["rtx4060-local"]:
        execution_label = "RTX 4060 local"
    else:
        execution_label = "mixed execution"
    summary = {
        "run_id": manifest.get("run_id", run_dir.name),
        "run_status": manifest.get("status", "unknown"),
        "profile": manifest.get("profile", {}).get("name", "unknown"),
        "run_complete": run_complete,
        "required_datasets": expected_datasets,
        "missing_datasets": missing_datasets,
        "selected_count": len(selected_ids),
        "completed_count": len(records),
        "failed_count": failed_count,
        "pending_count": pending_count,
        "execution_label": execution_label if origins else None,
        "by_execution": by_execution,
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
