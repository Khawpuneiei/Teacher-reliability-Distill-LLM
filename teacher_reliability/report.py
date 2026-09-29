"""Build calibration metrics, reliability plots, and a measured gate summary."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .hub_data import DATASET_REPOSITORIES
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


def _nested(record: dict[str, Any], first: str, second: str):
    value = record.get(first)
    return value.get(second) if isinstance(value, dict) else None


def _valid_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    return score if math.isfinite(score) and 0.0 <= score <= 1.0 else None


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


def _macro_and_recommendation(
    metric_rows: list[dict[str, Any]], datasets: list[str], run_complete: bool, profile: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    macro: dict[str, Any] = {}
    candidates: list[tuple[float, float, str]] = []
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
                "eligible_for_recommendation": all(
                    selected[dataset]["coverage"] == 1.0 for dataset in datasets
                ),
            }
            if macro[signal]["eligible_for_recommendation"]:
                candidates.append((macro_auc, -macro_ece, signal))
        else:
            macro[signal] = {
                "macro_auroc": None,
                "macro_ece": None,
                "minimum_dataset_auroc": None,
                "all_datasets_defined": False,
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
            "This is a candidate for a later KD ablation, not proof of student improvement."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_reliability_plots(run_dir: Path, datasets: list[str], records: list[dict[str, Any]]) -> None:
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
        figure.suptitle(f"Reliability diagrams: {dataset}")
        figure.savefig(run_dir / f"reliability_{dataset}.png", dpi=150)
        plt.close(figure)


def _write_markdown(path: Path, summary: dict[str, Any], metric_rows: list[dict[str, Any]]) -> None:
    recommendation = summary["recommendation"]
    lines = [
        "# Teacher reliability report",
        "",
        f"- Run: `{summary['run_id']}`",
        f"- Profile: `{summary['profile']}`",
        f"- Run status: `{summary['run_status']}`",
        f"- Completed rows: {summary['completed_count']} / {summary['selected_count']}",
        f"- Recommendation status: **{recommendation['status']}**",
        f"- Candidate signal: `{recommendation['signal'] or 'none'}`",
        f"- Reason: {recommendation['reason']}",
        "",
        "The primary greedy analysis predicts greedy-answer correctness. Majority-vote correctness is a separate analysis.",
        "The paired_primary rows compare entropy and self-consistency on their shared eligible cohort; greedy rows also show per-signal coverage.",
        "Smoke, pilot, and A100 time-bounded runs are exploratory and cannot justify a full-study KD gate.",
        "",
        "## Metrics",
        "",
        "| Analysis | Dataset | Signal | n | Coverage | Accuracy | ECE | AUROC |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
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
    if not manifest_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError("run_manifest.json and predictions.jsonl are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records_by_id = load_completed_predictions(predictions_path, repair_torn_tail=False)
    records = list(records_by_id.values())
    if not records:
        raise ValueError("the run contains no complete prediction records")

    selected_ids = manifest.get("selected_example_ids", [])
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("run manifest contains duplicate selected IDs")
    run_complete = (
        manifest.get("status") == "complete"
        and set(records_by_id) == set(selected_ids)
        and len(records_by_id) == manifest.get("selected_example_count")
    )
    expected_datasets = sorted(
        {identifier.split(":", 1)[0] for identifier in selected_ids}
        or {row.get("source", {}).get("dataset") for row in records}
    )
    if manifest.get("profile", {}).get("name") == "full":
        expected_datasets = sorted(set(expected_datasets) | set(DATASET_REPOSITORIES))
    represented_datasets = {row.get("source", {}).get("dataset") for row in records}
    missing_datasets = sorted(set(expected_datasets) - represented_datasets)
    run_complete = run_complete and not missing_datasets
    metric_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    dataset_summaries: dict[str, Any] = {}
    for dataset in expected_datasets:
        dataset_rows = [row for row in records if row.get("source", {}).get("dataset") == dataset]
        greedy_correct = [
            value for row in dataset_rows
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
        dataset_summaries[dataset] = {
            "total_rows": len(dataset_rows),
            "selected_rows": sum(identifier.startswith(dataset + ":") for identifier in selected_ids),
            "scorable_rows": len(greedy_correct),
            "unscorable_rows": len(dataset_rows) - len(greedy_correct),
            "greedy_accuracy": sum(greedy_correct) / len(greedy_correct) if greedy_correct else None,
            "prediction_parse_coverage": parsed_count / len(dataset_rows) if dataset_rows else 0.0,
            "sample_answer_parseable_count": sample_parseable_count,
            "sample_count": sample_count,
            "sample_answer_parse_coverage": (
                sample_parseable_count / sample_count if sample_count else 0.0
            ),
        }
        for signal, path in GREEDY_SIGNALS.items():
            row, bins = _collect_metric(
                dataset_rows,
                "greedy",
                dataset,
                signal,
                path,
                ("greedy", "correct"),
            )
            metric_rows.append(row)
            bin_rows.extend(bins)
        paired_rows = [
            row for row in dataset_rows
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
            dataset_rows,
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
    )
    summary = {
        "run_id": manifest.get("run_id", run_dir.name),
        "run_status": manifest.get("status", "unknown"),
        "profile": manifest.get("profile", {}).get("name", "unknown"),
        "run_complete": run_complete,
        "required_datasets": expected_datasets,
        "missing_datasets": missing_datasets,
        "selected_count": len(selected_ids),
        "completed_count": len(records),
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
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_reliability_plots(run_dir, expected_datasets, records)
    _write_markdown(run_dir / "report.md", summary, metric_rows)
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
