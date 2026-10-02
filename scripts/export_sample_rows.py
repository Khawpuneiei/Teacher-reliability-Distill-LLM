"""Export a compact, per-question view of a completed mini_12h teacher run.

Writes three files to ``--out-dir``:

- ``sample_rows.csv``: one row per question with answers and confidence scores;
- ``sample_rows.jsonl``: the same fields plus the full greedy solution text and
  the eight sampled answers (token-level arrays are omitted);
- ``SAMPLE.md``: a readable overview, worked examples, and the full table.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

DATASET_ORDER = {"gsm8k": 0, "math": 1, "gsm_plus": 2}
DATASET_LABEL = {"gsm8k": "GSM8K", "math": "MATH", "gsm_plus": "GSM-Plus"}
FIELDS = [
    "example_id", "dataset", "subject_or_perturbation", "gold_answer",
    "greedy_answer", "greedy_correct", "greedy_tokens", "greedy_truncated",
    "entropy_confidence", "sequence_probability", "answer_token_probability",
    "agreement_share", "majority_answer", "majority_share", "majority_correct",
    "sample_answers", "question",
]


def _round(value, digits=3):
    return None if value is None else round(float(value), digits)


def compact_row(row: dict) -> dict:
    source = row.get("source", {})
    greedy = row.get("greedy", {})
    consistency = row.get("self_consistency", {})
    dataset = source.get("dataset")
    return {
        "example_id": row["example_id"],
        "dataset": dataset,
        "subject_or_perturbation": source.get("subject") or source.get("config")
        if dataset == "math" else source.get("perturbation_type"),
        "gold_answer": row.get("gold_answer"),
        "greedy_answer": greedy.get("predicted_answer"),
        "greedy_correct": greedy.get("correct"),
        "greedy_tokens": len(greedy.get("token_ids") or []),
        "greedy_truncated": greedy.get("was_truncated"),
        "entropy_confidence": _round(greedy.get("entropy_confidence")),
        "sequence_probability": _round(greedy.get("sequence_geometric_probability")),
        "answer_token_probability": _round(greedy.get("answer_token_geometric_probability")),
        "agreement_share": _round(consistency.get("greedy_agreement_share")),
        "majority_answer": consistency.get("majority_answer"),
        "majority_share": _round(consistency.get("majority_vote_share")),
        "majority_correct": consistency.get("majority_correct"),
        "sample_answers": [sample.get("answer") for sample in row.get("samples", [])],
        "question": row.get("question"),
        "greedy_text": greedy.get("text"),
    }


def _cell(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "✓" if value else "✗"
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= 28 else text[:25] + "…"


def _example_block(title: str, row: dict) -> list[str]:
    solution = (row["greedy_text"] or "").strip()
    if len(solution) > 1200:
        solution = solution[:1200].rstrip() + "\n…(truncated for display)"
    return [
        f"### {title}",
        "",
        f"`{row['example_id']}` ({DATASET_LABEL[row['dataset']]})",
        "",
        "> " + (row["question"] or "").strip().replace("\n", "\n> "),
        "",
        f"- Gold answer: `{row['gold_answer']}`; teacher greedy answer: "
        f"`{row['greedy_answer']}` ({'correct' if row['greedy_correct'] else 'wrong'})",
        f"- Entropy confidence {row['entropy_confidence']}, sequence probability "
        f"{row['sequence_probability']}, agreement share {row['agreement_share']} "
        f"(majority `{row['majority_answer']}` at {row['majority_share']})",
        f"- Eight sampled answers: {', '.join(f'`{a}`' for a in row['sample_answers'])}",
        "",
        "<details><summary>Teacher greedy solution</summary>",
        "",
        "```text",
        solution,
        "```",
        "",
        "</details>",
        "",
    ]


def write_markdown(path: Path, rows: list[dict], run_id: str) -> None:
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    lines = [
        "# Sample experiment: 111-question teacher run",
        "",
        f"Run `{run_id}`: Qwen2.5-Math-7B-Instruct (NF4/FP16) on an RTX 4060 Laptop GPU,",
        "one greedy answer plus eight seeded samples (temperature 0.7, top-k 50) per",
        "question, 1,024-token cap, seed 42. Aggregate metrics and reliability plots",
        "are in [`report.md`](report.md); every row below is also in",
        "[`sample_rows.csv`](sample_rows.csv) and, with the full greedy solution text,",
        "[`sample_rows.jsonl`](sample_rows.jsonl).",
        "",
        "## Overview",
        "",
        "| Dataset | Questions | Scorable | Greedy correct | Majority correct | Mean agreement (correct) | Mean agreement (wrong) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in sorted(by_dataset, key=DATASET_ORDER.get):
        items = by_dataset[dataset]
        scored = [r for r in items if r["greedy_correct"] is not None]
        right = [r for r in scored if r["greedy_correct"]]
        wrong = [r for r in scored if not r["greedy_correct"]]
        majority = [r for r in scored if r["majority_correct"]]

        def mean_agreement(group):
            values = [r["agreement_share"] for r in group if r["agreement_share"] is not None]
            return f"{sum(values) / len(values):.2f}" if values else "—"

        lines.append(
            f"| {DATASET_LABEL[dataset]} | {len(items)} | {len(scored)} | "
            f"{len(right)} ({len(right) / len(scored):.1%}) | "
            f"{len(majority)} ({len(majority) / len(scored):.1%}) | "
            f"{mean_agreement(right)} | {mean_agreement(wrong)} |"
        )
    lines += [
        "",
        "Agreement share is the fraction of the eight samples whose answer matches",
        "the greedy answer. Wrong answers have much lower agreement on average, which",
        "is why self-consistency is a useful (though exploratory) correctness signal.",
        "",
    ]

    scored = [r for r in rows if r["greedy_correct"] is not None]
    confident_right = max(
        (r for r in scored if r["greedy_correct"] and r["dataset"] == "math"),
        key=lambda r: (r["agreement_share"] or 0, r["entropy_confidence"] or 0),
    )
    flagged_wrong = min(
        (r for r in scored if not r["greedy_correct"] and r["greedy_answer"] is not None),
        key=lambda r: (r["agreement_share"] if r["agreement_share"] is not None else 1),
    )
    confident_wrong = max(
        (r for r in scored if not r["greedy_correct"]),
        key=lambda r: (r["agreement_share"] or 0, r["entropy_confidence"] or 0),
    )
    lines += ["## Worked examples", ""]
    lines += _example_block("Confident and correct", confident_right)
    lines += _example_block("Wrong, and flagged by low agreement", flagged_wrong)
    lines += _example_block("Wrong despite high agreement (a gate would miss it)", confident_wrong)

    lines += [
        "## All 111 rows",
        "",
        "✓ = correct, ✗ = wrong, — = no gold answer. Entropy = entropy confidence;",
        "Agree = agreement share; Maj = majority vote share.",
        "",
    ]
    for dataset in sorted(by_dataset, key=DATASET_ORDER.get):
        lines += [
            f"### {DATASET_LABEL[dataset]}",
            "",
            "| # | Example | Subject / perturbation | Gold | Greedy | ✓ | Entropy | Agree | Majority | Maj | ✓ | Tokens |",
            "|---:|---|---|---|---|:-:|---:|---:|---|---:|:-:|---:|",
        ]
        for index, r in enumerate(by_dataset[dataset], start=1):
            lines.append(
                f"| {index} | `{r['example_id'].split(':')[-1]}` | {_cell(r['subject_or_perturbation'])} | "
                f"{_cell(r['gold_answer'])} | {_cell(r['greedy_answer'])} | {_cell(r['greedy_correct'])} | "
                f"{_cell(r['entropy_confidence'])} | {_cell(r['agreement_share'])} | "
                f"{_cell(r['majority_answer'])} | {_cell(r['majority_share'])} | "
                f"{_cell(r['majority_correct'])} | {r['greedy_tokens']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    with (args.run_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        rows = [compact_row(json.loads(line)) for line in handle if line.strip()]
    rows.sort(key=lambda r: (DATASET_ORDER[r["dataset"]], r["example_id"]))
    counts = Counter(r["dataset"] for r in rows)
    if len(rows) != manifest.get("selected_example_count"):
        raise ValueError("prediction rows do not match the manifest's selected count")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "sample_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "sample_answers": " | ".join(str(a) for a in row["sample_answers"])})
    with (args.out_dir / "sample_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_markdown(args.out_dir / "SAMPLE.md", rows, manifest.get("run_id", args.run_dir.name))
    print(f"Exported {len(rows)} rows {dict(counts)} to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
