"""Teacher calibration report contracts for the smoke and mini_12h profiles."""

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from teacher_reliability.report import main as report_main
from teacher_reliability.report import run_report
from teacher_reliability.teacher import derive_sample_seed


DATASETS = ("gsm8k", "math", "gsm_plus")


def result_row(dataset, index):
    labels = [True, False, True, False, None]
    entropy = [0.9, 0.1, 0.8, 0.2, 0.5]
    agreement = [0.8, 0.2, 0.7, 0.3, 0.5]
    answer = [0.8, None, 0.7, 0.3, None]
    sequence = [0.7, 0.3, 0.6, 0.4, 0.5]
    majority_labels = [False, False, True, True, None]
    majority_share = [0.9, 0.8, 0.7, 0.6, 0.5]
    example_id = f"{dataset}:fixture:test:{index}"
    return {
        "example_id": example_id,
        "status": "complete",
        "source": {"dataset": dataset, "gold_status": "parsed" if index < 4 else "unanswerable"},
        "gold_answer": "1" if index < 4 else None,
        "greedy": {
            "text": r"\boxed{1}",
            "token_ids": [10, 11],
            "token_entropies_nats": [0.4, 0.5],
            "token_logprobs": [-0.3, -0.4],
            "token_char_spans": [[0, 1], [1, 2]],
            "content_token_mask": [True, True],
            "token_alignment_status": "aligned",
            "eos_generated": True,
            "was_truncated": False,
            "termination_reason": "eos",
            "predicted_answer": "1",
            "answer_method": "boxed",
            "prediction_parseable": index < 4,
            "gold_parseable": index < 4,
            "correct": labels[index],
            "grading_status": "graded" if index < 4 else "missing_gold",
            "content_token_count": 2,
            "mean_entropy_nats": 0.45,
            "entropy_confidence": entropy[index],
            "sequence_logprob_sum": -0.7,
            "answer_token_count": 0,
            "answer_token_geometric_probability": answer[index],
            "sequence_geometric_probability": sequence[index],
            "answer_alignment_status": "no_answer_tokens",
        },
        "self_consistency": {
            "sample_count": 8,
            "sample_answer_parseable_count": 6,
            "sample_answer_parse_unknown_count": 0,
            "greedy_agreement_count": round(agreement[index] * 8),
            "greedy_agreement_unknown_count": 0,
            "greedy_agreement_share": agreement[index],
            "majority_correct": majority_labels[index],
            "majority_grading_status": "graded" if index < 4 else "missing_gold",
            "majority_vote_count": round(majority_share[index] * 8),
            "majority_vote_share": majority_share[index],
            "majority_comparison_unknown_count": 0,
        },
        "samples": [
            {
                "index": sample_index,
                "seed": derive_sample_seed(42, example_id, sample_index),
                "text": r"\boxed{1}",
                "token_ids": [13, 14],
                "answer": "1",
                "answer_method": "boxed",
                "eos_generated": True,
                "was_truncated": False,
                "termination_reason": "eos",
            }
            for sample_index in range(8)
        ],
        "requested_sample_count": 8,
    }


class ReportGenerationTests(unittest.TestCase):
    def make_run(self, run_dir, status="complete", records=None, profile="mini_12h"):
        run_dir.mkdir(parents=True)
        if records is None:
            records = [result_row(dataset, index) for dataset in DATASETS for index in range(5)]
        selected_ids = [row["example_id"] for row in records]
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "profile": {"name": profile},
                    "seed": 42,
                    "selected_example_ids": selected_ids,
                    "selected_example_count": len(selected_ids),
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records),
            encoding="utf-8",
        )

    def test_report_emits_dataset_metrics_bins_plots_and_an_exploratory_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            records = [result_row(dataset, index) for dataset in DATASETS for index in range(4)]
            self.make_run(run_dir, records=records)

            summary = run_report(run_dir)

            self.assertTrue(summary["run_complete"])
            self.assertEqual(summary["recommendation"]["status"], "exploratory")
            self.assertIsNone(summary["recommendation"]["signal"])
            self.assertAlmostEqual(
                summary["by_dataset"]["gsm8k"]["sample_answer_parse_coverage"], 0.75
            )
            self.assertAlmostEqual(summary["macro"]["entropy_confidence"]["macro_auroc"], 1.0)
            self.assertTrue((run_dir / "metrics.csv").exists())
            self.assertTrue((run_dir / "calibration_bins.csv").exists())
            self.assertTrue((run_dir / "report.md").exists())
            self.assertEqual(len(list(run_dir.glob("reliability_*.png"))), 3)
            with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            greedy_entropy = next(
                row for row in rows
                if row["analysis"] == "greedy"
                and row["dataset"] == "gsm8k"
                and row["signal"] == "entropy_confidence"
            )
            majority = next(
                row for row in rows
                if row["analysis"] == "majority" and row["dataset"] == "gsm8k"
            )
            self.assertEqual(greedy_entropy["n"], "4")
            self.assertAlmostEqual(float(greedy_entropy["auroc"]), 1.0)
            self.assertAlmostEqual(float(majority["auroc"]), 0.0)
            answer_token = next(
                row for row in rows
                if row["dataset"] == "gsm8k" and row["signal"] == "answer_token_probability"
            )
            self.assertAlmostEqual(float(answer_token["coverage"]), 0.75)

    def test_mini_and_smoke_profiles_explain_why_they_are_exploratory(self):
        for profile, phrase in (("mini_12h", "too small"), ("smoke", "setup/runtime check")):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary) / "run"
                self.make_run(run_dir, profile=profile)

                summary = run_report(run_dir)

                self.assertEqual(summary["recommendation"]["status"], "exploratory")
                self.assertIsNone(summary["recommendation"]["signal"])
                self.assertIn(phrase, summary["recommendation"]["reason"])

    def test_coverage_gate_reports_unknown_labels_and_low_source_coverage(self):
        records = []
        for dataset in DATASETS:
            for index in range(20):
                row = result_row(dataset, index % 4)
                row["example_id"] = f"{dataset}:fixture:mini:{index:02d}"
                for sample in row["samples"]:
                    sample["seed"] = derive_sample_seed(42, row["example_id"], sample["index"])
                records.append(row)
        records[0]["greedy"]["correct"] = None
        records[20]["source"]["gold_status"] = "unanswerable"
        records[20]["greedy"]["correct"] = None
        records[21]["source"]["gold_status"] = "unanswerable"
        records[21]["greedy"]["correct"] = None
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_run(run_dir, records=records)

            summary = run_report(run_dir)

            self.assertTrue(summary["run_complete"])
            self.assertFalse(summary["recommendation_coverage_gate"]["valid"])
            self.assertAlmostEqual(summary["by_dataset"]["gsm8k"]["outcome_label_coverage"], 0.95)
            self.assertAlmostEqual(summary["by_dataset"]["math"]["source_scorable_fraction"], 0.9)
            report = (run_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("Outcome label coverage", report)
            self.assertIn("95.0%", report)

    def test_report_keeps_unavailable_unknown_counts_unavailable(self):
        records = [result_row(dataset, index) for dataset in DATASETS for index in range(5)]
        for field in (
            "sample_answer_parse_unknown_count",
            "greedy_agreement_unknown_count",
            "majority_comparison_unknown_count",
        ):
            records[0]["self_consistency"][field] = None
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_run(run_dir, records=records)

            summary = run_report(run_dir)

            self.assertTrue(summary["run_complete"])
            for field in (
                "sample_answer_parse_unknown_count",
                "greedy_agreement_unknown_count",
                "majority_comparison_unknown_count",
            ):
                self.assertIsNone(summary["by_dataset"]["gsm8k"][field])
            report = (run_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("| gsm8k | unavailable | unavailable | unavailable |", report)

    def test_report_labels_local_rtx4060_execution_and_writes_per_origin_metrics(self):
        records = []
        for dataset in DATASETS:
            for index in range(2):
                row = result_row(dataset, index)
                row["execution_origin"] = "rtx4060-local"
                records.append(row)
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_run(run_dir, records=records)

            summary = run_report(run_dir)

            self.assertEqual(summary.get("execution_label"), "RTX 4060 local")
            self.assertEqual(summary["by_execution"]["rtx4060-local"]["row_count"], 6)
            self.assertTrue((run_dir / "metrics_by_execution.csv").exists())
            report = (run_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("RTX 4060 local", report)

    def test_local_report_recovers_failed_count_from_durable_stage_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_run(run_dir, status="partial", records=[result_row("gsm8k", 0)])
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update({
                "backend": "transformers-local", "run_signature": "local-signature",
                "selected_example_ids": ["gsm8k:fixture:test:0", "gsm8k:fixture:test:1", "gsm8k:fixture:test:2"],
                "selected_example_count": 3, "completed_example_count": 1,
                "failed_example_count": 0, "pending_example_count": 2,
            })
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            stages = run_dir / "stages"
            stages.mkdir()
            (stages / "failed.json").write_text(json.dumps({
                "run_signature": "local-signature", "example_id": "gsm8k:fixture:test:1",
                "greedy": {"text": "partial"}, "samples": {}, "failure": "CUDA OOM",
            }), encoding="utf-8")
            summary = run_report(run_dir)
            self.assertEqual(summary["failed_count"], 1)
            self.assertEqual(summary["pending_count"], 1)
            self.assertFalse(summary["run_complete"])

    def test_report_exposes_unknown_parse_and_majority_comparison_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            records = [result_row(dataset, index) for dataset in DATASETS for index in range(5)]
            records[0]["self_consistency"]["sample_answer_parse_unknown_count"] = 2
            records[0]["self_consistency"]["majority_comparison_unknown_count"] = 1
            self.make_run(run_dir, records=records)

            summary = run_report(run_dir)

            gsm8k = summary["by_dataset"]["gsm8k"]
            self.assertEqual(gsm8k.get("sample_answer_parse_unknown_count", 0), 2)
            self.assertEqual(gsm8k.get("majority_comparison_unknown_count", 0), 1)

    def test_unknown_greedy_comparison_removes_agreement_from_signal_coverage(self):
        records = [result_row(dataset, index) for dataset in DATASETS for index in range(4)]
        records[0]["self_consistency"]["greedy_agreement_share"] = None
        records[0]["self_consistency"]["greedy_agreement_unknown_count"] = 1
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_run(run_dir, records=records)

            summary = run_report(run_dir)

            gate = summary["recommendation_coverage_gate"]["by_dataset"]["gsm8k"]
            coverage = gate.get("signal_score_coverage", {}).get(
                "self_consistency_agreement", {}
            )
            self.assertEqual(
                coverage,
                {"scored_rows": 3, "source_scorable_rows": 4, "coverage": 0.75},
            )
            self.assertFalse(
                summary["macro"]["self_consistency_agreement"]["eligible_for_recommendation"]
            )
            self.assertEqual(
                summary["by_dataset"]["gsm8k"]["greedy_agreement_unknown_count"], 1
            )

    def test_incomplete_run_does_not_receive_a_kd_gate_recommendation(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_run(run_dir, status="running")

            summary = run_report(run_dir)

            self.assertFalse(summary["run_complete"])
            self.assertIsNone(summary["recommendation"]["signal"])

    def test_report_cli_handles_a_workspace_path_outside_the_console_encoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "เอกสาร" / "run"
            self.make_run(run_dir)
            output_bytes = io.BytesIO()
            output = io.TextIOWrapper(output_bytes, encoding="cp1252")

            try:
                with patch("sys.stdout", output):
                    result = report_main(["--run-dir", str(run_dir)])
                output.flush()
                printed = output_bytes.getvalue().decode("cp1252")
            finally:
                output.close()

            self.assertEqual(result, 0)
            self.assertIn("Report written to", printed)

    def test_primary_auroc_comparison_uses_the_same_rows_when_entropy_is_missing(self):
        records = [result_row(dataset, index) for dataset in DATASETS for index in range(5)]
        records[0]["greedy"]["entropy_confidence"] = None
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_run(run_dir, records=records)
            run_report(run_dir)

            with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
                paired = [
                    row for row in csv.DictReader(handle)
                    if row["analysis"] == "paired_primary" and row["dataset"] == "gsm8k"
                ]
            self.assertEqual(len(paired), 2)
            self.assertEqual({row["n"] for row in paired}, {"3"})
            self.assertEqual({row["scorable_n"] for row in paired}, {"4"})
            self.assertEqual({float(row["coverage"]) for row in paired}, {0.75})
            self.assertEqual({float(row["auroc"]) for row in paired}, {1.0})


if __name__ == "__main__":
    unittest.main()
