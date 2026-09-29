import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from teacher_reliability.report import main as report_main, run_report


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
        "greedy": {
            "correct": labels[index],
            "entropy_confidence": entropy[index],
            "answer_token_geometric_probability": answer[index],
            "sequence_geometric_probability": sequence[index],
            "prediction_parseable": index < 4,
        },
        "self_consistency": {
            "sample_count": 8,
            "sample_answer_parseable_count": 6,
            "greedy_agreement_share": agreement[index],
            "majority_correct": majority_labels[index],
            "majority_vote_share": majority_share[index],
        },
        "samples": [],
    }


class ReportGenerationTests(unittest.TestCase):
    def make_run(self, run_dir, status="complete", records=None, profile="full"):
        run_dir.mkdir(parents=True)
        if records is None:
            records = [result_row(dataset, index) for dataset in DATASETS for index in range(5)]
        selected_ids = [row["example_id"] for row in records]
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "profile": {"name": profile},
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

    def test_report_emits_dataset_metrics_bins_plots_and_a_grounded_recommendation(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_run(run_dir)

            summary = run_report(run_dir)

            self.assertTrue(summary["run_complete"])
            self.assertEqual(summary["recommendation"]["signal"], "entropy_confidence")
            self.assertAlmostEqual(
                summary["by_dataset"]["gsm8k"]["sample_answer_parse_coverage"], 0.75
            )
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

    def test_incomplete_run_does_not_receive_a_kd_gate_recommendation(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_run(run_dir, status="running")

            summary = run_report(run_dir)

            self.assertFalse(summary["run_complete"])
            self.assertIsNone(summary["recommendation"]["signal"])

    def test_time_bounded_profile_report_explains_why_it_is_exploratory(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_run(run_dir, profile="a100_12h")

            summary = run_report(run_dir)

            self.assertEqual(summary["recommendation"]["status"], "exploratory")
            self.assertIsNone(summary["recommendation"]["signal"])
            self.assertIn("time-bounded subset", summary["recommendation"]["reason"])
            self.assertIn("full benchmark", summary["recommendation"]["reason"])

    def test_full_report_requires_all_three_benchmarks_even_when_selected_ids_are_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_run(run_dir, records=[result_row("gsm8k", 0), result_row("gsm8k", 1)])

            summary = run_report(run_dir)

            self.assertFalse(summary["run_complete"])
            self.assertIsNone(summary["recommendation"]["signal"])
            self.assertEqual(summary["missing_datasets"], ["gsm_plus", "math"])
            self.assertEqual(set(summary["by_dataset"]), {"gsm8k", "math", "gsm_plus"})
            self.assertEqual(summary["by_dataset"]["math"]["total_rows"], 0)
            self.assertIsNone(summary["macro"]["entropy_confidence"]["macro_auroc"])

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

    def test_gate_ranking_does_not_reward_missing_scores_on_hard_rows(self):
        records = [result_row(dataset, index) for dataset in DATASETS for index in range(5)]
        for row in records:
            index = int(row["example_id"].rsplit(":", 1)[1])
            row["greedy"]["entropy_confidence"] = [0.8, 0.9, 0.7, 0.1, 0.5][index]
            row["greedy"]["sequence_geometric_probability"] = [0.2, 0.9, 0.1, 0.05, 0.5][index]
            row["self_consistency"]["greedy_agreement_share"] = [0.3, 0.9, 0.2, 0.1, 0.5][index]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_run(run_dir, records=records)
            summary = run_report(run_dir)

            self.assertEqual(summary["recommendation"]["signal"], "entropy_confidence")
            self.assertFalse(summary["macro"]["answer_token_probability"]["eligible_for_recommendation"])


if __name__ == "__main__":
    unittest.main()
