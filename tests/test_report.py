import csv
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from teacher_reliability.report import (
    _full_row_issues,
    _validate_full_run_revisions,
    _validate_local_record_receipts,
    _validate_preserved_a100_rows,
    main as report_main,
    run_report,
)
from teacher_reliability.run import _canonical_digest
from teacher_reliability.teacher import derive_sample_seed


DATASETS = ("gsm8k", "math", "gsm_plus")
MODEL_REVISION = "ef9926d75ab1d54532f6a30dd5e760355eb9aa4d"
DATASET_REVISIONS = {
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "EleutherAI/hendrycks_math": "21a5633873b6a120296cce3e2df9d5550074f4a3",
    "qintongli/GSM-Plus": "3b708db57b96a16e8e3368ed2956990c0809440e",
}
PINNED_A100_SHA256 = "7fa37f4a6e083e293e4f29606f7643f7ede940569d531410c47e8162130a4d82"
PINNED_A100_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "full-a10080-20260930T131901Z.zip"


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
    def make_run(self, run_dir, status="complete", records=None, profile="full", study_scope="synthetic"):
        run_dir.mkdir(parents=True)
        if records is None:
            records = [result_row(dataset, index) for dataset in DATASETS for index in range(5)]
        selected_ids = [row["example_id"] for row in records]
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "profile": {"name": profile},
                    "study_scope": study_scope,
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

    def make_full_mixed_origin_run(
        self, run_dir, *, preserve_integrity=False, legacy_unknown_counts=False,
        archive_source="pinned",
    ):
        if archive_source == "pinned":
            archive_bytes = PINNED_A100_FIXTURE.read_bytes()
            self.assertEqual(hashlib.sha256(archive_bytes).hexdigest(), PINNED_A100_SHA256)
            with zipfile.ZipFile(BytesIO(archive_bytes)) as source_archive:
                source_manifest_bytes = source_archive.read(next(
                    name for name in source_archive.namelist() if name.endswith("run_manifest.json")
                ))
                archived_rows = [json.loads(line) for line in source_archive.read(next(
                    name for name in source_archive.namelist() if name.endswith("predictions.jsonl")
                )).splitlines()]
            source_manifest = json.loads(source_manifest_bytes)
            imported_ids = [row["example_id"] for row in archived_rows]
            self.assertEqual(len(imported_ids), 110)
            archive_sha256 = PINNED_A100_SHA256
            source_run_id = source_manifest["run_id"]
            source_gpu = source_manifest["hardware"]["gpu_name"]
            source_identity = source_manifest["identity"]
            source_run_signature = source_manifest["run_signature"]
        elif archive_source == "synthetic":
            imported_ids = [f"gsm8k:a100:{index:03d}" for index in range(110)]
            archive_sha256 = "a" * 64
            source_run_id = "source-a100-run"
            source_gpu = "NVIDIA A100 80GB"
            source_identity = {"runtime": {"gpu_name": source_gpu}}
            source_run_signature = _canonical_digest(source_identity)
        else:
            raise ValueError("unsupported test archive source")
        local_gpu = "NVIDIA GeForce RTX 4060 Laptop GPU"
        local_hardware = {"gpu_name": local_gpu, "gpu_total_memory_mib": 8188}
        records = []
        selected_ids = []

        for dataset, origin, count in (
            ("gsm8k", "a100-import", 110),
            ("gsm8k", "rtx4060-local", 1209),
            ("math", "rtx4060-local", 350),
            ("gsm_plus", "rtx4060-local", 10552),
        ):
            for index in range(count):
                if origin == "a100-import" and archive_source == "pinned":
                    row = deepcopy(archived_rows[index])
                    for field in (
                        "sample_answer_parse_unknown_count",
                        "greedy_agreement_unknown_count",
                        "majority_comparison_unknown_count",
                    ):
                        row["self_consistency"].setdefault(field, None)
                else:
                    row = result_row(dataset, index % 5)
                if origin == "a100-import":
                    example_id = imported_ids[index]
                    row["execution_provenance"] = {
                        "archive_sha256": archive_sha256,
                        "source_run_id": source_run_id,
                        "source_run_signature": source_run_signature,
                        "source_gpu": source_gpu,
                    }
                else:
                    example_id = f"{dataset}:fixture:local:{index:05d}"
                row["example_id"] = example_id
                row["source"]["revision"] = DATASET_REVISIONS[
                    {
                        "gsm8k": "openai/gsm8k",
                        "math": "EleutherAI/hendrycks_math",
                        "gsm_plus": "qintongli/GSM-Plus",
                    }[dataset]
                ]
                if origin == "a100-import" and legacy_unknown_counts:
                    for field in (
                        "sample_answer_parse_unknown_count",
                        "greedy_agreement_unknown_count",
                        "majority_comparison_unknown_count",
                    ):
                        row["self_consistency"][field] = None
                for sample in row["samples"]:
                    sample["seed"] = derive_sample_seed(42, example_id, sample["index"])
                row["execution_origin"] = origin
                selected_ids.append(example_id)
                records.append(row)

        run_dir.mkdir(parents=True)
        manifest = {
                    "status": "complete",
                    "profile": {"name": "full"},
                    "study_scope": "full-mixed-origin",
                    "seed": 42,
                    "run_signature": "local-run-signature",
                    "model_revision": MODEL_REVISION,
                    "tokenizer_revision": MODEL_REVISION,
                    "dataset_revisions": dict(DATASET_REVISIONS),
                    "identity": {
                        "base_seed": 42,
                        "model": {"revision": MODEL_REVISION},
                        "datasets": dict(DATASET_REVISIONS),
                        "runtime": local_hardware,
                    },
                    "hardware": local_hardware,
                    "selected_example_ids": selected_ids,
                    "selected_example_count": 12221,
                    "imported_example_ids": imported_ids,
                    "imported_example_count": 110,
                    "import_provenance": {
                        "archive_sha256": archive_sha256,
                        "source_run_id": source_run_id,
                        "source_run_signature": source_run_signature,
                        "source_gpu": source_gpu,
                        "imported_count": 110,
                    },
                    "execution_origins": {
                        "a100-import": {
                            "gpu_name": source_gpu,
                            "archive_sha256": archive_sha256,
                            "source_run_id": source_run_id,
                            "source_run_signature": source_run_signature,
                            "imported_count": 110,
                        },
                        "rtx4060-local": {"gpu_name": local_gpu},
                    },
                }
        manifest["run_signature"] = _canonical_digest(manifest["identity"])
        if preserve_integrity:
            if archive_source == "synthetic":
                source_manifest = {
                    "run_id": source_run_id,
                    "identity": source_identity,
                    "hardware": source_identity["runtime"],
                    "run_signature": source_run_signature,
                }
                source_manifest_bytes = json.dumps(
                    source_manifest, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
                imported_records = [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"execution_origin", "execution_provenance"}
                    }
                    for row in records
                    if row["execution_origin"] == "a100-import"
                ]
                if legacy_unknown_counts:
                    for source_row in imported_records:
                        source_row["self_consistency"] = dict(source_row["self_consistency"])
                        for field in (
                            "sample_answer_parse_unknown_count",
                            "greedy_agreement_unknown_count",
                            "majority_comparison_unknown_count",
                        ):
                            source_row["self_consistency"].pop(field, None)
                predictions_bytes = "".join(
                    json.dumps(row, sort_keys=True) + "\n" for row in imported_records
                ).encode("utf-8")
                archive_buffer = BytesIO()
                with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr("source-run/run_manifest.json", source_manifest_bytes)
                    archive.writestr("source-run/predictions.jsonl", predictions_bytes)
                archive_bytes = archive_buffer.getvalue()
            archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
            manifest["import_provenance"]["archive_sha256"] = archive_sha256
            manifest["execution_origins"]["a100-import"]["archive_sha256"] = archive_sha256
            for row in records:
                if row["execution_origin"] == "a100-import":
                    row["execution_provenance"]["archive_sha256"] = archive_sha256
            provenance_dir = run_dir / "provenance"
            provenance_dir.mkdir(parents=True)
            (provenance_dir / "source_run.zip").write_bytes(archive_bytes)
            (provenance_dir / "source_run_manifest.json").write_bytes(source_manifest_bytes)
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (run_dir / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records),
            encoding="utf-8",
        )
        if preserve_integrity:
            integrity_dir = run_dir / "stages" / "integrity"
            integrity_dir.mkdir(parents=True)
            for row in records:
                if row["execution_origin"] != "rtx4060-local":
                    continue
                key = hashlib.sha256(row["example_id"].encode("utf-8")).hexdigest()
                receipt = {
                    "schema_version": 1,
                    "run_signature": manifest["run_signature"],
                    "example_id": row["example_id"],
                    "record_sha256": _canonical_digest(row),
                }
                (integrity_dir / f"{key}.json").write_text(
                    json.dumps(receipt), encoding="utf-8"
                )

    def read_prediction_rows(self, run_dir):
        return [
            json.loads(line)
            for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        ]

    def write_prediction_rows(self, run_dir, records):
        (run_dir / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records),
            encoding="utf-8",
        )

    def test_valid_full_mixed_origin_report_accepts_exact_import_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_full_mixed_origin_run(run_dir, preserve_integrity=True)

            summary = run_report(run_dir)

            self.assertTrue(summary["run_complete"])
            self.assertEqual(summary["completed_count"], 12221)
            self.assertEqual(
                summary["origin_count_validation"]["actual"],
                {
                    "a100-import": {"gsm8k": 110, "math": 0, "gsm_plus": 0},
                    "rtx4060-local": {"gsm8k": 1209, "math": 350, "gsm_plus": 10552},
                },
            )
            self.assertTrue(summary["origin_provenance_validation"]["valid"])

    def test_full_report_rejects_a_self_consistent_alternate_a100_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_full_mixed_origin_run(
                run_dir, preserve_integrity=True, archive_source="synthetic"
            )

            summary = run_report(run_dir)

            self.assertFalse(summary["run_complete"])
            self.assertIn(
                "source_archive_unpinned",
                summary["origin_provenance_validation"]["errors"],
            )

    def test_full_sized_report_enforces_origin_provenance_when_scope_is_synthetic(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_full_mixed_origin_run(run_dir, preserve_integrity=True)
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["study_scope"] = "synthetic"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            summary = run_report(run_dir)

            self.assertIsNotNone(summary["origin_count_validation"])
            self.assertTrue(summary["origin_count_validation"]["valid"])
            self.assertTrue(summary["origin_provenance_validation"]["valid"])
            self.assertTrue(summary["run_complete"])

    def test_legacy_import_unknown_counts_are_unavailable_and_withhold_recommendation(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_full_mixed_origin_run(
                run_dir, preserve_integrity=True, legacy_unknown_counts=True
            )

            summary = run_report(run_dir)

            self.assertTrue(summary["run_complete"])
            self.assertIsNone(summary["recommendation"]["signal"])
            validation = summary.get("unknown_count_validation", {})
            self.assertFalse(validation.get("valid", True))
            self.assertEqual(validation.get("unavailable_row_count"), 110)
            self.assertEqual(validation.get("unavailable_value_count"), 330)
            self.assertIsNone(
                summary["by_dataset"]["gsm8k"]["sample_answer_parse_unknown_count"]
            )
            report = (run_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("## Full-study unknown-count validation", report)
            self.assertIn("Validation: **failed**", report)
            self.assertIn("| gsm8k | unavailable | unavailable | unavailable |", report)

    def test_full_report_invalidates_model_tokenizer_dataset_and_row_revision_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_full_mixed_origin_run(run_dir, preserve_integrity=True)
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["model_revision"] = "5" * 40
            manifest["tokenizer_revision"] = "6" * 40
            manifest["dataset_revisions"]["openai/gsm8k"] = "7" * 40
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            records = self.read_prediction_rows(run_dir)
            local_math = next(
                row for row in records
                if row["execution_origin"] == "rtx4060-local"
                and row["source"]["dataset"] == "math"
            )
            local_math["source"]["revision"] = "8" * 40
            self.write_prediction_rows(run_dir, records)

            summary = run_report(run_dir)

            validation = summary.get("revision_validation", {})
            self.assertFalse(summary["run_complete"])
            self.assertFalse(validation.get("valid", True))
            self.assertEqual(
                set(validation.get("issues", [])),
                {
                    "model_revision_mismatch",
                    "model_revision_unpinned",
                    "tokenizer_revision_mismatch",
                    "dataset_revisions_mismatch",
                    "dataset_revisions_unpinned",
                    "source_revision_mismatch",
                },
            )
            report = (run_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("## Full-study revision validation", report)
            self.assertIn("Validation: **failed**", report)

    def test_full_revision_validator_rejects_self_consistent_alternate_revisions(self):
        alternate_model = "5" * 40
        alternate_datasets = {repository: "6" * 40 for repository in DATASET_REVISIONS}
        manifest = {
            "seed": 42, "model_revision": alternate_model,
            "tokenizer_revision": alternate_model,
            "dataset_revisions": alternate_datasets,
            "identity": {
                "base_seed": 42,
                "model": {"revision": alternate_model},
                "datasets": dict(alternate_datasets),
            },
        }
        records = [{
            "example_id": "gsm8k:fixture:test:0",
            "source": {"dataset": "gsm8k", "revision": alternate_datasets["openai/gsm8k"]},
        }]

        result = _validate_full_run_revisions(manifest, records)

        self.assertFalse(result["valid"])
        self.assertIn("model_revision_unpinned", result["issues"])
        self.assertIn("dataset_revisions_unpinned", result["issues"])

    def test_full_revision_validator_rejects_non_42_seed(self):
        manifest = {
            "seed": 43, "model_revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "dataset_revisions": dict(DATASET_REVISIONS),
            "identity": {
                "base_seed": 43,
                "model": {"revision": MODEL_REVISION},
                "datasets": dict(DATASET_REVISIONS),
            },
        }
        records = [{
            "example_id": "gsm8k:fixture:test:0",
            "source": {"dataset": "gsm8k", "revision": DATASET_REVISIONS["openai/gsm8k"]},
        }]

        result = _validate_full_run_revisions(manifest, records)

        self.assertFalse(result["valid"])
        self.assertIn("seed_unpinned", result["issues"])

    def test_report_keeps_legacy_unknown_counts_unavailable(self):
        records = [result_row(dataset, index) for dataset in DATASETS for index in range(5)]
        records[0]["execution_origin"] = "a100-import"
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
            self.assertEqual(summary["malformed_row_count"], 0)
            for field in (
                "sample_answer_parse_unknown_count",
                "greedy_agreement_unknown_count",
                "majority_comparison_unknown_count",
            ):
                self.assertIsNone(summary["by_dataset"]["gsm8k"][field])
            report = (run_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("| gsm8k | unavailable | unavailable | unavailable |", report)

    def test_preserved_archive_helper_rejects_changed_imported_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            example_id = "gsm8k:a100:compact-fixture"
            source_run_id = "source-a100-run"
            source_gpu = "NVIDIA A100 80GB"
            source_identity = {"runtime": {"gpu_name": source_gpu}}
            source_run_signature = _canonical_digest(source_identity)
            imported = result_row("gsm8k", 0)
            imported["example_id"] = example_id
            for sample in imported["samples"]:
                sample["seed"] = derive_sample_seed(42, example_id, sample["index"])
            source_row = {
                key: value
                for key, value in imported.items()
                if key not in {"execution_origin", "execution_provenance"}
            }
            source_manifest_bytes = json.dumps(
                {
                    "run_id": source_run_id,
                    "identity": source_identity,
                    "hardware": source_identity["runtime"],
                    "run_signature": source_run_signature,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            predictions_bytes = (json.dumps(source_row) + "\n").encode("utf-8")
            archive_buffer = BytesIO()
            with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("source-run/run_manifest.json", source_manifest_bytes)
                archive.writestr("source-run/predictions.jsonl", predictions_bytes)
            archive_bytes = archive_buffer.getvalue()
            provenance_dir = run_dir / "provenance"
            provenance_dir.mkdir(parents=True)
            (provenance_dir / "source_run.zip").write_bytes(archive_bytes)
            (provenance_dir / "source_run_manifest.json").write_bytes(source_manifest_bytes)
            imported["execution_origin"] = "a100-import"
            imported["execution_provenance"] = {
                "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
                "source_gpu": source_gpu,
            }
            import_provenance = {
                "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
                "source_run_id": source_run_id,
                "source_run_signature": source_run_signature,
                "source_gpu": source_gpu,
            }
            expected_lineage = {
                "source_run_id": source_run_id,
                "source_run_signature": source_run_signature,
                "source_gpu": source_gpu,
            }
            preserved_archive = (provenance_dir / "source_run.zip").read_bytes()
            preserved_manifest = (provenance_dir / "source_run_manifest.json").read_bytes()

            imported["greedy"]["correct"] = not imported["greedy"]["correct"]
            archive_valid, rows_valid = _validate_preserved_a100_rows(
                run_dir, import_provenance, [imported], expected_lineage
            )

            self.assertTrue(archive_valid)
            self.assertFalse(rows_valid)
            self.assertEqual((provenance_dir / "source_run.zip").read_bytes(), preserved_archive)
            self.assertEqual((provenance_dir / "source_run_manifest.json").read_bytes(), preserved_manifest)

    def test_local_receipt_helper_rejects_changed_checkpoint_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            run_signature = "local-run-signature"
            example_id = "gsm8k:fixture:local:compact"
            local = result_row("gsm8k", 0)
            local["example_id"] = example_id
            for sample in local["samples"]:
                sample["seed"] = derive_sample_seed(42, example_id, sample["index"])
            local["execution_origin"] = "rtx4060-local"
            receipt_key = hashlib.sha256(example_id.encode("utf-8")).hexdigest()
            receipt_path = run_dir / "stages" / "integrity" / f"{receipt_key}.json"
            receipt_path.parent.mkdir(parents=True)
            receipt_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_signature": run_signature,
                        "example_id": example_id,
                        "record_sha256": _canonical_digest(local),
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(_validate_local_record_receipts(run_dir, run_signature, [local]))

            local["greedy"]["entropy_confidence"] = 0.75

            self.assertFalse(_validate_local_record_receipts(run_dir, run_signature, [local]))

    def test_full_report_rejects_swapped_imported_id_with_origin_counts_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_full_mixed_origin_run(run_dir)
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["imported_example_ids"][0] = "gsm8k:fixture:local:00000"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            summary = run_report(run_dir)

            self.assertFalse(summary["run_complete"])
            self.assertTrue(summary["origin_count_validation"]["valid"])
            self.assertFalse(summary["origin_provenance_validation"]["valid"])
            self.assertIn(
                "imported_example_ids_mismatch",
                summary["origin_provenance_validation"]["errors"],
            )
            self.assertIn(
                "local_row_id_overlap",
                summary["origin_provenance_validation"]["errors"],
            )
            report = (run_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("Full-study origin provenance validation", report)
            self.assertIn("Validation: **failed**", report)

    def test_full_report_rejects_missing_row_archive_hash_with_origin_counts_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_full_mixed_origin_run(run_dir)
            records = self.read_prediction_rows(run_dir)
            imported = next(row for row in records if row["execution_origin"] == "a100-import")
            del imported["execution_provenance"]["archive_sha256"]
            self.write_prediction_rows(run_dir, records)

            summary = run_report(run_dir)

            self.assertFalse(summary["run_complete"])
            self.assertTrue(summary["origin_count_validation"]["valid"])
            self.assertFalse(summary["origin_provenance_validation"]["valid"])
            self.assertIn(
                "row_provenance_mismatch",
                summary["origin_provenance_validation"]["errors"],
            )

    def test_full_report_rejects_changed_row_source_signature_with_origin_counts_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_full_mixed_origin_run(run_dir)
            records = self.read_prediction_rows(run_dir)
            imported = next(row for row in records if row["execution_origin"] == "a100-import")
            imported["execution_provenance"]["source_run_signature"] = "c" * 64
            self.write_prediction_rows(run_dir, records)

            summary = run_report(run_dir)

            self.assertFalse(summary["run_complete"])
            self.assertTrue(summary["origin_count_validation"]["valid"])
            self.assertFalse(summary["origin_provenance_validation"]["valid"])
            self.assertIn(
                "row_provenance_mismatch",
                summary["origin_provenance_validation"]["errors"],
            )

    def test_full_report_rejects_changed_source_gpu_metadata_when_all_local_claims_agree(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_full_mixed_origin_run(run_dir, preserve_integrity=True)
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            changed_gpu = "NVIDIA GeForce RTX 4060 Laptop GPU"
            manifest["import_provenance"]["source_gpu"] = changed_gpu
            manifest["execution_origins"]["a100-import"]["gpu_name"] = changed_gpu
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            records = self.read_prediction_rows(run_dir)
            for row in records:
                if row["execution_origin"] == "a100-import":
                    row["execution_provenance"]["source_gpu"] = changed_gpu
            self.write_prediction_rows(run_dir, records)

            summary = run_report(run_dir)

            self.assertFalse(summary["run_complete"])
            self.assertFalse(summary["origin_provenance_validation"]["valid"])
            self.assertIn(
                "preserved_archive_invalid",
                summary["origin_provenance_validation"]["errors"],
            )

    def test_full_report_rejects_local_gpu_metadata_that_disagrees_with_run_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_full_mixed_origin_run(run_dir, preserve_integrity=True)
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            changed_gpu = "NVIDIA A100 80GB"
            manifest["hardware"]["gpu_name"] = changed_gpu
            manifest["execution_origins"]["rtx4060-local"]["gpu_name"] = changed_gpu
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            summary = run_report(run_dir)

            self.assertFalse(summary["run_complete"])
            self.assertFalse(summary["origin_provenance_validation"]["valid"])
            self.assertIn(
                "local_runtime_metadata_mismatch",
                summary["origin_provenance_validation"]["errors"],
            )

    def test_hybrid_report_counts_failed_and_pending_without_complete_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            (run_dir / "run_manifest.json").write_text(json.dumps({
                "backend": "vllm-hybrid", "status": "incomplete", "profile": {"name": "full"},
                "selected_example_ids": ["gsm8k:fixture:test:0", "gsm8k:fixture:test:1"],
                "selected_example_count": 2, "completed_example_count": 0,
                "failed_example_count": 1, "pending_example_count": 1,
            }), encoding="utf-8")
            try:
                summary = run_report(run_dir)
            except (FileNotFoundError, ValueError):
                summary = {}
            self.assertEqual(summary.get("completed_count"), 0)
            self.assertEqual(summary.get("failed_count"), 1)
            self.assertEqual(summary.get("pending_count"), 1)
            self.assertFalse(summary.get("run_complete", True))
            self.assertTrue((run_dir / "report.md").exists())

    def test_hybrid_report_reads_durable_stages_when_manifest_progress_is_stale(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_run(run_dir, status="running", records=[result_row("gsm8k", 0)])
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update({
                "backend": "vllm-hybrid", "run_signature": "fixture-signature",
                "selected_example_ids": ["gsm8k:fixture:test:0", "gsm8k:fixture:test:1", "gsm8k:fixture:test:2"],
                "selected_example_count": 3, "completed_example_count": 0,
                "failed_example_count": 0, "pending_example_count": 3,
            })
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            stages = run_dir / "stages"
            stages.mkdir()
            (stages / "failed.json").write_text(json.dumps({
                "run_signature": "fixture-signature", "example_id": "gsm8k:fixture:test:1",
                "greedy": None, "samples": {}, "failure": "CUDA OOM",
            }), encoding="utf-8")
            try:
                summary = run_report(run_dir)
            except ValueError:
                summary = {}
            self.assertEqual(summary.get("completed_count"), 1)
            self.assertEqual(summary.get("failed_count"), 1)
            self.assertEqual(summary.get("pending_count"), 1)

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

    def test_report_discloses_an_unqualified_fast_start_and_no_launch_eta(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_run(run_dir, status="partial")
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            qualification = {
                "status": "unqualified-fast-start",
                "qualified": False,
                "configuration": "default-greedy-1-sample-2",
                "dataset_weighted_eta_hours": None,
                "reason": "Owner-authorized run with the full batch matrix skipped.",
            }
            manifest["qualification"] = qualification
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            summary = run_report(run_dir)
            report = (run_dir / "report.md").read_text(encoding="utf-8")

        self.assertEqual(summary["qualification"], qualification)
        self.assertIn("Profile qualification: **unqualified-fast-start**", report)
        self.assertIn("default-greedy-1-sample-2", report)
        self.assertIn("No measured throughput ETA is available", report)

    def test_report_emits_dataset_metrics_bins_plots_and_a_grounded_recommendation(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            records = [result_row(dataset, index) for dataset in DATASETS for index in range(4)]
            self.make_run(run_dir, records=records)

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

    def test_full_report_accepts_unknown_majority_comparison_with_null_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_full_mixed_origin_run(run_dir, preserve_integrity=True)
            records = self.read_prediction_rows(run_dir)
            row = next(
                item for item in records
                if item.get("execution_origin") == "rtx4060-local"
            )
            row["self_consistency"].update({
                "majority_answer": None,
                "majority_vote_count": None,
                "majority_vote_share": None,
                "majority_correct": None,
                "majority_grading_status": "majority_comparison_incomplete",
                "majority_comparison_unknown_count": 1,
            })
            self.write_prediction_rows(run_dir, records)

            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            receipt_key = hashlib.sha256(row["example_id"].encode("utf-8")).hexdigest()
            receipt_path = run_dir / "stages" / "integrity" / f"{receipt_key}.json"
            receipt_path.write_text(json.dumps({
                "schema_version": 1,
                "run_signature": manifest["run_signature"],
                "example_id": row["example_id"],
                "record_sha256": _canonical_digest(row),
            }), encoding="utf-8")

            summary = run_report(run_dir)

            self.assertTrue(summary["run_complete"], summary["malformed_rows"][:1])
            self.assertEqual(summary["malformed_row_count"], 0)

    def test_full_row_validator_rejects_majority_metrics_when_comparisons_are_unknown(self):
        row = result_row("gsm8k", 0)
        row["self_consistency"].update({
            "majority_comparison_unknown_count": 1,
            "majority_vote_count": 7,
            "majority_vote_share": 0.875,
            "majority_answer": "1",
            "majority_correct": True,
            "majority_grading_status": "graded",
        })

        issues = _full_row_issues(row, seed=42)

        self.assertIn("self_consistency_counts_invalid", issues)

    def test_full_row_validator_rejects_consistency_counts_above_sample_limits(self):
        row = result_row("gsm8k", 0)
        row["self_consistency"]["sample_answer_parseable_count"] = 9

        issues = _full_row_issues(row, seed=42)

        self.assertIn("self_consistency_counts_invalid", issues)

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

    def test_full_run_withholds_recommendation_when_a_source_scorable_label_is_unknown(self):
        records = []
        for dataset in DATASETS:
            for index in range(20):
                row = result_row(dataset, index % 4)
                row["example_id"] = f"{dataset}:fixture:full:{index:02d}"
                for sample in row["samples"]:
                    sample["seed"] = derive_sample_seed(42, row["example_id"], sample["index"])
                records.append(row)
        records[0]["greedy"]["correct"] = None
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_run(run_dir, records=records)

            summary = run_report(run_dir)

            self.assertTrue(summary["run_complete"])
            self.assertEqual(summary["recommendation"]["status"], "insufficient")
            self.assertIsNone(summary["recommendation"]["signal"])
            self.assertIn("label coverage", summary["recommendation"]["reason"])
            self.assertAlmostEqual(
                summary["by_dataset"]["gsm8k"]["outcome_label_coverage"], 0.95
            )
            self.assertFalse(summary["recommendation_coverage_gate"]["valid"])
            report = (run_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("Outcome label coverage", report)
            self.assertIn("95.0%", report)

    def test_full_run_withholds_recommendation_when_source_scorable_fraction_is_low(self):
        records = []
        for dataset in DATASETS:
            for index in range(20):
                row = result_row(dataset, index % 4)
                row["example_id"] = f"{dataset}:fixture:full:{index:02d}"
                for sample in row["samples"]:
                    sample["seed"] = derive_sample_seed(42, row["example_id"], sample["index"])
                records.append(row)
        records[0]["source"]["gold_status"] = "unanswerable"
        records[0]["greedy"]["correct"] = None
        records[1]["source"]["gold_status"] = "unanswerable"
        records[1]["greedy"]["correct"] = None
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_run(run_dir, records=records)

            summary = run_report(run_dir)

            self.assertTrue(summary["run_complete"])
            self.assertEqual(summary["recommendation"]["status"], "insufficient")
            self.assertIsNone(summary["recommendation"]["signal"])
            self.assertIn("source coverage", summary["recommendation"]["reason"])
            self.assertAlmostEqual(
                summary["by_dataset"]["gsm8k"]["source_scorable_fraction"], 0.9
            )
            self.assertAlmostEqual(
                summary["by_dataset"]["gsm8k"]["outcome_label_coverage"], 1.0
            )

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
            self.make_run(
                run_dir, records=[result_row("gsm8k", 0), result_row("gsm8k", 1)],
                study_scope="full",
            )

            summary = run_report(run_dir)

            self.assertFalse(summary["run_complete"])
            self.assertIsNone(summary["recommendation"]["signal"])
            self.assertEqual(summary["missing_datasets"], ["gsm_plus", "math"])
            self.assertEqual(set(summary["by_dataset"]), {"gsm8k", "math", "gsm_plus"})
            self.assertEqual(summary["by_dataset"]["math"]["total_rows"], 0)
            self.assertIsNone(summary["macro"]["entropy_confidence"]["macro_auroc"])

    def test_report_separates_imported_a100_rows_from_local_rtx4060_metrics(self):
        records = []
        for dataset in DATASETS:
            for index in range(2):
                row = result_row(dataset, index)
                row["execution_origin"] = (
                    "a100-import" if dataset == "gsm8k" and index == 0 else "rtx4060-local"
                )
                records.append(row)
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_run(run_dir, records=records)

            summary = run_report(run_dir)

            self.assertEqual(summary.get("execution_label"), "mixed A100 / RTX 4060")
            self.assertEqual(summary["by_execution"]["a100-import"]["row_count"], 1)
            self.assertEqual(summary["by_execution"]["rtx4060-local"]["row_count"], 5)
            self.assertEqual(
                summary["by_execution"]["rtx4060-local"]["by_dataset"]["gsm8k"], 1
            )
            self.assertTrue((run_dir / "metrics_by_execution.csv").exists())
            with (run_dir / "metrics_by_execution.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                grouped = list(csv.DictReader(handle))
            a100_gsm8k = next(
                row for row in grouped
                if row["execution_origin"] == "a100-import"
                and row["analysis"] == "greedy"
                and row["dataset"] == "gsm8k"
            )
            local_gsm8k = next(
                row for row in grouped
                if row["execution_origin"] == "rtx4060-local"
                and row["analysis"] == "greedy"
                and row["dataset"] == "gsm8k"
            )
            self.assertEqual(a100_gsm8k["n"], "1")
            self.assertEqual(local_gsm8k["n"], "1")
            report = (run_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("mixed A100 / RTX 4060", report)

    def test_full_report_rejects_complete_rows_with_the_wrong_origin_dataset_counts(self):
        records = []
        for dataset in DATASETS:
            for index in range(5):
                row = result_row(dataset, index)
                row["execution_origin"] = (
                    "a100-import" if dataset == "gsm8k" and index == 0 else "rtx4060-local"
                )
                records.append(row)
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.make_run(run_dir, records=records, profile="full", study_scope="full")

            summary = run_report(run_dir)

            validation = summary.get("origin_count_validation", {})
            actual_a100 = validation.get("actual", {}).get("a100-import", {})
            expected_a100 = validation.get("expected", {}).get("a100-import", {})
            self.assertFalse(summary["run_complete"])
            self.assertFalse(validation.get("valid", True))
            self.assertEqual(actual_a100.get("gsm8k"), 1)
            self.assertEqual(expected_a100.get("gsm8k"), 110)
            self.assertIsNone(summary["recommendation"]["signal"])
            markdown = (run_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("Validation: **failed**", markdown)
            self.assertIn("| a100-import | gsm8k | 110 | 1 |", markdown)

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
        records = [result_row(dataset, index) for dataset in DATASETS for index in range(4)]
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

    def test_full_report_rejects_malformed_greedy_or_sample_records(self):
        malformed_rows = (
            (
                "greedy_score_alignment_invalid",
                lambda row: row["greedy"]["token_entropies_nats"].pop(),
            ),
            ("sample_count_invalid", lambda row: row["samples"].pop()),
            (
                "sample_indices_invalid",
                lambda row: row["samples"][1].update(index=0),
            ),
            (
                "sample_seed_invalid",
                lambda row: row["samples"][0].update(seed=123),
            ),
        )
        for expected_issue, corrupt in malformed_rows:
            with self.subTest(expected_issue=expected_issue), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary) / "run"
                records = [result_row(dataset, 0) for dataset in DATASETS]
                corrupt(records[0])
                self.make_run(run_dir, records=records, profile="full")
                manifest_path = run_dir / "run_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["backend"] = "transformers-local"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

                summary = run_report(run_dir)

                self.assertFalse(summary["run_complete"])
                self.assertEqual(summary["malformed_row_count"], 1)
                self.assertIn(expected_issue, " ".join(summary["malformed_rows"][0]["issues"]))


if __name__ == "__main__":
    unittest.main()
