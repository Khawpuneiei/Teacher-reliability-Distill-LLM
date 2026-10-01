"""Archive import validates provenance and row content without extracting files."""

import hashlib
import importlib
import importlib.util
import json
import math
import shutil
import tempfile
import unittest
import zipfile
from dataclasses import asdict
from pathlib import Path

from teacher_reliability.data import Example
from teacher_reliability.run import _canonical_digest
from teacher_reliability.selection import RunProfile
from teacher_reliability.teacher import MODEL_REPOSITORY, derive_sample_seed


MODEL_REVISION = "a" * 40
DATASET_REVISIONS = {
    "openai/gsm8k": "1" * 40,
    "EleutherAI/hendrycks_math": "2" * 40,
    "qintongli/GSM-Plus": "3" * 40,
}
PROFILE = RunProfile("full", None, 50, None, False, "test", 8, 1024)


def _fixture_example():
    return Example(
        example_id="gsm8k:main:test:7",
        dataset="gsm8k",
        source_revision=DATASET_REVISIONS["openai/gsm8k"],
        source_config="main",
        split="test",
        source_row_id="7",
        question="What is 9 plus 9?",
        gold_answer="18",
        gold_solution="9 + 9 = 18. #### 18",
    )


def _prediction(example, sample_count=8):
    greedy = {
        "token_ids": [10, 11, 12],
        "token_entropies_nats": [0.4, 0.5, 0.6],
        "token_logprobs": [-0.3, -0.4, -0.5],
        "content_token_mask": [True, True, True],
        "token_char_spans": [[0, 1], [1, 2], [2, 3]],
        "token_alignment_status": "aligned",
        "text": r"\boxed{18}",
        "eos_generated": True,
        "was_truncated": False,
        "termination_reason": "eos",
        "correct": True,
        "predicted_answer": "18",
        "answer_method": "boxed",
        "prediction_parseable": True,
        "gold_parseable": True,
        "grading_status": "graded",
        "content_token_count": 3,
        "mean_entropy_nats": 0.5,
        "entropy_confidence": 1 - 0.5 / math.log(152064),
        "sequence_logprob_sum": -1.2,
        "sequence_geometric_probability": math.exp(-0.4),
        "answer_token_count": 0,
        "answer_token_geometric_probability": None,
        "answer_alignment_status": "no_answer_tokens",
    }
    samples = [
        {
            "index": index,
            "seed": derive_sample_seed(42, example.example_id, index),
            "token_ids": [13, 14],
            "text": r"\boxed{18}",
            "answer": "18",
            "answer_method": "boxed",
            "eos_generated": True,
            "was_truncated": False,
            "termination_reason": "eos",
        }
        for index in range(sample_count)
    ]
    return {
        "example_id": example.example_id,
        "status": "complete",
        "question": example.question,
        "gold_answer": example.gold_answer,
        "requested_sample_count": 8,
        "source": {
            "dataset": example.dataset,
            "config": example.source_config,
            "split": example.split,
            "row_id": example.source_row_id,
            "revision": example.source_revision,
            "gold_status": example.gold_status,
            "subject": example.subject,
            "perturbation_type": example.perturbation_type,
            "seed_question_id": example.seed_question_id,
        },
        "greedy": greedy,
        "samples": samples,
        "self_consistency": {
            "sample_count": 8,
            "sample_answer_parseable_count": 8,
            "sample_answer_parse_unknown_count": 0,
            "greedy_agreement_count": 8,
            "greedy_agreement_unknown_count": 0,
            "greedy_agreement_share": 1.0,
            "majority_answer": "18",
            "majority_vote_count": 8,
            "majority_vote_share": 1.0,
            "majority_correct": True,
            "majority_grading_status": "graded",
            "majority_comparison_unknown_count": 0,
        },
    }


def _archive(
    path,
    examples,
    *,
    sample_count=8,
    unsafe=False,
    runtime_claim="default",
    omit_runtime=False,
    mutate_row=None,
):
    ids = [example.example_id for example in examples]
    hardware = {"gpu_name": "NVIDIA A100 80GB PCIe", "platform": "Linux"}
    identity = {
        "profile": PROFILE.as_dict(),
        "base_seed": 42,
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "quantization": {
                "load_in_4bit": True,
                "quant_type": "nf4",
                "double_quant": True,
                "compute_dtype": "float16",
            },
        },
        "datasets": DATASET_REVISIONS,
        "selected_ids_sha256": _canonical_digest(ids),
        "selected_examples_sha256": _canonical_digest([asdict(row) for row in examples]),
    }
    if not omit_runtime:
        identity["runtime"] = hardware if runtime_claim == "default" else runtime_claim
    manifest = {
        "run_id": "old-a100-run",
        "status": "running",
        "profile": PROFILE.as_dict(),
        "seed": 42,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "dataset_revisions": DATASET_REVISIONS,
        "selected_example_ids": ids,
        "selected_example_count": len(ids),
        "identity": identity,
        "run_signature": _canonical_digest(identity),
        "hardware": hardware,
    }
    rows = [_prediction(example, sample_count) for example in examples]
    if mutate_row is not None:
        mutate_row(rows[0])
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if unsafe:
            archive.writestr("../run_manifest.json", json.dumps(manifest))
            archive.writestr("old-a100-run/predictions.jsonl", "\n")
        else:
            archive.writestr("old-a100-run/run_manifest.json", json.dumps(manifest))
            archive.writestr(
                "old-a100-run/predictions.jsonl",
                "".join(json.dumps(row) + "\n" for row in rows),
            )
    return rows


class ArchiveImportTests(unittest.TestCase):
    def _importer(self):
        spec = importlib.util.find_spec("teacher_reliability.archive_import")
        self.assertIsNotNone(
            spec,
            "local runs need a provenance-checking importer for the archived A100 rows",
        )
        return importlib.import_module("teacher_reliability.archive_import").import_a100_run

    def test_import_preserves_complete_rows_and_binds_archive_hash_to_a100_provenance(self):
        import_a100_run = self._importer()
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "a100.zip"
            rows = _archive(archive_path, [_fixture_example()])

            imported = import_a100_run(
                archive_path,
                examples=[_fixture_example()],
                profile=PROFILE,
                seed=42,
                model_revision=MODEL_REVISION,
                dataset_revisions=DATASET_REVISIONS,
            )

            self.assertEqual(imported.rows, rows)
            self.assertEqual(
                imported.metadata["archive_sha256"],
                hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(imported.metadata["source_run_id"], "old-a100-run")
            self.assertEqual(imported.metadata["source_gpu"], "NVIDIA A100 80GB PCIe")
            imported.close()

    def test_import_rejects_a_different_valid_snapshot_when_expected_sha_is_pinned(self):
        import_a100_run = self._importer()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected_path = root / "expected.zip"
            alternate_path = root / "alternate.zip"
            rows = _archive(expected_path, [_fixture_example()])
            expected_sha = hashlib.sha256(expected_path.read_bytes()).hexdigest()
            with zipfile.ZipFile(expected_path) as source, zipfile.ZipFile(
                alternate_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as alternate:
                for member in source.infolist():
                    alternate.writestr(member.filename, source.read(member))
                alternate.writestr("old-a100-run/runner.log", "different valid archive")

            accepted_without_pin = import_a100_run(
                alternate_path, examples=[_fixture_example()], profile=PROFILE,
                seed=42, model_revision=MODEL_REVISION,
                dataset_revisions=DATASET_REVISIONS,
            )
            self.assertEqual(accepted_without_pin.rows, rows)
            accepted_without_pin.close()

            try:
                with self.assertRaisesRegex(ValueError, "pinned A100 archive SHA-256"):
                    import_a100_run(
                        alternate_path, examples=[_fixture_example()], profile=PROFILE,
                        seed=42, model_revision=MODEL_REVISION,
                        dataset_revisions=DATASET_REVISIONS,
                        expected_archive_sha256=expected_sha,
                    )
            except TypeError as exc:
                self.fail(f"full import needs an expected archive digest: {exc}")

    def test_import_marks_missing_legacy_unknown_counts_unavailable(self):
        import_a100_run = self._importer()
        unavailable_fields = (
            "sample_answer_parse_unknown_count",
            "greedy_agreement_unknown_count",
            "majority_comparison_unknown_count",
        )

        def make_legacy(row):
            for field in unavailable_fields:
                del row["self_consistency"][field]

        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "legacy-a100.zip"
            _archive(archive_path, [_fixture_example()], mutate_row=make_legacy)
            with zipfile.ZipFile(archive_path) as archive:
                raw_predictions = archive.read("old-a100-run/predictions.jsonl")

            try:
                imported = import_a100_run(
                    archive_path,
                    examples=[_fixture_example()],
                    profile=PROFILE,
                    seed=42,
                    model_revision=MODEL_REVISION,
                    dataset_revisions=DATASET_REVISIONS,
                )
            except ValueError as exc:
                self.fail(f"legacy unknown counters should import as unavailable: {exc}")

            consistency = imported.rows[0]["self_consistency"]
            for field in unavailable_fields:
                self.assertIsNone(consistency[field])
            self.assertEqual(consistency["sample_answer_parseable_count"], 8)
            self.assertEqual(consistency["greedy_agreement_count"], 8)
            self.assertEqual(consistency["majority_vote_count"], 8)
            self.assertEqual(
                imported.metadata["predictions_sha256"],
                hashlib.sha256(raw_predictions).hexdigest(),
            )
            imported.close()

    def test_import_rejects_partial_samples_and_wrong_local_question_content(self):
        import_a100_run = self._importer()
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "partial.zip"
            _archive(archive_path, [_fixture_example()], sample_count=7)
            with self.assertRaisesRegex(ValueError, "sample"):
                import_a100_run(
                    archive_path,
                    examples=[_fixture_example()],
                    profile=PROFILE,
                    seed=42,
                    model_revision=MODEL_REVISION,
                    dataset_revisions=DATASET_REVISIONS,
                )

    def test_import_rejects_greedy_output_missing_termination_metadata(self):
        import_a100_run = self._importer()

        def remove_termination(record):
            record["greedy"].pop("eos_generated")

        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "missing-termination.zip"
            _archive(archive_path, [_fixture_example()], mutate_row=remove_termination)

            with self.assertRaisesRegex(ValueError, "greedy"):
                import_a100_run(
                    archive_path,
                    examples=[_fixture_example()],
                    profile=PROFILE,
                    seed=42,
                    model_revision=MODEL_REVISION,
                    dataset_revisions=DATASET_REVISIONS,
                )

    def test_import_rejects_conflicting_greedy_termination_flags(self):
        import_a100_run = self._importer()

        def contradict_termination(record):
            record["greedy"]["was_truncated"] = True

        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "conflicting-termination.zip"
            _archive(archive_path, [_fixture_example()], mutate_row=contradict_termination)

            with self.assertRaisesRegex(ValueError, "greedy.*termination"):
                import_a100_run(
                    archive_path,
                    examples=[_fixture_example()],
                    profile=PROFILE,
                    seed=42,
                    model_revision=MODEL_REVISION,
                    dataset_revisions=DATASET_REVISIONS,
                )

    def test_import_rejects_non_text_greedy_termination_reason(self):
        import_a100_run = self._importer()

        def corrupt_termination_reason(record):
            record["greedy"]["termination_reason"] = ["eos"]

        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "invalid-termination-reason.zip"
            _archive(archive_path, [_fixture_example()], mutate_row=corrupt_termination_reason)

            with self.assertRaisesRegex(ValueError, "greedy.*termination"):
                import_a100_run(
                    archive_path,
                    examples=[_fixture_example()],
                    profile=PROFILE,
                    seed=42,
                    model_revision=MODEL_REVISION,
                    dataset_revisions=DATASET_REVISIONS,
                )

            valid_path = Path(temporary) / "valid.zip"
            _archive(valid_path, [_fixture_example()])
            changed = Example(
                **{**asdict(_fixture_example()), "question": "Altered input"}
            )
            with self.assertRaisesRegex(ValueError, "input|selection|question"):
                import_a100_run(
                    valid_path,
                    examples=[changed],
                    profile=PROFILE,
                    seed=42,
                    model_revision=MODEL_REVISION,
                    dataset_revisions=DATASET_REVISIONS,
                )

    def test_import_rejects_zip_path_traversal_without_extracting_members(self):
        import_a100_run = self._importer()
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "unsafe.zip"
            _archive(archive_path, [_fixture_example()], unsafe=True)
            with self.assertRaisesRegex(ValueError, "unsafe|path"):
                import_a100_run(
                    archive_path,
                    examples=[_fixture_example()],
                    profile=PROFILE,
                    seed=42,
                    model_revision=MODEL_REVISION,
                    dataset_revisions=DATASET_REVISIONS,
                )

    def test_import_requires_signed_runtime_to_confirm_the_top_level_a100_hardware(self):
        import_a100_run = self._importer()
        invalid_claims = (
            {"omit_runtime": True},
            {"runtime_claim": {"gpu_name": "NVIDIA RTX 4060 Laptop GPU", "platform": "Linux"}},
            {"runtime_claim": {"gpu_name": "NVIDIA A100 40GB PCIe", "platform": "Linux"}},
        )
        with tempfile.TemporaryDirectory() as temporary:
            for index, claim in enumerate(invalid_claims):
                archive_path = Path(temporary) / f"runtime-{index}.zip"
                _archive(archive_path, [_fixture_example()], **claim)
                with self.subTest(claim=claim), self.assertRaisesRegex(
                    ValueError, "runtime|hardware"
                ):
                    import_a100_run(
                        archive_path,
                        examples=[_fixture_example()],
                        profile=PROFILE,
                        seed=42,
                        model_revision=MODEL_REVISION,
                        dataset_revisions=DATASET_REVISIONS,
                    )

    def test_import_rejects_derived_fields_inconsistent_with_raw_generations(self):
        import_a100_run = self._importer()
        corruptions = (
            ("correctness", lambda row: row["greedy"].update(correct=False)),
            ("confidence", lambda row: row["greedy"].update(entropy_confidence=0.5)),
            (
                "self-consistency",
                lambda row: row["self_consistency"].update(greedy_agreement_count=7),
            ),
            ("sample answer/text", lambda row: row["samples"][0].update(answer="19")),
            ("sample token", lambda row: row["samples"][0].update(token_ids=[152064])),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for index, (label, corrupt) in enumerate(corruptions):
                archive_path = Path(temporary) / f"semantic-{index}.zip"
                _archive(archive_path, [_fixture_example()], mutate_row=corrupt)
                with self.subTest(label=label), self.assertRaises(ValueError):
                    import_a100_run(
                        archive_path,
                        examples=[_fixture_example()],
                        profile=PROFILE,
                        seed=42,
                        model_revision=MODEL_REVISION,
                        dataset_revisions=DATASET_REVISIONS,
                    )

    def test_import_snapshot_stays_bound_to_validated_bytes_after_source_path_replacement(self):
        import_a100_run = self._importer()
        from teacher_reliability.local_run import _archive_copy

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "source.zip"
            _archive(archive_path, [_fixture_example()])
            validated_bytes = archive_path.read_bytes()
            imported = import_a100_run(
                archive_path,
                examples=[_fixture_example()],
                profile=PROFILE,
                seed=42,
                model_revision=MODEL_REVISION,
                dataset_revisions=DATASET_REVISIONS,
            )
            replacement_path = root / "replacement.zip"
            _archive(replacement_path, [_fixture_example()])
            replacement_bytes = replacement_path.read_bytes() + b"replacement"
            replacement_path.write_bytes(replacement_bytes)
            shutil.copyfile(replacement_path, archive_path)

            snapshot = getattr(imported, "archive_snapshot", None)
            self.assertIsNotNone(snapshot, "validated archive bytes must remain available for preservation")
            self.assertEqual(
                imported.metadata["archive_sha256"], hashlib.sha256(validated_bytes).hexdigest()
            )
            preserved = root / "preserved.zip"
            _archive_copy(snapshot, preserved, imported.metadata["archive_sha256"])

            self.assertNotEqual(archive_path.read_bytes(), validated_bytes)
            self.assertEqual(preserved.read_bytes(), validated_bytes)
            self.assertEqual(imported.rows[0]["greedy"]["text"], r"\boxed{18}")
            imported.close()


if __name__ == "__main__":
    unittest.main()
