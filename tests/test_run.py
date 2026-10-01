import importlib.metadata
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from teacher_reliability.data import normalize_example
from teacher_reliability.selection import RunProfile
from teacher_reliability.teacher import GreedyGeneration, SampleGeneration
from teacher_reliability.run import run_experiment


def fixture_example():
    return normalize_example(
        "gsm8k",
        "main",
        "test",
        17,
        {"question": "How many apples remain?", "answer": "#### 12"},
        source_revision="1" * 40,
    )


class FakeTokenizer:
    vocab_size = 4


class RunPipelineTests(unittest.TestCase):
    def setUp(self):
        self.profile = RunProfile(
            "fixture", 1, 1, 1, True, "test", 2, 32
        )
        self.calls = {"greedy": 0, "sample": []}

        def greedy(_tokenizer, _model, question, _max_new_tokens):
            self.assertEqual(question, "How many apples remain?")
            self.calls["greedy"] += 1
            return GreedyGeneration(
                text=r"\boxed{12}",
                token_ids=[7, 8],
                token_entropies_nats=[0.4, 0.2],
                token_logprobs=[-0.3, -0.2],
                token_char_spans=[(7, 8), (8, 9)],
                content_token_mask=[True, True],
                eos_generated=True,
                was_truncated=False,
                termination_reason="eos",
            )

        def sample(_tokenizer, _model, question, _max_new_tokens, seed):
            self.assertEqual(question, "How many apples remain?")
            self.calls["sample"].append(seed)
            text = r"\boxed{12}" if len(self.calls["sample"]) % 2 else r"\boxed{13}"
            return SampleGeneration(text, [7, 8], False, True, "max_new_tokens")

        self.greedy = greedy
        self.sample = sample

    def run_with(self, run_dir, seed=5, resume=False, model_revision="b" * 40, examples=None):
        return run_experiment(
            run_dir=run_dir,
            profile=self.profile,
            seed=seed,
            examples=[fixture_example()] if examples is None else examples,
            model_revision=model_revision,
            dataset_revisions={
                "openai/gsm8k": "1" * 40,
                "EleutherAI/hendrycks_math": "2" * 40,
                "qintongli/GSM-Plus": "3" * 40,
            },
            tokenizer=FakeTokenizer(),
            model=object(),
            greedy_generator=self.greedy,
            sample_generator=self.sample,
            resume=resume,
        )

    def test_run_writes_auditable_predictions_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"

            manifest = self.run_with(run_dir)

            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["profile"]["sample_count"], 2)
            self.assertEqual(manifest["software"]["hf_xet"], "1.6.0")
            self.assertEqual(self.calls["greedy"], 1)
            self.assertEqual(len(self.calls["sample"]), 2)
            record = json.loads((run_dir / "predictions.jsonl").read_text(encoding="utf-8"))
            self.assertTrue(record["greedy"]["correct"])
            self.assertTrue(record["greedy"]["eos_generated"])
            self.assertFalse(record["greedy"]["was_truncated"])
            self.assertAlmostEqual(record["greedy"]["mean_entropy_nats"], 0.3)
            self.assertAlmostEqual(record["self_consistency"]["greedy_agreement_share"], 0.5)
            self.assertEqual(record["self_consistency"]["sample_answer_parseable_count"], 2)
            self.assertTrue(record["samples"][1]["was_truncated"])
            self.assertNotIn("gold_solution", record)

    def test_run_persists_unknown_math_verify_checks_separately(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "teacher_reliability.consistency.check_answer_parseability",
            return_value=None,
        ):
            run_dir = Path(temporary) / "run"
            self.run_with(run_dir)

            record = json.loads((run_dir / "predictions.jsonl").read_text(encoding="utf-8"))
            summary = record["self_consistency"]
            self.assertEqual(summary.get("sample_answer_parse_unknown_count", 0), 2)

    def test_resume_skips_complete_rows_without_duplicate_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.run_with(run_dir)

            manifest = self.run_with(run_dir, resume=True)

            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(self.calls["greedy"], 1)
            rows = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1)

    def test_resume_refuses_a_different_seed_or_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.run_with(run_dir)

            with self.assertRaisesRegex(ValueError, "configuration differs"):
                self.run_with(run_dir, seed=6, resume=True)

    def test_resume_rejects_a_changed_tokenizer_package_without_changing_checkpoints(self):
        actual_version = importlib.metadata.version
        tokenizer_version = "0.21.3"

        def version(package):
            return tokenizer_version if package == "tokenizers" else actual_version(package)

        with tempfile.TemporaryDirectory() as temporary, patch(
            "teacher_reliability.run.importlib.metadata.version", side_effect=version
        ):
            run_dir = Path(temporary) / "run"
            self.run_with(run_dir)
            self.run_with(run_dir, resume=True)
            prediction_before = (run_dir / "predictions.jsonl").read_bytes()
            manifest_before = (run_dir / "run_manifest.json").read_bytes()

            tokenizer_version = "0.21.4"
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                self.run_with(run_dir, resume=True)

            self.assertEqual((run_dir / "predictions.jsonl").read_bytes(), prediction_before)
            self.assertEqual((run_dir / "run_manifest.json").read_bytes(), manifest_before)
            self.assertEqual(self.calls["greedy"], 1)

    def test_run_rejects_a_revision_that_is_not_a_hex_commit_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "commit SHA"):
                self.run_with(Path(temporary) / "run", model_revision="z" * 40)

    @staticmethod
    def hub_fixture(repository, config, split, revision):
        if repository == "EleutherAI/hendrycks_math":
            return [{"problem": "How many apples remain?", "solution": r"\boxed{12}"}]
        if repository == "openai/gsm8k":
            return [{"question": "How many apples remain?", "answer": "#### 12"}]
        return [{"question": "How many apples remain?", "answer": "12"}]

    def test_explicit_dataset_revisions_are_used_for_loading(self):
        revisions = {
            "openai/gsm8k": "1" * 40,
            "EleutherAI/hendrycks_math": "2" * 40,
            "qintongli/GSM-Plus": "3" * 40,
        }
        with tempfile.TemporaryDirectory() as temporary, patch(
            "teacher_reliability.hub_data._resolve_dataset_revision", return_value="4" * 40
        ), patch(
            "teacher_reliability.hub_data._load_huggingface_split", self.hub_fixture
        ):
            run_dir = Path(temporary) / "run"
            manifest = run_experiment(
                run_dir, self.profile, model_revision="b" * 40,
                dataset_revisions=revisions, tokenizer=FakeTokenizer(), model=object(),
                greedy_generator=self.greedy, sample_generator=self.sample,
            )

            records = [
                json.loads(line)
                for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            record = next(row for row in records if row["source"]["dataset"] == "gsm8k")
            self.assertEqual(manifest["dataset_revisions"], revisions)
            self.assertEqual(record["source"]["revision"], "1" * 40)

    def test_resume_reuses_recorded_commits_after_hub_main_advances(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "teacher_reliability.hub_data._resolve_dataset_revision", return_value="4" * 40
        ), patch(
            "teacher_reliability.hub_data._load_huggingface_split", self.hub_fixture
        ), patch(
            "teacher_reliability.run.resolve_model_revision", return_value="c" * 40
        ):
            run_dir = Path(temporary) / "run"
            original = run_experiment(
                run_dir, self.profile, model_revision="b" * 40,
                dataset_revisions={
                    "openai/gsm8k": "1" * 40,
                    "EleutherAI/hendrycks_math": "2" * 40,
                    "qintongli/GSM-Plus": "3" * 40,
                },
                tokenizer=FakeTokenizer(), model=object(),
                greedy_generator=self.greedy, sample_generator=self.sample,
            )
            before = (run_dir / "predictions.jsonl").read_bytes()
            try:
                resumed = run_experiment(run_dir, self.profile, resume=True)
            except ValueError as exc:
                self.fail(f"a pinned run must remain resumable after Hub drift: {exc}")

            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(resumed["model_revision"], "b" * 40)
            self.assertEqual(resumed["run_signature"], original["run_signature"])
            self.assertEqual((run_dir / "predictions.jsonl").read_bytes(), before)
            self.assertEqual(self.calls["greedy"], 9)

    def test_preloaded_examples_must_match_manifest_dataset_revisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            mismatched = replace(fixture_example(), source_revision="4" * 40)
            with self.assertRaisesRegex(ValueError, "source revision"):
                self.run_with(Path(temporary) / "run", examples=[mismatched])

    def test_duplicate_preloaded_ids_cannot_be_reported_as_a_complete_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "duplicate"):
                self.run_with(
                    Path(temporary) / "run", examples=[fixture_example(), fixture_example()]
                )

    def test_resume_rejects_changed_input_content_with_unchanged_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.run_with(run_dir)
            changed_gold = replace(fixture_example(), gold_answer="13")
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                self.run_with(run_dir, resume=True, examples=[changed_gold])

    def test_resume_cannot_silently_mix_platforms_in_one_manifest(self):
        hardware = {
            "python": "3.10.4", "platform": "Windows-11",
            "torch_cuda_runtime": "12.1", "cuda_available": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            with patch("teacher_reliability.run._hardware_info", return_value=hardware):
                self.run_with(run_dir)
            linux_hardware = {**hardware, "platform": "Linux"}
            with patch("teacher_reliability.run._hardware_info", return_value=linux_hardware):
                with self.assertRaisesRegex(ValueError, "configuration differs"):
                    self.run_with(run_dir, resume=True)


if __name__ == "__main__":
    unittest.main()
