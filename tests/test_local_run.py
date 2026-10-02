"""Checkpoint, import, deadline and retry contracts for the local coordinator."""

import importlib.util
import hashlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from teacher_reliability.data import Example, normalize_example
from teacher_reliability.hybrid import CudaOutOfMemory, MemoryProfile
from teacher_reliability.selection import RunProfile, get_profile
from teacher_reliability.teacher import GreedyGeneration, SampleGeneration, derive_sample_seed

def run_local(*args, **kwargs):
    spec = importlib.util.find_spec("teacher_reliability.local_run")
    if spec is None:
        raise AssertionError("local coordinator module is missing")
    module = __import__("teacher_reliability.local_run", fromlist=["run_local"])
    return module.run_local(*args, **kwargs)


class TinyTokenizer:
    vocab_size = 151643

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return [1, 2, 3]


class LocalRunTests(unittest.TestCase):
    def setUp(self):
        self.examples = [
            normalize_example(
                "gsm8k", "main", "test", i,
                {"question": f"How many apples {i}?", "answer": "#### 12"},
                source_revision="1" * 40,
            )
            for i in range(3)
        ]
        self.profile = RunProfile("fixture", 3, 1, 1, True, "test", 2, 8)
        self.revisions = {
            "openai/gsm8k": "1" * 40,
            "EleutherAI/hendrycks_math": "2" * 40,
            "qintongli/GSM-Plus": "3" * 40,
        }
        self.now_value = datetime(2026, 10, 1, 0, tzinfo=timezone.utc)
        self.greedy_calls = []
        self.sample_calls = []

    def test_progress_updates_without_memory_keep_the_last_gpu_snapshot(self):
        module = __import__("teacher_reliability.local_run", fromlist=["_save_progress"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            manifest = {
                "run_id": "progress-test", "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_example_count": 0, "imported_example_count": 0,
                "failed_example_count": 0, "pending_example_count": 1,
            }
            expected_memory = {"free_mib": 900, "allocated_mib": 5200}
            module._save_progress(
                path, manifest, active_request="row-1", generated_tokens=100,
                memory=expected_memory,
            )

            progress = module._save_progress(
                path, manifest, active_request="row-1", generated_tokens=101,
            )

        self.assertEqual(progress["memory"], expected_memory)

    def greedy(self, batch, _profile, *, on_result=None):
        result = {}
        for row in batch:
            self.greedy_calls.append(row["request_id"])
            value = GreedyGeneration(
                r"\\boxed{12}", [7, 8], [0.4, 0.2], [-0.3, -0.2],
                [(0, 1), (1, 2)], [True, True], eos_generated=True,
                termination_reason="eos",
            )
            result[row["request_id"]] = value
            if on_result:
                on_result(row["request_id"], value)
        return result

    def sample(self, batch, _profile, *, on_result=None):
        result = {}
        for row in reversed(batch):
            self.sample_calls.append((row["request_id"], row["seed"]))
            value = SampleGeneration(r"\\boxed{12}", [7, 8], True, False, "eos")
            result[row["request_id"]] = value
            if on_result:
                on_result(row["request_id"], value)
        return result

    def invoke(self, run_dir, **kwargs):
        profile = kwargs.pop("profile", self.profile)
        settings = dict(
            seed=5, examples=self.examples,
            model_revision="b" * 40, dataset_revisions=self.revisions,
            tokenizer=TinyTokenizer(), model_vocab_size=152064,
            greedy_execute=self.greedy, sample_execute=self.sample,
            now=lambda: self.now_value,
        )
        settings.update(kwargs)
        return run_local(run_dir, profile, **settings)


    def test_run_records_vocab_and_local_origin_and_resumes_without_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "local"
            result = self.invoke(run_dir)
            self.assertEqual(result["backend"], "transformers-local")
            self.assertIsNone(result["deadline_utc"])
            self.assertEqual(result["status"], "complete")
            rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual({row["execution_origin"] for row in rows}, {"rtx4060-local"})
            self.assertEqual(rows[0]["greedy"]["entropy_confidence"], 1.0 - sum([0.4, 0.2]) / (2 * __import__("math").log(152064)))
            self.assertEqual(len(self.greedy_calls), 3)
            self.assertEqual(len(self.sample_calls), 6)
            resumed = self.invoke(run_dir, resume=True)
            self.assertEqual(resumed["completed_example_count"], 3)
            self.assertEqual(len((run_dir / "predictions.jsonl").read_text().splitlines()), 3)
            self.assertEqual(len(self.greedy_calls), 3)

    def test_resume_rejects_tampered_local_question_before_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "local"
            self.invoke(run_dir)
            predictions_path = run_dir / "predictions.jsonl"
            rows = [json.loads(line) for line in predictions_path.read_text().splitlines()]
            rows[0]["question"] = "A different but well-formed question?"
            predictions_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            altered_checkpoint = predictions_path.read_bytes()
            self.greedy_calls.clear()
            self.sample_calls.clear()

            with self.assertRaisesRegex(ValueError, "checkpoint|question|source"):
                self.invoke(run_dir, resume=True)

            self.assertEqual(predictions_path.read_bytes(), altered_checkpoint)
            self.assertEqual(self.greedy_calls, [])
            self.assertEqual(self.sample_calls, [])

    def test_resume_rejects_local_generation_mutations_against_stages(self):
        mutations = {
            "greedy text": lambda row: row["greedy"].__setitem__("text", "a different generated answer"),
            "same-length score value": lambda row: row["greedy"]["token_entropies_nats"].__setitem__(
                0, row["greedy"]["token_entropies_nats"][0] + 0.125
            ),
            "derived confidence value": lambda row: row["greedy"].__setitem__(
                "entropy_confidence", 0.0
            ),
            "valid grading label": lambda row: row["greedy"].__setitem__(
                "correct", not row["greedy"]["correct"]
            ),
            "valid self-consistency share": lambda row: row["self_consistency"].__setitem__(
                "majority_vote_share", 0.0
            ),
            "sample text": lambda row: row["samples"][0].__setitem__("text", "a different sampled answer"),
            "sample index with matching seed": lambda row: self._swap_sample_indices(row),
            "sample seed": lambda row: row["samples"][0].__setitem__(
                "seed", row["samples"][0]["seed"] + 1
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "local"
            self.invoke(run_dir)
            predictions_path = run_dir / "predictions.jsonl"
            original_checkpoint = predictions_path.read_bytes()

            for description, mutate in mutations.items():
                with self.subTest(mutation=description):
                    rows = [json.loads(line) for line in original_checkpoint.splitlines()]
                    mutate(rows[0])
                    altered_checkpoint = b"".join(
                        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                        for row in rows
                    )
                    predictions_path.write_bytes(altered_checkpoint)
                    self.greedy_calls.clear()
                    self.sample_calls.clear()

                    with self.assertRaisesRegex(ValueError, "checkpoint|sample"):
                        self.invoke(run_dir, resume=True)

                    self.assertEqual(predictions_path.read_bytes(), altered_checkpoint)
                    self.assertEqual(self.greedy_calls, [])
                    self.assertEqual(self.sample_calls, [])

            predictions_path.write_bytes(original_checkpoint)

    def _swap_sample_indices(self, row):
        first, second = row["samples"]
        first["index"], second["index"] = second["index"], first["index"]
        for sample in (first, second):
            sample["seed"] = derive_sample_seed(5, row["example_id"], sample["index"])

    def test_oom_keeps_completed_stages_retries_original_seed_and_reduces_setting(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "local"
            calls = []

            def sampling(batch, profile, *, on_result=None):
                calls.append((profile.batch_size, [(row["request_id"], row["seed"]) for row in batch]))
                if len(calls) == 1:
                    request = batch[0]
                    saved = SampleGeneration("sample", [9], False, False, "other_stop")
                    on_result(request["request_id"], saved)
                    raise CudaOutOfMemory("CUDA out of memory")
                return self.sample(batch, profile, on_result=on_result)

            result = self.invoke(run_dir, sample_execute=sampling)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["active_settings"]["sampling_batch_size"], 1)
            self.assertEqual(calls[0][0], 2)
            self.assertEqual(calls[1][0], 1)
            # The first result was checkpointed before OOM; only the unfinished
            # request retries, with its original id and seed.
            self.assertEqual(calls[0][1][1], calls[1][1][0])
            self.assertTrue(result["adaptations"])

    def test_greedy_and_sampling_cache_provenance_are_tracked_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "local"
            greedy_calls = 0

            def greedy_with_offloaded_fallback(batch, profile, *, on_result=None):
                nonlocal greedy_calls
                greedy_calls += 1
                if greedy_calls == 1:
                    raise CudaOutOfMemory("regular cache exhausted")
                return self.greedy(batch, profile, on_result=on_result)

            result = self.invoke(
                run_dir,
                profile=RunProfile("fixture-cache-provenance", 3, 1, 1, True, "test", 1, 8),
                greedy_execute=greedy_with_offloaded_fallback,
                greedy_profiles=[
                    MemoryProfile("greedy-1", 1),
                    MemoryProfile("greedy-1-offloaded", 1, cache_implementation="offloaded"),
                ],
                sample_profiles=[MemoryProfile("sample-1", 1)],
            )
            persisted = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

        for active_settings in (result["active_settings"], persisted["active_settings"]):
            self.assertIn("greedy_cache_implementation", active_settings)
            self.assertIn("sampling_cache_implementation", active_settings)
            self.assertEqual(active_settings["greedy_cache_implementation"], "offloaded")
            self.assertIsNone(active_settings["sampling_cache_implementation"])

    def test_host_memory_fallback_records_regular_cache_as_none(self):
        from teacher_reliability.hybrid_workers import WorkerExecutionError

        first = self.examples[0]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "local"

            def sampling_with_host_memory_fallback(batch, profile, *, on_result=None):
                if batch[0]["example_id"] == first.example_id:
                    raise WorkerExecutionError(
                        "InsufficientHostMemory", "offloaded KV cache needs more host RAM",
                    )
                return self.sample(batch, profile, on_result=on_result)

            result = self.invoke(
                run_dir,
                profile=RunProfile("fixture-cache-fallback", 3, 1, 1, True, "test", 1, 8),
                sample_execute=sampling_with_host_memory_fallback,
                sample_profiles=[
                    MemoryProfile("sample-1-offloaded", 1, cache_implementation="offloaded"),
                    MemoryProfile("sample-1", 1),
                ],
            )
            persisted = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(result["failed_example_count"], 1)
        for active_settings in (result["active_settings"], persisted["active_settings"]):
            self.assertIn("sampling_cache_implementation", active_settings)
            self.assertIsNone(active_settings["sampling_cache_implementation"])
            self.assertIsNone(active_settings["greedy_cache_implementation"])

    def test_resume_accepts_local_record_with_unknown_majority_comparisons(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "local"
            with patch("teacher_reliability.consistency.check_answers_equivalence", return_value=None):
                first = self.invoke(run_dir)
            record = json.loads((run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
            consistency = record["self_consistency"]
            self.assertGreaterEqual(consistency["majority_comparison_unknown_count"], 1)
            self.assertIsNone(consistency["majority_vote_count"])
            self.assertIsNone(consistency["majority_vote_share"])
            self.assertIsNone(consistency["majority_answer"])
            self.assertIsNone(consistency["majority_correct"])
            self.assertEqual(consistency["majority_grading_status"], "majority_comparison_incomplete")

            try:
                resumed = self.invoke(run_dir, resume=True)
            except ValueError as exc:
                self.fail(f"a valid incomplete-majority checkpoint must resume: {exc}")

        self.assertEqual(first["status"], "complete")
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(resumed["completed_example_count"], 3)

    def test_local_resume_count_validation_rejects_invalid_counts_and_combinations(self):
        module = __import__("teacher_reliability.local_run", fromlist=["_validate_local_checkpoint_record"])
        example = self.examples[0]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "local"
            self.invoke(run_dir)
            record = json.loads((run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
        record["self_consistency"] = {
            "sample_count": 2,
            "sample_answer_parseable_count": 2,
            "sample_answer_parse_unknown_count": 0,
            "greedy_agreement_count": 2,
            "greedy_agreement_unknown_count": 0,
            "greedy_agreement_share": 1.0,
            "majority_answer": "12",
            "majority_vote_count": 2,
            "majority_vote_share": 1.0,
            "majority_correct": True,
            "majority_grading_status": "correct",
            "majority_comparison_unknown_count": 0,
        }

        invalid_mutations = {
            "negative count": lambda consistency: consistency.__setitem__("sample_answer_parseable_count", -1),
            "non-integer count": lambda consistency: consistency.__setitem__("greedy_agreement_count", 1.5),
            "boolean count": lambda consistency: consistency.__setitem__("greedy_agreement_count", True),
            "count above sample total": lambda consistency: consistency.__setitem__("sample_answer_parseable_count", 3),
            "unknown comparisons above pair total": lambda consistency: consistency.__setitem__("majority_comparison_unknown_count", 2),
            "null majority without unknown comparisons": lambda consistency: consistency.__setitem__("majority_vote_count", None),
            "provisional majority exposed despite unknown comparison": lambda consistency: consistency.update({
                "majority_comparison_unknown_count": 1,
                "majority_answer": "12",
                "majority_vote_count": 1,
                "majority_vote_share": 0.5,
                "majority_correct": True,
                "majority_grading_status": "correct",
            }),
        }
        for description, mutate in invalid_mutations.items():
            with self.subTest(invalid=description):
                candidate = json.loads(json.dumps(record))
                mutate(candidate["self_consistency"])
                with self.assertRaisesRegex(ValueError, "self-consistency"):
                    module._validate_local_checkpoint_record(
                        candidate, example, seed=5, sample_count=2, max_new_tokens=8,
                    )

    def test_resume_rejects_corrupt_incomplete_stages_before_reusing_them(self):
        mutations = {
            "sample seed": lambda stage: stage["samples"]["0"].__setitem__(
                "seed", stage["samples"]["0"]["seed"] + 1
            ),
            "greedy score alignment": lambda stage: stage["greedy"]["token_logprobs"].pop(),
        }
        for description, mutate in mutations.items():
            with self.subTest(corruption=description), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary) / "local"

                def stop_after_one_sample(batch, _profile, *, on_result=None):
                    request = batch[0]
                    result = SampleGeneration("sample", [9], False, False, "other_stop")
                    on_result(request["request_id"], result)
                    raise RuntimeError("worker exited after saved stage")

                with self.assertRaisesRegex(RuntimeError, "after saved stage"):
                    self.invoke(run_dir, sample_execute=stop_after_one_sample)

                stage_path = next(
                    path for path in (run_dir / "stages").glob("*.json")
                    if json.loads(path.read_text(encoding="utf-8"))["example_id"] == self.examples[0].example_id
                )
                saved = json.loads(stage_path.read_text(encoding="utf-8"))
                mutate(saved)
                stage_path.write_text(json.dumps(saved), encoding="utf-8")
                self.greedy_calls.clear()
                self.sample_calls.clear()

                with self.assertRaises(ValueError):
                    self.invoke(run_dir, resume=True)

                self.assertEqual(self.greedy_calls, [])
                self.assertEqual(self.sample_calls, [])

    def test_resume_rejects_sample_stage_over_output_limit_before_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "local"

            def stop_after_one_sample(batch, _profile, *, on_result=None):
                request = batch[0]
                result = SampleGeneration("sample", [9], False, False, "other_stop")
                on_result(request["request_id"], result)
                raise RuntimeError("worker exited after saved stage")

            with self.assertRaisesRegex(RuntimeError, "after saved stage"):
                self.invoke(run_dir, sample_execute=stop_after_one_sample)

            stage_path = next(
                path for path in (run_dir / "stages").glob("*.json")
                if json.loads(path.read_text(encoding="utf-8"))["example_id"] == self.examples[0].example_id
            )
            saved = json.loads(stage_path.read_text(encoding="utf-8"))
            saved["samples"]["0"]["generation"]["token_ids"] = [9] * (self.profile.max_new_tokens + 1)
            stage_path.write_text(json.dumps(saved), encoding="utf-8")
            calls = []

            def resumed_sampling(batch, profile, *, on_result=None):
                calls.extend((row["request_id"], row["seed"]) for row in batch)
                return self.sample(batch, profile, on_result=on_result)

            with self.assertRaisesRegex(ValueError, "stage|token|output"):
                self.invoke(run_dir, sample_execute=resumed_sampling, resume=True)

            self.assertEqual(calls, [])

    def test_checkpoint_validator_rejects_completed_generation_over_profile_output_limit(self):
        module = __import__("teacher_reliability.local_run", fromlist=["_validate_local_checkpoint_record"])
        for component in ("greedy", "sample"):
            with self.subTest(component=component), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary) / "local"
                self.invoke(run_dir)

                predictions_path = run_dir / "predictions.jsonl"
                row = json.loads(predictions_path.read_text(encoding="utf-8").splitlines()[0])
                over_limit = self.profile.max_new_tokens + 1
                if component == "greedy":
                    greedy = row["greedy"]
                    greedy["token_ids"] = [7] * over_limit
                    greedy["token_entropies_nats"] = [0.4] * over_limit
                    greedy["token_logprobs"] = [-0.3] * over_limit
                    greedy["token_char_spans"] = [[0, 1]] * over_limit
                    greedy["content_token_mask"] = [True] * over_limit
                else:
                    row["samples"][0]["token_ids"] = [9] * over_limit

                with self.assertRaisesRegex(ValueError, "output limit"):
                    module._validate_local_checkpoint_record(
                        row, self.examples[0], 5, self.profile.sample_count,
                        self.profile.max_new_tokens,
                    )

    def test_batch_checkpoints_an_eos_row_before_a_later_row_crashes(self):
        from tests.test_teacher import _BatchFailureAfterShortEosModel, _BatchFixtureTokenizer
        from teacher_reliability.teacher import generate_greedy_batch

        examples = [
            normalize_example(
                "gsm8k", "main", "test", index,
                {"question": question, "answer": "#### 12"},
                source_revision="1" * 40,
            )
            for index, question in enumerate(("short", "long"))
        ]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "local"
            teacher_tokenizer = _BatchFixtureTokenizer()
            teacher_model = _BatchFailureAfterShortEosModel()

            def batched_greedy(batch, _profile, *, on_result=None):
                def emit(index, generation):
                    on_result(batch[index]["request_id"], generation)
                return generate_greedy_batch(
                    teacher_tokenizer, teacher_model,
                    [row["question"] for row in batch], 8,
                    on_result=emit,
                )

            try:
                self.invoke(
                    run_dir, examples=examples, greedy_execute=batched_greedy,
                    greedy_profiles=[MemoryProfile("greedy-2", 2)],
                )
            except (TypeError, RuntimeError) as exc:
                failure = exc
            else:
                self.fail("the later batch row should fail after the first row emits EOS")

            stages = {
                stage["example_id"]: stage
                for path in (run_dir / "stages").glob("*.json")
                for stage in [json.loads(path.read_text(encoding="utf-8"))]
            }
            self.assertIsInstance(
                failure, RuntimeError,
                f"decoder must stream completed rows before the later failure: {failure}",
            )
            self.assertIn("later batch row failed", str(failure))
            self.assertIsNotNone(stages[examples[0].example_id]["greedy"])
            self.assertIsNone(stages.get(examples[1].example_id, {}).get("greedy"))

    def test_failed_rows_are_incomplete_and_retry_failed_preserves_good_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "local"
            still_failing = {self.examples[0].example_id}

            def sampling(batch, profile, *, on_result=None):
                if batch[0]["example_id"] in still_failing:
                    raise CudaOutOfMemory("minimum profile exhausted")
                return self.sample(batch, profile, on_result=on_result)

            self.assertIn("cache_implementation", __import__("inspect").signature(MemoryProfile).parameters)
            profiles = [MemoryProfile("sample-1", 1), MemoryProfile("sample-1-offloaded", 1)]
            first = self.invoke(run_dir, sample_execute=sampling, sample_profiles=profiles)
            self.assertEqual(first["failed_example_count"], 1)
            self.assertEqual(first["pending_example_count"], 0)
            self.assertEqual(first["status"], "partial")
            still_failing.clear()
            retried = self.invoke(run_dir, sample_execute=sampling, sample_profiles=profiles, resume=True, retry_failed=True)
            self.assertEqual(retried["failed_example_count"], 0)
            self.assertEqual(retried["status"], "complete")
            stage_files = list((run_dir / "stages").glob("*.json"))
            stages = [json.loads(path.read_text()) for path in stage_files]
            retried_stage = next(stage for stage in stages if stage["example_id"] == self.examples[0].example_id)
            self.assertEqual(retried_stage["failure_history"], ["CudaOutOfMemory: minimum profile exhausted"])
            self.assertIsNotNone(retried_stage["greedy"])

    def test_insufficient_host_ram_fails_one_row_and_later_row_completes(self):
        from teacher_reliability.hybrid_workers import WorkerExecutionError

        first, second = self.examples[:2]
        two_example_profile = RunProfile("fixture-two", 2, 1, 1, True, "test", 1, 8)
        calls = []

        def sampling(batch, profile, *, on_result=None):
            request = batch[0]
            calls.append((request["request_id"], request["seed"], profile.name))
            if request["example_id"] == first.example_id and profile.cache_implementation is None:
                raise CudaOutOfMemory("regular cache exhausted")
            if request["example_id"] == first.example_id:
                raise WorkerExecutionError(
                    "InsufficientHostMemory", "offloaded KV cache needs more host RAM",
                )
            return self.sample(batch, profile, on_result=on_result)

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "local"
            result = self.invoke(
                run_dir,
                examples=[first, second],
                profile=two_example_profile,
                sample_execute=sampling,
                sample_profiles=[
                    MemoryProfile("sample-1", 1),
                    MemoryProfile("sample-1-offloaded", 1, cache_implementation="offloaded"),
                ],
            )
            stages = {
                stage["example_id"]: stage
                for path in (run_dir / "stages").glob("*.json")
                for stage in [json.loads(path.read_text(encoding="utf-8"))]
            }
            predictions = [
                json.loads(line)
                for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        first_seed = derive_sample_seed(5, first.example_id, 0)
        second_seed = derive_sample_seed(5, second.example_id, 0)
        self.assertEqual(result["failed_example_count"], 1)
        self.assertEqual(result["completed_example_count"], 1)
        self.assertEqual(result["pending_example_count"], 0)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(calls, [
            (f"{first.example_id}::sample::0", first_seed, "sample-1"),
            (f"{first.example_id}::sample::0", first_seed, "sample-1-offloaded"),
            (f"{second.example_id}::sample::0", second_seed, "sample-1"),
        ])
        self.assertEqual(stages[first.example_id]["failure"].split(":", 1)[0], "InsufficientHostMemory")
        self.assertIsNotNone(stages[first.example_id]["greedy"])
        self.assertEqual(stages[first.example_id]["samples"], {})
        self.assertEqual(stages[second.example_id]["samples"]["0"]["seed"], second_seed)
        self.assertEqual([row["example_id"] for row in predictions], [second.example_id])

    def test_optional_deadline_persists_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "local"
            first = self.invoke(run_dir, max_runtime_hours=1)
            self.assertEqual(first["deadline_utc"], "2026-10-01T01:00:00+00:00")
            self.now_value += timedelta(hours=2)
            second = self.invoke(run_dir, max_runtime_hours=1, resume=True)
            self.assertEqual(second["deadline_utc"], first["deadline_utc"])
            self.assertEqual(second["status"], "partial")

    def test_active_generation_progress_stops_at_deadline_and_keeps_partial_sample_retryable(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "local"
            deadline = self.now_value + timedelta(hours=1)
            streamed = []

            def stop_inside_sampling(batch, _profile, *, on_result=None, on_progress=None):
                request = batch[0]
                generation = SampleGeneration("partial sample", [9], False, False, "other_stop")
                on_result(request["request_id"], generation)
                streamed.append((request["request_id"], request["seed"]))
                self.now_value = deadline
                on_progress({"request_index": 0, "generated_tokens": 1})
                raise RuntimeError("generation continued past its deadline")

            result = self.invoke(
                run_dir,
                max_runtime_hours=1,
                sample_execute=stop_inside_sampling,
            )

            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["failed_example_count"], 0)
            self.assertEqual(result["pending_example_count"], 3)
            self.assertEqual(len(streamed), 1)
            request_id, seed = streamed[0]
            example_id, suffix = request_id.rsplit("::sample::", 1)
            sample_index = int(suffix)
            self.assertEqual(seed, derive_sample_seed(5, example_id, sample_index))
            self.assertFalse((run_dir / "predictions.jsonl").exists())
            stage_path = next(
                path for path in (run_dir / "stages").glob("*.json")
                if json.loads(path.read_text(encoding="utf-8"))["example_id"] == example_id
            )
            saved = json.loads(stage_path.read_text(encoding="utf-8"))
            self.assertEqual(set(saved["samples"]), {str(sample_index)})
            self.assertEqual(saved["samples"][str(sample_index)]["seed"], seed)
            self.assertIsNone(saved["failure"])

    def test_crashed_gpu_worker_restarts_and_retries_only_uncheckpointed_samples(self):
        from teacher_reliability.hybrid_workers import ManagedWorker

        script = r'''import json, os, sys
from pathlib import Path

counter = Path(sys.argv[1])
log_path = Path(sys.argv[2])
startup = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(startup))
print(json.dumps({"ready": True, "runtime": {"engine": "test-worker", "startup": startup}}), flush=True)
for line in sys.stdin:
    payload = json.loads(line)
    requests = payload["requests"]
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"stage": payload["stage"], "ids": [row["request_id"] for row in requests],
                              "seeds": [row.get("seed") for row in requests], "startup": startup}) + "\n")
    for index, row in enumerate(requests):
        if payload["stage"] == "greedy":
            value = {"request_id": row["request_id"], "greedy": {
                "text": "answer", "token_ids": [7], "token_entropies_nats": [0.1],
                "token_logprobs": [-0.1], "token_char_spans": [[0, 1]],
                "content_token_mask": [True], "eos_generated": True,
                "was_truncated": False, "termination_reason": "eos"}}
        else:
            value = {"request_id": row["request_id"], "sample": {
                "text": "sample", "token_ids": [8], "eos_generated": True,
                "was_truncated": False, "reason": "eos"}}
        print(json.dumps({"ok": True, "result": value}), flush=True)
        if payload["stage"] == "sample" and startup == 1:
            os._exit(17)
    print(json.dumps({"ok": True, "done": True}), flush=True)
'''

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker_script = root / "worker.py"
            worker_counter = root / "starts.txt"
            worker_log = root / "requests.jsonl"
            worker_script.write_text(script, encoding="utf-8")
            worker = ManagedWorker(
                lambda _profile: [sys.executable, str(worker_script), str(worker_counter), str(worker_log)],
                root / "worker.stderr.log",
            )

            def execute(stage):
                def call(batch, profile, *, on_result=None):
                    payload = {
                        "op": "generate",
                        "stage": stage,
                        "requests": [dict(row) for row in batch],
                    }

                    def receive(request_id, value):
                        generation = (
                            GreedyGeneration(**value["greedy"])
                            if stage == "greedy" else SampleGeneration(**value["sample"])
                        )
                        on_result(request_id, generation)

                    response = worker.request(
                        payload, profile.name, on_result=receive,
                    )
                    results = {}
                    for value in response["results"]:
                        generation = (
                            GreedyGeneration(**value["greedy"])
                            if stage == "greedy" else SampleGeneration(**value["sample"])
                        )
                        results[value["request_id"]] = generation
                    return results
                return call

            try:
                result = self.invoke(
                    root / "local",
                    greedy_execute=execute("greedy"),
                    sample_execute=execute("sample"),
                    greedy_profiles=[MemoryProfile("greedy-1", 1)],
                    sample_profiles=[MemoryProfile("sample-2", 2)],
                )
                calls = [json.loads(line) for line in worker_log.read_text(encoding="utf-8").splitlines()]
                worker_starts = int(worker_counter.read_text(encoding="utf-8"))
            finally:
                worker.close()

        self.assertEqual(result["status"], "complete")
        self.assertEqual(worker_starts, 2)
        sample_calls = [call for call in calls if call["stage"] == "sample"]
        self.assertGreaterEqual(len(sample_calls), 2)
        completed_request = sample_calls[0]["ids"][0]
        self.assertNotIn(completed_request, [request_id for call in sample_calls[1:] for request_id in call["ids"]])
        self.assertEqual(sample_calls[0]["seeds"], [
            derive_sample_seed(5, self.examples[0].example_id, 0),
            derive_sample_seed(5, self.examples[0].example_id, 1),
        ])


if __name__ == "__main__":
    unittest.main()



