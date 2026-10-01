"""Observable recovery and checkpoint contracts for the hybrid runner."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from teacher_reliability.data import normalize_example
from teacher_reliability.hybrid import (
    CudaOutOfMemory,
    MemoryProfile,
    StageStore,
    run_with_oom_backoff,
)
import teacher_reliability.hybrid as hybrid_module
from teacher_reliability.hybrid_run import run_hybrid
from teacher_reliability.selection import RunProfile
from teacher_reliability.teacher import GreedyGeneration, SampleGeneration


class HybridRecoveryTests(unittest.TestCase):
    def test_insufficient_host_memory_fails_exhausted_request_and_continues_with_regular_cache(self):
        from teacher_reliability.local_worker import InsufficientHostMemory

        requests = [
            {"request_id": "first", "example_id": "row-a", "seed": 101},
            {"request_id": "second", "example_id": "row-b", "seed": 202},
        ]
        profiles = [
            MemoryProfile("sample-1", 1),
            MemoryProfile("sample-1-offloaded", 1, cache_implementation="offloaded"),
        ]
        calls = []
        completed = []
        failures = []

        def execute(batch, profile):
            request = batch[0]
            calls.append((request["request_id"], request["seed"], profile.name))
            if request["request_id"] == "first" and profile.cache_implementation is None:
                raise CudaOutOfMemory("regular KV cache exhausted")
            if request["request_id"] == "first":
                raise InsufficientHostMemory("offloaded KV cache needs more host RAM")
            return {request["request_id"]: request["seed"]}

        run_with_oom_backoff(
            requests, profiles, execute,
            on_complete=lambda request, value: completed.append((request["request_id"], value)),
            on_failure=lambda request, error: failures.append(
                (request["request_id"], type(error).__name__, request["seed"])
            ),
        )

        self.assertEqual(calls, [
            ("first", 101, "sample-1"),
            ("first", 101, "sample-1-offloaded"),
            ("second", 202, "sample-1"),
        ])
        self.assertEqual(failures, [("first", "InsufficientHostMemory", 101)])
        self.assertEqual(completed, [("second", 202)])

    def test_worker_exit_restarts_and_retries_only_unfinished_requests(self):
        restart_error = getattr(hybrid_module, "WorkerRestartRequired", RuntimeError)
        requests = [
            {"request_id": "a", "seed": 101},
            {"request_id": "b", "seed": 202},
        ]
        calls = []
        completed = []

        def execute(batch, _profile, *, on_result=None):
            calls.append([(row["request_id"], row["seed"]) for row in batch])
            if len(calls) == 1:
                on_result("a", {"request_id": "a", "seed": 101})
                if restart_error is RuntimeError:
                    error = RuntimeError("worker exited before replying")
                else:
                    error = restart_error("worker exited before replying")
                    error.completed_results = {"a": {"request_id": "a", "seed": 101}}
                raise error
            return {row["request_id"]: row for row in batch}

        try:
            run_with_oom_backoff(
                requests, [MemoryProfile("same-profile", 2)], execute,
                on_complete=lambda request, result: completed.append(
                    (request["request_id"], request["seed"], result["seed"])
                ),
                on_failure=lambda request, error: self.fail(f"unexpected failure {request}: {error}"),
            )
        except RuntimeError:
            pass

        self.assertEqual(calls, [
            [("a", 101), ("b", 202)],
            [("b", 202)],
        ])
        self.assertEqual(completed, [("a", 101, 101), ("b", 202, 202)])

    def test_oom_controller_forwards_worker_progress_to_coordinator(self):
        import inspect

        self.assertIn("on_progress", inspect.signature(run_with_oom_backoff).parameters)
        events = []

        def execute(batch, _profile, *, on_result, on_progress):
            on_progress({"generated_tokens": 8})
            on_result(batch[0]["request_id"], "done")
            return {batch[0]["request_id"]: "done"}

        run_with_oom_backoff(
            [{"request_id": "a"}], [MemoryProfile("one", 1)], execute,
            on_complete=lambda *_: None, on_failure=lambda *_: self.fail("unexpected failure"),
            on_progress=lambda batch, event: events.append((batch[0]["request_id"], event)),
        )
        self.assertEqual(events, [("a", {"generated_tokens": 8})])

    def test_partial_batch_oom_retries_only_ids_not_already_completed(self):
        requests = [{"request_id": "a", "seed": 1}, {"request_id": "b", "seed": 2}]
        calls = []
        completed = []

        def execute(batch, _profile, *, on_result=None):
            calls.append([row["request_id"] for row in batch])
            if len(calls) == 1:
                on_result("a", "saved")
                raise CudaOutOfMemory("later request OOM")
            return {"b": "done"}

        try:
            run_with_oom_backoff(
                requests, [MemoryProfile("initial", 2), MemoryProfile("minimum", 1)], execute,
                on_complete=lambda request, result: completed.append((request["request_id"], result)),
                on_failure=lambda *_: self.fail("unexpected failure"),
            )
        except (TypeError, ValueError):
            pass
        self.assertEqual(calls, [["a", "b"], ["b"]])
        self.assertEqual(completed, [("a", "saved"), ("b", "done")])

    def test_failed_example_skips_its_remaining_sample_requests(self):
        requests = [
            {"request_id": "a:0", "example_id": "a"},
            {"request_id": "a:1", "example_id": "a"},
            {"request_id": "b:0", "example_id": "b"},
        ]
        failed = set()
        calls = []

        def execute(batch, _profile):
            calls.append(batch[0]["request_id"])
            if batch[0]["example_id"] == "a":
                raise CudaOutOfMemory("example a needs more memory")
            return {"b:0": "done"}

        try:
            run_with_oom_backoff(
                requests, [MemoryProfile("minimum", 1)], execute,
                on_complete=lambda *_: None,
                on_failure=lambda request, _error: failed.add(request["example_id"]),
                skip_request=lambda request: request["example_id"] in failed,
            )
        except TypeError:
            calls.append("missing skip_request support")
        self.assertEqual(calls, ["a:0", "b:0"])

    def test_oom_retries_only_unfinished_requests_with_their_original_ids_and_seeds(self):
        requests = [
            {"request_id": "a", "seed": 101},
            {"request_id": "b", "seed": 202},
            {"request_id": "c", "seed": 303},
        ]
        profiles = [
            MemoryProfile("initial", 4),
            MemoryProfile("reduced", 2),
            MemoryProfile("minimum", 1),
        ]
        calls = []
        completed = []
        adaptations = []

        def execute(batch, profile):
            calls.append((profile.name, [(row["request_id"], row["seed"]) for row in batch]))
            if profile.name == "initial":
                raise CudaOutOfMemory("allocation failed")
            return {row["request_id"]: {"token_ids": [row["seed"]]} for row in reversed(batch)}

        run_with_oom_backoff(
            requests,
            profiles,
            execute,
            on_complete=lambda request, result: completed.append(
                (request["request_id"], result["token_ids"])
            ),
            on_failure=lambda request, error: self.fail(f"unexpected failure: {request} {error}"),
            on_adapt=lambda profile, error: adaptations.append(profile.name),
        )

        self.assertEqual(calls, [
            ("initial", [("a", 101), ("b", 202), ("c", 303)]),
            ("reduced", [("a", 101), ("b", 202)]),
            ("reduced", [("c", 303)]),
        ])
        self.assertEqual(completed, [("a", [101]), ("b", [202]), ("c", [303])])
        self.assertEqual(adaptations, ["reduced"])

    def test_minimum_batch_oom_marks_that_request_failed_and_continues(self):
        requests = [{"request_id": "too-long"}, {"request_id": "short"}]
        complete = []
        failed = []

        def execute(batch, _profile):
            if batch[0]["request_id"] == "too-long":
                raise CudaOutOfMemory("minimum batch still failed")
            return {"short": "done"}

        run_with_oom_backoff(
            requests,
            [MemoryProfile("minimum", 1)],
            execute,
            on_complete=lambda request, result: complete.append((request["request_id"], result)),
            on_failure=lambda request, error: failed.append((request["request_id"], str(error))),
        )

        self.assertEqual(complete, [("short", "done")])
        self.assertEqual(failed, [("too-long", "minimum batch still failed")])

    def test_unrelated_error_does_not_turn_into_an_incomplete_example(self):
        with self.assertRaisesRegex(ValueError, "protocol"):
            run_with_oom_backoff(
                [{"request_id": "a"}],
                [MemoryProfile("minimum", 1)],
                lambda _batch, _profile: (_ for _ in ()).throw(ValueError("protocol")),
                on_complete=lambda *_: self.fail("completed"),
                on_failure=lambda *_: self.fail("recorded unrelated failure"),
            )


class HybridStageTests(unittest.TestCase):
    def test_resumes_saved_greedy_and_individual_samples_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = StageStore(root, "signature-1", {"example:a"})
            first.checkpoint_greedy("example:a", {"text": "answer", "token_ids": [7]})
            first.checkpoint_sample("example:a", 0, {"token_ids": [8], "seed": 111})

            resumed = StageStore(root, "signature-1", {"example:a"})
            self.assertEqual(resumed.get("example:a")["greedy"]["token_ids"], [7])
            self.assertEqual(resumed.get("example:a")["samples"], {
                "0": {"token_ids": [8], "seed": 111}
            })
            with self.assertRaisesRegex(ValueError, "already checkpointed"):
                resumed.checkpoint_sample("example:a", 0, {"token_ids": [9], "seed": 111})

    def test_retry_failed_stage_clears_failure_and_keeps_successful_components(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = StageStore(Path(temporary), "signature-1", {"example:a"})
            stage.checkpoint_greedy("example:a", {"text": "answer", "token_ids": [7]})
            stage.checkpoint_sample("example:a", 0, {"token_ids": [8], "seed": 111})
            stage.mark_failed("example:a", "CUDA OOM")
            clear_failure = getattr(stage, "clear_failure", None)
            self.assertTrue(
                callable(clear_failure),
                "retrying a failed request must preserve completed stages and failure history",
            )

            clear_failure("example:a")

            resumed = StageStore(Path(temporary), "signature-1", {"example:a"})
            saved = resumed.get("example:a")
            self.assertIsNone(saved["failure"])
            self.assertEqual(saved["greedy"]["token_ids"], [7])
            self.assertEqual(saved["samples"]["0"]["seed"], 111)
            self.assertEqual(saved["failure_history"], ["CUDA OOM"])

    def test_stage_identity_or_unknown_example_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            StageStore(root, "signature-1", {"example:a"}).checkpoint_greedy(
                "example:a", {"token_ids": [7]}
            )
            with self.assertRaisesRegex(ValueError, "signature"):
                StageStore(root, "signature-2", {"example:a"}).get("example:a")
            with self.assertRaisesRegex(ValueError, "outside"):
                StageStore(root, "signature-1", {"example:a"}).get("example:b")


class TinyTokenizer:
    vocab_size = 4

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return [1] * (3 if "apples" in messages[0]["content"] else 2)


class HybridRunTests(unittest.TestCase):
    def setUp(self):
        self.examples = [
            normalize_example(
                "gsm8k", "main", "test", index,
                {"question": question, "answer": "#### 12"}, source_revision="1" * 40,
            )
            for index, question in enumerate(("How many apples?", "How many pears?"))
        ]
        self.profile = RunProfile("fixture", 2, 1, 1, True, "test", 2, 32)
        self.revisions = {
            "openai/gsm8k": "1" * 40,
            "EleutherAI/hendrycks_math": "2" * 40,
            "qintongli/GSM-Plus": "3" * 40,
        }
        self.artifact = {"aggregate_sha256": "a" * 64, "model_revision": "b" * 40}
        self.now_value = datetime(2026, 9, 30, 12, tzinfo=timezone.utc)
        self.greedy_calls = []
        self.sample_calls = []

    def greedy_execute(self, batch, _memory_profile):
        self.greedy_calls.extend(row["request_id"] for row in batch)
        return {
            row["request_id"]: GreedyGeneration(
                text=r"\boxed{12}", token_ids=[7, 8],
                token_entropies_nats=[0.4, 0.2], token_logprobs=[-0.3, -0.2],
                token_char_spans=[(7, 8), (8, 9)], content_token_mask=[True, True],
                eos_generated=True, termination_reason="eos",
            )
            for row in batch
        }

    def sample_execute(self, batch, _memory_profile):
        self.sample_calls.extend((row["request_id"], row["seed"]) for row in batch)
        return {
            row["request_id"]: SampleGeneration(r"\boxed{12}", [7, 8], True, False, "eos")
            for row in batch
        }

    def invoke_run(self, run_dir, *, resume=False, sample_execute=None):
        return run_hybrid(
            run_dir, self.profile, seed=5, examples=self.examples,
            model_revision="b" * 40, dataset_revisions=self.revisions,
            artifact=self.artifact, tokenizer=TinyTokenizer(),
            greedy_execute=self.greedy_execute,
            sample_execute=sample_execute or self.sample_execute,
            max_runtime_hours=12, now=lambda: self.now_value, resume=resume,
        )

    def test_completes_and_resumes_without_repeating_generation_or_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            first = self.invoke_run(run_dir)
            self.assertEqual(first["status"], "complete")
            self.assertEqual(first["completed_example_count"], 2)
            self.assertEqual(first["failed_example_count"], 0)
            self.assertEqual(len(self.greedy_calls), 2)
            self.assertEqual(len(self.sample_calls), 4)

            second = self.invoke_run(run_dir, resume=True)
            self.assertEqual(second["status"], "complete")
            self.assertEqual(len(self.greedy_calls), 2)
            self.assertEqual(len(self.sample_calls), 4)
            self.assertEqual(len((run_dir / "predictions.jsonl").read_text().splitlines()), 2)

    def test_each_stage_is_durable_without_rewriting_large_manifest_per_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            from teacher_reliability.hybrid_run import atomic_write_json as real_write

            manifest_writes = []

            def counted_write(path, value):
                if path.name == "run_manifest.json":
                    manifest_writes.append(value["status"])
                return real_write(path, value)

            with patch("teacher_reliability.hybrid_run.atomic_write_json", side_effect=counted_write):
                result = self.invoke_run(run_dir)
            self.assertEqual(result["completed_example_count"], 2)
            self.assertLessEqual(len(manifest_writes), 3)
            self.assertEqual(len(list((run_dir / "stages").glob("*.json"))), 2)

    def test_resume_reuses_greedy_checkpoint_after_sampler_crash(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"

            def crash(_batch, _profile):
                raise ValueError("worker crashed")

            with self.assertRaisesRegex(ValueError, "worker crashed"):
                self.invoke_run(run_dir, sample_execute=crash)
            self.assertEqual(len(self.greedy_calls), 2)
            self.assertEqual(len(list((run_dir / "stages").glob("*.json"))), 2)

            recovered = self.invoke_run(run_dir, resume=True)
            self.assertEqual(recovered["status"], "complete")
            self.assertNotIn("last_error", recovered)
            self.assertNotIn("failed_at", recovered)
            self.assertEqual(len(self.greedy_calls), 2)
            self.assertEqual(len(self.sample_calls), 4)

    def test_resume_rejects_corrupt_incomplete_sample_stage_before_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"

            def stop_after_first_sample(batch, _profile, *, on_result=None):
                request = batch[0]
                result = SampleGeneration(r"\boxed{12}", [7, 8], True, False, "eos")
                on_result(request["request_id"], result)
                raise RuntimeError("worker exited after first saved sample")

            with self.assertRaisesRegex(RuntimeError, "after first saved sample"):
                self.invoke_run(run_dir, sample_execute=stop_after_first_sample)

            stage_path = next(
                path for path in (run_dir / "stages").glob("*.json")
                if json.loads(path.read_text(encoding="utf-8"))["example_id"] == self.examples[0].example_id
            )
            saved = json.loads(stage_path.read_text(encoding="utf-8"))
            saved["samples"]["0"]["seed"] += 1
            stage_path.write_text(json.dumps(saved), encoding="utf-8")
            self.greedy_calls.clear()
            self.sample_calls.clear()

            with self.assertRaisesRegex(ValueError, "stage|seed|checkpoint"):
                self.invoke_run(run_dir, resume=True)

            self.assertEqual(self.greedy_calls, [])
            self.assertEqual(self.sample_calls, [])

    def test_resume_rejects_stage_tokens_over_profile_limit_before_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"

            def stop_after_first_sample(batch, _profile, *, on_result=None):
                request = batch[0]
                result = SampleGeneration(r"\boxed{12}", [7, 8], True, False, "eos")
                on_result(request["request_id"], result)
                raise RuntimeError("worker exited after first saved sample")

            with self.assertRaisesRegex(RuntimeError, "after first saved sample"):
                self.invoke_run(run_dir, sample_execute=stop_after_first_sample)

            stage_path = next(
                path for path in (run_dir / "stages").glob("*.json")
                if json.loads(path.read_text(encoding="utf-8"))["example_id"] == self.examples[0].example_id
            )
            saved = json.loads(stage_path.read_text(encoding="utf-8"))
            saved["samples"]["0"]["generation"]["token_ids"] = [7] * (self.profile.max_new_tokens + 1)
            stage_path.write_text(json.dumps(saved), encoding="utf-8")
            self.greedy_calls.clear()
            self.sample_calls.clear()

            with self.assertRaisesRegex(ValueError, "stage|token|output"):
                self.invoke_run(run_dir, resume=True)

            self.assertEqual(self.greedy_calls, [])
            self.assertEqual(self.sample_calls, [])

    def test_resume_rejects_greedy_stage_over_profile_limit_before_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"

            def stop_after_first_sample(batch, _profile, *, on_result=None):
                request = batch[0]
                result = SampleGeneration(r"\boxed{12}", [7, 8], True, False, "eos")
                on_result(request["request_id"], result)
                raise RuntimeError("worker exited after first saved sample")

            with self.assertRaisesRegex(RuntimeError, "after first saved sample"):
                self.invoke_run(run_dir, sample_execute=stop_after_first_sample)

            stage_path = next(
                path for path in (run_dir / "stages").glob("*.json")
                if json.loads(path.read_text(encoding="utf-8"))["example_id"] == self.examples[0].example_id
            )
            saved = json.loads(stage_path.read_text(encoding="utf-8"))
            over_limit = self.profile.max_new_tokens + 1
            saved["greedy"]["token_ids"] = [7] * over_limit
            saved["greedy"]["token_entropies_nats"] = [0.1] * over_limit
            saved["greedy"]["token_logprobs"] = [-0.1] * over_limit
            saved["greedy"]["token_char_spans"] = None
            saved["greedy"]["content_token_mask"] = [True] * over_limit
            stage_path.write_text(json.dumps(saved), encoding="utf-8")
            self.greedy_calls.clear()
            self.sample_calls.clear()

            with self.assertRaisesRegex(ValueError, "stage|token|output"):
                self.invoke_run(run_dir, resume=True)

            self.assertEqual(self.greedy_calls, [])
            self.assertEqual(self.sample_calls, [])

    def test_deadline_survives_resume_and_preserves_partial_greedy(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"

            def slow_greedy(batch, profile):
                output = self.greedy_execute(batch, profile)
                self.now_value += timedelta(hours=13)
                return output

            first = run_hybrid(
                run_dir, self.profile, seed=5, examples=self.examples,
                model_revision="b" * 40, dataset_revisions=self.revisions,
                artifact=self.artifact, tokenizer=TinyTokenizer(),
                greedy_execute=slow_greedy, sample_execute=self.sample_execute,
                max_runtime_hours=12, now=lambda: self.now_value,
            )
            self.assertEqual(first["status"], "partial")
            self.assertEqual(first["pending_example_count"], 2)
            self.assertEqual(len(self.sample_calls), 0)
            self.assertEqual(first["deadline_utc"], "2026-10-01T00:00:00+00:00")

            resumed = self.invoke_run(run_dir, resume=True)
            self.assertEqual(resumed["deadline_utc"], first["deadline_utc"])
            self.assertEqual(resumed["status"], "partial")
            self.assertEqual(len(self.greedy_calls), 2)

    def test_worker_timeout_at_deadline_preserves_partial_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"

            def timeout_sample(_batch, _profile):
                self.now_value += timedelta(hours=12)
                raise TimeoutError("worker deadline expired")

            try:
                result = self.invoke_run(run_dir, sample_execute=timeout_sample)
            except TimeoutError:
                result = {"status": "failed", "completed_example_count": 0}
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["completed_example_count"], 0)
            self.assertEqual(len(list((run_dir / "stages").glob("*.json"))), 2)

    def test_reduced_greedy_profile_carries_into_next_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            examples = [
                normalize_example("gsm8k", "main", "test", index,
                                  {"question": f"How many apples {index}?", "answer": "#### 12"},
                                  source_revision="1" * 40)
                for index in range(9)
            ]
            seen = []

            def greedy(batch, memory_profile):
                seen.append((memory_profile.batch_size, len(batch)))
                if memory_profile.batch_size == 8:
                    raise CudaOutOfMemory("batch too large")
                return self.greedy_execute(batch, memory_profile)

            run_hybrid(
                Path(temporary) / "run", self.profile, seed=5, examples=examples,
                model_revision="b" * 40, dataset_revisions=self.revisions,
                artifact=self.artifact, tokenizer=TinyTokenizer(),
                greedy_execute=greedy, sample_execute=self.sample_execute,
                max_runtime_hours=12, now=lambda: self.now_value,
            )
            self.assertEqual(seen, [(8, 8), (4, 4), (4, 4), (4, 1)])


if __name__ == "__main__":
    unittest.main()
