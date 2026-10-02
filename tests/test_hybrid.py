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


if __name__ == "__main__":
    unittest.main()
