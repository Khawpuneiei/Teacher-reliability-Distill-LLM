"""Process-boundary tests without loading either GPU model."""

import json
import inspect
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from teacher_reliability.hybrid import CudaOutOfMemory, MemoryProfile, run_with_oom_backoff
from teacher_reliability import hybrid_workers
from teacher_reliability.hybrid_workers import LineWorker, ManagedWorker


WORKER_SCRIPT = r'''
import json
import os
import sys
import time
from pathlib import Path

print(json.dumps({"ready": True, "runtime": {"engine": sys.argv[1]}}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request["op"] == "slow":
        time.sleep(3)
    elif request["op"] == "streamoom":
        print(json.dumps({"ok": True, "result": {"request_id": "a", "seed": 31}}), flush=True)
        print(json.dumps({"ok": False, "error_type": "cuda_oom", "message": "later request CUDA OOM"}), flush=True)
    elif request["op"] == "oom":
        print(json.dumps({"ok": False, "error_type": "cuda_oom", "message": "fake CUDA OOM"}), flush=True)
    elif request["op"] == "progress":
        print(json.dumps({"ok": True, "progress": {"generated_tokens": 17}}), flush=True)
        print(json.dumps({"ok": True, "results": [{"request_id": "a", "seed": 31}]}), flush=True)
    elif request["op"] == "bad":
        print(json.dumps({"ok": False, "error_type": "ValueError", "message": "invalid request value"}), flush=True)
    elif request["op"] == "crashonce":
        marker = Path(sys.argv[2])
        if not marker.exists():
            marker.write_text("crashed")
            first = request["requests"][0]
            print(json.dumps({"ok": True, "result": first}), flush=True)
            os._exit(17)
        print(json.dumps({"ok": True, "results": request["requests"]}), flush=True)
    else:
        print(json.dumps({"ok": True, "results": [{"request_id": row["request_id"], "seed": row["seed"]} for row in request["requests"]]}), flush=True)
'''


class WorkerBoundaryTests(unittest.TestCase):
    def test_worker_request_forwards_progress_events_without_ending_request(self):
        self.assertIn("on_progress", inspect.signature(LineWorker.request).parameters)
        worker = LineWorker(self.command(type("P", (), {"name": "progress"})()), self.root / "stderr.log")
        self.addCleanup(worker.close)
        progress = []
        result = worker.request(
            {"op": "progress", "requests": [{"request_id": "a", "seed": 31}]},
            deadline=datetime.now(timezone.utc) + timedelta(seconds=5),
            on_progress=progress.append,
        )
        self.assertEqual(progress, [{"generated_tokens": 17}])
        self.assertEqual(result["results"][0]["request_id"], "a")

    def test_worker_supports_no_request_timeout_when_run_has_no_deadline(self):
        self.assertIn("unbounded", inspect.signature(LineWorker.request).parameters)
        worker = LineWorker(self.command(type("P", (), {"name": "no-deadline"})()), self.root / "stderr.log")
        self.addCleanup(worker.close)
        result = worker.request(
            {"op": "generate", "requests": [{"request_id": "a", "seed": 31}]},
            unbounded=True,
        )
        self.assertEqual(result["results"][0]["request_id"], "a")

    def test_worker_errors_keep_their_remote_exception_classification(self):
        worker = LineWorker(self.command(type("P", (), {"name": "bad"})()), self.root / "stderr.log")
        self.addCleanup(worker.close)
        with self.assertRaises(RuntimeError) as caught:
            worker.request({"op": "bad"})
        self.assertEqual(getattr(caught.exception, "worker_error_type", None), "ValueError")

    def test_global_gpu_memory_snapshot_parses_nvidia_smi_mib(self):
        with patch("teacher_reliability.hybrid_workers.subprocess.check_output", return_value="50240, 81920\n"):
            probe = getattr(hybrid_workers, "gpu_memory_snapshot", lambda: {})
            result = probe()
        self.assertEqual(result.get("free_mib"), 50240)
        self.assertEqual(result.get("total_mib"), 81920)

    def test_streaming_worker_preserves_completed_results_before_oom(self):
        worker = LineWorker(self.command(type("P", (), {"name": "stream"})()), self.root / "stderr.log")
        self.addCleanup(worker.close)
        observed = []
        try:
            worker.request({"op": "streamoom"}, deadline=datetime.now(timezone.utc) + timedelta(seconds=5),
                           on_result=lambda request_id, result: observed.append((request_id, result)))
        except CudaOutOfMemory as error:
            result = getattr(error, "completed_results", {})
        else:
            result = {}
        self.assertEqual(result, {"a": {"request_id": "a", "seed": 31}})
        self.assertEqual(observed, [("a", {"request_id": "a", "seed": 31})])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.script = self.root / "fake_worker.py"
        self.script.write_text(WORKER_SCRIPT, encoding="utf-8")

    def command(self, profile):
        return [sys.executable, str(self.script), profile.name, str(self.root / "crash.marker")]

    def test_line_worker_transfers_request_identity_and_runtime(self):
        worker = LineWorker(self.command(type("P", (), {"name": "first"})()), self.root / "stderr.log")
        self.addCleanup(worker.close)
        self.assertEqual(worker.runtime["engine"], "first")
        result = worker.request({
            "op": "generate",
            "requests": [{"request_id": "a", "seed": 7}, {"request_id": "b", "seed": 9}],
        }, deadline=datetime.now(timezone.utc) + timedelta(seconds=5))
        self.assertEqual(result["results"], [
            {"request_id": "a", "seed": 7}, {"request_id": "b", "seed": 9},
        ])

    def test_managed_worker_restarts_after_oom_and_uses_new_profile(self):
        managed = ManagedWorker(self.command, self.root / "stderr.log")
        self.addCleanup(managed.close)
        first = type("P", (), {"name": "initial"})()
        second = type("P", (), {"name": "reduced"})()
        with self.assertRaisesRegex(CudaOutOfMemory, "fake CUDA OOM"):
            managed.request({"op": "oom"}, first)
        self.assertEqual(managed.request({
            "op": "generate", "requests": [{"request_id": "id", "seed": 101}],
        }, second)["results"][0]["seed"], 101)
        self.assertEqual(managed.runtime["engine"], "reduced")

    def test_unexpected_process_exit_restarts_and_retries_only_unfinished_rows(self):
        managed = ManagedWorker(self.command, self.root / "stderr.log")
        self.addCleanup(managed.close)
        profile = MemoryProfile("same-profile", 2)
        requests = [
            {"request_id": "a", "seed": 31},
            {"request_id": "b", "seed": 47},
        ]
        submitted = []
        completed = []

        def execute(batch, _profile, *, on_result=None):
            submitted.append([(row["request_id"], row["seed"]) for row in batch])
            response = managed.request(
                {"op": "crashonce", "requests": batch}, _profile,
                on_result=on_result,
            )
            return {row["request_id"]: row for row in response["results"]}

        try:
            run_with_oom_backoff(
                requests, [profile], execute,
                on_complete=lambda request, result: completed.append(
                    (request["request_id"], request["seed"], result["seed"])
                ),
                on_failure=lambda request, error: self.fail(f"unexpected failure {request}: {error}"),
            )
        except RuntimeError:
            pass

        self.assertEqual(submitted, [
            [("a", 31), ("b", 47)],
            [("b", 47)],
        ])
        self.assertEqual(completed, [("a", 31, 31), ("b", 47, 47)])

    def test_deadline_stops_a_hung_worker(self):
        worker = LineWorker(self.command(type("P", (), {"name": "slow"})()), self.root / "stderr.log")
        self.addCleanup(worker.close)
        with self.assertRaisesRegex(TimeoutError, "deadline"):
            worker.request({"op": "slow"}, deadline=datetime.now(timezone.utc) + timedelta(milliseconds=100))
        self.assertIsNotNone(worker.process.poll())


if __name__ == "__main__":
    unittest.main()
