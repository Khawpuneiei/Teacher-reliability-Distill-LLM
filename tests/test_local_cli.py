"""Windows local runner dispatch, path and lifecycle command contracts."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from teacher_reliability.run import main


class LocalCliTests(unittest.TestCase):
    def invoke(self, argv):
        try:
            return main(argv)
        except SystemExit as error:
            return error.code

    def test_local_backend_dispatches_mini_run_and_keeps_deadline_unset_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "local-run"
            expected = {
                "run_id": "local-run", "status": "partial",
                "completed_example_count": 40, "selected_example_count": 111,
            }
            with patch("teacher_reliability.run._default_local_run_dir", return_value=run_dir, create=True), patch(
                "teacher_reliability.run._execute_local_cli", return_value=expected, create=True
            ) as execute, patch("sys.stdout", new_callable=io.StringIO) as stdout:
                result = self.invoke([
                    "--backend", "transformers-local", "--profile", "mini_12h",
                    "--device-preset", "rtx4060-8gb", "--retry-failed",
                ])
            self.assertEqual(result, 0)
            args = execute.call_args.args[0]
            self.assertEqual(args.backend, "transformers-local")
            self.assertEqual(args.profile, "mini_12h")
            self.assertEqual(args.device_preset, "rtx4060-8gb")
            self.assertTrue(args.retry_failed)
            self.assertIsNone(args.max_runtime_hours)
            self.assertIn("40/111 examples", stdout.getvalue())
            pointer = root / "current_run_dir.txt"
            self.assertTrue(pointer.is_file(), "local runs need a durable path for later status and resume commands")
            self.assertEqual(pointer.read_text(encoding="utf-8"), str(run_dir.resolve()))

    def test_removed_full_study_options_are_rejected(self):
        for argv in (
            ["--profile", "full"],
            ["--backend", "vllm-hybrid"],
            ["--import-run-zip", "source.zip"],
            ["--fast-start-unqualified"],
        ):
            with self.subTest(argv=argv), patch("sys.stderr", new_callable=io.StringIO):
                self.assertNotEqual(self.invoke(argv), 0)

    def test_local_revision_resolution_pins_every_dataset_once_before_offline_loading(self):
        import inspect

        from teacher_reliability import run

        resolver = getattr(run, "_local_dataset_revisions", None)
        self.assertTrue(callable(resolver), "local runs need one pinned commit for each cached dataset")
        from teacher_reliability.hub_data import DATASET_REPOSITORIES

        with patch("teacher_reliability.hub_data._resolve_dataset_revision", side_effect=lambda repository: repository.replace("/", "-").ljust(40, "x")) as resolve:
            revisions = resolver(None)
        self.assertEqual(set(revisions), set(DATASET_REPOSITORIES.values()))
        self.assertEqual(resolve.call_count, 3)

    def test_local_status_reads_the_persisted_progress_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            (run_dir / "run_manifest.json").write_text(json.dumps({
                "backend": "transformers-local", "status": "partial",
                "selected_example_count": 111, "completed_example_count": 40,
                "selected_example_ids": ["long-id-list-that-status-should-not-repeat"],
            }))
            (run_dir / "progress.json").write_text(json.dumps({"completed_rows": 40, "pending_rows": 71}))
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                result = self.invoke(["--status", "--run-dir", str(run_dir)])
            self.assertEqual(result, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["progress"]["completed_rows"], 40)
            self.assertEqual(payload["manifest"]["selected_example_count"], 111)
            self.assertNotIn("selected_example_ids", payload["manifest"])

    def test_stop_requires_running_local_manifest_and_writes_a_request_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            (run_dir / "run_manifest.json").write_text(json.dumps({
                "backend": "transformers-local", "status": "running", "coordinator_pid": 1234,
            }))
            with patch("sys.stdout", new_callable=io.StringIO):
                result = self.invoke(["--stop", "--run-dir", str(run_dir)])
            self.assertEqual(result, 0)
            self.assertTrue((run_dir / "stop.requested").is_file())


if __name__ == "__main__":
    unittest.main()
