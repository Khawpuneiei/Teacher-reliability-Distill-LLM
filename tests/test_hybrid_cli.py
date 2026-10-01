"""CLI dispatch contracts without model downloads or a GPU."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from teacher_reliability import run
from teacher_reliability.hybrid import CudaOutOfMemory, MemoryProfile, WorkerStartupOutOfMemory
from teacher_reliability.run import main


class HybridCliTests(unittest.TestCase):
    def test_hybrid_default_run_directory_is_shared_with_dispatch_and_output(self):
        expected = Path("outputs") / "fixture"
        with patch("teacher_reliability.run._default_run_dir", return_value=expected), patch(
            "teacher_reliability.run._execute_hybrid_cli",
            return_value={"run_id": "fixture", "status": "partial", "completed_example_count": 0, "selected_example_count": 1},
        ) as execute, patch("sys.stdout", new_callable=io.StringIO) as stdout:
            result = main(["--backend", "vllm-hybrid"])
        self.assertEqual(result, 0)
        self.assertEqual(execute.call_args.args[0].run_dir, expected)
        self.assertIn(str(expected / "run_manifest.json"), stdout.getvalue())

    def test_vllm_environment_manifest_captures_transitive_package_versions(self):
        payload = '{"vllm":"0.30.0","vllm-bnb-plugin":"0.0.3","bitsandbytes":"0.50.2","numpy":"2.4.0"}'
        with patch("teacher_reliability.run.subprocess.check_output", return_value=payload) as command:
            versions = run._vllm_package_versions(Path(".venv-vllm/bin/python"))
        self.assertEqual(versions["numpy"], "2.4.0")
        self.assertIn("distributions", command.call_args.args[0][2])

    def test_global_free_memory_preflight_rejects_overbudget_vllm_load(self):
        check = getattr(run, "_require_vllm_headroom", lambda *_: None)
        profile = MemoryProfile("vllm-16", 16, 4096, 0.60)
        try:
            check({"free_mib": 30000, "total_mib": 81920}, profile)
        except CudaOutOfMemory:
            rejected = True
        else:
            rejected = False
        self.assertTrue(rejected)
        check({"free_mib": 60000, "total_mib": 81920}, profile)
        with self.assertRaises(WorkerStartupOutOfMemory):
            check({"free_mib": 10000, "total_mib": 81920}, MemoryProfile("vllm-1", 1, 1024, 0.40, True))

    def test_hybrid_flags_dispatch_to_isolated_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            with patch("teacher_reliability.run._execute_hybrid_cli", create=True) as run_hybrid_cli:
                run_hybrid_cli.return_value = {
                    "run_id": "run", "status": "partial",
                    "completed_example_count": 0, "selected_example_count": 2,
                }
                with patch("sys.stdout", new_callable=io.StringIO), patch("sys.stderr", new_callable=io.StringIO):
                    try:
                        result = main([
                            "--backend", "vllm-hybrid", "--profile", "full",
                            "--run-dir", str(run_dir), "--model-dir", "models/qwen-nf4",
                            "--vllm-python", ".venv-vllm/bin/python",
                            "--max-runtime-hours", "12",
                        ])
                    except SystemExit as error:
                        result = error.code
            self.assertEqual(result, 0)
            self.assertEqual(run_hybrid_cli.call_count, 1)
            args = run_hybrid_cli.call_args.args[0]
            self.assertEqual(args.backend, "vllm-hybrid")
            self.assertEqual(args.max_runtime_hours, 12)


if __name__ == "__main__":
    unittest.main()
