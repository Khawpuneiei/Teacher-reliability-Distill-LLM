"""Windows local runner dispatch, path and lifecycle command contracts."""

import io
import json
import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from teacher_reliability.run import main


class LocalCliTests(unittest.TestCase):
    def invoke(self, argv):
        try:
            return main(argv)
        except SystemExit as error:
            return error.code

    def test_local_backend_dispatches_import_and_keeps_deadline_unset_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "local-run"
            archive = root / "source.zip"
            qualification = root / "qualification_manifest.json"
            expected = {
                "run_id": "local-run", "status": "partial",
                "completed_example_count": 110, "selected_example_count": 12221,
                "estimated_remaining_hours": 10.0,
            }
            with patch("teacher_reliability.run._default_local_run_dir", return_value=run_dir, create=True), patch(
                "teacher_reliability.run._execute_local_cli", return_value=expected, create=True
            ) as execute, patch("sys.stdout", new_callable=io.StringIO) as stdout:
                result = self.invoke([
                    "--backend", "transformers-local", "--profile", "full",
                    "--device-preset", "rtx4060-8gb", "--import-run-zip", str(archive),
                    "--qualification-manifest", str(qualification),
                    "--retry-failed",
                ])
            self.assertEqual(result, 0)
            args = execute.call_args.args[0]
            self.assertEqual(args.backend, "transformers-local")
            self.assertEqual(args.device_preset, "rtx4060-8gb")
            self.assertEqual(args.import_run_zip, archive)
            self.assertEqual(args.qualification_manifest, qualification)
            self.assertTrue(args.retry_failed)
            self.assertIsNone(args.max_runtime_hours)
            self.assertIn("qualification ETA at launch: 10.0 hours", stdout.getvalue())
            pointer = root / "current_run_dir.txt"
            self.assertTrue(pointer.is_file(), "local runs need a durable path for later status and resume commands")
            self.assertEqual(pointer.read_text(encoding="utf-8"), str(run_dir.resolve()))

    def test_full_local_run_cannot_start_without_import_archive_and_qualification(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            with patch("sys.stdout", new_callable=io.StringIO), patch(
                "sys.stderr", new_callable=io.StringIO
            ), patch("teacher_reliability.run._execute_local_cli") as execute:
                result = self.invoke([
                    "--backend", "transformers-local", "--profile", "full",
                    "--run-dir", str(run_dir),
                ])

            self.assertEqual(result, 1)
            execute.assert_not_called()

    def test_full_local_run_accepts_explicit_unqualified_fast_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "local-run"
            archive = root / "a100.zip"
            expected = {
                "run_id": "local-run", "status": "partial",
                "completed_example_count": 110, "selected_example_count": 12221,
                "estimated_remaining_hours": None,
            }
            with patch("teacher_reliability.run._execute_local_cli", return_value=expected) as execute, patch(
                "sys.stdout", new_callable=io.StringIO
            ):
                result = self.invoke([
                    "--backend", "transformers-local", "--profile", "full",
                    "--device-preset", "rtx4060-8gb", "--import-run-zip", str(archive),
                    "--fast-start-unqualified", "--run-dir", str(run_dir),
                ])

            self.assertEqual(result, 0)
            args = execute.call_args.args[0]
            self.assertTrue(args.fast_start_unqualified)
            self.assertIsNone(args.qualification_manifest)

    def test_full_local_run_rejects_non_42_seed_before_dataset_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as source:
                source.writestr("source/run_manifest.json", "{}")
                source.writestr("source/predictions.jsonl", "{}\n")
            with patch("teacher_reliability.run._load_cached_examples", side_effect=AssertionError(
                "unpinned seed reached dataset loading"
            )), patch("sys.stderr", new_callable=io.StringIO) as stderr:
                result = self.invoke([
                    "--backend", "transformers-local", "--profile", "full",
                    "--device-preset", "rtx4060-8gb", "--seed", "43",
                    "--import-run-zip", str(archive), "--fast-start-unqualified",
                    "--run-dir", str(root / "run"),
                ])

            self.assertEqual(result, 1)
            self.assertIn("full local study seed must be 42", stderr.getvalue())

    def test_full_local_run_rejects_alternate_zip_using_imported_bytes(self):
        from teacher_reliability.run import LOCAL_MODEL_REVISION

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "alternate.zip"
            with zipfile.ZipFile(archive, "w") as source:
                source.writestr("source/run_manifest.json", json.dumps({
                    "model_revision": LOCAL_MODEL_REVISION,
                    "dataset_revisions": {
                        "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
                        "EleutherAI/hendrycks_math": "21a5633873b6a120296cce3e2df9d5550074f4a3",
                        "qintongli/GSM-Plus": "3b708db57b96a16e8e3368ed2956990c0809440e",
                    },
                }))
                source.writestr("source/predictions.jsonl", "{}\n")
            with patch("teacher_reliability.run._load_cached_examples", return_value=([], {
                "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
                "EleutherAI/hendrycks_math": "21a5633873b6a120296cce3e2df9d5550074f4a3",
                "qintongli/GSM-Plus": "3b708db57b96a16e8e3368ed2956990c0809440e",
            })), patch("transformers.AutoTokenizer.from_pretrained", return_value=SimpleNamespace(
                is_fast=True
            )), patch("sys.stderr", new_callable=io.StringIO) as stderr:
                result = self.invoke([
                    "--backend", "transformers-local", "--profile", "full",
                    "--device-preset", "rtx4060-8gb", "--seed", "42",
                    "--import-run-zip", str(archive), "--fast-start-unqualified",
                    "--run-dir", str(root / "run"),
                ])

            self.assertEqual(result, 1)
            self.assertIn("pinned A100 archive SHA-256", stderr.getvalue())

    def test_fast_start_qualification_record_is_explicitly_unqualified(self):
        from teacher_reliability.run import _fast_start_qualification_record

        record = _fast_start_qualification_record()

        self.assertEqual(record["status"], "unqualified-fast-start")
        self.assertFalse(record["qualified"])
        self.assertEqual(record["configuration"], "default-greedy-1-sample-2")
        self.assertEqual(record["actual_greedy_profile"]["batch_size"], 1)
        self.assertEqual(record["actual_sample_profile"]["batch_size"], 2)
        self.assertIsNone(record["dataset_weighted_eta_hours"])
        self.assertIn("matrix was skipped", record["reason"])

    def test_full_run_reuses_the_recommended_measured_recovery_profiles(self):
        from teacher_reliability import run

        derive = getattr(run, "_recovery_profiles_from_qualification", None)
        self.assertTrue(callable(derive), "full run must start from its qualified memory profiles")
        configuration = {
            "actual_greedy_profile": {"name": "greedy-2", "batch_size": 2, "cache_implementation": None},
            "actual_sample_profile": {"name": "sample-4", "batch_size": 4, "cache_implementation": None},
        }

        greedy, sampling = derive(configuration)

        self.assertEqual([profile.batch_size for profile in greedy], [2, 1, 1])
        self.assertEqual([profile.cache_implementation for profile in greedy], [None, None, "offloaded"])
        self.assertEqual([profile.batch_size for profile in sampling], [4, 2, 1, 1])

    def test_full_run_binds_selected_qualification_to_archive_model_data_and_hardware(self):
        from teacher_reliability import run

        load = getattr(run, "_load_local_qualification", None)
        self.assertTrue(callable(load), "full run must verify the qualification artifact before launch")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_revisions = {"openai/gsm8k": "1" * 40}
            hardware = {
                "platform": "Windows-test", "gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
                "gpu_total_memory_mib": 8192,
            }
            qualification_identity = {
                "model_revision": "m" * 40, "dataset_revisions": dataset_revisions,
                "source_archive_sha256": "a" * 64, "seed": 42,
                "source_snapshot_sha256": run._source_snapshot(Path(run.__file__).resolve().parents[1]),
                "software": {"torch": "2.5.1+cu121"},
                "hardware": {
                    "platform": hardware["platform"], "gpu_name": hardware["gpu_name"],
                    "gpu_total_memory_bytes": 8192 * 1024**2,
                },
            }
            qualification_signature = run._canonical_digest(qualification_identity)
            config_name = "g1-s1"
            config_dir = root / config_name
            config_dir.mkdir()
            config = {
                "run_signature": run._canonical_digest({
                    "qualification": qualification_signature, "config": config_name,
                }),
                "status": "complete", "greedy": {str(i): {} for i in range(24)},
                "samples": {str(i): {} for i in range(192)}, "failures": [],
                "actual_greedy_profile": {
                    "name": "greedy-1-offloaded", "batch_size": 1,
                    "cache_implementation": "offloaded",
                },
                "actual_sample_profile": {
                    "name": "sample-1-offloaded", "batch_size": 1,
                    "cache_implementation": "offloaded",
                },
                "rows_per_hour": 2.0,
                "rows_per_hour_by_dataset": {"gsm8k": 1.0, "math": 2.0, "gsm_plus": 3.0},
                "memory": {"minimum_global_free_mib": 900},
            }
            config_path = config_dir / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            qualification = {
                "status": "complete", "qualification_signature": qualification_signature,
                "identity": qualification_identity, "recommended_configuration": config_name,
                "settings": {config_name: {"status": "complete"}},
                "dataset_weighted_eta_hours": 3.0,
                "remaining_rows_by_dataset": {"gsm8k": 1, "math": 2, "gsm_plus": 3},
                "stress_test": {"status": "complete", "global_memory": {"free_mib": 1200}},
            }
            qualification_path = root / "qualification_manifest.json"
            qualification_path.write_text(json.dumps(qualification), encoding="utf-8")

            selected = load(
                qualification_path, model_revision="m" * 40,
                dataset_revisions=dataset_revisions, seed=42,
                source_archive_sha256="a" * 64,
                package_versions={"torch": "2.5.1+cu121"}, hardware=hardware,
                expected_remaining_rows_by_dataset={"gsm8k": 1, "math": 2, "gsm_plus": 3},
            )

            self.assertEqual(selected["configuration"], config_name)
            self.assertEqual(selected["dataset_weighted_eta_hours"], 3.0)
            self.assertEqual(selected["manifest_sha256"], hashlib.sha256(qualification_path.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, "remaining row counts differ"):
                load(
                    qualification_path, model_revision="m" * 40,
                    dataset_revisions=dataset_revisions, seed=42,
                    source_archive_sha256="a" * 64,
                    package_versions={"torch": "2.5.1+cu121"}, hardware=hardware,
                    expected_remaining_rows_by_dataset={"gsm8k": 2, "math": 2, "gsm_plus": 3},
                )
            qualification["stress_test"] = {"status": "oom", "global_memory": {"free_mib": 400}}
            qualification_path.write_text(json.dumps(qualification), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "1,024-token stress check"):
                load(
                    qualification_path, model_revision="m" * 40,
                    dataset_revisions=dataset_revisions, seed=42,
                    source_archive_sha256="a" * 64,
                    package_versions={"torch": "2.5.1+cu121"}, hardware=hardware,
                    expected_remaining_rows_by_dataset={"gsm8k": 1, "math": 2, "gsm_plus": 3},
                )
            qualification["stress_test"] = {"status": "complete", "global_memory": {"free_mib": 1200}}
            qualification_identity["source_snapshot_sha256"] = "f" * 64
            qualification["identity"] = qualification_identity
            qualification_signature = run._canonical_digest(qualification_identity)
            qualification["qualification_signature"] = qualification_signature
            config["run_signature"] = run._canonical_digest({
                "qualification": qualification_signature, "config": config_name,
            })
            config_path.write_text(json.dumps(config), encoding="utf-8")
            qualification_path.write_text(json.dumps(qualification), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source code snapshot differs"):
                load(
                    qualification_path, model_revision="m" * 40,
                    dataset_revisions=dataset_revisions, seed=42,
                    source_archive_sha256="a" * 64,
                    package_versions={"torch": "2.5.1+cu121"}, hardware=hardware,
                    expected_remaining_rows_by_dataset={"gsm8k": 1, "math": 2, "gsm_plus": 3},
                )

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
                "selected_example_count": 12221, "completed_example_count": 110,
                "qualification": {"configuration": "g1-s1"},
                "estimated_remaining_hours": 123.5,
                "selected_example_ids": ["long-id-list-that-status-should-not-repeat"],
            }))
            (run_dir / "progress.json").write_text(json.dumps({"completed_rows": 110, "pending_rows": 12111}))
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                result = self.invoke(["--status", "--run-dir", str(run_dir)])
            self.assertEqual(result, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["progress"]["completed_rows"], 110)
            self.assertEqual(payload["manifest"]["selected_example_count"], 12221)
            self.assertEqual(payload["manifest"]["qualification"]["configuration"], "g1-s1")
            self.assertEqual(payload["manifest"]["estimated_remaining_hours"], 123.5)
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
