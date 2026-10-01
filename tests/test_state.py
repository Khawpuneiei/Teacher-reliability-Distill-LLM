import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from teacher_reliability.state import (
    RunAlreadyActive,
    atomic_write_json,
    load_completed_predictions,
    run_lock,
)


class RunStateTests(unittest.TestCase):
    def test_run_lock_rejects_a_second_writer_and_releases_on_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            with run_lock(run_dir):
                with self.assertRaises(RunAlreadyActive):
                    with run_lock(run_dir):
                        pass

            with run_lock(run_dir):
                pass

    def test_only_complete_checkpoints_are_returned(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "example_id": "gsm8k:main:test:1",
                        "status": "complete",
                        "greedy": {"text": "..."},
                        "samples": ["1"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            records = load_completed_predictions(path)

            self.assertEqual(set(records), {"gsm8k:main:test:1"})
            self.assertEqual(records["gsm8k:main:test:1"]["samples"], ["1"])

    def test_incomplete_trailing_jsonl_line_is_removed_for_safe_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            complete = (
                json.dumps(
                    {
                        "example_id": "gsm8k:main:test:1",
                        "status": "complete",
                        "greedy": {"text": "..."},
                        "samples": ["1"],
                    }
                )
                + "\n"
            ).encode("utf-8")
            path.write_bytes(complete + b'{"example_id":"partial')

            records = load_completed_predictions(path)

            self.assertEqual(set(records), {"gsm8k:main:test:1"})
            self.assertEqual(path.read_bytes(), complete)

    def test_read_only_checkpoint_load_leaves_a_torn_tail_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            complete = (
                json.dumps(
                    {
                        "example_id": "gsm8k:main:test:1",
                        "status": "complete",
                        "greedy": {"text": "..."},
                        "samples": ["1"],
                    }
                )
                + "\n"
            ).encode("utf-8")
            original = complete + b'{"example_id":"partial'
            path.write_bytes(original)

            records = load_completed_predictions(path, repair_torn_tail=False)

            self.assertEqual(set(records), {"gsm8k:main:test:1"})
            self.assertEqual(path.read_bytes(), original)

    def test_duplicate_completed_checkpoint_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            record = {
                "example_id": "gsm8k:main:test:1",
                "status": "complete",
                "greedy": {"text": "..."},
                "samples": ["1"],
            }
            path.write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate checkpoint"):
                load_completed_predictions(path)

    def test_json_manifest_is_atomically_replaced_with_canonical_valid_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run_manifest.json"

            atomic_write_json(path, {"z": 2, "a": 1})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": 1, "z": 2})
            self.assertEqual(list(Path(temporary).iterdir()), [path])

    def test_json_manifest_retries_transient_replace_denial_and_preserves_old_file_on_persistent_denial(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "qualification.json"
            path.write_text('{"version": 1}\n', encoding="utf-8")
            real_replace = os.replace
            attempts = []

            def deny_twice_then_replace(source, destination):
                attempts.append((Path(source), Path(destination)))
                if len(attempts) < 3:
                    raise PermissionError("temporary Windows file lock")
                return real_replace(source, destination)

            failure = None
            with patch("teacher_reliability.state.os.replace", side_effect=deny_twice_then_replace):
                try:
                    atomic_write_json(path, {"version": 2})
                except PermissionError as exc:
                    failure = exc

            self.assertIsNone(failure, "a transient replace denial must be retried")
            self.assertEqual(len(attempts), 3)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"version": 2})
            self.assertEqual(list(Path(temporary).iterdir()), [path])

            previous_payload = path.read_bytes()
            persistent_failure = None
            persistent_attempts = []

            def deny_persistently(_source, _destination):
                persistent_attempts.append(None)
                raise PermissionError("persistent Windows file lock")

            with patch(
                "teacher_reliability.state.os.replace",
                side_effect=deny_persistently,
            ):
                try:
                    atomic_write_json(path, {"version": 3})
                except PermissionError as exc:
                    persistent_failure = exc

            self.assertIsInstance(persistent_failure, PermissionError)
            self.assertIn("persistent Windows file lock", str(persistent_failure))
            self.assertIn("after 5 attempts", str(persistent_failure))
            self.assertEqual(len(persistent_attempts), 5)
            self.assertEqual(path.read_bytes(), previous_payload)
            self.assertEqual(list(Path(temporary).iterdir()), [path])

    def test_json_manifest_handles_integer_and_string_device_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run_manifest.json"

            atomic_write_json(
                path,
                {"model_load": {"max_memory": {0: "6GiB", "cpu": "12GiB"}}},
            )

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"model_load": {"max_memory": {"0": "6GiB", "cpu": "12GiB"}}},
            )


if __name__ == "__main__":
    unittest.main()
