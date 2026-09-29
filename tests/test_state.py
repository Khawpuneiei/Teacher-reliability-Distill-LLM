import json
import tempfile
import unittest
from pathlib import Path

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
