import json
import tempfile
import unittest
from pathlib import Path

from scripts.mini_distill import load_training_rows
from teacher_reliability.teacher import derive_sample_seed


SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
DATASET_REVISIONS = {
    "EleutherAI/hendrycks_math": "21a5633873b6a120296cce3e2df9d5550074f4a3",
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "qintongli/GSM-Plus": "3b708db57b96a16e8e3368ed2956990c0809440e",
}
DATASET_ROW_REVISIONS = {
    "gsm8k": DATASET_REVISIONS["openai/gsm8k"],
    "math": DATASET_REVISIONS["EleutherAI/hendrycks_math"],
    "gsm_plus": DATASET_REVISIONS["qintongli/GSM-Plus"],
}


def mini_manifest(rows):
    return {
        "status": "complete",
        "profile": {
            "name": "mini_12h",
            "gsm8k_limit": 40,
            "math_per_subject": 3,
            "gsm_plus_limit": 50,
            "gsm_plus_one_per_seed": True,
            "gsm_plus_split": "test",
            "sample_count": 8,
            "max_new_tokens": 1024,
            "target_hours": 12,
        },
        "seed": 42,
        "model_repository": "Qwen/Qwen2.5-Math-7B-Instruct",
        "model_revision": "ef9926d75ab1d54532f6a30dd5e760355eb9aa4d",
        "dataset_revisions": dict(DATASET_REVISIONS),
        "identity": {"datasets": dict(DATASET_REVISIONS)},
        "selected_example_count": 111,
        "selected_example_ids": [row["example_id"] for row in rows],
        "completed_example_count": 111,
        "failed_example_count": 0,
        "pending_example_count": 0,
        "imported_example_count": 0,
    }


def mini_rows():
    rows = []
    for dataset, config, count in (
        ("gsm8k", "main", 40),
        *(("math", subject, 3) for subject in SUBJECTS),
        ("gsm_plus", "default", 50),
    ):
        for index in range(count):
            example_id = f"{dataset}:{config}:test:{index}"
            rows.append({
                "example_id": example_id,
                "status": "complete",
                "source": {
                    "dataset": dataset,
                    "config": config,
                    "revision": DATASET_ROW_REVISIONS[dataset],
                },
                "requested_sample_count": 8,
                "samples": [{"index": sample_index,
                             "seed": derive_sample_seed(42, example_id, sample_index)}
                            for sample_index in range(8)],
            })
    return rows


def write_teacher_run(directory: Path, manifest, rows):
    (directory / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (directory / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class MiniTeacherInputTests(unittest.TestCase):
    def test_load_training_rows_accepts_only_a_complete_mini_12h_teacher_run(self):
        rows = mini_rows()
        manifest = mini_manifest(rows)
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            write_teacher_run(run_dir, manifest, rows)
            loaded = load_training_rows(run_dir / "predictions.jsonl")
        self.assertEqual(len(loaded), 111)

    def test_load_training_rows_rejects_a_full_profile_teacher_run(self):
        rows = mini_rows()
        manifest = mini_manifest(rows)
        manifest["profile"]["name"] = "full"
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            write_teacher_run(run_dir, manifest, rows)
            with self.assertRaisesRegex(ValueError, "mini_12h"):
                load_training_rows(run_dir / "predictions.jsonl")

    def test_load_training_rows_rejects_missing_sample_outputs(self):
        rows = mini_rows()
        rows[0]["samples"].pop()
        manifest = mini_manifest(mini_rows())
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            write_teacher_run(run_dir, manifest, rows)
            with self.assertRaisesRegex(ValueError, "sample"):
                load_training_rows(run_dir / "predictions.jsonl")

    def test_load_training_rows_rejects_a_teacher_run_that_is_not_complete(self):
        rows = mini_rows()
        manifest = mini_manifest(rows)
        manifest["status"] = "running"
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            write_teacher_run(run_dir, manifest, rows)
            with self.assertRaisesRegex(ValueError, "complete"):
                load_training_rows(run_dir / "predictions.jsonl")

    def test_load_training_rows_rejects_sample_seeds_that_do_not_match_the_run_seed(self):
        rows = mini_rows()
        rows[0]["samples"][0]["seed"] = 1
        manifest = mini_manifest(mini_rows())
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            write_teacher_run(run_dir, manifest, rows)
            with self.assertRaisesRegex(ValueError, "seed"):
                load_training_rows(run_dir / "predictions.jsonl")

    def test_load_training_rows_rejects_an_alternate_dataset_revision(self):
        rows = mini_rows()
        manifest = mini_manifest(rows)
        manifest["dataset_revisions"]["openai/gsm8k"] = "a" * 40
        manifest["identity"]["datasets"]["openai/gsm8k"] = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            write_teacher_run(run_dir, manifest, rows)
            with self.assertRaisesRegex(ValueError, "dataset revision"):
                load_training_rows(run_dir / "predictions.jsonl")

    def test_load_training_rows_rejects_a_row_from_an_alternate_revision(self):
        rows = mini_rows()
        rows[0]["source"]["revision"] = "a" * 40
        manifest = mini_manifest(mini_rows())
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            write_teacher_run(run_dir, manifest, rows)
            with self.assertRaisesRegex(ValueError, "dataset revision"):
                load_training_rows(run_dir / "predictions.jsonl")


if __name__ == "__main__":
    unittest.main()
