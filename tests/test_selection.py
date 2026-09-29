import unittest

from teacher_reliability.data import Example
from teacher_reliability.selection import PROFILES, RunProfile, get_profile, select_examples


def example(dataset, config, row_id, seed_question_id=None):
    return Example(
        example_id=f"{dataset}:{config}:test:{row_id}",
        dataset=dataset,
        source_revision="fixture-sha",
        source_config=config,
        split="test",
        source_row_id=str(row_id),
        question=f"Question {row_id}?",
        gold_answer="1",
        seed_question_id=seed_question_id,
    )


class StableSelectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            example("gsm8k", "main", i) for i in range(4)
        ] + [
            example("math", "algebra", i) for i in range(3)
        ] + [
            example("math", "geometry", i) for i in range(2)
        ] + [
            example("gsm_plus", "default", 0, "seed-a"),
            example("gsm_plus", "default", 1, "seed-a"),
            example("gsm_plus", "default", 2, "seed-b"),
            example("gsm_plus", "default", 3, "seed-c"),
        ]
        self.profile = RunProfile(
            name="fixture",
            gsm8k_limit=2,
            math_per_subject=1,
            gsm_plus_limit=2,
            gsm_plus_one_per_seed=True,
            gsm_plus_split="test",
            sample_count=2,
            max_new_tokens=32,
        )

    def test_caps_are_stable_and_math_quota_is_per_subject(self):
        first = select_examples(self.rows, self.profile, seed=42)
        reordered = select_examples(list(reversed(self.rows)), self.profile, seed=42)

        self.assertEqual([row.example_id for row in first], [row.example_id for row in reordered])
        selected = {row.example_id for row in first}
        self.assertEqual(
            {item for item in selected if item.startswith("gsm8k:")},
            {"gsm8k:main:test:1", "gsm8k:main:test:3"},
        )
        self.assertEqual(
            {item for item in selected if item.startswith("math:")},
            {"math:algebra:test:2", "math:geometry:test:0"},
        )

    def test_gsm_plus_pilot_avoids_duplicate_seed_questions(self):
        selected = select_examples(self.rows, self.profile, seed=42)
        gsm_plus = [row for row in selected if row.dataset == "gsm_plus"]

        self.assertEqual(
            {row.example_id for row in gsm_plus},
            {"gsm_plus:default:test:2", "gsm_plus:default:test:3"},
        )
        self.assertEqual(len({row.seed_question_id for row in gsm_plus}), 2)

    def test_selection_rejects_duplicate_example_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate example_id"):
            select_examples(self.rows + [self.rows[0]], self.profile, seed=42)

    def test_profile_rejects_zero_sample_count(self):
        with self.assertRaisesRegex(ValueError, "sample_count"):
            RunProfile(
                name="bad",
                gsm8k_limit=1,
                math_per_subject=1,
                gsm_plus_limit=1,
                gsm_plus_one_per_seed=True,
                gsm_plus_split="test",
                sample_count=0,
                max_new_tokens=32,
            )

    def test_smoke_profile_allows_reasoned_answers_to_reach_the_final_answer(self):
        smoke = get_profile("smoke")

        self.assertEqual(smoke.sample_count, 2)
        self.assertGreaterEqual(smoke.max_new_tokens, 1024)

    def test_a100_time_budget_profiles_bound_deterministic_representative_workload(self):
        twelve_hour = PROFILES.get("a100_12h")
        twenty_four_hour = PROFILES.get("a100_24h")
        self.assertIsNotNone(twelve_hour, "a100_12h profile is required")
        self.assertIsNotNone(twenty_four_hour, "a100_24h profile is required")
        if twelve_hour is None or twenty_four_hour is None:
            return

        self.assertEqual(
            (twelve_hour.target_hours, twelve_hour.gsm8k_limit,
             twelve_hour.math_per_subject, twelve_hour.gsm_plus_limit,
             twelve_hour.sample_count),
            (12, 150, 50, 220, 8),
        )
        self.assertEqual(
            (twenty_four_hour.target_hours, twenty_four_hour.gsm8k_limit,
             twenty_four_hour.math_per_subject, twenty_four_hour.gsm_plus_limit,
             twenty_four_hour.sample_count),
            (24, 300, 50, 790, 8),
        )
        self.assertEqual(twelve_hour.as_dict()["target_hours"], 12)
        self.assertEqual(twenty_four_hour.as_dict()["target_hours"], 24)
        self.assertTrue(twelve_hour.gsm_plus_one_per_seed)
        self.assertTrue(twenty_four_hour.gsm_plus_one_per_seed)

        subjects = (
            "algebra", "counting_and_probability", "geometry",
            "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
        )
        rows = (
            [example("gsm8k", "main", index) for index in range(300)]
            + [example("math", subject, index) for subject in subjects for index in range(50)]
            + [example("gsm_plus", "default", index, f"seed-{index}")
               for index in range(790)]
        )
        selected_12 = select_examples(rows, twelve_hour, seed=42)
        selected_24 = select_examples(rows, twenty_four_hour, seed=42)
        counts_12 = {dataset: sum(row.dataset == dataset for row in selected_12)
                     for dataset in ("gsm8k", "math", "gsm_plus")}
        counts_24 = {dataset: sum(row.dataset == dataset for row in selected_24)
                     for dataset in ("gsm8k", "math", "gsm_plus")}
        self.assertEqual(counts_12, {"gsm8k": 150, "math": 350, "gsm_plus": 220})
        self.assertEqual(counts_24, {"gsm8k": 300, "math": 350, "gsm_plus": 790})
        self.assertEqual(
            {subject: sum(row.source_config == subject for row in selected_12 if row.dataset == "math")
             for subject in subjects},
            {subject: 50 for subject in subjects},
        )
        self.assertEqual((1 + twelve_hour.sample_count) * len(selected_12), 6_480)
        self.assertEqual((1 + twenty_four_hour.sample_count) * len(selected_24), 12_960)
        self.assertTrue({row.example_id for row in selected_12}.issubset(
            {row.example_id for row in selected_24}
        ))


if __name__ == "__main__":
    unittest.main()
