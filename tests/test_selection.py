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

    def test_gsm_plus_selection_avoids_duplicate_seed_questions(self):
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

    def test_mini_profile_selects_the_111_row_study_mix(self):
        mini = PROFILES.get("mini_12h")
        self.assertIsNotNone(mini, "mini_12h profile is required")
        self.assertEqual(set(PROFILES), {"smoke", "mini_12h"})
        self.assertEqual(
            (mini.target_hours, mini.gsm8k_limit, mini.math_per_subject,
             mini.gsm_plus_limit, mini.sample_count, mini.max_new_tokens),
            (12, 40, 3, 50, 8, 1024),
        )
        self.assertTrue(mini.gsm_plus_one_per_seed)

        subjects = (
            "algebra", "counting_and_probability", "geometry",
            "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
        )
        rows = (
            [example("gsm8k", "main", index) for index in range(100)]
            + [example("math", subject, index) for subject in subjects for index in range(10)]
            + [example("gsm_plus", "default", index, f"seed-{index % 80}")
               for index in range(160)]
        )
        selected = select_examples(rows, mini, seed=42)
        counts = {dataset: sum(row.dataset == dataset for row in selected)
                  for dataset in ("gsm8k", "math", "gsm_plus")}
        self.assertEqual(counts, {"gsm8k": 40, "math": 21, "gsm_plus": 50})
        self.assertEqual(
            {subject: sum(row.source_config == subject for row in selected if row.dataset == "math")
             for subject in subjects},
            {subject: 3 for subject in subjects},
        )
        plus_seeds = [row.seed_question_id for row in selected if row.dataset == "gsm_plus"]
        self.assertEqual(len(plus_seeds), len(set(plus_seeds)))
        self.assertEqual(select_examples(list(reversed(rows)), mini, seed=42), selected)

if __name__ == "__main__":
    unittest.main()
