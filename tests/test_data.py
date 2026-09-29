import unittest

from teacher_reliability.data import normalize_example


class DatasetNormalizationTests(unittest.TestCase):
    def test_gsm8k_keeps_only_the_reference_after_the_last_marker(self):
        example = normalize_example(
            "gsm8k",
            "main",
            "test",
            17,
            {
                "question": "How many are left?",
                "answer": "First try: #### 9\nRecheck: 20 - 8 = 12. #### 12",
            },
            source_revision="gsm8k-sha",
        )

        self.assertEqual(example.gold_answer, "12")
        self.assertEqual(example.example_id, "gsm8k:main:test:17")
        self.assertEqual(example.source_revision, "gsm8k-sha")
        self.assertTrue(example.scorable)

    def test_math_uses_the_last_boxed_solution_answer_and_subject(self):
        example = normalize_example(
            "math",
            "algebra",
            "test",
            4,
            {
                "problem": "Find x.",
                "solution": r"Try x=1, so \boxed{1}. Correct value is \boxed{\frac{3}{2}}.",
                "level": "Level 3",
                "type": "Algebra",
            },
            source_revision="math-sha",
        )

        self.assertEqual(example.gold_answer, r"\frac{3}{2}")
        self.assertEqual(example.subject, "algebra")
        self.assertEqual(
            example.gold_solution,
            r"Try x=1, so \boxed{1}. Correct value is \boxed{\frac{3}{2}}.",
        )

    def test_gsm_plus_null_answer_is_retained_as_unscorable(self):
        example = normalize_example(
            "gsm_plus",
            "default",
            "test",
            9,
            {
                "question": "How many did she sell?",
                "solution": "The starting quantity is missing.",
                "answer": None,
                "perturbation_type": "critical thinking",
                "seed_question": "A source question.",
            },
            source_revision="gsm-plus-sha",
        )

        self.assertIsNone(example.gold_answer)
        self.assertFalse(example.scorable)
        self.assertEqual(example.perturbation_type, "critical thinking")
        same_seed = normalize_example(
            "gsm_plus",
            "default",
            "test",
            10,
            {
                "question": "A changed version of the source question.",
                "solution": "A solution.",
                "answer": "3",
                "perturbation_type": "numerical substitution",
                "seed_question": "A source question.",
            },
            source_revision="gsm-plus-sha",
        )
        self.assertEqual(example.seed_question_id, same_seed.seed_question_id)

    def test_example_ids_include_dataset_configuration_and_split(self):
        first = normalize_example(
            "gsm8k",
            "main",
            "test",
            2,
            {"question": "Q?", "answer": "#### 2"},
            source_revision="r1",
        )
        second = normalize_example(
            "gsm8k",
            "socratic",
            "train",
            2,
            {"question": "Q?", "answer": "#### 2"},
            source_revision="r1",
        )

        self.assertNotEqual(first.example_id, second.example_id)

    def test_unknown_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_example("unknown", "default", "test", 0, {}, "sha")


if __name__ == "__main__":
    unittest.main()
