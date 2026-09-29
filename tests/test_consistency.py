import unittest

from teacher_reliability.consistency import summarize_consistency


class SelfConsistencyTests(unittest.TestCase):
    def test_greedy_agreement_and_majority_use_math_equivalence(self):
        summary = summarize_consistency(
            greedy_answer="1",
            sample_answers=["1", "1.0", r"\frac{2}{2}", "2"],
            gold_answer="1",
        )

        self.assertEqual(summary.sample_count, 4)
        self.assertEqual(summary.greedy_agreement_count, 3)
        self.assertAlmostEqual(summary.greedy_agreement_share, 0.75)
        self.assertEqual(summary.majority_answer, "1")
        self.assertEqual(summary.majority_vote_count, 3)
        self.assertAlmostEqual(summary.majority_vote_share, 0.75)
        self.assertTrue(summary.majority_correct)

    def test_first_sample_breaks_a_vote_tie_reproducibly(self):
        summary = summarize_consistency("1", ["2", "1"], "1")

        self.assertEqual(summary.majority_answer, "2")
        self.assertEqual(summary.majority_vote_count, 1)
        self.assertAlmostEqual(summary.majority_vote_share, 0.5)
        self.assertFalse(summary.majority_correct)

    def test_missing_sample_answers_vote_as_abstentions_and_missing_gold_is_unscorable(self):
        summary = summarize_consistency(None, [None, None, "2"], None)

        self.assertEqual(summary.greedy_agreement_count, 0)
        self.assertEqual(summary.greedy_agreement_share, 0.0)
        self.assertIsNone(summary.majority_answer)
        self.assertEqual(summary.majority_vote_count, 2)
        self.assertIsNone(summary.majority_correct)

    def test_empty_sample_list_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            summarize_consistency("1", [], "1")

    def test_sample_parse_coverage_counts_only_math_verify_expressions(self):
        summary = summarize_consistency("12", ["12", r"\frac{1}{", None], "12")

        self.assertEqual(summary.sample_answer_parseable_count, 1)


if __name__ == "__main__":
    unittest.main()
