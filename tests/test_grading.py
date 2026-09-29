import unittest

from teacher_reliability.grading import answer_parseable, answers_equivalent, grade_prediction


class MathAnswerGradingTests(unittest.TestCase):
    def test_equivalent_fraction_and_decimal_are_mathematically_equal(self):
        self.assertTrue(answers_equivalent(r"\frac{1}{2}", "0.5"))

    def test_greedy_answer_is_compared_with_the_extracted_final_box(self):
        result = grade_prediction(
            "First I got \\boxed{9}. Rechecking gives \\boxed{12}.", "12"
        )

        self.assertTrue(result.correct)
        self.assertTrue(result.prediction_parseable)
        self.assertTrue(result.gold_parseable)
        self.assertEqual(result.status, "graded")

    def test_unparseable_prediction_is_incorrect_when_gold_is_parseable(self):
        result = grade_prediction("No final answer is provided.", "12")

        self.assertFalse(result.correct)
        self.assertFalse(result.prediction_parseable)
        self.assertTrue(result.gold_parseable)
        self.assertEqual(result.status, "missing_prediction_answer")

    def test_unparseable_gold_is_unscorable_instead_of_counted_wrong(self):
        result = grade_prediction(r"\boxed{12}", r"\frac{1}{")

        self.assertIsNone(result.correct)
        self.assertTrue(result.prediction_parseable)
        self.assertFalse(result.gold_parseable)
        self.assertEqual(result.status, "unparseable_gold")

    def test_missing_reference_is_unscorable(self):
        result = grade_prediction(r"\boxed{12}", None)

        self.assertIsNone(result.correct)
        self.assertEqual(result.status, "missing_gold")

    def test_parse_coverage_distinguishes_valid_expression_from_missing_or_malformed(self):
        self.assertTrue(answer_parseable("12"))
        self.assertFalse(answer_parseable(r"\frac{1}{"))
        self.assertFalse(answer_parseable(None))


if __name__ == "__main__":
    unittest.main()
