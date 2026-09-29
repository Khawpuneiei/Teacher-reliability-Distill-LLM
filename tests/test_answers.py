import unittest

from teacher_reliability.answers import extract_final_answer_span


class FinalAnswerSpanTests(unittest.TestCase):
    def test_last_boxed_answer_supports_nested_latex_braces(self):
        response = r"Work gives \boxed{2}; therefore \boxed{\frac{1}{2}}."

        span = extract_final_answer_span(response)

        self.assertIsNotNone(span)
        self.assertEqual(span.value, r"\frac{1}{2}")
        self.assertEqual(response[span.start : span.end], span.value)
        self.assertEqual(span.method, "boxed")

    def test_final_answer_label_is_used_when_no_box_is_present(self):
        response = "Reasoning line.\nFinal answer: 42"

        span = extract_final_answer_span(response)

        self.assertEqual(span.value, "42")
        self.assertEqual(span.method, "answer_label")

    def test_unclosed_box_does_not_invent_an_answer(self):
        self.assertIsNone(extract_final_answer_span(r"The result is \boxed{1+"))

    def test_escaped_braces_inside_box_do_not_end_the_answer(self):
        response = r"Final: \boxed{\{1,2\}}"

        span = extract_final_answer_span(response)

        self.assertEqual(span.value, r"\{1,2\}")


if __name__ == "__main__":
    unittest.main()
