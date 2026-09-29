import math
import unittest

from teacher_reliability.answers import AnswerSpan
from teacher_reliability.confidence import aggregate_confidence


class TokenConfidenceTests(unittest.TestCase):
    def test_special_token_is_excluded_and_answer_span_selects_overlapping_tokens(self):
        span = AnswerSpan("23", 1, 3, "boxed")
        result = aggregate_confidence(
            token_entropies=[math.log(2), math.log(2), 100.0],
            token_logprobs=[math.log(0.5), math.log(0.25), -100.0],
            token_char_spans=[(0, 2), (2, 4), None],
            content_token_mask=[True, True, False],
            answer_span=span,
            vocab_size=2,
        )

        self.assertEqual(result.content_token_count, 2)
        self.assertAlmostEqual(result.mean_entropy_nats, math.log(2))
        self.assertAlmostEqual(result.entropy_confidence, 0.0)
        self.assertAlmostEqual(result.sequence_logprob_sum, math.log(0.125))
        self.assertAlmostEqual(result.sequence_geometric_probability, math.sqrt(0.125))
        self.assertAlmostEqual(result.answer_token_geometric_probability, math.sqrt(0.125))
        self.assertEqual(result.answer_token_count, 2)
        self.assertEqual(result.answer_alignment_status, "aligned")

    def test_missing_token_offsets_leave_answer_probability_missing(self):
        result = aggregate_confidence(
            token_entropies=[0.2],
            token_logprobs=[math.log(0.8)],
            token_char_spans=None,
            content_token_mask=[True],
            answer_span=AnswerSpan("8", 0, 1, "boxed"),
            vocab_size=4,
        )

        self.assertIsNone(result.answer_token_geometric_probability)
        self.assertEqual(result.answer_alignment_status, "token_alignment_unavailable")

    def test_score_vectors_must_match_each_other(self):
        with self.assertRaisesRegex(ValueError, "same token count"):
            aggregate_confidence(
                token_entropies=[0.1],
                token_logprobs=[-0.2, -0.3],
                token_char_spans=[],
                content_token_mask=[True],
                answer_span=None,
                vocab_size=4,
            )


if __name__ == "__main__":
    unittest.main()
