"""Greedy worker protocol validation without starting CUDA."""

import unittest

from teacher_reliability.greedy_worker import _error_response, handle_generate_request
from teacher_reliability.teacher import GreedyGeneration


class GreedyWorkerProtocolTests(unittest.TestCase):
    def test_host_memory_error_is_not_classified_as_cuda_oom(self):
        self.assertEqual(_error_response(MemoryError("out of memory"))["error_type"], "other")

    def test_batch_results_keep_request_identity_and_scalar_scores(self):
        calls = []

        def batcher(_tokenizer, _model, questions, max_new_tokens):
            calls.append((questions, max_new_tokens))
            return [
                GreedyGeneration(
                    text=question, token_ids=[index + 7],
                    token_entropies_nats=[0.2], token_logprobs=[-0.3],
                    token_char_spans=[(0, 1)], content_token_mask=[True],
                )
                for index, question in enumerate(questions)
            ]

        response = handle_generate_request({
            "op": "generate",
            "requests": [
                {"request_id": "later", "question": "pears", "max_new_tokens": 32},
                {"request_id": "first", "question": "apples", "max_new_tokens": 32},
            ],
        }, object(), object(), batcher=batcher)

        self.assertEqual(calls, [(["pears", "apples"], 32)])
        self.assertEqual([row["request_id"] for row in response["results"]], ["later", "first"])
        self.assertEqual(response["results"][1]["greedy"]["token_logprobs"], [-0.3])

    def test_mixed_output_limits_are_rejected_before_inference(self):
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            handle_generate_request({
                "op": "generate",
                "requests": [
                    {"request_id": "a", "question": "a", "max_new_tokens": 32},
                    {"request_id": "b", "question": "b", "max_new_tokens": 16},
                ],
            }, object(), object(), batcher=lambda *_: self.fail("called"))


if __name__ == "__main__":
    unittest.main()
