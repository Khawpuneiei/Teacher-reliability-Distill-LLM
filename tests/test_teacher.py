import unittest

from teacher_reliability.teacher import (
    GreedyGeneration,
    SampleGeneration,
    build_chat_messages,
    classify_termination,
    derive_sample_seed,
    isolated_torch_rng,
)


class TeacherPromptTests(unittest.TestCase):
    def test_prompt_uses_only_the_problem_and_a_boxed_final_answer_instruction(self):
        messages = build_chat_messages("How many apples remain?")

        self.assertEqual(
            messages,
            [
                {
                    "role": "user",
                    "content": (
                        "Solve the following math problem carefully. Show your "
                        "reasoning and put the final answer in \\boxed{...}.\n\n"
                        "Problem:\nHow many apples remain?"
                    ),
                }
            ],
        )

    def test_each_example_sample_pair_gets_a_repeatable_independent_seed(self):
        first = derive_sample_seed(42, "gsm8k:main:test:17", 3)

        self.assertEqual(first, 5792900222203665036)
        self.assertEqual(first, derive_sample_seed(42, "gsm8k:main:test:17", 3))
        self.assertNotEqual(first, derive_sample_seed(42, "gsm8k:main:test:17", 2))

    def test_seed_rejects_negative_sample_index(self):
        with self.assertRaisesRegex(ValueError, "sample_index"):
            derive_sample_seed(42, "example-1", -1)

    def test_greedy_record_preserves_the_token_alignment_failure_reason(self):
        generation = GreedyGeneration(
            text="answer",
            token_ids=[1],
            token_entropies_nats=[0.4],
            token_logprobs=[-0.2],
            token_char_spans=None,
            content_token_mask=[True],
            token_alignment_status="token_roundtrip_mismatch",
        )

        self.assertEqual(generation.token_alignment_status, "token_roundtrip_mismatch")

    def test_generation_records_distinguish_eos_from_max_token_truncation(self):
        completed = classify_termination([7, 8, 99], [99], 10)
        capped = classify_termination([7, 8], [99], 2)

        self.assertTrue(completed.eos_generated)
        self.assertFalse(completed.was_truncated)
        self.assertEqual(completed.reason, "eos")
        self.assertFalse(capped.eos_generated)
        self.assertTrue(capped.was_truncated)
        self.assertEqual(capped.reason, "max_new_tokens")

    def test_sample_generation_keeps_token_termination_metadata(self):
        sample = SampleGeneration("answer", [7, 99], True, False, "eos")

        self.assertEqual(sample.token_ids, [7, 99])
        self.assertEqual(sample.reason, "eos")

    def test_sample_rng_is_repeatable_and_restores_the_callers_state(self):
        import torch

        torch.manual_seed(7)
        original_state = torch.random.get_rng_state().clone()
        with isolated_torch_rng(101, torch.device("cpu")):
            first = torch.rand(4)

        self.assertTrue(torch.equal(original_state, torch.random.get_rng_state()))
        with isolated_torch_rng(101, torch.device("cpu")):
            second = torch.rand(4)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
