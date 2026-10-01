import math
import unittest
from types import SimpleNamespace

from teacher_reliability.teacher import (
    GreedyGeneration,
    SampleGeneration,
    build_chat_messages,
    classify_termination,
    derive_sample_seed,
    generate_greedy,
    generate_greedy_batch,
    isolated_torch_rng,
)


class _BatchFixtureTokenizer:
    eos_token_id = 3
    pad_token_id = 3
    all_special_ids = [3, 4]

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_tensors,
        return_dict,
    ):
        import torch

        question = messages[0]["content"].split("Problem:\n", 1)[1]
        prompt_ids = {"short": [10, 11], "long": [20, 21, 22]}[question]
        return {
            "input_ids": torch.tensor([prompt_ids]),
            "attention_mask": torch.ones((1, len(prompt_ids)), dtype=torch.long),
        }

    def decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        return "".join(
            chr(ord("a") + int(token_id))
            for token_id in token_ids
            if int(token_id) not in self.all_special_ids
        )

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        token_ids = [ord(character) - ord("a") for character in text]
        return {
            "input_ids": token_ids,
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


class _BatchFixtureModel:
    def __init__(self):
        import torch

        self._torch = torch
        self._embedding = SimpleNamespace(weight=torch.empty(0, device="cpu"))
        self.generation_config = SimpleNamespace(eos_token_id=[3, 4])
        self.calls = []

    def get_input_embeddings(self):
        return self._embedding

    @staticmethod
    def _step_logits(prompt_marker, step):
        fixture = {
            (11, 0): [0.0, 1.0, -2.0, -4.0, -5.0],
            (11, 1): [0.0, -1.0, -2.0, 3.0, -5.0],
            (22, 0): [-2.0, -1.0, 2.0, -5.0, -6.0],
            (22, 1): [0.0, 2.0, -1.0, -3.0, -4.0],
            (22, 2): [2.0, 1.0, -1.0, -3.0, -4.0],
            (22, 3): [-2.0, -1.0, -3.0, -4.0, 4.0],
        }
        return fixture.get((prompt_marker, step), [-3.0, 0.0, -2.0, 1.0, -4.0])

    def generate(self, *, input_ids, attention_mask, **kwargs):
        torch = self._torch
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask.clone(),
                **kwargs,
            }
        )
        sequences = input_ids.clone()
        prompt_markers = input_ids[:, -1].tolist()
        unfinished = torch.ones(input_ids.shape[0], dtype=torch.bool)
        saved_scores = []
        for step in range(kwargs["max_new_tokens"]):
            scores = torch.tensor(
                [self._step_logits(int(marker), step) for marker in prompt_markers],
                dtype=torch.float16,
            )
            for processor in kwargs.get("logits_processor", []):
                scores = processor(sequences, scores)
            if kwargs["output_scores"]:
                saved_scores.append(scores.clone())
            next_tokens = scores.argmax(dim=-1)
            next_tokens = torch.where(
                unfinished, next_tokens, torch.full_like(next_tokens, 3)
            )
            sequences = torch.cat((sequences, next_tokens[:, None]), dim=1)
            for row, token_id in enumerate(next_tokens.tolist()):
                if unfinished[row] and int(token_id) in {3, 4}:
                    unfinished[row] = False
            if not bool(unfinished.any()):
                break
        result = SimpleNamespace(sequences=sequences)
        if kwargs["output_scores"]:
            result.scores = tuple(saved_scores)
        return result


class _BatchFailureAfterShortEosModel(_BatchFixtureModel):
    def generate(self, *, input_ids, attention_mask, **kwargs):
        torch = self._torch
        sequences = input_ids.clone()
        prompt_markers = input_ids[:, -1].tolist()
        for step in (0, 1):
            scores = torch.tensor(
                [self._step_logits(int(marker), step) for marker in prompt_markers],
                dtype=torch.float16,
            )
            for processor in kwargs["logits_processor"]:
                scores = processor(sequences, scores)
            sequences = torch.cat((sequences, scores.argmax(dim=-1)[:, None]), dim=1)
        raise RuntimeError("later batch row failed after the short row emitted EOS")


def _expected_entropy_and_logprob(logits, chosen_token):
    log_normalizer = math.log(sum(math.exp(value) for value in logits))
    probabilities = [math.exp(value - log_normalizer) for value in logits]
    entropy = log_normalizer - sum(
        probability * value for probability, value in zip(probabilities, logits)
    )
    return entropy, logits[chosen_token] - log_normalizer


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

    def test_greedy_batch_left_pads_and_preserves_each_row_through_its_eos(self):
        tokenizer = _BatchFixtureTokenizer()
        model = _BatchFixtureModel()

        generations = generate_greedy_batch(tokenizer, model, ["short", "long"], 6)

        self.assertEqual([item.token_ids for item in generations], [[1, 3], [2, 1, 0, 4]])
        self.assertEqual([item.termination_reason for item in generations], ["eos", "eos"])
        self.assertEqual([item.eos_generated for item in generations], [True, True])
        self.assertEqual([item.was_truncated for item in generations], [False, False])
        batch_call = model.calls[0]
        self.assertEqual(batch_call["input_ids"].tolist(), [[3, 10, 11], [20, 21, 22]])
        self.assertEqual(batch_call["attention_mask"].tolist(), [[0, 1, 1], [1, 1, 1]])

    def test_greedy_batch_emits_an_eos_row_before_a_later_row_fails(self):
        tokenizer = _BatchFixtureTokenizer()
        model = _BatchFailureAfterShortEosModel()
        observed = []

        try:
            generate_greedy_batch(
                tokenizer, model, ["short", "long"], 6,
                on_result=lambda row, result: observed.append((row, result)),
            )
        except TypeError as exc:
            self.fail(f"batched greedy decoding must expose completed rows: {exc}")
        except RuntimeError as exc:
            self.assertIn("later batch row failed", str(exc))

        self.assertEqual(len(observed), 1)
        row, generation = observed[0]
        self.assertEqual(row, 0)
        self.assertEqual(generation.token_ids, [1, 3])
        self.assertEqual(len(generation.token_ids), len(generation.token_entropies_nats))
        self.assertEqual(len(generation.token_ids), len(generation.token_logprobs))
        self.assertEqual(generation.termination_reason, "eos")

    def test_greedy_batch_scores_full_logits_without_returning_vocabulary_tensors(self):
        tokenizer = _BatchFixtureTokenizer()
        model = _BatchFixtureModel()

        generations = generate_greedy_batch(tokenizer, model, ["short", "long"], 6)
        reference = generate_greedy(tokenizer, model, "short", 6)

        self.assertFalse(model.calls[0]["output_scores"])
        self.assertEqual(generations[0].token_entropies_nats, reference.token_entropies_nats)
        self.assertEqual(generations[0].token_logprobs, reference.token_logprobs)
        expected_logits = [
            ([0.0, 1.0, -2.0, -4.0, -5.0], 1),
            ([0.0, -1.0, -2.0, 3.0, -5.0], 3),
        ]
        expected_entropy = []
        expected_logprobs = []
        for logits, chosen_token in expected_logits:
            entropy, logprob = _expected_entropy_and_logprob(logits, chosen_token)
            expected_entropy.append(entropy)
            expected_logprobs.append(logprob)
        for actual, expected in zip(generations[0].token_entropies_nats, expected_entropy):
            self.assertAlmostEqual(actual, expected, places=3)
        for actual, expected in zip(generations[0].token_logprobs, expected_logprobs):
            self.assertAlmostEqual(actual, expected, places=3)
        self.assertEqual(len(generations[0].token_ids), len(generations[0].token_logprobs))
        self.assertEqual(len(generations[1].token_ids), len(generations[1].token_entropies_nats))


if __name__ == "__main__":
    unittest.main()
