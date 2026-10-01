"""Local sampled-batch semantics against the pinned Transformers scalar path."""

import unittest
import inspect

import teacher_reliability.teacher as teacher


class _TinyTokenizer:
    eos_token_id = 2
    all_special_ids = [0, 2]

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
        prompt_ids = {
            "short": [1, 4, 5],
            "medium": [1, 6, 7, 8],
            "long": [1, 9, 10, 11, 12],
        }[question]
        ids = torch.tensor([prompt_ids], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        return ",".join(
            str(int(token_id))
            for token_id in token_ids
            if int(token_id) not in self.all_special_ids
        )


class LocalBatchSamplingTests(unittest.TestCase):
    def test_batch_generators_support_cache_fallback_and_heartbeat_progress(self):
        greedy_signature = inspect.signature(teacher.generate_greedy_batch).parameters
        sample_signature = inspect.signature(teacher.generate_sample_batch).parameters
        for signature in (greedy_signature, sample_signature):
            self.assertIn("cache_implementation", signature)
            self.assertIn("on_progress", signature)

    def test_streamed_result_callback_reports_each_original_request_index(self):
        generate_sample_batch = getattr(teacher, "generate_sample_batch", None)
        self.assertTrue(callable(generate_sample_batch))
        self.assertIn(
            "on_result",
            inspect.signature(generate_sample_batch).parameters,
            "completed samples must be checkpointable before the whole batch returns",
        )

        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(1357)
        model = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=32,
                eos_token_id=2,
                pad_token_id=0,
                bos_token_id=1,
            )
        ).eval()
        tokenizer = _TinyTokenizer()
        seen = []
        results = generate_sample_batch(
            tokenizer,
            model,
            ["short", "long"],
            4,
            [41, 83],
            on_result=lambda index, result: seen.append((index, result)),
        )

        self.assertEqual({index for index, _ in seen}, {0, 1})
        self.assertEqual({index: result for index, result in seen}, dict(enumerate(results)))

    def test_per_request_seeded_batches_match_scalar_sampling_independent_of_order(self):
        generate_sample_batch = getattr(teacher, "generate_sample_batch", None)
        self.assertTrue(
            callable(generate_sample_batch),
            "local inference needs a batched sampler with a per-request seed",
        )

        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(2468)
        model = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=32,
                eos_token_id=2,
                pad_token_id=0,
                bos_token_id=1,
            )
        ).eval()
        model.generation_config.top_k = 50
        model.generation_config.top_p = 1.0
        tokenizer = _TinyTokenizer()
        requests = [("short", 101), ("long", 202), ("medium", 303)]
        expected = {
            question: teacher.generate_sample(tokenizer, model, question, 8, seed)
            for question, seed in requests
        }

        first = generate_sample_batch(
            tokenizer,
            model,
            [question for question, _ in requests],
            8,
            [seed for _, seed in requests],
        )
        reordered = [("medium", 303), ("short", 101), ("long", 202)]
        second = generate_sample_batch(
            tokenizer,
            model,
            [question for question, _ in reordered],
            8,
            [seed for _, seed in reordered],
        )

        self.assertEqual(
            {question: result for (question, _), result in zip(requests, first)},
            expected,
        )
        self.assertEqual(
            {question: result for (question, _), result in zip(reordered, second)},
            expected,
        )

    def test_batch_draw_uses_temperature_and_top_k_before_sampling(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        model = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=32,
                eos_token_id=2,
                pad_token_id=0,
                bos_token_id=1,
            )
        ).eval()
        model.generation_config.top_k = 50
        model.generation_config.top_p = 1.0

        # The first 50 candidate tokens are ids 3..52. The remaining 11
        # candidates have enough raw probability for seed 0 to select id 56,
        # but are excluded by the configured top_k warper. Temperature 0.7
        # then makes the pinned scalar path select id 19 from the retained set.
        logits = torch.full((64,), -10.0)
        logits[3:53] = torch.linspace(0.5, -0.4, 50)
        logits[53:] = -0.5

        def controlled_lm_head(_module, _inputs, output):
            return logits.to(output).view(1, 1, -1).expand_as(output)

        hook = model.lm_head.register_forward_hook(controlled_lm_head)
        self.addCleanup(hook.remove)

        tokenizer = _TinyTokenizer()
        expected = teacher.generate_sample(tokenizer, model, "short", 1, seed=0)
        actual = teacher.generate_sample_batch(
            tokenizer, model, ["short"], 1, [0]
        )[0]

        self.assertEqual(expected.token_ids, [19])
        self.assertEqual(actual, expected)
        self.assertLess(actual.token_ids[0], 53)
        self.assertFalse(actual.eos_generated)
        self.assertTrue(actual.was_truncated)
        self.assertEqual(actual.reason, "max_new_tokens")


if __name__ == "__main__":
    unittest.main()
