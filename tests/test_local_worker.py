"""Single-GPU worker request validation and streamed-result contracts."""

import importlib.util
import unittest
from types import SimpleNamespace

from teacher_reliability.hybrid import CudaOutOfMemory, MemoryProfile
from teacher_reliability.teacher import GreedyGeneration, SampleGeneration


def get_worker_module(test):
    spec = importlib.util.find_spec("teacher_reliability.local_worker")
    test.assertIsNotNone(spec, "the persistent local GPU worker is missing")
    return __import__("teacher_reliability.local_worker", fromlist=["LocalWorkerService"])


class LocalWorkerTests(unittest.TestCase):
    def test_offloaded_cache_host_ram_shortage_is_not_classified_as_cuda_oom(self):
        module = get_worker_module(self)
        config = SimpleNamespace(
            num_hidden_layers=28,
            num_key_value_heads=4,
            num_attention_heads=28,
            hidden_size=3584,
        )

        with self.assertRaisesRegex(RuntimeError, "offloaded KV cache needs") as caught:
            module._check_offloaded_cache_ram(config, 1, 2 * 1024**3)

        self.assertEqual(type(caught.exception).__name__, "InsufficientHostMemory")
        self.assertFalse(module.is_cuda_oom(caught.exception))

    def test_each_request_releases_unused_cache_before_the_headroom_check(self):
        module = get_worker_module(self)
        events = []
        service = module.LocalWorkerService(
            object(), SimpleNamespace(config=SimpleNamespace(vocab_size=152064)),
            cache_releaser=lambda: events.append("release"),
            memory_probe=lambda: (events.append("probe") or {"free_mib": 1200, "total_mib": 8192}),
            greedy_generator=lambda *_args, **_kwargs: (
                events.append("generate") or [GreedyGeneration("a", [1], [0.2], [-0.3], [(0, 1)], [True])]
            ),
            sample_generator=lambda *_args, **_kwargs: [],
        )

        service.handle({"op": "generate", "stage": "greedy", "requests": [
            {"request_id": "x", "question": "q", "max_new_tokens": 2}
        ]})

        self.assertEqual(events, ["release", "probe", "generate"])

    def test_offloaded_cache_recovery_uses_its_lower_recorded_vram_floor(self):
        module = get_worker_module(self)
        service = module.LocalWorkerService(
            object(), SimpleNamespace(config=SimpleNamespace(
                vocab_size=152064, num_hidden_layers=28, num_key_value_heads=4,
                num_attention_heads=28, hidden_size=3584,
            )),
            cache_releaser=lambda: None,
            memory_probe=lambda: {"free_mib": 900, "total_mib": 8192},
            available_ram_bytes=lambda: 8 * 1024**3,
            greedy_generator=lambda *_args, **_kwargs: [
                GreedyGeneration("a", [1], [0.2], [-0.3], [(0, 1)], [True])
            ],
            sample_generator=lambda *_args, **_kwargs: [],
        )

        result = service.handle({"op": "generate", "stage": "greedy", "cache_implementation": "offloaded", "requests": [
            {"request_id": "x", "question": "q", "max_new_tokens": 2}
        ]})

        self.assertEqual(result["x"]["greedy"]["text"], "a")

    def test_worker_stdio_is_configured_for_utf8_unicode_ipc(self):
        module = get_worker_module(self)

        class Stream:
            def __init__(self):
                self.encoding = None

            def reconfigure(self, *, encoding):
                self.encoding = encoding

        stdin = Stream()
        stdout = Stream()
        configure = getattr(module, "_configure_worker_stdio", None)
        self.assertTrue(callable(configure), "worker must configure Unicode-safe IPC streams")

        configure(stdin, stdout)

        self.assertEqual((stdin.encoding, stdout.encoding), ("utf-8", "utf-8"))

    def test_model_load_cache_is_released_before_free_memory_is_snapshotted(self):
        module = get_worker_module(self)
        events = []
        torch_module = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: events.append("empty_cache")))

        release = getattr(module, "_release_model_load_cache", None)
        self.assertTrue(callable(release), "worker must release unused model-load allocations")
        memory = release(
            torch_module,
            lambda: (events.append("snapshot") or {"free_mib": 1111, "total_mib": 8192}),
        )

        self.assertEqual(events, ["empty_cache", "snapshot"])
        self.assertEqual(memory["free_mib"], 1111)

    def test_one_worker_dispatches_batch_and_streams_original_request_ids_and_seeds(self):
        module = get_worker_module(self)
        seen = {}

        def greedy(_tokenizer, _model, questions, max_new_tokens, **kwargs):
            seen["greedy"] = (questions, max_new_tokens, kwargs["cache_implementation"])
            return [
                GreedyGeneration("answer", [7], [0.2], [-0.3], [(0, 1)], [True])
                for _ in questions
            ]

        def sample(_tokenizer, _model, questions, max_new_tokens, seeds, **kwargs):
            seen["sample"] = (questions, max_new_tokens, seeds, kwargs["cache_implementation"])
            results = [SampleGeneration(str(seed), [seed], False, False, "other_stop") for seed in seeds]
            for index, result in reversed(list(enumerate(results))):
                kwargs["on_result"](index, result)
            return results

        service = module.LocalWorkerService(
            object(), SimpleNamespace(config=SimpleNamespace(
                vocab_size=152064, num_hidden_layers=28, num_key_value_heads=4,
                num_attention_heads=28, hidden_size=3584,
            )),
            memory_probe=lambda: {"free_mib": 4096, "total_mib": 8192},
            available_ram_bytes=lambda: 8 * 1024**3,
            greedy_generator=greedy, sample_generator=sample,
        )
        emitted = []
        request = {
            "op": "generate", "stage": "sample", "cache_implementation": "offloaded",
            "requests": [
                {"request_id": "b", "question": "second", "seed": 202, "max_new_tokens": 9},
                {"request_id": "a", "question": "first", "seed": 101, "max_new_tokens": 9},
            ],
        }
        results = service.handle(request, on_result=lambda request_id, result: emitted.append((request_id, result)))
        self.assertEqual(seen["sample"], (["second", "first"], 9, [202, 101], "offloaded"))
        self.assertEqual([request_id for request_id, _ in emitted], ["a", "b"])
        self.assertEqual(set(results), {"a", "b"})

    def test_greedy_worker_streams_completed_rows_before_later_batch_failure(self):
        module = get_worker_module(self)
        completed = GreedyGeneration("answer", [7], [0.2], [-0.3], [(0, 1)], [True])

        def greedy(_tokenizer, _model, _questions, _limit, *, on_result, **_kwargs):
            on_result(0, completed)
            raise RuntimeError("later row failed")

        service = module.LocalWorkerService(
            object(), SimpleNamespace(config=SimpleNamespace(vocab_size=152064)),
            memory_probe=lambda: {"free_mib": 4096, "total_mib": 8192},
            greedy_generator=greedy,
        )
        emitted = []
        try:
            service.handle(
                {"op": "generate", "stage": "greedy", "requests": [
                    {"request_id": "first", "question": "short", "max_new_tokens": 8},
                    {"request_id": "second", "question": "long", "max_new_tokens": 8},
                ]},
                on_result=lambda request_id, result: emitted.append((request_id, result["greedy"])),
            )
        except (TypeError, RuntimeError) as caught:
            failure = caught
        else:
            self.fail("the later batch row should fail after the first result streams")

        self.assertIsInstance(failure, RuntimeError)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0][0], "first")
        self.assertEqual(emitted[0][1]["text"], "answer")

    def test_memory_preflight_and_oom_classification_distinguish_worker_failures(self):
        module = get_worker_module(self)
        service = module.LocalWorkerService(
            object(), SimpleNamespace(config=SimpleNamespace(vocab_size=152064)),
            memory_probe=lambda: {"free_mib": 900, "total_mib": 8192},
            greedy_generator=lambda *_args, **_kwargs: self.fail("generation must not start"),
            sample_generator=lambda *_args, **_kwargs: self.fail("generation must not start"),
        )
        with self.assertRaisesRegex(CudaOutOfMemory, "900 MiB"):
            service.handle({"op": "generate", "stage": "greedy", "requests": [
                {"request_id": "x", "question": "q", "max_new_tokens": 2}
            ]})
        self.assertTrue(module.is_cuda_oom(CudaOutOfMemory("global GPU free memory is below 1024 MiB")))
        self.assertTrue(module.is_cuda_oom(RuntimeError("CUDA out of memory. allocated")))
        self.assertFalse(module.is_cuda_oom(RuntimeError("tokenizer worker crashed")))

    def test_invalid_duplicate_ids_and_mixed_generation_limits_are_rejected(self):
        module = get_worker_module(self)
        service = module.LocalWorkerService(
            object(), SimpleNamespace(config=SimpleNamespace(vocab_size=152064)),
            memory_probe=lambda: {"free_mib": 4096, "total_mib": 8192},
            greedy_generator=lambda *_args, **_kwargs: [],
            sample_generator=lambda *_args, **_kwargs: [],
        )
        duplicates = [{"request_id": "x", "question": "a", "max_new_tokens": 2} for _ in range(2)]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            service.handle({"op": "generate", "stage": "greedy", "requests": duplicates})
        mixed = [
            {"request_id": "x", "question": "a", "max_new_tokens": 2},
            {"request_id": "y", "question": "b", "max_new_tokens": 3},
        ]
        with self.assertRaisesRegex(ValueError, "same output limit"):
            service.handle({"op": "generate", "stage": "greedy", "requests": mixed})


if __name__ == "__main__":
    unittest.main()
