import importlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    vllm_worker = importlib.import_module("teacher_reliability.vllm_worker")
except ModuleNotFoundError:
    vllm_worker = None


class VllmWorkerTests(unittest.TestCase):
    def test_worker_module_is_available_without_vllm_installed(self):
        self.assertIsNotNone(vllm_worker, "the isolated worker module is missing")
        importlib.import_module("teacher_reliability")
        self.assertNotIn("vllm", sys.modules)

    @unittest.skipIf(vllm_worker is None, "worker module is not implemented yet")
    def test_generate_preserves_request_seed_mapping_when_vllm_returns_out_of_order(self):
        class SamplingParams:
            def __init__(self, **kwargs):
                self.options = kwargs

        class Candidate:
            def __init__(self, token_ids, text, finish_reason):
                self.token_ids = token_ids
                self.text = text
                self.finish_reason = finish_reason

        class Output:
            def __init__(self, request_id, prompt_token_ids, token_ids, text, finish_reason="stop"):
                self.request_id = request_id
                self.prompt_token_ids = prompt_token_ids
                self.outputs = [Candidate(token_ids, text, finish_reason)]
                self.finished = True

        class Tokenizer:
            eos_token_id = 99

        class ReorderingEngine:
            def __init__(self):
                self.calls = []
                self.pending = []
                self.step_count = 0

            def add_request(self, request_id, prompt, sampling_params):
                self.calls.append((request_id, prompt, sampling_params))
                seed = sampling_params.options["seed"]
                self.pending.append(
                    Output(
                        request_id,
                        prompt["prompt_token_ids"],
                        [seed],
                        f"seed-{seed}",
                    )
                )
                return request_id

            def has_unfinished_requests(self):
                return bool(self.pending)

            def step(self):
                self.step_count += 1
                results = list(reversed(self.pending))
                self.pending.clear()
                return results

        class LLM:
            def __init__(self, engine):
                self.llm_engine = engine

        requests = [
            {"request_id": "first", "prompt_token_ids": [1, 10], "seed": 101, "max_new_tokens": 12},
            {"request_id": "second", "prompt_token_ids": [2, 20], "seed": 202, "max_new_tokens": 13},
            {"request_id": "same-prompt-later", "prompt_token_ids": [1, 10], "seed": 303, "max_new_tokens": 14},
        ]
        engine = ReorderingEngine()

        results = vllm_worker.generate_requests(
            LLM(engine),
            Tokenizer(),
            requests,
            sampling_params_type=SamplingParams,
        )

        self.assertEqual(
            results,
            [
                {
                    "request_id": "first",
                    "token_ids": [101],
                    "text": "seed-101",
                    "eos_generated": False,
                    "was_truncated": False,
                    "reason": "other_stop",
                },
                {
                    "request_id": "second",
                    "token_ids": [202],
                    "text": "seed-202",
                    "eos_generated": False,
                    "was_truncated": False,
                    "reason": "other_stop",
                },
                {
                    "request_id": "same-prompt-later",
                    "token_ids": [303],
                    "text": "seed-303",
                    "eos_generated": False,
                    "was_truncated": False,
                    "reason": "other_stop",
                },
            ],
        )
        self.assertEqual(len(engine.calls), 3)
        self.assertEqual(engine.step_count, 1)
        self.assertEqual(
            [params.options for _, _, params in engine.calls],
            [
                {"temperature": 0.7, "top_k": 50, "top_p": 1.0, "max_tokens": 12, "stop_token_ids": [99], "seed": 101},
                {"temperature": 0.7, "top_k": 50, "top_p": 1.0, "max_tokens": 13, "stop_token_ids": [99], "seed": 202},
                {"temperature": 0.7, "top_k": 50, "top_p": 1.0, "max_tokens": 14, "stop_token_ids": [99], "seed": 303},
            ],
        )

    @unittest.skipIf(vllm_worker is None, "worker module is not implemented yet")
    def test_each_completed_request_can_be_streamed_before_batch_finishes(self):
        class SamplingParams:
            def __init__(self, **kwargs):
                self.options = kwargs

        class Candidate:
            token_ids = [1]
            text = "done"
            finish_reason = "stop"

        class Output:
            finished = True

            def __init__(self, request_id):
                self.request_id = request_id
                self.outputs = [Candidate()]

        class Tokenizer:
            eos_token_id = 99

        class Engine:
            def __init__(self):
                self.pending = []

            def add_request(self, request_id, _prompt, _params):
                self.pending.append(Output(request_id))
                return request_id

            def has_unfinished_requests(self):
                return bool(self.pending)

            def step(self):
                values = list(reversed(self.pending))
                self.pending.clear()
                return values

        requests = [
            {"request_id": "a", "prompt_token_ids": [1], "seed": 1, "max_new_tokens": 2},
            {"request_id": "b", "prompt_token_ids": [2], "seed": 2, "max_new_tokens": 2},
        ]
        streamed = []
        try:
            result = vllm_worker.generate_requests(
                type("LLM", (), {"llm_engine": Engine()})(), Tokenizer(), requests,
                sampling_params_type=SamplingParams, on_result=streamed.append,
            )
        except TypeError:
            result = []
        self.assertEqual([row["request_id"] for row in streamed], ["b", "a"])
        self.assertEqual([row["request_id"] for row in result], ["a", "b"])

    @unittest.skipIf(vllm_worker is None, "worker module is not implemented yet")
    def test_model_config_eos_ids_are_used_when_tokenizer_has_no_eos_id(self):
        class SamplingParams:
            def __init__(self, **kwargs):
                self.options = kwargs

        class Candidate:
            token_ids = [100]
            text = "done"
            finish_reason = "stop"

        class Output:
            request_id = "r"
            prompt_token_ids = [4, 5]
            outputs = [Candidate()]
            finished = True

        class Tokenizer:
            eos_token_id = None

        class ModelConfig:
            class HfConfig:
                eos_token_id = [99, 100]

            hf_config = HfConfig()

        class Engine:
            def __init__(self):
                self.pending = []

            def add_request(self, request_id, prompt, sampling_params):
                self.stop_token_ids = sampling_params.options["stop_token_ids"]
                self.pending.append(Output())
                return request_id

            def has_unfinished_requests(self):
                return bool(self.pending)

            def step(self):
                pending, self.pending = self.pending, []
                return pending

        class LLM:
            def __init__(self, engine):
                self.llm_engine = engine

        engine = Engine()
        result = vllm_worker.generate_requests(
            LLM(engine),
            Tokenizer(),
            [{"request_id": "r", "prompt_token_ids": [4, 5], "seed": 7, "max_new_tokens": 8}],
            sampling_params_type=SamplingParams,
            model_config=ModelConfig(),
        )

        self.assertEqual(engine.stop_token_ids, [99, 100])
        self.assertEqual(result[0]["token_ids"], [100])
        self.assertTrue(result[0]["eos_generated"])
        self.assertEqual(result[0]["reason"], "eos")

    @unittest.skipIf(vllm_worker is None, "worker module is not implemented yet")
    def test_cli_accepts_all_engine_memory_controls(self):
        args = vllm_worker.parse_args(
            [
                "--model-dir", "models/nf4",
                "--memory-utilization", "0.7",
                "--max-num-seqs", "3",
                "--max-num-batched-tokens", "1000",
                "--max-model-len", "2048",
                "--enforce-eager",
            ]
        )

        self.assertEqual(args.model_dir, "models/nf4")
        self.assertEqual(args.memory_utilization, 0.7)
        self.assertEqual(args.max_num_seqs, 3)
        self.assertEqual(args.max_num_batched_tokens, 1000)
        self.assertEqual(args.max_model_len, 2048)
        self.assertTrue(args.enforce_eager)

    @unittest.skipIf(vllm_worker is None, "worker module is not implemented yet")
    def test_startup_validates_artifact_and_constructs_local_single_gpu_engine(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory) / "models" / "nf4"
            recorded_options = []
            tokenizer = object()
            model_config = object()

            class FakeLLM:
                def __init__(self, **options):
                    recorded_options.append(options)
                    self.renderer = types.SimpleNamespace(tokenizer=tokenizer)
                    self.llm_engine = types.SimpleNamespace(model_config=model_config)

            vllm_module = types.ModuleType("vllm")
            vllm_module.__version__ = "0.30.0"
            vllm_module.LLM = FakeLLM
            artifact = {
                "model_revision": "a" * 40,
                "aggregate_sha256": "b" * 64,
                "quantization": {
                    "load_in_4bit": True,
                    "quant_type": "nf4",
                    "double_quant": True,
                    "compute_dtype": "float16",
                },
            }
            args = vllm_worker.parse_args(
                [
                    "--model-dir", str(model_dir),
                    "--memory-utilization", "0.71",
                    "--max-num-seqs", "4",
                    "--max-num-batched-tokens", "2048",
                    "--max-model-len", "3072",
                    "--enforce-eager",
                ]
            )

            with patch.dict(sys.modules, {"vllm": vllm_module}), patch.object(
                vllm_worker, "validate_artifact", return_value=artifact
            ) as validate:
                worker = vllm_worker.create_worker(args)

        self.assertIsNotNone(worker, "startup should return the constructed worker")
        validate.assert_called_once_with(model_dir.resolve())
        self.assertIs(worker.tokenizer, tokenizer)
        self.assertIs(worker.model_config, model_config)
        self.assertEqual(recorded_options, [{
            "model": str(model_dir.resolve()),
            "tokenizer": str(model_dir.resolve()),
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.71,
            "max_num_seqs": 4,
            "max_num_batched_tokens": 2048,
            "max_model_len": 3072,
            "quantization": "bitsandbytes",
            "load_format": "bitsandbytes",
            "dtype": "float16",
            "enforce_eager": True,
            "trust_remote_code": False,
        }])
        self.assertEqual(worker.runtime["model_revision"], "a" * 40)
        self.assertEqual(worker.runtime["version"], "0.30.0")

    @unittest.skipIf(vllm_worker is None, "worker module is not implemented yet")
    def test_main_emits_ready_then_one_json_response_line(self):
        class Worker:
            runtime = {"engine": "vllm", "model_revision": "a" * 40}

            def generate(self, requests, *, on_result=None):
                results = [{
                    "request_id": requests[0]["request_id"],
                    "token_ids": [17],
                    "text": "ok",
                    "eos_generated": False,
                    "was_truncated": False,
                    "reason": "other_stop",
                }]
                if on_result is not None:
                    for result in results:
                        on_result(result)
                return results

        stdin = io.StringIO(
            '{"op":"generate","requests":[{"request_id":"r","prompt_token_ids":[1],"seed":2,"max_new_tokens":3}]}\n'
        )
        stdout = io.StringIO()
        with patch.object(vllm_worker, "create_worker", return_value=Worker()), patch(
            "sys.stdin", stdin
        ), patch("sys.stdout", stdout):
            exit_code = vllm_worker.main(["--model-dir", "models/nf4"])

        lines = stdout.getvalue().splitlines()
        self.assertTrue(lines, "startup should emit its readiness record")
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(lines[0]), {"ready": True, "runtime": Worker.runtime})
        self.assertEqual(json.loads(lines[1]), {
            "ok": True,
            "result": {
                "request_id": "r",
                "token_ids": [17],
                "text": "ok",
                "eos_generated": False,
                "was_truncated": False,
                "reason": "other_stop",
            },
        })
        self.assertEqual(json.loads(lines[2]), {"ok": True, "done": True})

    @unittest.skipIf(vllm_worker is None, "worker module is not implemented yet")
    def test_startup_cuda_oom_emits_typed_protocol_error(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(
            vllm_worker,
            "create_worker",
            side_effect=RuntimeError("CUDA out of memory while loading weights"),
        ), patch("sys.stdout", stdout), patch("sys.stderr", stderr):
            exit_code = vllm_worker.main(["--model-dir", "models/nf4"])

        lines = stdout.getvalue().splitlines()
        self.assertTrue(lines, "confirmed startup CUDA OOM should be written as protocol JSON")
        self.assertEqual(exit_code, 2)
        self.assertEqual(
            json.loads(lines[0]),
            {
                "ok": False,
                "error_type": "cuda_oom",
                "message": "CUDA out of memory while loading weights",
            },
        )
        self.assertEqual(stderr.getvalue(), "")

    @unittest.skipIf(vllm_worker is None, "worker module is not implemented yet")
    def test_startup_cuda_memory_admission_failure_emits_typed_protocol_error(self):
        message = (
            "Free memory on device cuda:0 (6.44/11.63 GiB) on startup is less than "
            "desired GPU memory utilization (0.9, 10.47 GiB). Decrease GPU memory "
            "utilization or reduce GPU memory used by other processes."
        )
        stdout = io.StringIO()
        with patch.object(
            vllm_worker,
            "create_worker",
            side_effect=ValueError(message),
        ), patch("sys.stdout", stdout), patch("sys.stderr", io.StringIO()):
            exit_code = vllm_worker.main(["--model-dir", "models/nf4"])

        lines = stdout.getvalue().splitlines()
        self.assertTrue(lines, "vLLM CUDA memory admission failure should be sent as protocol JSON")
        self.assertEqual(exit_code, 2)
        self.assertEqual(
            json.loads(lines[0]),
            {"ok": False, "error_type": "cuda_oom", "message": message},
        )

    @unittest.skipIf(vllm_worker is None, "worker module is not implemented yet")
    def test_malformed_request_is_returned_as_a_protocol_error(self):
        class Worker:
            def generate(self, _requests):
                self.fail("malformed input must not reach generation")

        response = json.loads(vllm_worker.process_line('{"op":"generate","requests":[{"request_id":"x","prompt_token_ids":[true],"seed":1,"max_new_tokens":2}]}', Worker()))

        self.assertEqual(response["ok"], False)
        self.assertEqual(response["error_type"], "other")
        self.assertIn("prompt_token_ids", response["message"])

    @unittest.skipIf(vllm_worker is None, "worker module is not implemented yet")
    def test_success_protocol_line_serializes_only_the_declared_response_fields(self):
        class Worker:
            def generate(self, requests):
                self.assert_request_ids = [row["request_id"] for row in requests]
                return [
                    {
                        "request_id": "r-1",
                        "token_ids": [42],
                        "text": "answer",
                        "eos_generated": True,
                        "was_truncated": False,
                        "reason": "eos",
                    }
                ]

        worker = Worker()
        line = json.dumps(
            {
                "op": "generate",
                "requests": [
                    {"request_id": "r-1", "prompt_token_ids": [5], "seed": 8, "max_new_tokens": 3}
                ],
            }
        )
        encoded = vllm_worker.process_line(line, worker)

        self.assertEqual(worker.assert_request_ids, ["r-1"])
        self.assertEqual(
            json.loads(encoded),
            {
                "ok": True,
                "results": [
                    {
                        "request_id": "r-1",
                        "token_ids": [42],
                        "text": "answer",
                        "eos_generated": True,
                        "was_truncated": False,
                        "reason": "eos",
                    }
                ],
            },
        )

    @unittest.skipIf(vllm_worker is None, "worker module is not implemented yet")
    def test_cuda_out_of_memory_is_a_typed_error(self):
        class Worker:
            def generate(self, _requests):
                raise RuntimeError("CUDA out of memory while allocating tensor")

        response = json.loads(
            vllm_worker.process_line(
                '{"op":"generate","requests":[{"request_id":"r","prompt_token_ids":[1],"seed":1,"max_new_tokens":2}]}',
                Worker(),
            )
        )

        self.assertEqual(response["ok"], False)
        self.assertEqual(response["error_type"], "cuda_oom")
        self.assertIn("CUDA out of memory", response["message"])

        class HostOomWorker:
            def generate(self, _requests):
                raise RuntimeError("host ran out of memory")

        host_oom = json.loads(vllm_worker.process_line(
            '{"op":"generate","requests":[{"request_id":"r","prompt_token_ids":[1],"seed":1,"max_new_tokens":2}]}',
            HostOomWorker(),
        ))
        self.assertEqual(host_oom["error_type"], "other")


if __name__ == "__main__":
    unittest.main()
