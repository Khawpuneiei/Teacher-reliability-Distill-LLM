"""Fixed 24-question qualification selection and profile-matrix contracts."""

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from teacher_reliability.data import normalize_example
from teacher_reliability.hybrid import CudaOutOfMemory, MemoryProfile
from teacher_reliability.state import RunAlreadyActive, run_lock
from teacher_reliability.teacher import GreedyGeneration, SampleGeneration


def benchmark_module(test):
    spec = importlib.util.find_spec("teacher_reliability.local_benchmark")
    test.assertIsNotNone(spec, "the local GPU qualification harness is missing")
    return __import__("teacher_reliability.local_benchmark", fromlist=["select_benchmark_examples"])


class LocalBenchmarkSelectionTests(unittest.TestCase):
    def setUp(self):
        self.examples = []
        for index in range(12):
            self.examples.append(normalize_example(
                "gsm8k", "main", "test", index,
                {"question": f"GSM8K question {index}", "answer": "#### 1"},
                source_revision="1" * 40,
            ))
        subjects = (
            "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus",
        )
        for subject in subjects:
            for index in range(2):
                self.examples.append(normalize_example(
                    "math", subject, "test", index,
                    {"problem": f"{subject} math problem {index}", "solution": "#### 2"},
                    source_revision="2" * 40,
                ))
        for index in range(12):
            self.examples.append(normalize_example(
                "gsm_plus", "default", "test", index,
                {"question": f"GSM Plus question {index}", "answer": "#### 3"},
                source_revision="3" * 40,
            ))
        self.prompt_tokens = {row.example_id: 20 + int(row.source_row_id) for row in self.examples}
        self.longest = {
            dataset: max((row for row in self.examples if row.dataset == dataset), key=lambda row: self.prompt_tokens[row.example_id])
            for dataset in ("gsm8k", "math", "gsm_plus")
        }
        self.archived_ids = {row.example_id for row in self.examples if row.dataset == "gsm8k" and int(row.source_row_id) < 5}

    def test_selection_has_eight_per_dataset_all_math_subjects_long_prompts_and_archived_rows(self):
        module = benchmark_module(self)
        selected = module.select_benchmark_examples(
            self.examples, self.prompt_tokens, archived_ids=self.archived_ids, seed=91,
        )
        self.assertEqual(len(selected), 24)
        self.assertEqual({dataset: sum(row.dataset == dataset for row in selected) for dataset in ("gsm8k", "math", "gsm_plus")},
                         {"gsm8k": 8, "math": 8, "gsm_plus": 8})
        self.assertEqual({row.source_config for row in selected if row.dataset == "math"}, {
            "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus",
        })
        self.assertTrue(set(self.longest.values()).issubset(set(selected)))
        self.assertGreaterEqual(sum(row.example_id in self.archived_ids for row in selected), 3)

    def test_selection_is_order_independent_and_requires_all_math_subjects(self):
        module = benchmark_module(self)
        first = module.select_benchmark_examples(
            self.examples, self.prompt_tokens, archived_ids=self.archived_ids, seed=91,
        )
        second = module.select_benchmark_examples(
            list(reversed(self.examples)), self.prompt_tokens, archived_ids=self.archived_ids, seed=91,
        )
        self.assertEqual([row.example_id for row in first], [row.example_id for row in second])
        incomplete = [row for row in self.examples if not (row.dataset == "math" and row.source_config == "precalculus")]
        with self.assertRaisesRegex(ValueError, "seven MATH subjects"):
            module.select_benchmark_examples(incomplete, self.prompt_tokens, archived_ids=self.archived_ids, seed=91)

    def test_qualification_matrix_covers_serial_and_every_requested_batch_size(self):
        module = benchmark_module(self)
        matrix = module.qualification_matrix()
        self.assertEqual(matrix[0], (1, 1))
        self.assertEqual(set(matrix), {
            (greedy, sample) for greedy in (1, 2, 4) for sample in (1, 2, 4, 8)
        })

    def test_configuration_filter_runs_a_profile_without_shrinking_the_matrix_identity(self):
        module = benchmark_module(self)
        select = getattr(module, "select_qualification_configurations", None)
        self.assertTrue(callable(select), "qualification must support staged profile runs")

        matrix = [(g, s) for g in (1, 2, 4) for s in (1, 2, 4, 8)]
        self.assertEqual(select(None), matrix)
        self.assertEqual(select(["g1-s1"]), [(1, 1)])
        self.assertEqual(select(["g2-s4", "g1-s1"]), [(1, 1), (2, 4)])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select(["g1-s1", "g1-s1"])
        with self.assertRaisesRegex(ValueError, "unsupported"):
            select(["g8-s8"])

    def test_each_oom_fallback_halves_batch_and_enables_offload_at_one(self):
        module = benchmark_module(self)
        profiles = module.memory_profiles(4, "sample")
        self.assertEqual([item.batch_size for item in profiles], [4, 2, 1, 1])
        self.assertEqual([item.cache_implementation for item in profiles], [None, None, None, "offloaded"])

    def test_selection_rejects_a_prompt_that_cannot_fit_with_the_full_output_budget(self):
        module = benchmark_module(self)
        token_counts = dict(self.prompt_tokens)
        token_counts[self.longest["gsm8k"].example_id] = 4096 - 1024 + 1
        with self.assertRaisesRegex(ValueError, "context limit"):
            module.select_benchmark_examples(
                self.examples, token_counts, archived_ids=self.archived_ids, seed=91,
            )

    def test_remaining_counts_subtract_validated_imports_by_selected_example_id(self):
        module = benchmark_module(self)
        count_remaining = getattr(module, "remaining_rows_by_dataset", None)
        self.assertTrue(callable(count_remaining), "qualification must count imports by validated selected IDs")
        imported_rows = [
            {"example_id": self.examples[0].example_id, "source": {"dataset": "wrong-label"}},
            {"example_id": next(row.example_id for row in self.examples if row.dataset == "math"), "source": {}},
            {"example_id": next(row.example_id for row in self.examples if row.dataset == "gsm_plus")},
        ]

        remaining = count_remaining(self.examples, imported_rows)

        self.assertEqual(remaining, {"gsm8k": 11, "math": 13, "gsm_plus": 11})
        with self.assertRaisesRegex(ValueError, "outside the selected examples"):
            count_remaining(self.examples, [{"example_id": "not-selected"}])


class LocalBenchmarkMemoryTests(unittest.TestCase):
    def test_qualification_releases_prompt_token_ids_before_selection(self):
        import gc
        import weakref

        module = benchmark_module(self)
        examples = [normalize_example(
            "gsm8k", "main", "test", index,
            {"question": f"Question {index}", "answer": "#### 1"},
            source_revision="1" * 40,
        ) for index in range(2)]

        class TokenIds:
            def __len__(self):
                return 48

        class TrackingTokenizer:
            def __init__(self):
                self.references = []

            def apply_chat_template(self, *_args, **_kwargs):
                token_ids = TokenIds()
                self.references.append(weakref.ref(token_ids))
                return token_ids

        class SelectionReached(Exception):
            pass

        tokenizer = TrackingTokenizer()

        def inspect_selection_inputs(rows, token_counts, **_kwargs):
            gc.collect()
            self.assertEqual(rows, examples)
            self.assertEqual(token_counts, {row.example_id: 48 for row in examples})
            self.assertEqual(
                [reference() for reference in tokenizer.references],
                [None, None],
                "qualification should release each prompt token list after counting it",
            )
            raise SelectionReached()

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.dict(sys.modules, {"torch": types.ModuleType("torch")}),
                patch.object(module, "select_benchmark_examples", side_effect=inspect_selection_inputs),
            ):
                with self.assertRaises(SelectionReached):
                    module._run_qualification_body(
                        Path(temp_dir), examples, tokenizer, object(), base_seed=42,
                        archived_ids=set(), dataset_revisions={}, model_revision="model-revision",
                        source_archive_sha256="a" * 64,
                    )

    def test_generation_memory_monitor_fails_when_global_headroom_dips_mid_generation(self):
        import threading
        import time

        module = benchmark_module(self)
        entered_low_memory = threading.Event()
        observations = [0]
        memory = {"minimum_global_free_mib": None}

        def probe():
            observations[0] += 1
            if observations[0] == 1:
                return {"free_mib": 2048, "total_mib": 8192}
            entered_low_memory.set()
            return {"free_mib": 900, "total_mib": 8192}

        monitor_type = getattr(module, "GenerationMemoryMonitor", None)
        self.assertTrue(callable(monitor_type), "qualification must continuously observe global GPU memory")
        monitor = monitor_type(
            MemoryProfile("greedy-1", 1), memory, probe, poll_interval_seconds=0.001,
        )

        with self.assertRaisesRegex(CudaOutOfMemory, "900 MiB"):
            with monitor:
                self.assertTrue(entered_low_memory.wait(1))
                time.sleep(0.005)

        self.assertGreaterEqual(observations[0], 2)
        self.assertEqual(memory["minimum_global_free_mib"], 900)

    def test_qualification_records_the_lowest_observed_available_host_ram(self):
        module = benchmark_module(self)
        memory = {"peak_available_ram_mib": 0}

        module._max_memory(memory, {"available_ram_mib": 5600})
        module._max_memory(memory, {"available_ram_mib": 1900})
        module._max_memory(memory, {"available_ram_mib": 4100})

        self.assertEqual(memory["peak_available_ram_mib"], 5600)
        self.assertEqual(memory.get("minimum_available_ram_mib"), 1900)

    def test_dataset_weighted_eta_uses_remaining_origin_counts_and_measured_rates(self):
        module = benchmark_module(self)
        estimate = getattr(module, "estimate_dataset_weighted_eta_hours", None)
        self.assertTrue(callable(estimate), "qualification must report a dataset-weighted ETA")

        hours = estimate(
            {"gsm8k": 1600, "math": 240, "gsm_plus": 10000},
            {"gsm8k": 80, "math": 40, "gsm_plus": 20},
        )

        self.assertEqual(hours, 526.0)
        self.assertIsNone(estimate({"gsm8k": 5}, {}))

    def test_complete_offloaded_configuration_can_qualify_at_its_recorded_floor(self):
        module = benchmark_module(self)
        eligible = getattr(module, "is_qualified_configuration", None)
        self.assertTrue(callable(eligible), "qualification must evaluate the selected recovery floor")
        stress_cache = getattr(module, "stress_cache_implementation", None)
        self.assertTrue(callable(stress_cache), "stress evidence must follow the selected cache mode")
        offloaded = {
            "status": "complete",
            "actual_greedy_profile": {"cache_implementation": "offloaded"},
            "actual_sample_profile": {"cache_implementation": "offloaded"},
            "memory": {"minimum_global_free_mib": 900},
        }
        default_cache = {
            "status": "complete",
            "actual_greedy_profile": {"cache_implementation": None},
            "actual_sample_profile": {"cache_implementation": None},
            "memory": {"minimum_global_free_mib": 900},
        }

        self.assertTrue(eligible(offloaded))
        self.assertFalse(eligible(default_cache))
        self.assertEqual(stress_cache(offloaded), "offloaded")
        self.assertIsNone(stress_cache(default_cache))
        self.assertIsNone(stress_cache({
            **offloaded,
            "actual_sample_profile": {"cache_implementation": None},
        }))

    def test_mixed_cache_configuration_must_meet_normal_cache_headroom_floor(self):
        module = benchmark_module(self)
        eligible = module.is_qualified_configuration
        mixed = {
            "status": "complete",
            "actual_greedy_profile": {"cache_implementation": "offloaded"},
            "actual_sample_profile": {"cache_implementation": None},
            "memory": {"minimum_global_free_mib": 900},
        }

        self.assertFalse(eligible(mixed))

    def test_raw_torch_oom_is_normalized_for_the_shared_backoff_controller(self):
        module = benchmark_module(self)
        normalize = getattr(module, "normalize_cuda_oom", None)
        self.assertTrue(callable(normalize), "benchmark must normalize raw CUDA OOMs for backoff")

        normalized = normalize(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
        ordinary = ValueError("bad benchmark input")

        self.assertIsInstance(normalized, CudaOutOfMemory)
        self.assertIs(normalize(ordinary), ordinary)

    def test_offloaded_cache_profile_uses_and_reports_its_emergency_vram_floor(self):
        module = benchmark_module(self)
        validate = getattr(module, "validate_memory_snapshot", None)
        self.assertTrue(callable(validate), "qualification must use the runtime recovery memory floor")

        floor = validate(
            MemoryProfile("greedy-1-offloaded", 1, cache_implementation="offloaded"),
            {"free_mib": 900, "total_mib": 8192},
        )

        self.assertEqual(floor, 512)
        with self.assertRaisesRegex(CudaOutOfMemory, "768 MiB"):
            validate(MemoryProfile("greedy-1", 1), {"free_mib": 768, "total_mib": 8192})

    def test_1024_token_stress_uses_the_selected_cache_implementation(self):
        import torch

        module = benchmark_module(self)
        example = normalize_example(
            "gsm8k", "main", "test", 0,
            {"question": "Stress prompt", "answer": "#### 1"},
            source_revision="1" * 40,
        )

        class TinyTokenizer:
            eos_token_id = 2

            def apply_chat_template(self, *_args, **_kwargs):
                return {"input_ids": torch.ones((1, 20), dtype=torch.long)}

        class StressModel:
            generation_config = types.SimpleNamespace(eos_token_id=2)
            config = types.SimpleNamespace(num_hidden_layers=28)

            def __init__(self):
                self.kwargs = None
                self.weight = torch.empty(1)

            def get_input_embeddings(self):
                return types.SimpleNamespace(weight=self.weight)

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return types.SimpleNamespace(
                    sequences=torch.ones((1, 1044), dtype=torch.long)
                )

        model = StressModel()
        memory = {"free_mib": 2048, "total_mib": 8192}
        with (
            patch.object(torch.cuda, "reset_peak_memory_stats"),
            patch.object(torch.cuda, "max_memory_allocated", return_value=0),
            patch.object(torch.cuda, "max_memory_reserved", return_value=0),
            patch.object(module, "_check_offloaded_cache_ram"),
        ):
            result = module.run_1024_token_stress(
                [example], {example.example_id: 20}, TinyTokenizer(), model,
                cache_implementation="offloaded", memory_probe=lambda: memory,
            )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["generated_tokens"], 1024)
        self.assertEqual(model.kwargs["cache_implementation"], "offloaded")

    def test_stress_selection_tries_next_fastest_profile_after_memory_gate_failure(self):
        module = benchmark_module(self)
        select = getattr(module, "select_stress_candidate", None)
        self.assertTrue(callable(select), "qualification must try the next measured candidate after stress failure")
        candidates = [
            ("g4-s8", {"actual_greedy_profile": {}, "actual_sample_profile": {}}),
            ("g1-s1", {
                "actual_greedy_profile": {"cache_implementation": "offloaded"},
                "actual_sample_profile": {"cache_implementation": "offloaded"},
            }),
        ]
        calls = []

        def stress(name, _configuration):
            calls.append(name)
            if name == "g4-s8":
                return {"status": "oom", "global_memory": {"free_mib": 700}}
            return {"status": "complete", "global_memory": {"free_mib": 1024}}

        selected, result, results = select(candidates, stress)

        self.assertEqual(calls, ["g4-s8", "g1-s1"])
        self.assertEqual(selected, "g1-s1")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(results["g4-s8"]["status"], "oom")

    def test_stress_selection_reuses_checkpoints_without_rerunning_failed_profiles(self):
        module = benchmark_module(self)
        select = getattr(module, "select_stress_candidate", None)
        self.assertTrue(callable(select), "stress attempts must be resumable")
        candidates = [
            ("g4-s8", {"actual_greedy_profile": {}, "actual_sample_profile": {}}),
            ("g1-s1", {"actual_greedy_profile": {}, "actual_sample_profile": {}}),
        ]
        prior = {
            "g4-s8": {"status": "oom", "global_memory": {"free_mib": 500}},
        }
        calls = []

        def stress(name, _configuration):
            calls.append(name)
            return {"status": "complete", "global_memory": {"free_mib": 2048}}

        selected, _result, results = select(candidates, stress, previous_results=prior)

        self.assertEqual(calls, ["g1-s1"])
        self.assertEqual(selected, "g1-s1")
        self.assertEqual(results["g4-s8"]["status"], "oom")


class LocalBenchmarkQualificationResumeTests(unittest.TestCase):
    def setUp(self):
        self.examples = []
        for index in range(12):
            self.examples.append(normalize_example(
                "gsm8k", "main", "test", index,
                {"question": f"GSM8K question {index}", "answer": "#### 1"},
                source_revision="1" * 40,
            ))
        subjects = (
            "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus",
        )
        for subject in subjects:
            for index in range(2):
                self.examples.append(normalize_example(
                    "math", subject, "test", index,
                    {"problem": f"{subject} math problem {index}", "solution": "#### 1"},
                    source_revision="2" * 40,
                ))
        for index in range(12):
            self.examples.append(normalize_example(
                "gsm_plus", "default", "test", index,
                {"question": f"GSM Plus question {index}", "answer": "#### 1"},
                source_revision="3" * 40,
            ))
        self.prompt_tokens = {row.example_id: 20 + int(row.source_row_id) for row in self.examples}
        self.archived_ids = {
            row.example_id for row in self.examples
            if row.dataset == "gsm8k" and int(row.source_row_id) < 5
        }

    def test_qualification_refuses_to_write_an_output_directory_owned_by_another_process(self):
        module = benchmark_module(self)

        class QualificationReached(Exception):
            pass

        class TinyTokenizer:
            def apply_chat_template(self, _messages, *, tokenize, add_generation_prompt):
                assert tokenize and add_generation_prompt
                return [1] * 20

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "qualification"
            with run_lock(output_dir):
                with patch.object(
                    module, "select_benchmark_examples", side_effect=QualificationReached
                ):
                    with self.assertRaises(RunAlreadyActive):
                        module.run_qualification(
                            output_dir, self.examples, TinyTokenizer(), None, base_seed=42,
                            archived_ids=self.archived_ids, dataset_revisions={},
                        model_revision="model-revision", source_archive_sha256="a" * 64,
                        )

    def test_cli_acquires_output_lock_before_loading_a_second_gpu_model(self):
        module = benchmark_module(self)

        class ModelLoadReached(Exception):
            pass

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def set_per_process_memory_fraction(_fraction, _device):
                return None

        fake_torch = types.SimpleNamespace(cuda=FakeCuda())
        revisions = {"gsm8k": "1", "math": "2", "gsm_plus": "3"}
        imported = types.SimpleNamespace(rows=[], metadata={"archive_sha256": "a" * 64})
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "qualification"
            archive_path = Path(temporary) / "archive.zip"
            archive_path.write_bytes(b"placeholder; archive reader is patched")
            with (
                patch.dict(sys.modules, {"torch": fake_torch}),
                patch(
                    "teacher_reliability.run._read_import_manifest",
                    return_value={"model_revision": "model-revision", "dataset_revisions": revisions},
                ),
                patch(
                    "teacher_reliability.run._load_cached_examples",
                    return_value=(self.examples, revisions),
                ),
                patch("teacher_reliability.archive_import.import_a100_run", return_value=imported),
                patch(
                    "teacher_reliability.teacher.load_teacher",
                    side_effect=ModelLoadReached,
                ) as load_teacher,
            ):
                with run_lock(output_dir):
                    status = module.main([
                        "--output-dir", str(output_dir), "--import-run-zip", str(archive_path),
                    ])

                self.assertEqual(status, 1)
                load_teacher.assert_not_called()

    def test_cli_closes_imported_archive_before_loading_teacher(self):
        import gc
        import weakref

        from teacher_reliability import archive_import

        module = benchmark_module(self)

        class ModelLoadReached(Exception):
            pass

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def set_per_process_memory_fraction(_fraction, _device):
                return None

        class Snapshot:
            def close(self):
                closed.append(True)

        class ImportedResult:
            def __init__(self):
                self.rows = []
                self.metadata = {"archive_sha256": "a" * 64}
                self.archive_snapshot = Snapshot()

            def close(self):
                self.archive_snapshot.close()

        class Importer:
            def __init__(self):
                self.result_ref = None

            def __call__(self, *_args, **_kwargs):
                result = ImportedResult()
                self.result_ref = weakref.ref(result)
                return result

        fake_torch = types.SimpleNamespace(cuda=FakeCuda())
        revisions = {"gsm8k": "1", "math": "2", "gsm_plus": "3"}
        closed = []
        importer = Importer()
        load_observations = []

        def load_teacher(*_args, **_kwargs):
            gc.collect()
            load_observations.append({
                "snapshot_closed": bool(closed),
                "imported_result_alive": importer.result_ref() is not None,
            })
            raise ModelLoadReached()

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "qualification"
            archive_path = Path(temporary) / "archive.zip"
            archive_path.write_bytes(b"placeholder; archive reader is patched")
            with (
                patch.dict(sys.modules, {
                    "torch": fake_torch,
                    "teacher_reliability.archive_import": archive_import,
                }),
                patch(
                    "teacher_reliability.run._read_import_manifest",
                    return_value={"model_revision": "model-revision", "dataset_revisions": revisions},
                ),
                patch(
                    "teacher_reliability.run._load_cached_examples",
                    return_value=(self.examples, revisions),
                ),
                patch("teacher_reliability.archive_import.import_a100_run", side_effect=importer),
                patch("teacher_reliability.teacher.load_teacher", side_effect=load_teacher),
            ):
                status = module.main([
                    "--output-dir", str(output_dir), "--import-run-zip", str(archive_path),
                ])

        self.assertEqual(status, 1)
        self.assertEqual(
            load_observations,
            [{"snapshot_closed": True, "imported_result_alive": False}],
        )

    def test_matrix_with_persisted_incomplete_profile_is_not_recommended_or_stress_tested(self):
        module = benchmark_module(self)

        class TinyTokenizer:
            eos_token_id = 2
            vocab_size = 4

            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                assert tokenize and add_generation_prompt
                return [1] * 20

        class FakeCuda:
            @staticmethod
            def get_device_name(_index):
                return "test GPU"

            @staticmethod
            def get_device_properties(_index):
                return types.SimpleNamespace(total_memory=8 * 1024**3)

            @staticmethod
            def empty_cache():
                return None

            @staticmethod
            def reset_peak_memory_stats(_index):
                return None

            @staticmethod
            def max_memory_allocated(_index):
                return 0

            @staticmethod
            def max_memory_reserved(_index):
                return 0

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = FakeCuda
        revisions = {
            "openai/gsm8k": "1" * 40,
            "EleutherAI/hendrycks_math": "2" * 40,
            "qintongli/GSM-Plus": "3" * 40,
        }
        memory = {
            "free_mib": 8192,
            "total_mib": 8192,
            "allocated_mib": 0,
            "reserved_mib": 0,
            "available_ram_mib": 8192,
        }
        greedy = GreedyGeneration(
            text="1", token_ids=[1], token_entropies_nats=[0.1],
            token_logprobs=[-0.1], token_char_spans=[(0, 1)],
            content_token_mask=[True], eos_generated=True,
            termination_reason="eos",
        )

        def generate_greedy(_tokenizer, _model, questions, _max_new_tokens,
                            *, on_progress=None, **_kwargs):
            return [greedy for _question in questions]

        def generate_samples(_tokenizer, _model, questions, _max_new_tokens, seeds,
                             *, on_result, **_kwargs):
            outputs = []
            for index, _request in enumerate(questions):
                generation = SampleGeneration("1", [1], True, False, "eos")
                on_result(index, generation)
                outputs.append(generation)
            return outputs

        memory_profiles = lambda size, stage: [MemoryProfile(f"{stage}-{size}", size)]
        stress_calls = []
        stress_options = []

        def fake_stress(*_args, **kwargs):
            stress_calls.append("stress")
            stress_options.append(kwargs)
            return {"status": "complete", "global_memory": {"free_mib": 8192}}

        def write_json(path, value):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding="utf-8")

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "qualification"
            with (
                patch.dict(sys.modules, {"torch": fake_torch}),
                patch.object(module, "memory_profiles", side_effect=memory_profiles),
                patch.object(module, "stress_cache_implementation", return_value="offloaded"),
                patch.object(module, "atomic_write_json", side_effect=write_json),
                patch.object(module, "generate_greedy_batch", side_effect=generate_greedy),
                patch.object(module, "generate_sample_batch", side_effect=generate_samples),
                patch.object(module, "_prediction_record", side_effect=lambda row, *_args: {"example_id": row.example_id}),
                patch.object(module, "run_1024_token_stress", side_effect=fake_stress),
            ):
                module.run_qualification(
                    output_dir, self.examples, TinyTokenizer(),
                    types.SimpleNamespace(config=types.SimpleNamespace(vocab_size=4)),
                    base_seed=42, archived_ids=self.archived_ids,
                    dataset_revisions=revisions, model_revision="model-revision",
                    source_archive_sha256="a" * 64,
                    remaining_rows_by_dataset={"gsm8k": 1, "math": 1, "gsm_plus": 1},
                    memory_probe=lambda: dict(memory),
                )

                expected_names = {
                    f"g{greedy_size}-s{sample_size}"
                    for greedy_size, sample_size in module.qualification_matrix()
                }
                persisted_names = {
                    path.parent.name for path in output_dir.glob("g*-s*/config.json")
                }
                self.assertEqual(persisted_names, expected_names)
                self.assertEqual(len(stress_calls), 1)
                self.assertEqual(stress_options[0]["cache_implementation"], "offloaded")

                incomplete_name = "g4-s8"
                incomplete_path = output_dir / incomplete_name / "config.json"
                incomplete = json.loads(incomplete_path.read_text(encoding="utf-8"))
                incomplete["status"] = "incomplete"
                incomplete_path.write_text(json.dumps(incomplete), encoding="utf-8")
                manifest_path = output_dir / "qualification_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["settings"][incomplete_name]["status"] = "incomplete"
                manifest.pop("stress_test", None)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

                result = module.run_qualification(
                    output_dir, self.examples, TinyTokenizer(),
                    types.SimpleNamespace(config=types.SimpleNamespace(vocab_size=4)),
                    base_seed=42, archived_ids=self.archived_ids,
                    dataset_revisions=revisions, model_revision="model-revision",
                    source_archive_sha256="a" * 64,
                    remaining_rows_by_dataset={"gsm8k": 1, "math": 1, "gsm_plus": 1},
                    memory_probe=lambda: dict(memory),
                    configurations=["g1-s1"],
                )

        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["recommended_configuration"])
        self.assertEqual(result["stress_test"]["status"], "pending")
        self.assertEqual(stress_calls, ["stress"])

    def test_transient_config_replace_denial_resumes_only_missing_samples_with_original_seeds(self):
        module = benchmark_module(self)
        selected = module.select_benchmark_examples(
            self.examples,
            {row.example_id: 20 for row in self.examples},
            archived_ids=self.archived_ids, seed=42,
        )

        class TinyTokenizer:
            eos_token_id = 2
            vocab_size = 4

            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                assert tokenize and add_generation_prompt
                return [1] * 20

        class FakeCuda:
            @staticmethod
            def get_device_name(_index):
                return "test GPU"

            @staticmethod
            def get_device_properties(_index):
                return types.SimpleNamespace(total_memory=8 * 1024**3)

            @staticmethod
            def empty_cache():
                return None

            @staticmethod
            def reset_peak_memory_stats(_index):
                return None

            @staticmethod
            def max_memory_allocated(_index):
                return 0

            @staticmethod
            def max_memory_reserved(_index):
                return 0

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = FakeCuda
        revisions = {
            "openai/gsm8k": "1" * 40,
            "EleutherAI/hendrycks_math": "2" * 40,
            "qintongli/GSM-Plus": "3" * 40,
        }
        memory = {
            "free_mib": 8192,
            "total_mib": 8192,
            "allocated_mib": 0,
            "reserved_mib": 0,
            "available_ram_mib": 8192,
        }
        greedy = GreedyGeneration(
            text="1", token_ids=[1], token_entropies_nats=[0.1],
            token_logprobs=[-0.1], token_char_spans=[(0, 1)],
            content_token_mask=[True], eos_generated=True,
            termination_reason="eos",
        )
        greedy_inputs = [[], []]
        sample_attempts = [[], []]
        active_attempt = [0]
        interrupt_after_checkpoint = [True]
        inject_replace_denial = [False]
        denial_count = [2]

        def generate_greedy(_tokenizer, _model, questions, _max_new_tokens, **_kwargs):
            greedy_inputs[active_attempt[0]].extend(questions)
            return [greedy for _question in questions]

        def generate_samples(_tokenizer, _model, questions, _max_new_tokens, seeds,
                             *, on_result, **_kwargs):
            outputs = []
            for index, (question, seed) in enumerate(zip(questions, seeds)):
                sample_attempts[active_attempt[0]].append((question, seed))
                if active_attempt[0] == 0 and len(sample_attempts[0]) == 1:
                    inject_replace_denial[0] = True
                if active_attempt[0] == 0 and len(sample_attempts[0]) == 2 and interrupt_after_checkpoint[0]:
                    interrupt_after_checkpoint[0] = False
                    raise RuntimeError("simulated coordinator interruption")
                generation = SampleGeneration("1", [1], True, False, "eos")
                on_result(index, generation)
                outputs.append(generation)
            return outputs

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "qualification"
            config_path = output_dir / "g1-s1" / "config.json"
            real_replace = os.replace

            def transient_config_lock(source, destination):
                if inject_replace_denial[0] and Path(destination) == config_path and denial_count[0]:
                    denial_count[0] -= 1
                    raise PermissionError("injected transient qualification config lock")
                return real_replace(source, destination)

            with (
                patch.dict(sys.modules, {"torch": fake_torch}),
                patch("teacher_reliability.state.os.replace", side_effect=transient_config_lock),
                patch.object(module, "memory_profiles", side_effect=lambda size, stage: [MemoryProfile(f"{stage}-{size}", size)]),
                patch.object(module, "generate_greedy_batch", side_effect=generate_greedy),
                patch.object(module, "generate_sample_batch", side_effect=generate_samples),
                patch.object(module, "_prediction_record", side_effect=lambda row, *_args: {"example_id": row.example_id}),
                patch.object(module, "run_1024_token_stress", return_value={"status": "complete", "global_memory": {"free_mib": 8192}}),
            ):
                transient_or_interrupt = None
                try:
                    module.run_qualification(
                        output_dir, self.examples, TinyTokenizer(),
                        types.SimpleNamespace(config=types.SimpleNamespace(vocab_size=4)),
                        base_seed=42, archived_ids=self.archived_ids,
                        dataset_revisions=revisions, model_revision="model-revision",
                        source_archive_sha256="a" * 64,
                        remaining_rows_by_dataset={"gsm8k": 1, "math": 1, "gsm_plus": 1},
                        memory_probe=lambda: dict(memory), configurations=["g1-s1"],
                    )
                except (PermissionError, RuntimeError) as exc:
                    transient_or_interrupt = exc

                self.assertIsInstance(
                    transient_or_interrupt, RuntimeError,
                    "qualification must recover from the transient config lock and reach the later interruption",
                )
                self.assertEqual(str(transient_or_interrupt), "simulated coordinator interruption")
                self.assertEqual(denial_count[0], 0)
                first_question, first_seed = sample_attempts[0][0]
                self.assertEqual(len(sample_attempts[0]), 2)
                self.assertEqual(json.loads(config_path.read_text(encoding="utf-8"))["status"], "failed")

                active_attempt[0] = 1
                resumed = module.run_qualification(
                    output_dir, self.examples, TinyTokenizer(),
                    types.SimpleNamespace(config=types.SimpleNamespace(vocab_size=4)),
                    base_seed=42, archived_ids=self.archived_ids,
                    dataset_revisions=revisions, model_revision="model-revision",
                    source_archive_sha256="a" * 64,
                    remaining_rows_by_dataset={"gsm8k": 1, "math": 1, "gsm_plus": 1},
                    memory_probe=lambda: dict(memory), configurations=["g1-s1"],
                )

            expected_remaining = {
                (row.question, module.derive_sample_seed(42, row.example_id, sample_index))
                for row in selected for sample_index in range(8)
            }
            expected_remaining.remove((first_question, first_seed))
            self.assertEqual(greedy_inputs[1], [])
            self.assertEqual(len(sample_attempts[1]), 191)
            self.assertEqual(set(sample_attempts[1]), expected_remaining)
            self.assertEqual(resumed["settings"]["g1-s1"]["status"], "complete")
            self.assertEqual(resumed["status"], "incomplete")

    def test_incomplete_qualification_retries_failed_stage_without_changing_id_or_seed(self):
        module = benchmark_module(self)
        selected = module.select_benchmark_examples(
            self.examples, self.prompt_tokens, archived_ids=self.archived_ids, seed=42,
        )
        failed_example = selected[0]
        failed_input = []
        attempt_inputs = [[], []]
        active_attempt = [0]
        fail_once = [True]
        greedy_inputs = []

        class TinyTokenizer:
            eos_token_id = 2
            vocab_size = 4

            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                assert tokenize and add_generation_prompt
                return [1] * 20

        class FakeCuda:
            @staticmethod
            def get_device_name(_index):
                return "test GPU"

            @staticmethod
            def get_device_properties(_index):
                return types.SimpleNamespace(total_memory=8 * 1024**3)

            @staticmethod
            def empty_cache():
                return None

            @staticmethod
            def reset_peak_memory_stats(_index):
                return None

            @staticmethod
            def max_memory_allocated(_index):
                return 0

            @staticmethod
            def max_memory_reserved(_index):
                return 0

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = FakeCuda

        def generate_greedy(_tokenizer, _model, questions, _max_new_tokens,
                            *, cache_implementation, **_kwargs):
            if cache_implementation is None:
                raise CudaOutOfMemory("default greedy profile OOM")
            greedy_inputs.extend(questions)
            return [
                GreedyGeneration(
                    text="1", token_ids=[1], token_entropies_nats=[0.1],
                    token_logprobs=[-0.1], token_char_spans=[(0, 1)],
                    content_token_mask=[True], eos_generated=True,
                    termination_reason="eos",
                )
                for _question in questions
            ]

        def generate_samples(_tokenizer, _model, questions, _max_new_tokens, seeds,
                             *, on_result, cache_implementation, **_kwargs):
            if cache_implementation is None:
                raise CudaOutOfMemory("default sample profile OOM")
            outputs = []
            for index, (question, seed) in enumerate(zip(questions, seeds)):
                request = (question, seed)
                attempt_inputs[active_attempt[0]].append(request)
                if fail_once[0] and question == failed_example.question:
                    failed_input.append(request)
                    fail_once[0] = False
                    raise CudaOutOfMemory("injected sample OOM")
                generation = SampleGeneration("1", [1], True, False, "eos")
                on_result(index, generation)
                outputs.append(generation)
            return outputs

        memory = {
            "free_mib": 8192,
            "total_mib": 8192,
            "allocated_mib": 0,
            "reserved_mib": 0,
            "available_ram_mib": 8192,
        }
        revisions = {
            "openai/gsm8k": "1" * 40,
            "EleutherAI/hendrycks_math": "2" * 40,
            "qintongli/GSM-Plus": "3" * 40,
        }

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "qualification"
            patched_profiles = lambda size, stage: [
                MemoryProfile(f"{stage}-{size}", size),
                MemoryProfile(f"{stage}-{size}-offloaded", 1, cache_implementation="offloaded"),
            ]
            with (
                patch.dict(sys.modules, {"torch": fake_torch}),
                patch.object(module, "memory_profiles", side_effect=patched_profiles),
                patch.object(module, "_check_offloaded_cache_ram", return_value=1),
                patch.object(module, "generate_greedy_batch", side_effect=generate_greedy),
                patch.object(module, "generate_sample_batch", side_effect=generate_samples),
                patch.object(
                    module, "_prediction_record",
                    side_effect=lambda row, *_args: {"example_id": row.example_id},
                ),
                patch.object(
                    module, "run_1024_token_stress",
                    return_value={"status": "complete", "generated_tokens": 1024},
                ),
            ):
                for attempt in range(2):
                    active_attempt[0] = attempt
                    module.run_qualification(
                        output_dir, self.examples, TinyTokenizer(),
                        types.SimpleNamespace(config=types.SimpleNamespace(vocab_size=4)),
                        base_seed=42, archived_ids=self.archived_ids,
                        dataset_revisions=revisions, model_revision="model-revision",
                        source_archive_sha256="a" * 64,
                        remaining_rows_by_dataset={"gsm8k": 1, "math": 1, "gsm_plus": 1},
                        memory_probe=lambda: dict(memory),
                        configurations=["g1-s1"],
                    )

            config = json.loads((output_dir / "g1-s1" / "config.json").read_text(encoding="utf-8"))
            manifest = json.loads((output_dir / "qualification_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(len(failed_input), 1)
        self.assertIn(failed_input[0], attempt_inputs[0])
        self.assertIn(failed_input[0], attempt_inputs[1])
        self.assertEqual(attempt_inputs[1].count(failed_input[0]), 1)
        self.assertEqual(len(attempt_inputs[1]), 8)
        self.assertEqual({question for question, _seed in attempt_inputs[1]}, {failed_example.question})
        self.assertEqual(len(greedy_inputs), 24)
        self.assertEqual(config["status"], "complete")
        self.assertEqual(config["failures"], [])
        self.assertEqual(config["actual_greedy_profile"]["cache_implementation"], "offloaded")
        self.assertEqual(config["actual_sample_profile"]["cache_implementation"], "offloaded")
        self.assertEqual(len(config["failure_history"]), 1)
        self.assertEqual(config["failure_history"][0]["message"], "injected sample OOM")
        self.assertEqual(len(config["samples"]), 192)
        self.assertIn(f"{failed_example.example_id}::sample::0", config["samples"])
        self.assertEqual(len(manifest["identity"]["matrix"]), 12)
        self.assertEqual(set(manifest["settings"]), {"g1-s1"})
        self.assertEqual(manifest["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
