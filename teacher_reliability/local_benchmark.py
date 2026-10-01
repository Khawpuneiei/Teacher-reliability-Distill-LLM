"""Local RTX 4060 batching qualification for a fixed 24-question cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hybrid import CudaOutOfMemory, MemoryProfile, run_with_oom_backoff
from .local_worker import (
    MINIMUM_GLOBAL_FREE_MIB,
    MINIMUM_OFFLOADED_FREE_MIB,
    InsufficientHostMemory,
    _check_offloaded_cache_ram,
    _memory_snapshot,
    is_cuda_oom,
)
from .run import _canonical_digest, _package_versions, _prediction_record, _source_snapshot
from .selection import RunProfile
from .state import atomic_write_json, run_lock
from .teacher import (
    GreedyGeneration,
    SampleGeneration,
    build_chat_messages,
    derive_sample_seed,
    generate_greedy_batch,
    generate_sample_batch,
)


QUALIFICATION_SEED = 20261001
GREEDY_BATCH_SIZES = (1, 2, 4)
SAMPLE_BATCH_SIZES = (1, 2, 4, 8)
MATH_SUBJECTS = (
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
)
MAX_NEW_TOKENS = 1024


def qualification_matrix() -> list[tuple[int, int]]:
    return [(greedy, sample) for greedy in GREEDY_BATCH_SIZES for sample in SAMPLE_BATCH_SIZES]


def select_qualification_configurations(requested: list[str] | None = None) -> list[tuple[int, int]]:
    """Select profiles to execute while retaining the full qualification identity."""
    matrix = qualification_matrix()
    if requested is None:
        return matrix
    names = [f"g{greedy}-s{sample}" for greedy, sample in matrix]
    if not requested:
        raise ValueError("at least one qualification configuration must be selected")
    if len(requested) != len(set(requested)):
        raise ValueError("qualification configurations contain a duplicate")
    unsupported = set(requested) - set(names)
    if unsupported:
        raise ValueError(f"unsupported qualification configuration: {sorted(unsupported)[0]}")
    requested_names = set(requested)
    return [pair for pair, name in zip(matrix, names) if name in requested_names]


def memory_profiles(batch_size: int, stage: str) -> list[MemoryProfile]:
    if stage not in {"greedy", "sample"}:
        raise ValueError("stage must be greedy or sample")
    if batch_size not in ({1, 2, 4} if stage == "greedy" else {1, 2, 4, 8}):
        raise ValueError("batch size is outside the local qualification matrix")
    sizes = [batch_size]
    while sizes[-1] > 1:
        sizes.append(max(1, sizes[-1] // 2))
    profiles = [MemoryProfile(f"{stage}-{size}", size) for size in sizes]
    profiles.append(MemoryProfile(f"{stage}-1-offloaded", 1, cache_implementation="offloaded"))
    return profiles


def _rank(example, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{example.example_id}".encode("utf-8")).hexdigest()


def remaining_rows_by_dataset(examples, imported_rows) -> dict[str, int]:
    """Subtract imported rows using validated selected IDs, never archive labels."""
    rows = list(examples)
    examples_by_id = {row.example_id: row for row in rows}
    if len(examples_by_id) != len(rows):
        raise ValueError("selected examples contain duplicate IDs")
    imported_ids = []
    for row in imported_rows:
        example_id = row.get("example_id") if isinstance(row, dict) else None
        if example_id not in examples_by_id:
            raise ValueError("imported example ID is outside the selected examples")
        imported_ids.append(example_id)
    if len(imported_ids) != len(set(imported_ids)):
        raise ValueError("imported rows contain duplicate example IDs")
    datasets = ("gsm8k", "math", "gsm_plus")
    imported_set = set(imported_ids)
    return {
        dataset: sum(
            row.dataset == dataset and row.example_id not in imported_set
            for row in rows
        )
        for dataset in datasets
    }


def _take_longest(rows, prompt_token_counts, count: int, excluded: set[str] | None = None):
    excluded = excluded or set()
    candidates = [row for row in rows if row.example_id not in excluded]
    return sorted(candidates, key=lambda row: (-prompt_token_counts[row.example_id], row.example_id))[:count]


def select_benchmark_examples(
    examples,
    prompt_token_counts: dict[str, int],
    *,
    archived_ids: set[str],
    seed: int = QUALIFICATION_SEED,
):
    """Choose eight examples per dataset, covering every MATH subject and long prompts."""
    rows = list(examples)
    ids = [row.example_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("benchmark inputs contain duplicate IDs")
    if set(ids) - set(prompt_token_counts):
        raise ValueError("prompt lengths are missing for benchmark inputs")
    by_dataset = {dataset: [row for row in rows if row.dataset == dataset] for dataset in ("gsm8k", "math", "gsm_plus")}
    if any(len(group) < 8 for group in by_dataset.values()):
        raise ValueError("qualification needs at least eight examples per dataset")
    math_subjects = {row.source_config for row in by_dataset["math"]}
    if not set(MATH_SUBJECTS).issubset(math_subjects):
        raise ValueError("qualification requires all seven MATH subjects")

    selected = []
    gsm_rows = by_dataset["gsm8k"]
    archived = sorted(
        (row for row in gsm_rows if row.example_id in archived_ids),
        key=lambda row: (_rank(row, seed), row.example_id),
    )
    if len(archived) < 3:
        raise ValueError("qualification needs several archived A100 questions for comparison")
    gsm_selected = archived[:4]
    gsm_selected.extend(_take_longest(gsm_rows, prompt_token_counts, 1, {row.example_id for row in gsm_selected}))
    already = {row.example_id for row in gsm_selected}
    gsm_selected.extend(
        row for row in sorted(gsm_rows, key=lambda row: (_rank(row, seed), row.example_id))
        if row.example_id not in already
    )
    selected.extend(gsm_selected[:8])

    math_rows = by_dataset["math"]
    math_selected = []
    for subject in MATH_SUBJECTS:
        group = [row for row in math_rows if row.source_config == subject]
        math_selected.extend(_take_longest(group, prompt_token_counts, 1))
    math_ids = {row.example_id for row in math_selected}
    math_selected.extend(
        row for row in sorted(math_rows, key=lambda row: (-prompt_token_counts[row.example_id], _rank(row, seed), row.example_id))
        if row.example_id not in math_ids
    )
    selected.extend(math_selected[:8])

    plus_rows = by_dataset["gsm_plus"]
    plus_selected = _take_longest(plus_rows, prompt_token_counts, 2)
    plus_ids = {row.example_id for row in plus_selected}
    plus_selected.extend(
        row for row in sorted(plus_rows, key=lambda row: (_rank(row, seed), row.example_id))
        if row.example_id not in plus_ids
    )
    selected.extend(plus_selected[:8])
    if len(selected) != 24 or len({row.example_id for row in selected}) != 24:
        raise ValueError("qualification selection is not 24 unique examples")
    if any(prompt_token_counts[row.example_id] + MAX_NEW_TOKENS > 4096 for row in selected):
        raise ValueError("qualification prompt plus output exceeds the 4,096-token context limit")
    return sorted(selected, key=lambda row: (row.dataset, row.example_id))


def _stage_path(config_dir: Path, request_id: str) -> Path:
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    return config_dir / "stages" / f"{digest}.json"


def _save_stage(config_dir: Path, signature: str, request_id: str, generation: Any) -> None:
    atomic_write_json(_stage_path(config_dir, request_id), {
        "run_signature": signature,
        "request_id": request_id,
        "generation": asdict(generation),
    })


def _load_stages(config_dir: Path, signature: str) -> dict[str, Any]:
    results = {}
    for path in (config_dir / "stages").glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("run_signature") != signature or not isinstance(value.get("request_id"), str):
            raise ValueError("qualification stage identity differs from the configuration")
        results[value["request_id"]] = value["generation"]
    return results


def _max_memory(memory: dict[str, int], snapshot: dict[str, Any]) -> None:
    for source, target in (("allocated_mib", "peak_allocated_mib"),
                           ("reserved_mib", "peak_reserved_mib"),
                           ("available_ram_mib", "peak_available_ram_mib")):
        memory[target] = max(memory.get(target, 0), int(snapshot.get(source, 0)))
    available_ram = snapshot.get("available_ram_mib")
    if available_ram is not None:
        previous_minimum = memory.get("minimum_available_ram_mib")
        observed = int(available_ram)
        memory["minimum_available_ram_mib"] = (
            observed if previous_minimum is None else min(previous_minimum, observed)
        )
    free = int(snapshot.get("free_mib", 0))
    previous_free = memory.get("minimum_global_free_mib")
    memory["minimum_global_free_mib"] = free if previous_free is None else min(previous_free, free)


def validate_memory_snapshot(profile: MemoryProfile, snapshot: dict[str, Any]) -> int:
    """Apply the same normal and offloaded-cache VRAM floors as the runner."""
    minimum = (
        MINIMUM_OFFLOADED_FREE_MIB
        if profile.cache_implementation == "offloaded"
        else MINIMUM_GLOBAL_FREE_MIB
    )
    free_mib = int(snapshot.get("free_mib", 0))
    if free_mib < minimum:
        raise CudaOutOfMemory(
            f"global GPU free memory is {free_mib} MiB, below the {minimum} MiB "
            f"qualification minimum for cache={profile.cache_implementation or 'default'}"
        )
    return minimum


class GenerationMemoryMonitor:
    """Sample global GPU headroom while a generation call is in progress."""

    def __init__(
        self,
        profile: MemoryProfile,
        memory: dict[str, Any],
        memory_probe,
        *,
        poll_interval_seconds: float = 0.25,
        minimum_free_mib: int | None = None,
    ):
        if not math.isfinite(poll_interval_seconds) or poll_interval_seconds <= 0:
            raise ValueError("memory monitor poll interval must be positive")
        if minimum_free_mib is not None and (
            isinstance(minimum_free_mib, bool)
            or not isinstance(minimum_free_mib, int)
            or minimum_free_mib < 1
        ):
            raise ValueError("memory monitor minimum free memory must be a positive integer")
        self.profile = profile
        self.memory = memory
        self.memory_probe = memory_probe
        self.poll_interval_seconds = poll_interval_seconds
        self.minimum_free_mib = minimum_free_mib if minimum_free_mib is not None else (
            MINIMUM_OFFLOADED_FREE_MIB
            if profile.cache_implementation == "offloaded"
            else MINIMUM_GLOBAL_FREE_MIB
        )
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._headroom_error: CudaOutOfMemory | None = None
        self._probe_error: Exception | None = None

    def _observe(self) -> None:
        snapshot = self.memory_probe()
        _max_memory(self.memory, snapshot)
        free_mib = int(snapshot.get("free_mib", 0))
        if free_mib < self.minimum_free_mib:
            with self._lock:
                if self._headroom_error is None:
                    self._headroom_error = CudaOutOfMemory(
                        f"global GPU free memory reached {free_mib} MiB during generation, "
                        f"below the {self.minimum_free_mib} MiB qualification floor"
                    )

    def _poll(self) -> None:
        while not self._stop.wait(self.poll_interval_seconds):
            try:
                self._observe()
            except Exception as exc:
                with self._lock:
                    self._probe_error = exc
                return

    def check(self) -> None:
        with self._lock:
            headroom_error = self._headroom_error
            probe_error = self._probe_error
        if probe_error is not None:
            raise RuntimeError("global GPU memory sampling failed during generation") from probe_error
        if headroom_error is not None:
            raise headroom_error

    def __enter__(self):
        self._observe()
        self.check()
        self._thread = threading.Thread(target=self._poll, name="gpu-memory-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, error_type, error, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.poll_interval_seconds * 4))
            if self._thread.is_alive():
                with self._lock:
                    self._probe_error = RuntimeError("global GPU memory sampler did not stop")
            else:
                try:
                    self._observe()
                except Exception as exc:
                    with self._lock:
                        self._probe_error = exc
        if error_type is None:
            self.check()
        return False


def normalize_cuda_oom(error: BaseException) -> BaseException:
    """Convert raw PyTorch CUDA OOMs into the coordinator's backoff signal."""
    if isinstance(error, CudaOutOfMemory):
        return error
    if is_cuda_oom(error):
        return CudaOutOfMemory(str(error))
    return error


def is_qualified_configuration(result: dict[str, Any]) -> bool:
    """Require completion and enough measured headroom for the used cache mode."""
    if result.get("status") != "complete":
        return False
    profiles = (
        result.get("actual_greedy_profile") or {},
        result.get("actual_sample_profile") or {},
    )
    all_stages_offloaded = all(
        profile.get("cache_implementation") == "offloaded" for profile in profiles
    )
    required_free_mib = (
        MINIMUM_OFFLOADED_FREE_MIB if all_stages_offloaded else MINIMUM_GLOBAL_FREE_MIB
    )
    observed_free_mib = result.get("memory", {}).get("minimum_global_free_mib")
    return isinstance(observed_free_mib, int) and observed_free_mib >= required_free_mib


def stress_cache_implementation(configuration: dict[str, Any]) -> str | None:
    """Use offloaded KV cache only when both qualified stages use it."""
    profiles = (
        configuration.get("actual_greedy_profile"),
        configuration.get("actual_sample_profile"),
    )
    if not all(isinstance(profile, dict) for profile in profiles):
        raise ValueError("qualified configuration omits an actual generation profile")
    if all(profile.get("cache_implementation") == "offloaded" for profile in profiles):
        return "offloaded"
    return None


def select_stress_candidate(
    candidates,
    stress_runner,
    *,
    previous_results: dict[str, Any] | None = None,
    on_result=None,
):
    """Choose the fastest measured profile whose long-output stress passes."""
    results = dict(previous_results or {})
    if any(not isinstance(name, str) or not name for name in results):
        raise ValueError("stress results contain an invalid configuration name")
    for name, configuration in candidates:
        result = results.get(name)
        if result is None:
            result = stress_runner(name, configuration)
            if not isinstance(result, dict):
                raise ValueError("stress runner must return a result object")
            results[name] = result
            if on_result is not None:
                on_result(name, result)
        if not isinstance(result, dict):
            raise ValueError(f"saved stress result is invalid: {name}")
        memory = result.get("global_memory")
        free_mib = memory.get("free_mib") if isinstance(memory, dict) else None
        if (
            result.get("status") == "complete"
            and isinstance(free_mib, int) and not isinstance(free_mib, bool)
            and free_mib >= MINIMUM_GLOBAL_FREE_MIB
            and isinstance(
                result.get("minimum_global_free_mib", free_mib), int
            )
            and not isinstance(result.get("minimum_global_free_mib", free_mib), bool)
            and result.get("minimum_global_free_mib", free_mib) >= MINIMUM_GLOBAL_FREE_MIB
        ):
            return name, result, results
    return None, None, results


def estimate_dataset_weighted_eta_hours(
    remaining_rows_by_dataset: dict[str, int],
    rows_per_hour_by_dataset: dict[str, float],
) -> float | None:
    """Sum per-dataset remaining work using measured qualification rates."""
    total_hours = 0.0
    for dataset, remaining in remaining_rows_by_dataset.items():
        if isinstance(remaining, bool) or not isinstance(remaining, int) or remaining < 0:
            raise ValueError("remaining dataset row counts must be non-negative integers")
        if remaining == 0:
            continue
        rate = rows_per_hour_by_dataset.get(dataset)
        if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not math.isfinite(rate) or rate <= 0:
            return None
        total_hours += remaining / float(rate)
    return round(total_hours, 3)


def _allocate_elapsed_by_dataset(
    dataset_work_seconds: dict[str, float],
    batch: list[dict[str, Any]],
    dataset_by_id: dict[str, str],
    elapsed_seconds: float,
) -> None:
    if not batch:
        return
    counts: dict[str, int] = {}
    for request in batch:
        dataset = dataset_by_id[request["example_id"]]
        counts[dataset] = counts.get(dataset, 0) + 1
    for dataset, count in counts.items():
        dataset_work_seconds[dataset] = (
            dataset_work_seconds.get(dataset, 0.0) + elapsed_seconds * count / len(batch)
        )


def _compare_to_baseline(current: dict[str, Any], baseline: dict[str, Any], examples) -> dict[str, Any]:
    greedy_equal = []
    entropies = []
    logprobs = []
    sample_equal = []
    for example in examples:
        example_id = example.example_id
        left = current["greedy"].get(example_id)
        right = baseline["greedy"].get(example_id)
        if left is not None and right is not None:
            greedy_equal.append(left["token_ids"] == right["token_ids"])
            for field, target in (("token_entropies_nats", entropies), ("token_logprobs", logprobs)):
                for a, b in zip(left[field], right[field]):
                    target.append(abs(float(a) - float(b)))
        for index in range(8):
            key = f"{example_id}::sample::{index}"
            a = current["samples"].get(key)
            b = baseline["samples"].get(key)
            if a is not None and b is not None:
                sample_equal.append(a["token_ids"] == b["token_ids"])
    return {
        "greedy_exact_token_agreement": sum(greedy_equal) / len(greedy_equal) if greedy_equal else None,
        "sample_exact_token_agreement": sum(sample_equal) / len(sample_equal) if sample_equal else None,
        "max_absolute_entropy_difference": max(entropies, default=0.0),
        "max_absolute_logprob_difference": max(logprobs, default=0.0),
        "greedy_compared_rows": len(greedy_equal),
        "sample_compared_generations": len(sample_equal),
    }


def _qualification_profile(name: str, sample_count: int = 8) -> RunProfile:
    return RunProfile(name, None, None, None, False, "test", sample_count, MAX_NEW_TOKENS)


def _run_qualification_body(
    output_dir: Path,
    examples,
    tokenizer: Any,
    model: Any,
    *,
    base_seed: int,
    archived_ids: set[str],
    dataset_revisions: dict[str, str],
    model_revision: str,
    source_archive_sha256: str,
    remaining_rows_by_dataset: dict[str, int] | None = None,
    memory_probe=_memory_snapshot,
    progress_callback=None,
    configurations: list[str] | None = None,
):
    """Benchmark all 12 local batch combinations, checkpointing each result."""
    import torch

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_token_counts = {
        row.example_id: len(tokenizer.apply_chat_template(
            build_chat_messages(row.question), tokenize=True, add_generation_prompt=True
        ))
        for row in examples
    }
    selected = select_benchmark_examples(
        examples, prompt_token_counts, archived_ids=archived_ids, seed=base_seed
    )
    selected_ids = [row.example_id for row in selected]
    selection_signature = _canonical_digest([
        {"example_id": row.example_id, "question": row.question, "answer": row.gold_answer,
         "source": row.source_revision, "prompt_tokens": prompt_token_counts[row.example_id]}
        for row in selected
    ])
    complete_matrix = qualification_matrix()
    selected_matrix = select_qualification_configurations(configurations)
    qualification_identity = {
        "model_revision": model_revision,
        "dataset_revisions": dataset_revisions,
        "source_archive_sha256": source_archive_sha256,
        "source_snapshot_sha256": _source_snapshot(Path(__file__).resolve().parents[1]),
        "seed": base_seed,
        "selection_sha256": selection_signature,
        "selected_ids": selected_ids,
        "matrix": complete_matrix,
        "greedy_batch_sizes": GREEDY_BATCH_SIZES,
        "sample_batch_sizes": SAMPLE_BATCH_SIZES,
        "minimum_global_gpu_free_mib": MINIMUM_GLOBAL_FREE_MIB,
        "minimum_offloaded_cache_gpu_free_mib": MINIMUM_OFFLOADED_FREE_MIB,
        "remaining_rows_by_dataset": remaining_rows_by_dataset,
        "max_new_tokens": MAX_NEW_TOKENS,
        "samples_per_question": 8,
        "software": _package_versions(),
        "hardware": {
            "platform": platform.platform(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        },
    }
    signature = _canonical_digest(qualification_identity)
    manifest_path = output_dir / "qualification_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("qualification_signature") != signature:
            raise ValueError("qualification request differs from the saved manifest")
    else:
        manifest = {
            "schema_version": 1,
            "qualification_id": output_dir.name,
            "qualification_signature": signature,
            "identity": qualification_identity,
            "status": "running",
            "selection": {"count": len(selected), "sha256": selection_signature,
                          "example_ids": selected_ids,
                          "prompt_tokens": prompt_token_counts},
            "settings": {},
            "remaining_rows_by_dataset": remaining_rows_by_dataset or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(manifest_path, manifest)

    config_results = {}
    dataset_profile = _qualification_profile("qualification")
    for greedy_batch_size, sample_batch_size in selected_matrix:
        name = f"g{greedy_batch_size}-s{sample_batch_size}"
        config_dir = output_dir / name
        config_dir.mkdir(exist_ok=True)
        config_signature = _canonical_digest({"qualification": signature, "config": name})
        config_manifest_path = config_dir / "config.json"
        if config_manifest_path.is_file():
            result = json.loads(config_manifest_path.read_text(encoding="utf-8"))
            if result.get("run_signature") != config_signature:
                raise ValueError(f"saved qualification configuration identity differs: {name}")
        else:
            result = {
                "run_signature": config_signature,
                "configuration": {"greedy_batch_size": greedy_batch_size,
                                  "sampling_batch_size": sample_batch_size},
                "status": "running",
                "greedy": {},
                "samples": {},
                "failures": [],
                "adaptations": [],
                "timing_seconds": {"greedy_score_generation": 0.0,
                                    "sampling_generation": 0.0, "grading": 0.0},
                "memory": {"minimum_global_free_mib": None, "peak_allocated_mib": 0,
                           "peak_reserved_mib": 0, "peak_available_ram_mib": 0,
                           "minimum_available_ram_mib": None},
                "dataset_work_seconds": {dataset: 0.0 for dataset in ("gsm8k", "math", "gsm_plus")},
            }
            atomic_write_json(config_manifest_path, result)
        stages = _load_stages(config_dir, config_signature)
        result["greedy"] = {
            row.example_id: stages[f"greedy::{row.example_id}"]
            for row in selected if f"greedy::{row.example_id}" in stages
        }
        result["samples"] = {
            key.removeprefix("sample::"): value for key, value in stages.items()
            if key.startswith("sample::")
        }
        result["timing_seconds"] = result.get("timing_seconds", {
            "greedy_score_generation": 0.0, "sampling_generation": 0.0, "grading": 0.0
        })
        memory = result.setdefault("memory", {})
        memory.setdefault("minimum_global_free_mib", None)
        memory.setdefault("peak_allocated_mib", 0)
        memory.setdefault("peak_reserved_mib", 0)
        memory.setdefault("peak_available_ram_mib", 0)
        memory.setdefault("minimum_available_ram_mib", None)
        dataset_work_seconds = result.setdefault(
            "dataset_work_seconds", {dataset: 0.0 for dataset in ("gsm8k", "math", "gsm_plus")}
        )
        failure_history = result.setdefault("failure_history", [])
        if not isinstance(failure_history, list):
            raise ValueError(f"saved qualification failure history is invalid: {name}")
        prior_failures = result.get("failures", [])
        if not isinstance(prior_failures, list):
            raise ValueError(f"saved qualification failures are invalid: {name}")
        failure_history.extend(prior_failures)
        failures: list[dict[str, Any]] = []
        result["failures"] = failures
        failed_examples: set[str] = set()
        if prior_failures:
            atomic_write_json(config_manifest_path, result)
        dataset_by_id = {row.example_id: row.dataset for row in selected}

        def record_result(request, generation, stage):
            if stage == "greedy":
                result["greedy"][request["example_id"]] = asdict(generation)
                key = f"greedy::{request['example_id']}"
            else:
                request_id = request["request_id"]
                result["samples"][request_id] = asdict(generation)
                key = f"sample::{request_id}"
            _save_stage(config_dir, config_signature, key, generation)
            atomic_write_json(config_manifest_path, result)

        def profile_adapt(stage_name):
            def adapt(profile, error):
                result["adaptations"].append({
                    "stage": stage_name, "profile": asdict(profile),
                    "reason": str(error)[:500],
                    "at": datetime.now(timezone.utc).isoformat(),
                })
                atomic_write_json(config_manifest_path, result)
            return adapt

        def safe_memory(profile):
            torch.cuda.empty_cache()
            current = memory_probe()
            _max_memory(memory, current)
            validate_memory_snapshot(profile, current)
            if profile.cache_implementation == "offloaded":
                import psutil
                _check_offloaded_cache_ram(model.config, profile.batch_size, psutil.virtual_memory().available)

        greedy_requests = [
            {"request_id": row.example_id, "example_id": row.example_id,
             "question": row.question, "max_new_tokens": MAX_NEW_TOKENS}
            for row in selected if row.example_id not in result["greedy"]
            and row.example_id not in failed_examples
        ]

        def execute_greedy(batch, memory_profile, *, on_result, on_progress):
            safe_memory(memory_profile)
            started = time.perf_counter()
            try:
                with GenerationMemoryMonitor(memory_profile, memory, memory_probe) as monitor:
                    def generation_progress(event):
                        monitor.check()
                        if on_progress is not None:
                            on_progress(event)

                    generated = generate_greedy_batch(
                        tokenizer, model, [row["question"] for row in batch], MAX_NEW_TOKENS,
                        cache_implementation=memory_profile.cache_implementation,
                        on_progress=generation_progress,
                    )
            except BaseException as exc:
                elapsed = time.perf_counter() - started
                result["timing_seconds"]["greedy_score_generation"] += elapsed
                _allocate_elapsed_by_dataset(dataset_work_seconds, batch, dataset_by_id, elapsed)
                atomic_write_json(config_manifest_path, result)
                normalized = normalize_cuda_oom(exc)
                if normalized is exc:
                    raise
                raise normalized from exc
            elapsed = time.perf_counter() - started
            result["timing_seconds"]["greedy_score_generation"] += elapsed
            _allocate_elapsed_by_dataset(dataset_work_seconds, batch, dataset_by_id, elapsed)
            mapping = {}
            for request, generation in zip(batch, generated):
                record_result(request, generation, "greedy")
                mapping[request["request_id"]] = generation
                on_result(request["request_id"], generation)
            _max_memory(memory, memory_probe())
            atomic_write_json(config_manifest_path, result)
            return mapping

        def fail_greedy(request, error):
            failure = {"example_id": request["example_id"], "stage": "greedy",
                       "error_type": type(error).__name__, "message": str(error)[:500]}
            failures.append(failure)
            failed_examples.add(request["example_id"])
            atomic_write_json(config_manifest_path, result)

        try:
            torch.cuda.reset_peak_memory_stats(0)
            if greedy_requests:
                used_greedy_profile = run_with_oom_backoff(
                    greedy_requests, memory_profiles(greedy_batch_size, "greedy"), execute_greedy,
                    on_complete=lambda *_: None, on_failure=fail_greedy,
                    on_adapt=profile_adapt("greedy"),
                    skip_request=lambda request: request["example_id"] in failed_examples,
                )
                result["actual_greedy_profile"] = asdict(used_greedy_profile)

            sample_requests = []
            for row in selected:
                if row.example_id in failed_examples:
                    continue
                for index in range(8):
                    request_id = f"{row.example_id}::sample::{index}"
                    if request_id not in result["samples"]:
                        sample_requests.append({
                            "request_id": request_id, "example_id": row.example_id,
                            "question": row.question, "sample_index": index,
                            "seed": derive_sample_seed(base_seed, row.example_id, index),
                            "max_new_tokens": MAX_NEW_TOKENS,
                        })

            def execute_sample(batch, memory_profile, *, on_result, on_progress):
                safe_memory(memory_profile)
                started = time.perf_counter()
                def stream_sample(index, generation):
                    request = batch[index]
                    record_result(request, generation, "sample")
                    on_result(request["request_id"], generation)

                try:
                    with GenerationMemoryMonitor(memory_profile, memory, memory_probe) as monitor:
                        def generation_progress(event):
                            monitor.check()
                            if on_progress is not None:
                                on_progress(event)

                        generated = generate_sample_batch(
                            tokenizer, model, [row["question"] for row in batch], MAX_NEW_TOKENS,
                            [row["seed"] for row in batch],
                            on_result=stream_sample,
                            cache_implementation=memory_profile.cache_implementation,
                            on_progress=generation_progress,
                        )
                except BaseException as exc:
                    elapsed = time.perf_counter() - started
                    result["timing_seconds"]["sampling_generation"] += elapsed
                    _allocate_elapsed_by_dataset(dataset_work_seconds, batch, dataset_by_id, elapsed)
                    atomic_write_json(config_manifest_path, result)
                    normalized = normalize_cuda_oom(exc)
                    if normalized is exc:
                        raise
                    raise normalized from exc
                elapsed = time.perf_counter() - started
                result["timing_seconds"]["sampling_generation"] += elapsed
                _allocate_elapsed_by_dataset(dataset_work_seconds, batch, dataset_by_id, elapsed)
                mapping = {}
                for request, generation in zip(batch, generated):
                    mapping[request["request_id"]] = generation
                _max_memory(memory, memory_probe())
                atomic_write_json(config_manifest_path, result)
                return mapping

            def fail_sample(request, error):
                failure = {"example_id": request["example_id"], "stage": "sample",
                           "sample_index": request["sample_index"],
                           "error_type": type(error).__name__, "message": str(error)[:500]}
                failures.append(failure)
                failed_examples.add(request["example_id"])
                atomic_write_json(config_manifest_path, result)

            if sample_requests:
                used_sample_profile = run_with_oom_backoff(
                    sample_requests, memory_profiles(sample_batch_size, "sample"), execute_sample,
                    on_complete=lambda *_: None, on_failure=fail_sample,
                    on_adapt=profile_adapt("sample"),
                    skip_request=lambda request: request["example_id"] in failed_examples,
                )
                result["actual_sample_profile"] = asdict(used_sample_profile)
            result["memory"]["peak_allocated_mib"] = max(
                result["memory"].get("peak_allocated_mib", 0),
                int(torch.cuda.max_memory_allocated(0) / (1024**2)),
            )
            result["memory"]["peak_reserved_mib"] = max(
                result["memory"].get("peak_reserved_mib", 0),
                int(torch.cuda.max_memory_reserved(0) / (1024**2)),
            )
            result["status"] = "complete" if not failures and len(result["greedy"]) == 24 and len(result["samples"]) == 192 else "incomplete"

            if result["status"] == "complete":
                started = time.perf_counter()
                predictions = []
                for row in selected:
                    greedy = GreedyGeneration(**result["greedy"][row.example_id])
                    samples = [
                        SampleGeneration(**result["samples"][f"{row.example_id}::sample::{index}"])
                        for index in range(8)
                    ]
                    seeds = [derive_sample_seed(base_seed, row.example_id, index) for index in range(8)]
                    predictions.append(_prediction_record(row, greedy, samples, seeds, tokenizer, model, 8))
                grading_elapsed = time.perf_counter() - started
                result["timing_seconds"]["grading"] = grading_elapsed
                for dataset in ("gsm8k", "math", "gsm_plus"):
                    count = sum(row.dataset == dataset for row in selected)
                    dataset_work_seconds[dataset] += grading_elapsed * count / len(selected)
                result["rows_per_hour_by_dataset"] = {
                    dataset: sum(row.dataset == dataset for row in selected) * 3600 / seconds
                    for dataset, seconds in dataset_work_seconds.items() if seconds > 0
                }
                result["rows_per_hour"] = 24 * 3600 / max(1e-9, sum(dataset_work_seconds.values()))
                result["peak_memory"] = result["memory"]
                result["predictions"] = predictions
            result["finished_at"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(config_manifest_path, result)
            config_results[name] = result
            manifest["settings"][name] = {
                "status": result["status"],
                "rows_per_hour": result.get("rows_per_hour"),
                "rows_per_hour_by_dataset": result.get("rows_per_hour_by_dataset"),
                "actual_greedy_profile": result.get("actual_greedy_profile"),
                "actual_sample_profile": result.get("actual_sample_profile"),
                "memory": result.get("memory"),
            }
            atomic_write_json(manifest_path, manifest)
            if progress_callback:
                progress_callback({"configuration": name, "status": result["status"],
                                   "rows_per_hour": result.get("rows_per_hour")})
        except BaseException as exc:
            result["status"] = "failed"
            result["last_error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
            atomic_write_json(config_manifest_path, result)
            manifest["status"] = "failed"
            manifest["failed_configuration"] = name
            atomic_write_json(manifest_path, manifest)
            raise

    for greedy_batch_size, sample_batch_size in complete_matrix:
        name = f"g{greedy_batch_size}-s{sample_batch_size}"
        if name in config_results:
            continue
        config_manifest_path = output_dir / name / "config.json"
        if not config_manifest_path.is_file():
            continue
        result = json.loads(config_manifest_path.read_text(encoding="utf-8"))
        config_signature = _canonical_digest({"qualification": signature, "config": name})
        if result.get("run_signature") != config_signature:
            raise ValueError(f"saved qualification configuration identity differs: {name}")
        config_results[name] = result
        manifest["settings"].setdefault(name, {
            "status": result.get("status"),
            "rows_per_hour": result.get("rows_per_hour"),
            "rows_per_hour_by_dataset": result.get("rows_per_hour_by_dataset"),
            "actual_greedy_profile": result.get("actual_greedy_profile"),
            "actual_sample_profile": result.get("actual_sample_profile"),
            "memory": result.get("memory"),
        })

    baseline = config_results.get("g1-s1")
    for name, result in config_results.items():
        result["comparison_to_serial"] = _compare_to_baseline(result, baseline, selected) if baseline else None
        atomic_write_json(output_dir / name / "config.json", result)
        manifest["settings"][name]["comparison_to_serial"] = result["comparison_to_serial"]

    matrix_complete = all(
        (
            result := config_results.get(f"g{greedy}-s{sample}")
        ) is not None and result.get("status") == "complete"
        for greedy, sample in complete_matrix
    )
    candidates = [
        (name, result) for name, result in config_results.items()
        if matrix_complete and is_qualified_configuration(result)
    ]
    candidates.sort(key=lambda pair: (-(pair[1].get("rows_per_hour") or 0), pair[0]))
    stress_results = manifest.get("stress_tests", {})
    if not isinstance(stress_results, dict):
        raise ValueError("saved qualification stress results are invalid")
    selected_stress_configuration = None
    selected_stress_result = None
    if matrix_complete and candidates:
        def run_stress(configuration_name, configuration):
            cache_implementation = stress_cache_implementation(configuration)
            stress_result = run_1024_token_stress(
                selected, prompt_token_counts, tokenizer, model,
                memory_probe=memory_probe,
                cache_implementation=cache_implementation,
            )
            if not isinstance(stress_result, dict):
                raise ValueError("stress test must return a result object")
            return {
                **stress_result,
                "configuration": configuration_name,
                "cache_implementation": cache_implementation,
            }

        def save_stress(configuration_name, result):
            stress_results[configuration_name] = result
            manifest["stress_tests"] = stress_results
            atomic_write_json(manifest_path, manifest)

        selected_stress_configuration, selected_stress_result, stress_results = (
            select_stress_candidate(
                candidates, run_stress, previous_results=stress_results,
                on_result=save_stress,
            )
        )
        manifest["stress_tests"] = stress_results
        manifest["stress_test"] = (
            selected_stress_result
            if selected_stress_result is not None
            else {
                "status": "failed",
                "reason": "no measured configuration passed the 1,024-token stress and 1 GiB headroom gate",
                "tested_configurations": list(stress_results),
            }
        )
    elif not matrix_complete:
        manifest["stress_test"] = {
            "status": "pending",
            "reason": "qualification matrix is incomplete",
        }
    else:
        manifest["stress_test"] = {
            "status": "unavailable",
            "reason": "no measured configuration met its regular memory headroom floor",
        }
    stress_test = manifest.get("stress_test", {})
    stress_memory = stress_test.get("global_memory")
    stress_free_mib = stress_memory.get("free_mib") if isinstance(stress_memory, dict) else None
    stress_minimum_free_mib = stress_test.get("minimum_global_free_mib", stress_free_mib)
    stress_passed = (
        selected_stress_configuration is not None
        and stress_test.get("status") == "complete"
        and isinstance(stress_free_mib, int) and not isinstance(stress_free_mib, bool)
        and stress_free_mib >= MINIMUM_GLOBAL_FREE_MIB
        and isinstance(stress_minimum_free_mib, int)
        and not isinstance(stress_minimum_free_mib, bool)
        and stress_minimum_free_mib >= MINIMUM_GLOBAL_FREE_MIB
    )
    manifest["recommended_configuration"] = selected_stress_configuration if stress_passed else None
    manifest["status"] = "complete" if matrix_complete and candidates and stress_passed else "incomplete"
    manifest["remaining_rows_by_dataset"] = remaining_rows_by_dataset or {}
    recommended = config_results.get(manifest["recommended_configuration"])
    manifest["rows_per_hour_by_dataset"] = (
        recommended.get("rows_per_hour_by_dataset") if recommended else None
    )
    manifest["dataset_weighted_eta_hours"] = (
        estimate_dataset_weighted_eta_hours(
            remaining_rows_by_dataset or {}, recommended["rows_per_hour_by_dataset"]
        )
        if recommended and recommended.get("rows_per_hour_by_dataset") else None
    )
    atomic_write_json(manifest_path, manifest)
    return manifest


def run_qualification(output_dir: Path, *args, **kwargs):
    """Run or resume qualification while exclusively owning its output directory."""
    with run_lock(Path(output_dir).resolve()):
        return _run_qualification_body(output_dir, *args, **kwargs)


def run_1024_token_stress(
    examples, prompt_token_counts, tokenizer, model, *,
    memory_probe=_memory_snapshot, cache_implementation=None,
):
    """Force the longest selected prompt to retain a full 1,024-token KV cache."""
    import torch
    from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

    if cache_implementation not in (None, "offloaded"):
        raise ValueError("unsupported stress-test cache implementation")
    example = max(examples, key=lambda row: (prompt_token_counts[row.example_id], row.example_id))
    encoded = tokenizer.apply_chat_template(
        build_chat_messages(example.question), tokenize=True,
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(device) for key, value in encoded.items()}
    prompt_width = int(inputs["input_ids"].shape[-1])
    eos = getattr(model.generation_config, "eos_token_id", None)
    eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
    memory = {"minimum_global_free_mib": None}
    stress_profile = MemoryProfile("stress-1024", 1, cache_implementation=cache_implementation)

    class SuppressEos(LogitsProcessor):
        def __init__(self, memory_monitor):
            self.memory_monitor = memory_monitor

        def __call__(self, input_ids, scores):
            generated_count = int(input_ids.shape[-1]) - prompt_width
            if generated_count % 16 == 0:
                self.memory_monitor.check()
            if int(input_ids.shape[-1]) - prompt_width < MAX_NEW_TOKENS:
                for token_id in eos_ids:
                    scores[:, int(token_id)] = float("-inf")
            return scores

    if cache_implementation == "offloaded":
        import psutil

        try:
            _check_offloaded_cache_ram(
                model.config, 1, int(psutil.virtual_memory().available)
            )
        except InsufficientHostMemory as exc:
            return {
                "status": "insufficient_ram",
                "example_id": example.example_id,
                "prompt_tokens": prompt_width,
                "requested_new_tokens": MAX_NEW_TOKENS,
                "cache_implementation": cache_implementation,
                "error": str(exc)[:500],
                "global_memory": memory_probe(),
            }

    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    try:
        cache_kwargs = (
            {"cache_implementation": cache_implementation}
            if cache_implementation else {}
        )
        with GenerationMemoryMonitor(
            stress_profile, memory, memory_probe,
            minimum_free_mib=MINIMUM_GLOBAL_FREE_MIB,
        ) as memory_monitor:
            with torch.inference_mode():
                output = model.generate(
                    **inputs, do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS,
                    return_dict_in_generate=True, output_scores=False, use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    logits_processor=LogitsProcessorList([SuppressEos(memory_monitor)]),
                    **cache_kwargs,
                )
        current = memory_probe()
        _max_memory(memory, current)
        generated_tokens = int(output.sequences.shape[-1]) - prompt_width
        return {
            "status": (
                "complete"
                if generated_tokens == MAX_NEW_TOKENS
                and memory.get("minimum_global_free_mib", 0) >= MINIMUM_GLOBAL_FREE_MIB
                else "incomplete"
            ),
            "example_id": example.example_id,
            "prompt_tokens": prompt_width,
            "requested_new_tokens": MAX_NEW_TOKENS,
            "generated_tokens": generated_tokens,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_mib": int(torch.cuda.max_memory_allocated(0) / (1024**2)),
            "peak_reserved_mib": int(torch.cuda.max_memory_reserved(0) / (1024**2)),
            "cache_implementation": cache_implementation,
            "global_memory": current,
            "minimum_global_free_mib": memory.get("minimum_global_free_mib"),
        }
    except BaseException as exc:
        if not is_cuda_oom(exc):
            raise
        torch.cuda.empty_cache()
        current = memory_probe()
        _max_memory(memory, current)
        return {"status": "oom", "example_id": example.example_id,
                "prompt_tokens": prompt_width, "requested_new_tokens": MAX_NEW_TOKENS,
                "cache_implementation": cache_implementation,
                "elapsed_seconds": time.perf_counter() - started,
                "error": str(exc)[:500], "global_memory": current,
                "minimum_global_free_mib": memory.get("minimum_global_free_mib")}


def _default_qualification_dir() -> Path:
    root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / "TeacherReliability" / "qualifications" / f"rtx4060-{timestamp}-{uuid.uuid4().hex[:6]}"


def _run_qualification_cli(args, output_dir: Path) -> int:
    from .archive_import import import_a100_run
    from .local_run import MODEL_OUTPUT_VOCAB_SIZE
    from .run import _load_cached_examples, _read_import_manifest
    from .selection import get_profile
    from .teacher import MODEL_REPOSITORY, load_teacher

    archive_manifest = _read_import_manifest(args.import_run_zip.resolve(strict=True))
    revision = archive_manifest["model_revision"]
    revisions = archive_manifest["dataset_revisions"]
    examples, revisions = _load_cached_examples(get_profile("full"), args.seed, revisions)
    imported = import_a100_run(
        args.import_run_zip, examples=examples, profile=get_profile("full"),
        seed=args.seed, model_revision=revision, dataset_revisions=revisions,
    )
    try:
        archived_ids = {row["example_id"] for row in imported.rows}
        remaining_counts = remaining_rows_by_dataset(examples, imported.rows)
        source_archive_sha256 = imported.metadata["archive_sha256"]
    finally:
        imported.close()
    del imported
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local RTX 4060 qualification")
    torch.cuda.set_per_process_memory_fraction(0.85, 0)
    tokenizer, model = load_teacher(revision, repository=MODEL_REPOSITORY, local_files_only=True)
    if int(model.config.vocab_size) != MODEL_OUTPUT_VOCAB_SIZE:
        raise RuntimeError("qualification model vocabulary is not 152064")
    result = _run_qualification_body(
        output_dir, examples, tokenizer, model, base_seed=args.seed,
        archived_ids=archived_ids, dataset_revisions=revisions,
        model_revision=revision, source_archive_sha256=source_archive_sha256,
        remaining_rows_by_dataset=remaining_counts,
        progress_callback=lambda event: print(json.dumps(event), flush=True),
        configurations=args.configuration,
    )
    print(json.dumps({"output_dir": str(output_dir.resolve()),
                      "recommended_configuration": result.get("recommended_configuration"),
                      "dataset_weighted_eta_hours": result.get("dataset_weighted_eta_hours"),
                      "status": result["status"]}, ensure_ascii=False))
    return 0 if result["status"] == "complete" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--import-run-zip", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--configuration", action="append",
        choices=[f"g{greedy}-s{sample}" for greedy, sample in qualification_matrix()],
        help="run selected profiles first; repeat to select more; omit for all 12",
    )
    args = parser.parse_args(argv)
    output_dir = (args.output_dir or _default_qualification_dir()).resolve()
    try:
        with run_lock(output_dir):
            return _run_qualification_cli(args, output_dir)
    except Exception as exc:
        print(f"Local qualification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
