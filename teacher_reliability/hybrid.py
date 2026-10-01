"""OOM adaptation and durable intermediate checkpoints for hybrid inference."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .state import atomic_write_json


class InsufficientHostMemory(RuntimeError):
    """The offloaded KV cache cannot fit in available system RAM."""


class CudaOutOfMemory(RuntimeError):
    """A worker reported a confirmed CUDA allocation failure."""


class WorkerStartupOutOfMemory(CudaOutOfMemory):
    """The model could not be loaded at the requested memory setting."""


class WorkerRestartRequired(RuntimeError):
    """A worker process or pipe failed; unfinished requests can be retried."""

    def __init__(self, message: str, *, completed_results: dict[str, Any] | None = None):
        super().__init__(message)
        self.completed_results = completed_results or {}


@dataclass(frozen=True)
class MemoryProfile:
    name: str
    batch_size: int
    max_num_batched_tokens: int | None = None
    gpu_memory_utilization: float | None = None
    enforce_eager: bool = False
    cache_implementation: str | None = None

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.cache_implementation not in (None, "offloaded"):
            raise ValueError("cache_implementation must be None or 'offloaded'")


def run_with_oom_backoff(
    requests, profiles, execute, *, on_complete, on_failure, on_adapt=None,
    skip_request=None, on_progress=None,
):
    """Finish requests in input order, retaining their IDs and seeds on retry."""
    pending = list(requests)
    if not profiles or any(profile.batch_size < 1 for profile in profiles):
        raise ValueError("memory profiles must have positive batch sizes")
    ids = [request["request_id"] for request in pending]
    if len(ids) != len(set(ids)):
        raise ValueError("request IDs must be unique")
    level = 0
    restart_attempts: dict[str, int] = {}
    while pending:
        if skip_request is not None:
            pending = [request for request in pending if not skip_request(request)]
            if not pending:
                break
        profile = profiles[level]
        batch = pending[:profile.batch_size]
        request_by_id = {request["request_id"]: request for request in batch}
        completed_ids: set[str] = set()

        def receive_result(request_id, result):
            if request_id not in request_by_id:
                raise ValueError("streamed worker result ID is outside the submitted batch")
            if request_id not in completed_ids:
                on_complete(request_by_id[request_id], result)
                completed_ids.add(request_id)

        try:
            parameters = inspect.signature(execute).parameters
            callbacks = {}
            if "on_result" in parameters:
                callbacks["on_result"] = receive_result
            if "on_progress" in parameters:
                callbacks["on_progress"] = (
                    lambda event: on_progress(batch, event)
                    if on_progress is not None else None
                )
            results = execute(batch, profile, **callbacks)
        except WorkerStartupOutOfMemory as exc:
            if level == len(profiles) - 1:
                raise
            level += 1
            if on_adapt:
                on_adapt(profiles[level], exc)
            continue
        except WorkerRestartRequired as exc:
            completed_results = exc.completed_results
            if not isinstance(completed_results, dict) or set(completed_results) - set(request_by_id):
                raise ValueError("partial worker result IDs do not match the submitted batch") from exc
            for request in batch:
                request_id = request["request_id"]
                if request_id in completed_results:
                    receive_result(request_id, completed_results[request_id])
            pending_after_batch = pending[len(batch):]
            retry = []
            for request in batch:
                request_id = request["request_id"]
                if request_id in completed_ids:
                    continue
                attempts = restart_attempts.get(request_id, 0)
                if attempts >= 1:
                    on_failure(request, exc)
                else:
                    restart_attempts[request_id] = attempts + 1
                    retry.append(request)
            pending = retry + pending_after_batch
            continue
        except CudaOutOfMemory as exc:
            completed_results = getattr(exc, "completed_results", {})
            if completed_results:
                batch_ids = {request["request_id"] for request in batch}
                if not isinstance(completed_results, dict) or set(completed_results) - batch_ids:
                    raise ValueError("partial worker result IDs do not match the submitted batch") from exc
                for request in batch:
                    if request["request_id"] in completed_results:
                        receive_result(request["request_id"], completed_results[request["request_id"]])
                completed_ids.update(completed_results)
            if completed_ids:
                pending = [request for request in pending if request["request_id"] not in completed_ids]
                if not pending:
                    break
                batch = [request for request in batch if request["request_id"] not in completed_ids]
            if level < len(profiles) - 1:
                level += 1
                if on_adapt:
                    on_adapt(profiles[level], exc)
                continue
            if len(batch) != 1:
                raise ValueError("minimum OOM profile must have batch size one") from exc
            on_failure(batch[0], exc)
            del pending[:1]
            continue
        except Exception as exc:
            if not (
                isinstance(exc, InsufficientHostMemory)
                or getattr(exc, "worker_error_type", None) == "InsufficientHostMemory"
            ):
                raise
            completed_results = getattr(exc, "completed_results", {})
            if not isinstance(completed_results, dict) or set(completed_results) - set(request_by_id):
                raise ValueError("partial host-memory results do not match the submitted batch") from exc
            for request in batch:
                request_id = request["request_id"]
                if request_id in completed_results:
                    receive_result(request_id, completed_results[request_id])
            for request in batch:
                if request["request_id"] not in completed_ids:
                    on_failure(request, exc)
            pending = pending[len(batch):]
            regular_profiles = [
                index for index, item in enumerate(profiles)
                if item.cache_implementation is None and item.batch_size <= profile.batch_size
            ]
            if regular_profiles:
                fallback_level = regular_profiles[-1]
                if fallback_level != level:
                    level = fallback_level
                    if on_adapt:
                        on_adapt(profiles[level], exc)
            continue
        if not isinstance(results, dict) or set(results) != {r["request_id"] for r in batch}:
            raise ValueError("worker result IDs do not match the submitted batch")
        for request in batch:
            receive_result(request["request_id"], results[request["request_id"]])
        del pending[:len(batch)]
    return profiles[level]


class StageStore:
    """One atomically replaced checkpoint per selected question."""

    def __init__(self, run_dir: Path, run_signature: str, selected_ids: Iterable[str]):
        self.root = Path(run_dir) / "stages"
        self.signature = run_signature
        self.selected_ids = set(selected_ids)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, example_id: str) -> Path:
        if example_id not in self.selected_ids:
            raise ValueError(f"{example_id} is outside the selected examples")
        return self.root / f"{hashlib.sha256(example_id.encode('utf-8')).hexdigest()}.json"

    def get(self, example_id):
        path = self._path(example_id)
        if not path.exists():
            return {"greedy": None, "samples": {}, "failure": None, "failure_history": []}
        stage = json.loads(path.read_text(encoding="utf-8"))
        if stage.get("run_signature") != self.signature or stage.get("example_id") != example_id:
            raise ValueError("stage signature or example identity differs from this run")
        if not isinstance(stage.get("samples"), dict):
            raise ValueError("invalid sample checkpoint")
        stage.setdefault("failure_history", [])
        if not isinstance(stage["failure_history"], list):
            raise ValueError("invalid failure history checkpoint")
        return stage

    def _write(self, example_id: str, stage: dict[str, Any]) -> None:
        atomic_write_json(self._path(example_id), {
            "run_signature": self.signature,
            "example_id": example_id,
            "greedy": stage["greedy"],
            "samples": stage["samples"],
            "failure": stage["failure"],
            "failure_history": stage.get("failure_history", []),
        })

    def checkpoint_greedy(self, example_id, greedy):
        stage = self.get(example_id)
        if stage["greedy"] is not None:
            raise ValueError("greedy result already checkpointed")
        stage["greedy"] = greedy
        self._write(example_id, stage)

    def checkpoint_sample(self, example_id, index, sample):
        stage = self.get(example_id)
        key = str(index)
        if key in stage["samples"]:
            raise ValueError("sample already checkpointed")
        stage["samples"][key] = sample
        self._write(example_id, stage)

    def mark_failed(self, example_id, reason):
        stage = self.get(example_id)
        stage["failure"] = reason
        self._write(example_id, stage)

    def clear_failure(self, example_id):
        stage = self.get(example_id)
        if stage["failure"] is None:
            return False
        stage["failure_history"].append(stage["failure"])
        stage["failure"] = None
        self._write(example_id, stage)
        return True


def validate_stage_payload(
    stage: dict[str, Any],
    *,
    example_id: str,
    base_seed: int,
    sample_count: int,
    max_new_tokens: int,
    tokenizer: Any | None = None,
) -> None:
    """Reject malformed saved generation components before reuse or finalization."""
    from .teacher import derive_sample_seed

    def invalid(detail: str) -> ValueError:
        return ValueError(f"stage {detail} differs for {example_id}")

    if not isinstance(stage, dict) or stage.get("example_id", example_id) != example_id:
        raise invalid("example identity")
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 1:
        raise invalid("sample count")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
        raise invalid("output token limit")

    greedy = stage.get("greedy")
    if greedy is not None:
        fields = {
            "text", "token_ids", "token_entropies_nats", "token_logprobs",
            "token_char_spans", "content_token_mask", "token_alignment_status",
            "eos_generated", "was_truncated", "termination_reason",
        }
        if not isinstance(greedy, dict) or set(greedy) != fields:
            raise invalid("greedy structure")
        tokens = greedy["token_ids"]
        entropies = greedy["token_entropies_nats"]
        logprobs = greedy["token_logprobs"]
        mask = greedy["content_token_mask"]
        if not isinstance(greedy["text"], str) or not isinstance(tokens, list) or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in tokens
        ):
            raise invalid("greedy generated data")
        if len(tokens) > max_new_tokens:
            raise invalid("greedy output token limit")
        if not all(isinstance(vector, list) and len(vector) == len(tokens) for vector in (entropies, logprobs, mask)):
            raise invalid("greedy score/token alignment")
        if any(not _finite(value) or value < 0 for value in entropies) or any(
            not _finite(value) or value > 1e-7 for value in logprobs
        ) or any(not isinstance(value, bool) for value in mask):
            raise invalid("greedy score values")
        spans = greedy["token_char_spans"]
        if spans is not None and (
            not isinstance(spans, list) or len(spans) != len(tokens)
            or any(span is not None and (
                not isinstance(span, list) or len(span) != 2
                or any(not isinstance(offset, int) or isinstance(offset, bool) for offset in span)
                or span[0] < 0 or span[1] < span[0]
            ) for span in spans)
        ):
            raise invalid("greedy text/token alignment")
        if not isinstance(greedy["token_alignment_status"], str) or not all(
            isinstance(greedy[field], bool) for field in ("eos_generated", "was_truncated")
        ) or not isinstance(greedy["termination_reason"], str):
            raise invalid("greedy termination fields")
        if callable(getattr(tokenizer, "decode", None)) and tokenizer.decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ) != greedy["text"]:
            raise invalid("greedy text/token alignment")

    samples = stage.get("samples")
    if not isinstance(samples, dict):
        raise invalid("sample map")
    for key, sample in samples.items():
        try:
            index = int(key)
        except (TypeError, ValueError) as exc:
            raise invalid("sample index") from exc
        if not isinstance(key, str) or str(index) != key or not 0 <= index < sample_count:
            raise invalid("sample index")
        if not isinstance(sample, dict) or set(sample) != {"seed", "generation"}:
            raise invalid("sample structure")
        expected_seed = derive_sample_seed(base_seed, example_id, index)
        if (
            not isinstance(sample["seed"], int) or isinstance(sample["seed"], bool)
            or sample["seed"] != expected_seed
        ):
            raise invalid("sample seed")
        generation = sample["generation"]
        if not isinstance(generation, dict) or set(generation) != {
            "text", "token_ids", "eos_generated", "was_truncated", "reason"
        }:
            raise invalid("sample generation structure")
        tokens = generation["token_ids"]
        if not isinstance(generation["text"], str) or not isinstance(tokens, list) or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in tokens
        ):
            raise invalid("sample generated data")
        if len(tokens) > max_new_tokens:
            raise invalid("sample output token limit")
        if not all(isinstance(generation[field], bool) for field in ("eos_generated", "was_truncated")) or not isinstance(generation["reason"], str):
            raise invalid("sample termination fields")
        if callable(getattr(tokenizer, "decode", None)) and tokenizer.decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ) != generation["text"]:
            raise invalid("sample text/token alignment")


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
