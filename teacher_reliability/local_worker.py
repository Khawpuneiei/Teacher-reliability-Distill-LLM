"""Persistent, single-GPU Transformers worker used by the Windows coordinator."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from dataclasses import asdict
from typing import Any, Callable

from .hybrid import CudaOutOfMemory, InsufficientHostMemory, WorkerStartupOutOfMemory
from .hybrid_workers import gpu_memory_snapshot
from .teacher import MODEL_REPOSITORY, generate_greedy_batch, generate_sample_batch, load_teacher


OUTPUT_VOCAB_SIZE = 152_064
MINIMUM_GLOBAL_FREE_MIB = 1024
MINIMUM_OFFLOADED_FREE_MIB = 512


def is_cuda_oom(error: BaseException) -> bool:
    """Return true only for a confirmed CUDA allocator failure."""
    if isinstance(error, CudaOutOfMemory):
        return True
    try:
        import torch

        if isinstance(error, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    text = str(error).lower()
    return "cuda out of memory" in text or "cuda error: out of memory" in text


def _memory_snapshot() -> dict[str, Any]:
    import psutil
    import torch

    snapshot = gpu_memory_snapshot()
    snapshot["allocated_mib"] = int(torch.cuda.memory_allocated(0) / (1024**2))
    snapshot["reserved_mib"] = int(torch.cuda.memory_reserved(0) / (1024**2))
    snapshot["available_ram_mib"] = int(psutil.virtual_memory().available / (1024**2))
    return snapshot


def _empty_cuda_cache() -> None:
    import torch

    torch.cuda.empty_cache()


def _release_model_load_cache(
    torch_module: Any,
    memory_probe: Callable[[], dict[str, Any]] = _memory_snapshot,
) -> dict[str, Any]:
    """Release unused allocator blocks after loading before enforcing headroom."""
    torch_module.cuda.empty_cache()
    return memory_probe()


def _offloaded_cache_bytes(config: Any, batch_size: int, context_limit: int = 4096) -> int:
    layers = getattr(config, "num_hidden_layers", None)
    kv_heads = getattr(config, "num_key_value_heads", None) or getattr(config, "num_attention_heads", None)
    head_size = getattr(config, "head_dim", None)
    if head_size is None:
        hidden = getattr(config, "hidden_size", None)
        heads = getattr(config, "num_attention_heads", None)
        head_size = hidden // heads if hidden and heads else None
    if not all(isinstance(value, int) and value > 0 for value in (layers, kv_heads, head_size, batch_size)):
        raise ValueError("model config does not expose dimensions needed to estimate the KV cache")
    # K and V, fp16 cache precision.
    return context_limit * batch_size * layers * kv_heads * head_size * 2 * 2


def _check_offloaded_cache_ram(config: Any, batch_size: int, available_ram_bytes: int) -> int:
    required = _offloaded_cache_bytes(config, batch_size) + 2 * (1024**3)
    if available_ram_bytes < required:
        raise InsufficientHostMemory(
            f"offloaded KV cache needs {required} bytes including a 2 GiB reserve; "
            f"only {available_ram_bytes} bytes are available"
        )
    return required


class LocalWorkerService:
    """Validates one batch, dispatches to the resident model and streams rows."""

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        *,
        memory_probe: Callable[[], dict[str, Any]] = _memory_snapshot,
        greedy_generator: Callable[..., list[Any]] = generate_greedy_batch,
        sample_generator: Callable[..., list[Any]] = generate_sample_batch,
        available_ram_bytes: Callable[[], int] | None = None,
        cache_releaser: Callable[[], None] = _empty_cuda_cache,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.memory_probe = memory_probe
        self.cache_releaser = cache_releaser
        self.greedy_generator = greedy_generator
        self.sample_generator = sample_generator
        if available_ram_bytes is None:
            def current_ram():
                import psutil
                return int(psutil.virtual_memory().available)
            available_ram_bytes = current_ram
        self.available_ram_bytes = available_ram_bytes
        if int(getattr(getattr(model, "config", None), "vocab_size", 0)) != OUTPUT_VOCAB_SIZE:
            raise ValueError("loaded model output vocabulary is not 152064")

    def handle(
        self,
        payload: dict[str, Any],
        *,
        on_result: Callable[[str, dict[str, Any]], None] | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(payload, dict) or payload.get("op") != "generate":
            raise ValueError("worker op must be 'generate'")
        stage = payload.get("stage")
        if stage not in {"greedy", "sample"}:
            raise ValueError("worker stage must be greedy or sample")
        requests = payload.get("requests")
        if not isinstance(requests, list) or not requests:
            raise ValueError("worker requests must be a non-empty list")
        ids = [row.get("request_id") if isinstance(row, dict) else None for row in requests]
        if any(not isinstance(item, str) or not item for item in ids):
            raise ValueError("worker request IDs must be non-empty strings")
        if len(set(ids)) != len(ids):
            raise ValueError("worker request IDs contain duplicates")
        questions = [row.get("question") for row in requests]
        if any(not isinstance(item, str) or not item.strip() for item in questions):
            raise ValueError("worker questions must be non-empty strings")
        limits = {row.get("max_new_tokens") for row in requests}
        if len(limits) != 1 or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in limits):
            raise ValueError("worker requests must use the same output limit")
        max_new_tokens = limits.pop()
        cache_implementation = payload.get("cache_implementation")
        if cache_implementation not in (None, "offloaded"):
            raise ValueError("unsupported cache implementation")

        self.cache_releaser()
        memory = self.memory_probe()
        minimum_free_mib = (
            MINIMUM_OFFLOADED_FREE_MIB
            if cache_implementation == "offloaded"
            else MINIMUM_GLOBAL_FREE_MIB
        )
        free_mib = int(memory.get("free_mib", 0))
        if free_mib < minimum_free_mib:
            raise CudaOutOfMemory(
                f"global GPU free memory is {free_mib} MiB, below the "
                f"{minimum_free_mib} MiB minimum for cache={cache_implementation or 'default'}"
            )
        if cache_implementation == "offloaded":
            _check_offloaded_cache_ram(
                self.model.config, len(requests), self.available_ram_bytes()
            )

        results: dict[str, dict[str, Any]] = {}

        def emit(index: int, result: Any) -> None:
            request_id = ids[index]
            value = asdict(result)
            if stage == "greedy":
                row = {"request_id": request_id, "greedy": value}
            else:
                row = {"request_id": request_id, "sample": value}
            if request_id in results:
                raise ValueError(f"worker emitted duplicate result for {request_id}")
            results[request_id] = row
            if on_result is not None:
                on_result(request_id, row)

        def progress(event: dict[str, Any]) -> None:
            if on_progress is None:
                return
            event = dict(event)
            index = event.pop("request_index", None)
            event["active_request"] = ids[index] if isinstance(index, int) and 0 <= index < len(ids) else ids[0]
            event["memory"] = self.memory_probe()
            event["last_progress_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            on_progress(event)

        if stage == "greedy":
            generation_kwargs = {
                "cache_implementation": cache_implementation,
                "on_progress": progress,
            }
            parameters = inspect.signature(self.greedy_generator).parameters.values()
            if any(
                parameter.name == "on_result" or parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            ):
                generation_kwargs["on_result"] = lambda index, result: emit(index, result)
            generated = self.greedy_generator(
                self.tokenizer, self.model, questions, max_new_tokens, **generation_kwargs,
            )
            if len(generated) != len(requests):
                raise RuntimeError("greedy result count differs from request count")
            for index, result in enumerate(generated):
                if ids[index] not in results:
                    emit(index, result)
        else:
            seeds = [row.get("seed") for row in requests]
            if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in seeds):
                raise ValueError("sample requests require non-negative integer seeds")

            def streamed(index: int, result: Any) -> None:
                emit(index, result)

            generated = self.sample_generator(
                self.tokenizer, self.model, questions, max_new_tokens, seeds,
                on_result=streamed, cache_implementation=cache_implementation,
                on_progress=progress,
            )
            if len(generated) != len(requests):
                raise RuntimeError("sample result count differs from request count")
            for index, result in enumerate(generated):
                if ids[index] not in results:
                    emit(index, result)
        return results


def _emit(stream, value: dict[str, Any]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    stream.flush()


def _configure_worker_stdio(input_stream=None, output_stream=None, error_stream=None) -> None:
    for stream in (input_stream or sys.stdin, output_stream or sys.stdout, error_stream or sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


def serve(service: LocalWorkerService, *, input_stream=None, output_stream=None) -> None:
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    for line in input_stream:
        try:
            payload = json.loads(line)

            def result(request_id, value):
                _emit(output_stream, {"ok": True, "result": value})

            def progress(event):
                _emit(output_stream, {"ok": True, "progress": event})

            service.handle(payload, on_result=result, on_progress=progress)
            _emit(output_stream, {"ok": True, "done": True})
        except BaseException as exc:
            if is_cuda_oom(exc):
                _emit(output_stream, {"ok": False, "error_type": "cuda_oom", "message": str(exc)[:1000]})
            else:
                _emit(output_stream, {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:1000],
                })


def main(argv: list[str] | None = None) -> int:
    _configure_worker_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--memory-fraction", type=float, default=0.85)
    args = parser.parse_args(argv)
    try:
        if not 0 < args.memory_fraction <= 1:
            raise ValueError("memory fraction must be between 0 and 1")
        memory = gpu_memory_snapshot()
        if memory["free_mib"] < MINIMUM_GLOBAL_FREE_MIB:
            raise WorkerStartupOutOfMemory("less than 1 GiB global GPU memory is free before model load")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the RTX 4060 local backend")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=0)
        tokenizer, model = load_teacher(
            args.model_revision, repository=MODEL_REPOSITORY, local_files_only=True
        )
        memory_after_load = _release_model_load_cache(torch)
        if memory_after_load.get("free_mib", 0) < MINIMUM_GLOBAL_FREE_MIB:
            raise WorkerStartupOutOfMemory(
                f"global GPU free memory after model load is below {MINIMUM_GLOBAL_FREE_MIB} MiB"
            )
        vocab_size = int(getattr(model.config, "vocab_size", 0))
        if vocab_size != OUTPUT_VOCAB_SIZE:
            raise RuntimeError(f"unexpected model output vocabulary size: {vocab_size}")
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError("the local worker requires the pinned fast tokenizer")
        runtime = {
            "model_repository": MODEL_REPOSITORY,
            "model_revision": args.model_revision,
            "vocab_size": vocab_size,
            "gpu_name": torch.cuda.get_device_name(0),
            "total_gpu_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "allocator_fraction": args.memory_fraction,
            "minimum_global_free_mib": MINIMUM_GLOBAL_FREE_MIB,
            "minimum_offloaded_cache_free_mib": MINIMUM_OFFLOADED_FREE_MIB,
            "memory_after_load": memory_after_load,
            "quantization": model._teacher_reliability_load_info["quantization"],
            "model_load": model._teacher_reliability_load_info,
        }
        _emit(sys.stdout, {"ready": True, "runtime": runtime})
        serve(LocalWorkerService(tokenizer, model))
        return 0
    except BaseException as exc:
        if is_cuda_oom(exc):
            _emit(sys.stdout, {"ready": False, "error_type": "cuda_oom", "message": str(exc)[:1000]})
        else:
            _emit(sys.stdout, {
                "ready": False,
                "error_type": type(exc).__name__,
                "message": str(exc)[:1000],
            })
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
