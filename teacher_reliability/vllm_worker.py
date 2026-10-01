"""Line-delimited JSON subprocess for isolated vLLM sampled generation."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable, TextIO

from .model_artifact import validate_artifact
from .teacher import classify_termination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.memory_utilization <= 1:
        parser.error("--memory-utilization must be greater than zero and at most one")
    for name in ("max_num_seqs", "max_num_batched_tokens", "max_model_len"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def validate_request_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("op") != "generate":
        raise ValueError("request op must be 'generate'")
    requests = payload.get("requests")
    if not isinstance(requests, list):
        raise ValueError("requests must be a list")

    seen_ids: set[str] = set()
    validated = []
    for index, row in enumerate(requests):
        if not isinstance(row, dict):
            raise ValueError(f"requests[{index}] must be an object")
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(f"requests[{index}].request_id must be non-empty text")
        if request_id in seen_ids:
            raise ValueError(f"request_id must be unique within the batch: {request_id}")
        seen_ids.add(request_id)

        prompt_ids = row.get("prompt_token_ids")
        if not isinstance(prompt_ids, list) or not prompt_ids:
            raise ValueError(f"requests[{index}].prompt_token_ids must be a non-empty list")
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in prompt_ids
        ):
            raise ValueError(f"requests[{index}].prompt_token_ids must contain non-negative integers")
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"requests[{index}].seed must be a non-negative integer")
        max_new_tokens = row.get("max_new_tokens")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 1
        ):
            raise ValueError(f"requests[{index}].max_new_tokens must be a positive integer")
        validated.append(
            {
                "request_id": request_id,
                "prompt_token_ids": list(prompt_ids),
                "seed": seed,
                "max_new_tokens": max_new_tokens,
            }
        )
    return validated


def resolve_eos_token_ids(tokenizer: Any, model_config: Any | None = None) -> list[int]:
    candidates = []
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is not None:
        candidates.append(getattr(hf_config, "eos_token_id", None))
    if model_config is not None:
        candidates.append(getattr(model_config, "eos_token_id", None))
    candidates.append(getattr(tokenizer, "eos_token_id", None))

    resolved: list[int] = []
    for candidate in candidates:
        items = candidate if isinstance(candidate, (list, tuple, set)) else [candidate]
        for token_id in items:
            if token_id is None:
                continue
            if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                raise ValueError("EOS token IDs must be non-negative integers")
            if token_id not in resolved:
                resolved.append(token_id)
    if not resolved:
        raise ValueError("model config and tokenizer do not provide an EOS token ID")
    return resolved


def _make_sampling_params(
    request: dict[str, Any], eos_ids: list[int], sampling_params_type: Any
) -> Any:
    return sampling_params_type(
        temperature=0.7,
        top_k=50,
        top_p=1.0,
        max_tokens=request["max_new_tokens"],
        stop_token_ids=eos_ids,
        seed=request["seed"],
    )


def _make_result(output: Any, request: dict[str, Any], eos_ids: list[int]) -> dict[str, Any]:
    candidates = getattr(output, "outputs", None)
    if not candidates:
        raise RuntimeError(f"vLLM returned no generation for request {request['request_id']}")
    candidate = candidates[0]
    token_ids = [int(token_id) for token_id in candidate.token_ids]
    termination = classify_termination(token_ids, eos_ids, request["max_new_tokens"])
    if not termination.eos_generated and getattr(output, "stop_reason", None) in eos_ids:
        eos_generated, was_truncated, reason = True, False, "eos"
    elif not termination.eos_generated and getattr(candidate, "finish_reason", None) == "length":
        eos_generated, was_truncated, reason = False, True, "max_new_tokens"
    else:
        eos_generated = termination.eos_generated
        was_truncated = termination.was_truncated
        reason = termination.reason
    text = getattr(candidate, "text", "")
    return {
        "request_id": request["request_id"],
        "token_ids": token_ids,
        "text": text if isinstance(text, str) else "",
        "eos_generated": eos_generated,
        "was_truncated": was_truncated,
        "reason": reason,
    }


def generate_requests(
    llm: Any,
    tokenizer: Any,
    requests: list[dict[str, Any]],
    *,
    sampling_params_type: Any | None = None,
    model_config: Any | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    if not requests:
        return []
    if sampling_params_type is None:
        from vllm import SamplingParams

        sampling_params_type = SamplingParams
    eos_ids = resolve_eos_token_ids(tokenizer, model_config)
    engine = llm.llm_engine
    results: list[dict[str, Any] | None] = [None] * len(requests)
    by_engine_id: dict[str, tuple[int, dict[str, Any]]] = {}

    for index, request in enumerate(requests):
        params = _make_sampling_params(request, eos_ids, sampling_params_type)
        engine_id = engine.add_request(
            request["request_id"],
            {"prompt_token_ids": request["prompt_token_ids"]},
            params,
        )
        if not isinstance(engine_id, str) or not engine_id:
            raise RuntimeError("vLLM did not return an engine request ID")
        if engine_id in by_engine_id:
            raise RuntimeError(f"vLLM returned a duplicate engine request ID: {engine_id}")
        by_engine_id[engine_id] = (index, request)

    completed: set[str] = set()
    while engine.has_unfinished_requests():
        for output in engine.step():
            engine_id = getattr(output, "request_id", None)
            if engine_id not in by_engine_id:
                raise RuntimeError(f"vLLM returned an unknown engine request ID: {engine_id}")
            if not getattr(output, "finished", True):
                continue
            if engine_id in completed:
                raise RuntimeError(f"vLLM returned a duplicate final output: {engine_id}")
            index, request = by_engine_id[engine_id]
            result = _make_result(output, request, eos_ids)
            results[index] = result
            if on_result is not None:
                on_result(result)
            completed.add(engine_id)

    if completed != set(by_engine_id):
        missing = sorted(set(by_engine_id) - completed)
        raise RuntimeError(f"vLLM stopped before completing requests: {missing}")
    return [result for result in results if result is not None]


def _exception_response(error: BaseException) -> dict[str, str | bool]:
    message = str(error) or type(error).__name__
    kind = type(error).__name__.lower()
    text = message.lower()
    module = type(error).__module__.lower()
    is_cuda_oom = "cudaoutofmemory" in kind or (
        "outofmemory" in kind and module.startswith("torch.cuda")
    ) or (
        "cuda" in text and ("out of memory" in text or "oom" in text)
    ) or ("free memory on device cuda:" in text and "desired gpu memory utilization" in text)
    return {
        "ok": False,
        "error_type": "cuda_oom" if is_cuda_oom else "other",
        "message": message,
    }


def process_line(line: str, worker: Any) -> str:
    try:
        requests = validate_request_payload(json.loads(line))
        response = {"ok": True, "results": worker.generate(requests)}
    except Exception as error:
        response = _exception_response(error)
    return json.dumps(response, ensure_ascii=False, separators=(",", ":"))


class VllmSamplingWorker:
    def __init__(self, llm: Any, tokenizer: Any, model_config: Any, runtime: dict[str, Any]):
        self.llm = llm
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.runtime = runtime

    def generate(
        self, requests: list[dict[str, Any]], *,
        on_result: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        with redirect_stdout(sys.stderr):
            return generate_requests(
                self.llm,
                self.tokenizer,
                requests,
                model_config=self.model_config,
                on_result=on_result,
            )


def _model_config_for(llm: Any) -> Any:
    return getattr(getattr(llm, "llm_engine", None), "model_config", None)


def create_worker(args: argparse.Namespace) -> VllmSamplingWorker:
    model_dir = Path(args.model_dir).resolve()
    artifact = validate_artifact(model_dir)
    with redirect_stdout(sys.stderr):
        import vllm
        from vllm import LLM

        llm = LLM(
            model=str(model_dir),
            tokenizer=str(model_dir),
            tensor_parallel_size=1,
            gpu_memory_utilization=args.memory_utilization,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_model_len=args.max_model_len,
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            dtype="float16",
            enforce_eager=args.enforce_eager,
            trust_remote_code=False,
        )
        tokenizer = llm.renderer.tokenizer

    runtime = {
        "engine": "vllm",
        "version": str(getattr(vllm, "__version__", "unknown")),
        "model_dir": str(model_dir),
        "model_revision": artifact["model_revision"],
        "aggregate_sha256": artifact["aggregate_sha256"],
        "quantization": artifact["quantization"],
        "single_gpu": True,
        "memory_utilization": args.memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_model_len": args.max_model_len,
        "enforce_eager": args.enforce_eager,
    }
    return VllmSamplingWorker(llm, tokenizer, _model_config_for(llm), runtime)


def serve(input_stream: TextIO, output_stream: TextIO, worker: VllmSamplingWorker) -> None:
    def write_event(event):
        output_stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        output_stream.flush()

    for line in input_stream:
        try:
            requests = validate_request_payload(json.loads(line))
            worker.generate(requests, on_result=lambda result: write_event({"ok": True, "result": result}))
            write_event({"ok": True, "done": True})
        except Exception as error:
            write_event(_exception_response(error))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        worker = create_worker(args)
    except Exception as error:
        response = _exception_response(error)
        if response["error_type"] == "cuda_oom":
            protocol_out = sys.stdout
            protocol_out.write(
                json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            protocol_out.flush()
        else:
            print(f"vLLM worker startup failed: {error}", file=sys.stderr)
        return 2

    protocol_out = sys.stdout
    protocol_out.write(
        json.dumps({"ready": True, "runtime": worker.runtime}, ensure_ascii=False) + "\n"
    )
    protocol_out.flush()
    serve(sys.stdin, protocol_out, worker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
