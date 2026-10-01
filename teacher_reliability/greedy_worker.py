"""Isolated Transformers worker for batched NF4 greedy generation."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path


def handle_generate_request(message, tokenizer, model, *, batcher=None):
    from .teacher import generate_greedy_batch

    if not isinstance(message, dict) or message.get("op") != "generate":
        raise ValueError("request op must be generate")
    requests = message.get("requests")
    if not isinstance(requests, list):
        raise ValueError("requests must be a list")
    ids = [row.get("request_id") for row in requests if isinstance(row, dict)]
    if len(ids) != len(requests) or any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("request_id must be unique non-empty text")
    caps = {row.get("max_new_tokens") for row in requests}
    if len(caps) != 1 or not all(isinstance(cap, int) and not isinstance(cap, bool) and cap > 0 for cap in caps):
        raise ValueError("max_new_tokens must be the same positive integer for every question")
    questions = [row.get("question") for row in requests]
    if any(not isinstance(question, str) or not question.strip() for question in questions):
        raise ValueError("questions must be non-empty text")
    generate = batcher or generate_greedy_batch
    generations = generate(tokenizer, model, questions, caps.pop())
    if len(generations) != len(requests):
        raise RuntimeError("batch generator returned the wrong number of results")
    return {"ok": True, "results": [
        {"request_id": request_id, "greedy": asdict(generation)}
        for request_id, generation in zip(ids, generations)
    ]}


def _error_response(error):
    message = str(error) or type(error).__name__
    kind = type(error).__name__.lower()
    module = type(error).__module__.lower()
    is_oom = "cudaoutofmemory" in kind or (
        "outofmemory" in kind and module.startswith("torch.cuda")
    ) or ("cuda" in message.lower() and "out of memory" in message.lower())
    return {"ok": False, "error_type": "cuda_oom" if is_oom else "other", "message": message}


def _load_model(model_dir: Path, memory_fraction: float):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .model_artifact import validate_artifact

    artifact = validate_artifact(model_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the NF4 greedy worker")
    torch.cuda.memory.set_per_process_memory_fraction(memory_fraction, device=0)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("greedy confidence requires a fast tokenizer")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16, device_map={"": 0},
        local_files_only=True, low_cpu_mem_usage=True,
    )
    model.eval()
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("local model did not load as 4-bit NF4")
    return tokenizer, model, artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--memory-fraction", type=float, default=0.25)
    args = parser.parse_args(argv)
    if not 0 < args.memory_fraction <= 1:
        parser.error("--memory-fraction must be within (0, 1]")
    try:
        with redirect_stdout(sys.stderr):
            tokenizer, model, artifact = _load_model(args.model_dir, args.memory_fraction)
        print(json.dumps({"ready": True, "runtime": {
            "engine": "transformers-greedy", "model_revision": artifact["model_revision"],
            "artifact_sha256": artifact["aggregate_sha256"],
            "memory_fraction": args.memory_fraction,
        }}), flush=True)
    except Exception as error:
        print(json.dumps(_error_response(error)), flush=True)
        return 1
    for line in sys.stdin:
        try:
            message = json.loads(line)
            with redirect_stdout(sys.stderr):
                response = handle_generate_request(message, tokenizer, model)
        except Exception as error:
            response = _error_response(error)
            if response["error_type"] == "cuda_oom":
                try:
                    import gc
                    import torch

                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        print(json.dumps(response, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
