"""Qwen teacher loading and single-example decoding with token scores."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from .alignment import align_generated_tokens


MODEL_REPOSITORY = "Qwen/Qwen2.5-Math-7B-Instruct"
GREEDY_DECODING = {
    "do_sample": False,
    "num_beams": 1,
}
SAMPLE_TEMPERATURE = 0.7


@dataclass(frozen=True, slots=True)
class GreedyGeneration:
    text: str
    token_ids: list[int]
    token_entropies_nats: list[float]
    token_logprobs: list[float]
    token_char_spans: list[tuple[int, int] | None] | None
    content_token_mask: list[bool]
    token_alignment_status: str = "aligned"
    eos_generated: bool = False
    was_truncated: bool = False
    termination_reason: str = "unknown"


@dataclass(frozen=True, slots=True)
class SampleGeneration:
    text: str
    token_ids: list[int]
    eos_generated: bool
    was_truncated: bool
    reason: str


@dataclass(frozen=True, slots=True)
class GenerationTermination:
    eos_generated: bool
    was_truncated: bool
    reason: str


def classify_termination(
    token_ids: list[int], eos_token_ids: int | list[int] | None, max_new_tokens: int
) -> GenerationTermination:
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    eos_ids = (
        {eos_token_ids}
        if isinstance(eos_token_ids, int)
        else set(eos_token_ids or [])
    )
    eos_generated = bool(token_ids and token_ids[-1] in eos_ids)
    was_truncated = not eos_generated and len(token_ids) >= max_new_tokens
    reason = "eos" if eos_generated else "max_new_tokens" if was_truncated else "other_stop"
    return GenerationTermination(eos_generated, was_truncated, reason)


def build_chat_messages(question: str) -> list[dict[str, str]]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty text")
    return [
        {
            "role": "user",
            "content": (
                "Solve the following math problem carefully. Show your reasoning "
                "and put the final answer in \\boxed{...}.\n\n"
                f"Problem:\n{question.strip()}"
            ),
        }
    ]


def derive_sample_seed(base_seed: int, example_id: str, sample_index: int) -> int:
    if not isinstance(base_seed, int) or isinstance(base_seed, bool) or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    if not isinstance(example_id, str) or not example_id:
        raise ValueError("example_id must be non-empty text")
    if not isinstance(sample_index, int) or isinstance(sample_index, bool) or sample_index < 0:
        raise ValueError("sample_index must be a non-negative integer")
    material = f"{base_seed}\0{example_id}\0{sample_index}".encode("utf-8")
    return int(hashlib.sha256(material).hexdigest()[:16], 16) & ((1 << 63) - 1)


@contextmanager
def isolated_torch_rng(seed: int, device: Any) -> Iterator[None]:
    """Seed one sampling call without changing the caller's Torch RNG state."""
    import torch

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    cuda_devices: list[int] = []
    if getattr(device, "type", None) == "cuda":
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        cuda_devices = [device_index]

    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(seed)
        if cuda_devices:
            torch.cuda.default_generators[cuda_devices[0]].manual_seed(seed)
        yield


def resolve_model_revision(repository: str = MODEL_REPOSITORY) -> str:
    from huggingface_hub import HfApi

    revision = HfApi().model_info(repository).sha
    if not isinstance(revision, str) or len(revision) < 40:
        raise ValueError(f"{repository} did not resolve to a model commit SHA")
    return revision


def load_teacher(
    revision: str,
    repository: str = MODEL_REPOSITORY,
    *,
    local_files_only: bool = False,
) -> tuple[Any, Any]:
    """Load an inference-only 4-bit NF4 model and matching fast tokenizer."""
    import psutil
    import torch
    from accelerate import infer_auto_device_map
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the configured 4-bit teacher run")

    tokenizer = AutoTokenizer.from_pretrained(
        repository,
        revision=revision,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("A fast tokenizer is required for answer-token offsets")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    free_gpu_bytes, total_gpu_bytes = torch.cuda.mem_get_info()
    free_gpu_gib = free_gpu_bytes / (1024**3)
    gpu_limit_gib = max(2, int(free_gpu_gib - 1.0))
    available_ram_gib = psutil.virtual_memory().available / (1024**3)
    cpu_limit_gib = max(4, int(available_ram_gib * 0.75))
    max_memory = {0: f"{gpu_limit_gib}GiB", "cpu": f"{cpu_limit_gib}GiB"}

    model = AutoModelForCausalLM.from_pretrained(
        repository,
        revision=revision,
        quantization_config=quantization,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    model.eval()
    model.config.use_cache = True
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model._teacher_reliability_load_info = {
        "quantization": {
            "load_in_4bit": True,
            "quant_type": "nf4",
            "double_quant": True,
            "compute_dtype": "float16",
        },
        "device_map": getattr(model, "hf_device_map", None),
        "gpu_total_bytes_at_load": int(total_gpu_bytes),
        "gpu_free_bytes_at_load": int(free_gpu_bytes),
        "max_memory": max_memory,
    }
    return tokenizer, model


def _encode_prompt(tokenizer: Any, question: str, device: Any):
    messages = build_chat_messages(question)
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise RuntimeError("the model tokenizer has no chat template")
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    if "input_ids" not in encoded:
        raise RuntimeError("chat template did not return input_ids")
    return {key: value.to(device) for key, value in encoded.items()}


def generate_greedy(
    tokenizer: Any,
    model: Any,
    question: str,
    max_new_tokens: int,
) -> GreedyGeneration:
    import torch
    import torch.nn.functional as functional

    device = model.get_input_embeddings().weight.device
    inputs = _encode_prompt(tokenizer, question, device)
    prompt_length = int(inputs["input_ids"].shape[-1])
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            **GREEDY_DECODING,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    token_ids = output.sequences[0, prompt_length:].detach().cpu().tolist()
    if len(output.scores) != len(token_ids):
        raise RuntimeError(
            f"generated-token/score count mismatch: {len(token_ids)} tokens, "
            f"{len(output.scores)} score rows"
        )

    entropies: list[float] = []
    selected_logprobs: list[float] = []
    for token_id, scores in zip(token_ids, output.scores):
        log_probs = functional.log_softmax(scores[0].float(), dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum()
        entropies.append(float(entropy.item()))
        selected_logprobs.append(float(log_probs[int(token_id)].item()))

    alignment = align_generated_tokens(token_ids, tokenizer)
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    content_mask = [token_id not in special_ids for token_id in token_ids]
    eos_token_ids = getattr(model.generation_config, "eos_token_id", None)
    if eos_token_ids is None:
        eos_token_ids = tokenizer.eos_token_id
    termination = classify_termination(token_ids, eos_token_ids, max_new_tokens)
    return GreedyGeneration(
        text=alignment.text,
        token_ids=[int(token_id) for token_id in token_ids],
        token_entropies_nats=entropies,
        token_logprobs=selected_logprobs,
        token_char_spans=alignment.character_spans,
        content_token_mask=content_mask,
        token_alignment_status=alignment.status,
        eos_generated=termination.eos_generated,
        was_truncated=termination.was_truncated,
        termination_reason=termination.reason,
    )


def generate_sample(
    tokenizer: Any,
    model: Any,
    question: str,
    max_new_tokens: int,
    seed: int,
    temperature: float = SAMPLE_TEMPERATURE,
) -> SampleGeneration:
    import torch

    device = model.get_input_embeddings().weight.device
    inputs = _encode_prompt(tokenizer, question, device)
    prompt_length = int(inputs["input_ids"].shape[-1])
    with isolated_torch_rng(seed, device):
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=True,
                num_beams=1,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    token_ids = output.sequences[0, prompt_length:].detach().cpu().tolist()
    eos_token_ids = getattr(model.generation_config, "eos_token_id", None)
    if eos_token_ids is None:
        eos_token_ids = tokenizer.eos_token_id
    termination = classify_termination(token_ids, eos_token_ids, max_new_tokens)
    text = tokenizer.decode(
        token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return SampleGeneration(
        text=text,
        token_ids=[int(token_id) for token_id in token_ids],
        eos_generated=termination.eos_generated,
        was_truncated=termination.was_truncated,
        reason=termination.reason,
    )
