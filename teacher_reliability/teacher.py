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
SAMPLE_TOP_K = 50


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


def _left_pad_prompt_rows(prompt_rows: list[dict[str, Any]], tokenizer: Any, torch: Any):
    if any(set(row) != set(prompt_rows[0]) for row in prompt_rows[1:]):
        raise RuntimeError("chat template returned inconsistent fields for the batch")
    lengths = [int(row["input_ids"].shape[-1]) for row in prompt_rows]
    width = max(lengths)
    pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise RuntimeError("the tokenizer needs an eos_token_id for left-padded generation")
    pad_token_id = int(pad_token_id)

    inputs: dict[str, Any] = {}
    for key in prompt_rows[0]:
        values = [row[key] for row in prompt_rows]
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise RuntimeError(f"chat template field {key!r} is not a tensor")
        is_sequence = key in {"input_ids", "attention_mask"} or all(
            value.ndim >= 2 and int(value.shape[-1]) == length
            for value, length in zip(values, lengths)
        )
        if is_sequence:
            batched_values = []
            for value, length in zip(values, lengths):
                padding_width = width - length
                if padding_width:
                    shape = list(value.shape)
                    shape[-1] = padding_width
                    fill_value = pad_token_id if key == "input_ids" else 0
                    prefix = torch.full(
                        shape, fill_value, dtype=value.dtype, device=value.device
                    )
                    value = torch.cat((prefix, value), dim=-1)
                batched_values.append(value)
            inputs[key] = torch.cat(batched_values, dim=0)
        else:
            if any(value.shape != values[0].shape for value in values[1:]):
                raise RuntimeError(f"chat template field {key!r} cannot be batched")
            inputs[key] = torch.cat(values, dim=0)

    if "attention_mask" not in inputs:
        masks = []
        for row, length in zip(prompt_rows, lengths):
            mask = torch.ones_like(row["input_ids"])
            padding_width = width - length
            if padding_width:
                prefix = torch.zeros(
                    (*mask.shape[:-1], padding_width), dtype=mask.dtype, device=mask.device
                )
                mask = torch.cat((prefix, mask), dim=-1)
            masks.append(mask)
        inputs["attention_mask"] = torch.cat(masks, dim=0)
    return inputs, width, pad_token_id


def generate_greedy_batch(
    tokenizer: Any,
    model: Any,
    questions: list[str],
    max_new_tokens: int,
    *,
    cache_implementation: str | None = None,
    on_progress: Any = None,
    on_result: Any = None,
) -> list[GreedyGeneration]:
    """Greedily decode a batch while retaining only token-level score summaries."""
    import torch
    import torch.nn.functional as functional
    import time
    from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or max_new_tokens < 1
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    if not isinstance(questions, list):
        raise ValueError("questions must be a list of non-empty text")
    if not questions:
        return []

    device = model.get_input_embeddings().weight.device
    prompt_rows = [_encode_prompt(tokenizer, question, device) for question in questions]
    inputs, prompt_width, pad_token_id = _left_pad_prompt_rows(
        prompt_rows, tokenizer, torch
    )

    configured_eos = getattr(model.generation_config, "eos_token_id", None)
    if configured_eos is None:
        configured_eos = tokenizer.eos_token_id
    eos_ids = (
        {int(configured_eos)}
        if isinstance(configured_eos, int)
        else {int(token_id) for token_id in (configured_eos or [])}
    )
    special_ids = set(getattr(tokenizer, "all_special_ids", []))

    def build_generation(row: int, token_ids: list[int]) -> GreedyGeneration:
        if len(token_ids) != len(capture.entropies[row]) or len(token_ids) != len(
            capture.logprobs[row]
        ):
            raise RuntimeError(
                f"generated-token/score count mismatch for batch row {row}: "
                f"{len(token_ids)} tokens, {len(capture.entropies[row])} score rows"
            )
        alignment = align_generated_tokens(token_ids, tokenizer)
        termination = classify_termination(token_ids, configured_eos, max_new_tokens)
        return GreedyGeneration(
            text=alignment.text,
            token_ids=list(token_ids),
            token_entropies_nats=list(capture.entropies[row]),
            token_logprobs=list(capture.logprobs[row]),
            token_char_spans=alignment.character_spans,
            content_token_mask=[token_id not in special_ids for token_id in token_ids],
            token_alignment_status=alignment.status,
            eos_generated=termination.eos_generated,
            was_truncated=termination.was_truncated,
            termination_reason=termination.reason,
        )

    class CaptureTokenScores(LogitsProcessor):
        def __init__(self, batch_size: int):
            self.active_rows = [True] * batch_size
            self.token_ids = [[] for _ in range(batch_size)]
            self.entropies = [[] for _ in range(batch_size)]
            self.logprobs = [[] for _ in range(batch_size)]
            self.results: dict[int, GreedyGeneration] = {}
            self.last_progress_at = time.monotonic()

        def complete(self, row: int) -> None:
            generation = build_generation(row, self.token_ids[row])
            self.results[row] = generation
            if on_result is not None:
                on_result(row, generation)

        def __call__(self, input_ids, scores):
            log_probs = functional.log_softmax(scores.float(), dim=-1)
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            selected = scores.argmax(dim=-1)
            for row, active in enumerate(self.active_rows):
                if active:
                    token_id = int(selected[row].item())
                    self.token_ids[row].append(token_id)
                    self.entropies[row].append(float(entropy[row].item()))
                    self.logprobs[row].append(float(log_probs[row, token_id].item()))
                    if token_id in eos_ids:
                        self.active_rows[row] = False
                        self.complete(row)
            now = time.monotonic()
            if on_progress is not None and now - self.last_progress_at >= 30:
                self.last_progress_at = now
                on_progress({
                    "stage": "greedy",
                    "request_index": next(
                        (row for row, active in enumerate(self.active_rows) if active), 0
                    ),
                    "generated_tokens": sum(map(len, self.entropies)),
                })
            return scores

    capture = CaptureTokenScores(len(questions))
    with torch.inference_mode():
        cache_kwargs = {"cache_implementation": cache_implementation} if cache_implementation else {}
        output = model.generate(
            **inputs,
            **GREEDY_DECODING,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=False,
            logits_processor=LogitsProcessorList([capture]),
            use_cache=True,
            pad_token_id=pad_token_id,
            **cache_kwargs,
        )

    if int(output.sequences.shape[0]) != len(questions):
        raise RuntimeError("generated sequence count does not match the prompt batch")
    generations = []
    for row in range(len(questions)):
        token_ids = output.sequences[row, prompt_width:].detach().cpu().tolist()
        for index, token_id in enumerate(token_ids):
            if int(token_id) in eos_ids:
                if all(int(value) == pad_token_id for value in token_ids[index + 1 :]):
                    token_ids = token_ids[: index + 1]
                break
        token_ids = [int(token_id) for token_id in token_ids]
        if token_ids != capture.token_ids[row]:
            raise RuntimeError(
                f"captured token IDs differ from the generated sequence for batch row {row}"
            )
        generation = capture.results.get(row)
        if generation is None:
            generation = build_generation(row, token_ids)
            capture.results[row] = generation
            if on_result is not None:
                on_result(row, generation)
        generations.append(generation)
    return generations

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


def generate_sample_batch(
    tokenizer: Any,
    model: Any,
    questions: list[str],
    max_new_tokens: int,
    seeds: list[int],
    *,
    on_result: Any = None,
    cache_implementation: str | None = None,
    on_progress: Any = None,
) -> list[SampleGeneration]:
    """Sample a prompt batch with independent, request-stable RNG streams."""
    import torch
    import time
    from transformers.generation.logits_process import (
        LogitsProcessor,
        LogitsProcessorList,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
    )

    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if not isinstance(questions, list):
        raise ValueError("questions must be a list of non-empty text")
    if not isinstance(seeds, list) or len(seeds) != len(questions):
        raise ValueError("seeds must have one entry per question")
    if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative integers")
    if not questions:
        return []

    device = model.get_input_embeddings().weight.device
    prompt_rows = [_encode_prompt(tokenizer, question, device) for question in questions]
    inputs, prompt_width, pad_token_id = _left_pad_prompt_rows(prompt_rows, tokenizer, torch)
    configured_eos = getattr(model.generation_config, "eos_token_id", None)
    if configured_eos is None:
        configured_eos = tokenizer.eos_token_id
    eos_ids = (
        {int(configured_eos)}
        if isinstance(configured_eos, int)
        else {int(token_id) for token_id in (configured_eos or [])}
    )
    generated_tokens: list[list[int]] = [[] for _ in questions]
    generated_results: list[SampleGeneration | None] = [None] * len(questions)
    last_progress_at = [time.monotonic()]

    def finish_row(row: int) -> SampleGeneration:
        token_ids = generated_tokens[row]
        termination = classify_termination(token_ids, configured_eos, max_new_tokens)
        text = tokenizer.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        result = SampleGeneration(
            text=text,
            token_ids=list(token_ids),
            eos_generated=termination.eos_generated,
            was_truncated=termination.was_truncated,
            reason=termination.reason,
        )
        generated_results[row] = result
        if on_result is not None:
            on_result(row, result)
        return result

    class SeededSamplingProcessor(LogitsProcessor):
        def __init__(self) -> None:
            self.generators = [
                torch.Generator(device=device).manual_seed(seed) for seed in seeds
            ]
            self.finished = [False] * len(seeds)
            # HF appends built-in sampling warpers after custom processors.
            # Apply them here so each multinomial draw sees the transformed scores.
            self.sampling_warpers = LogitsProcessorList(
                [
                    TemperatureLogitsWarper(SAMPLE_TEMPERATURE),
                    TopKLogitsWarper(top_k=SAMPLE_TOP_K, min_tokens_to_keep=1),
                ]
            )

        def __call__(self, input_ids, scores):
            sampling_scores = self.sampling_warpers(input_ids, scores)
            selected_ids: list[int] = []
            for row, generator in enumerate(self.generators):
                if self.finished[row]:
                    selected_ids.append(pad_token_id)
                    continue
                probabilities = torch.softmax(sampling_scores[row], dim=-1)
                selected = int(
                    torch.multinomial(probabilities, 1, generator=generator).item()
                )
                selected_ids.append(selected)
                generated_tokens[row].append(selected)
                if selected in eos_ids:
                    self.finished[row] = True
                    finish_row(row)
            now = time.monotonic()
            if on_progress is not None and now - last_progress_at[0] >= 30:
                last_progress_at[0] = now
                on_progress({
                    "stage": "sampling",
                    "request_index": next(
                        (row for row, finished in enumerate(self.finished) if not finished), 0
                    ),
                    "generated_tokens": sum(map(len, generated_tokens)),
                })
            forced_scores = torch.full_like(scores, float("-inf"))
            forced_scores.scatter_(
                1,
                torch.tensor(selected_ids, dtype=torch.long, device=scores.device)[:, None],
                0.0,
            )
            return forced_scores

    with torch.inference_mode():
        cache_kwargs = {"cache_implementation": cache_implementation} if cache_implementation else {}
        output = model.generate(
            **inputs,
            do_sample=True,
            num_beams=1,
            # Avoid applying temperature and top-k again after the custom draw.
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=False,
            logits_processor=LogitsProcessorList([SeededSamplingProcessor()]),
            use_cache=True,
            pad_token_id=pad_token_id,
            **cache_kwargs,
        )

    if int(output.sequences.shape[0]) != len(questions):
        raise RuntimeError("generated sequence count does not match the sampling batch")
    results: list[SampleGeneration] = []
    for row in range(len(questions)):
        token_ids = [
            int(token_id)
            for token_id in output.sequences[row, prompt_width:].detach().cpu().tolist()
        ]
        for index, token_id in enumerate(token_ids):
            if token_id in eos_ids:
                token_ids = token_ids[: index + 1]
                break
        if generated_tokens[row] != token_ids:
            raise RuntimeError(f"sampled token history differs from generated row {row}")
        result = generated_results[row] or finish_row(row)
        results.append(result)
    return results
