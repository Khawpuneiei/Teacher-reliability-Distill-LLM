"""Resumable single-GPU RTX 4060 experiment coordinator."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from .answers import extract_final_answer_span
from .hybrid import MemoryProfile, StageStore, run_with_oom_backoff, validate_stage_payload
from .hub_data import DATASET_REPOSITORIES, load_examples
from .run import (
    _canonical_digest,
    _git_commit,
    _hardware_info,
    _package_versions,
    _prediction_record,
    _run_identity,
    _validate_revisions,
)
from .selection import RunProfile
from .state import (
    append_complete_prediction,
    atomic_write_json,
    load_completed_predictions,
    run_lock,
)
from .teacher import (
    GreedyGeneration,
    SampleGeneration,
    build_chat_messages,
    derive_sample_seed,
)


RTX4060_DEVICE = "rtx4060-8gb"
MAX_CONTEXT_TOKENS = 4096
MODEL_OUTPUT_VOCAB_SIZE = 152_064
DEFAULT_GREEDY_PROFILES = [
    MemoryProfile("greedy-1", 1),
    MemoryProfile("greedy-1-offloaded", 1, cache_implementation="offloaded"),
]
DEFAULT_SAMPLE_PROFILES = [
    MemoryProfile("sample-2", 2),
    MemoryProfile("sample-1", 1),
    MemoryProfile("sample-1-offloaded", 1, cache_implementation="offloaded"),
]
WINDOW_SIZE = 8


class StopRequested(Exception):
    """Raised at a generation-batch boundary when the owner requests a pause."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _execution_profile(profile: MemoryProfile) -> dict[str, Any]:
    return asdict(profile)


def _source_identity(example) -> dict[str, Any]:
    return {
        "dataset": example.dataset,
        "config": example.source_config,
        "split": example.split,
        "row_id": example.source_row_id,
        "revision": example.source_revision,
        "gold_status": example.gold_status,
        "subject": example.subject,
        "perturbation_type": example.perturbation_type,
        "seed_question_id": example.seed_question_id,
    }


def _checkpoint_error(example_id: str, detail: str) -> ValueError:
    return ValueError(f"checkpoint {detail} differs for {example_id}")


def _local_record_integrity_path(run_dir: Path, example_id: str) -> Path:
    key = hashlib.sha256(example_id.encode("utf-8")).hexdigest()
    return Path(run_dir) / "stages" / "integrity" / f"{key}.json"


def _write_local_record_integrity(
    run_dir: Path, run_signature: str, record: dict[str, Any]
) -> None:
    atomic_write_json(
        _local_record_integrity_path(run_dir, record["example_id"]),
        {
            "schema_version": 1,
            "run_signature": run_signature,
            "example_id": record["example_id"],
            "record_sha256": _canonical_digest(record),
        },
    )


def _validate_local_record_integrity(
    run_dir: Path, run_signature: str, record: dict[str, Any]
) -> None:
    example_id = record.get("example_id")
    path = _local_record_integrity_path(run_dir, example_id)
    try:
        integrity = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _checkpoint_error(example_id, "integrity receipt is missing or invalid") from exc
    if (
        integrity.get("schema_version") != 1
        or integrity.get("run_signature") != run_signature
        or integrity.get("example_id") != example_id
        or integrity.get("record_sha256") != _canonical_digest(record)
    ):
        raise _checkpoint_error(example_id, "record integrity receipt")


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _validate_local_checkpoint_record(
    record: dict[str, Any], example, seed: int, sample_count: int, max_new_tokens: int
) -> None:
    example_id = example.example_id
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
        raise _checkpoint_error(example_id, "output token limit")
    if (
        record.get("example_id") != example_id
        or record.get("status") != "complete"
        or record.get("question") != example.question
        or record.get("gold_answer") != example.gold_answer
        or record.get("source") != _source_identity(example)
    ):
        raise _checkpoint_error(example_id, "identity or source content")
    if record.get("execution_origin") != "rtx4060-local":
        raise _checkpoint_error(example_id, "execution origin")

    greedy = record.get("greedy")
    greedy_fields = {
        "text", "token_ids", "token_entropies_nats", "token_logprobs",
        "token_char_spans", "content_token_mask", "token_alignment_status",
        "eos_generated", "was_truncated", "termination_reason", "predicted_answer",
        "answer_method", "prediction_parseable", "gold_parseable", "correct",
        "grading_status", "content_token_count", "mean_entropy_nats",
        "entropy_confidence", "sequence_logprob_sum", "sequence_geometric_probability",
        "answer_token_count", "answer_token_geometric_probability", "answer_alignment_status",
    }
    if not isinstance(greedy, dict) or not greedy_fields.issubset(greedy):
        raise _checkpoint_error(example_id, "greedy record structure")

    token_ids = greedy["token_ids"]
    if not isinstance(token_ids, list) or any(
        not isinstance(token, int) or isinstance(token, bool) or token < 0
        for token in token_ids
    ):
        raise _checkpoint_error(example_id, "greedy token IDs")
    token_count = len(token_ids)
    if token_count > max_new_tokens:
        raise _checkpoint_error(example_id, "greedy output limit")
    entropies = greedy["token_entropies_nats"]
    logprobs = greedy["token_logprobs"]
    content_mask = greedy["content_token_mask"]
    if not all(
        isinstance(vector, list) and len(vector) == token_count
        for vector in (entropies, logprobs, content_mask)
    ):
        raise _checkpoint_error(example_id, "greedy score/token alignment")
    if any(not _is_finite_number(value) or value < 0 for value in entropies) or any(
        not _is_finite_number(value) or value > 1e-7 for value in logprobs
    ) or any(not isinstance(value, bool) for value in content_mask):
        raise _checkpoint_error(example_id, "greedy score values")
    spans = greedy["token_char_spans"]
    if spans is not None and (
        not isinstance(spans, list) or len(spans) != token_count
        or any(
            span is not None and (
                not isinstance(span, list) or len(span) != 2
                or any(not isinstance(offset, int) or isinstance(offset, bool) for offset in span)
                or span[0] < 0 or span[1] < span[0]
            )
            for span in spans
        )
    ):
        raise _checkpoint_error(example_id, "greedy character-span alignment")
    if not isinstance(greedy["text"], str) or not all(
        isinstance(greedy[field], bool) for field in ("eos_generated", "was_truncated")
    ) or not all(
        isinstance(greedy[field], str)
        for field in ("token_alignment_status", "termination_reason", "grading_status", "answer_alignment_status")
    ):
        raise _checkpoint_error(example_id, "greedy result fields")
    if any(
        greedy[field] is not None and not isinstance(greedy[field], str)
        for field in ("predicted_answer", "answer_method")
    ) or any(
        greedy[field] is not None and not isinstance(greedy[field], bool)
        for field in ("prediction_parseable", "gold_parseable", "correct")
    ):
        raise _checkpoint_error(example_id, "greedy grading fields")
    for field in ("content_token_count", "answer_token_count"):
        value = greedy[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise _checkpoint_error(example_id, "greedy token counts")
    for field in (
        "mean_entropy_nats", "entropy_confidence", "sequence_logprob_sum",
        "sequence_geometric_probability", "answer_token_geometric_probability",
    ):
        value = greedy[field]
        if value is not None and not _is_finite_number(value):
            raise _checkpoint_error(example_id, "greedy confidence scores")

    requested_samples = record.get("requested_sample_count")
    if (
        not isinstance(requested_samples, int) or isinstance(requested_samples, bool)
        or requested_samples != sample_count
    ):
        raise _checkpoint_error(example_id, "requested sample count")
    samples = record.get("samples")
    if not isinstance(samples, list) or len(samples) != sample_count:
        raise _checkpoint_error(example_id, "sample records")
    seen_indices: set[int] = set()
    sample_fields = {
        "index", "seed", "text", "token_ids", "answer", "answer_method",
        "eos_generated", "was_truncated", "termination_reason",
    }
    for sample in samples:
        if not isinstance(sample, dict) or not sample_fields.issubset(sample):
            raise _checkpoint_error(example_id, "sample record structure")
        index = sample["index"]
        if (
            not isinstance(index, int) or isinstance(index, bool)
            or not 0 <= index < sample_count or index in seen_indices
            or not isinstance(sample["seed"], int) or isinstance(sample["seed"], bool)
            or sample["seed"] != derive_sample_seed(seed, example_id, index)
        ):
            raise _checkpoint_error(example_id, "sample index or seed")
        seen_indices.add(index)
        sample_token_ids = sample["token_ids"]
        if not isinstance(sample["text"], str) or not isinstance(sample_token_ids, list) or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in sample_token_ids
        ):
            raise _checkpoint_error(example_id, "sample generation")
        if len(sample_token_ids) > max_new_tokens:
            raise _checkpoint_error(example_id, "sample output limit")
        if sample["answer"] is not None and not isinstance(sample["answer"], str):
            raise _checkpoint_error(example_id, "sample answer")
        if sample["answer_method"] is not None and not isinstance(sample["answer_method"], str):
            raise _checkpoint_error(example_id, "sample answer method")
        if not isinstance(sample["eos_generated"], bool) or not isinstance(sample["was_truncated"], bool):
            raise _checkpoint_error(example_id, "sample termination flags")
        if not isinstance(sample["termination_reason"], str):
            raise _checkpoint_error(example_id, "sample termination reason")
    if seen_indices != set(range(sample_count)):
        raise _checkpoint_error(example_id, "sample indices")

    consistency = record.get("self_consistency")
    consistency_fields = {
        "sample_count", "sample_answer_parseable_count", "sample_answer_parse_unknown_count",
        "greedy_agreement_count", "greedy_agreement_unknown_count", "greedy_agreement_share",
        "majority_answer", "majority_vote_count", "majority_vote_share", "majority_correct",
        "majority_grading_status", "majority_comparison_unknown_count",
    }
    if not isinstance(consistency, dict) or not consistency_fields.issubset(consistency):
        raise _checkpoint_error(example_id, "self-consistency summary")
    count_fields = consistency_fields - {
        "greedy_agreement_share", "majority_answer", "majority_vote_share",
        "majority_correct", "majority_grading_status", "majority_vote_count",
    }
    if any(
        not isinstance(consistency[field], int) or isinstance(consistency[field], bool)
        or consistency[field] < 0
        for field in count_fields
    ) or consistency["sample_count"] != sample_count:
        raise _checkpoint_error(example_id, "self-consistency counts")
    majority_vote_count = consistency["majority_vote_count"]
    if majority_vote_count is not None and (
        not isinstance(majority_vote_count, int) or isinstance(majority_vote_count, bool)
        or not 1 <= majority_vote_count <= sample_count
    ):
        raise _checkpoint_error(example_id, "self-consistency counts")
    count_limits = {
        "sample_answer_parseable_count": sample_count,
        "sample_answer_parse_unknown_count": sample_count,
        "greedy_agreement_count": sample_count,
        "greedy_agreement_unknown_count": sample_count,
        "majority_comparison_unknown_count": sample_count * (sample_count - 1) // 2,
    }
    if any(consistency[field] > limit for field, limit in count_limits.items()):
        raise _checkpoint_error(example_id, "self-consistency counts")
    if (
        consistency["sample_answer_parseable_count"]
        + consistency["sample_answer_parse_unknown_count"] > sample_count
        or consistency["greedy_agreement_count"]
        + consistency["greedy_agreement_unknown_count"] > sample_count
    ):
        raise _checkpoint_error(example_id, "self-consistency counts")
    majority_unknown = consistency["majority_comparison_unknown_count"] > 0
    if majority_unknown:
        if any(consistency[field] is not None for field in (
            "majority_answer", "majority_vote_count", "majority_vote_share", "majority_correct",
        )) or consistency["majority_grading_status"] != "majority_comparison_incomplete":
            raise _checkpoint_error(example_id, "self-consistency majority comparison state")
    elif (
        majority_vote_count is None
        or consistency["majority_vote_share"] is None
        or consistency["majority_grading_status"] == "majority_comparison_incomplete"
    ):
        raise _checkpoint_error(example_id, "self-consistency majority comparison state")
    for field in ("greedy_agreement_share", "majority_vote_share"):
        value = consistency[field]
        if value is not None and (not _is_finite_number(value) or not 0 <= value <= 1):
            raise _checkpoint_error(example_id, "self-consistency shares")
    if consistency["majority_answer"] is not None and not isinstance(consistency["majority_answer"], str):
        raise _checkpoint_error(example_id, "self-consistency majority answer")
    if consistency["majority_correct"] is not None and not isinstance(consistency["majority_correct"], bool):
        raise _checkpoint_error(example_id, "self-consistency majority grade")
    if not isinstance(consistency["majority_grading_status"], str):
        raise _checkpoint_error(example_id, "self-consistency grading status")


def _validate_local_checkpoint_stage(
    record: dict[str, Any], stage: dict[str, Any], example, seed: int, sample_count: int
) -> None:
    example_id = example.example_id
    staged_greedy = stage.get("greedy")
    generated_fields = {
        "text", "token_ids", "token_entropies_nats", "token_logprobs",
        "token_char_spans", "content_token_mask", "token_alignment_status",
        "eos_generated", "was_truncated", "termination_reason",
    }
    if not isinstance(staged_greedy, dict) or not generated_fields.issubset(staged_greedy):
        raise _checkpoint_error(example_id, "greedy stage structure")
    checkpoint_greedy = record["greedy"]
    if any(checkpoint_greedy[field] != staged_greedy[field] for field in generated_fields):
        raise _checkpoint_error(example_id, "greedy generation differs from its stage")

    staged_samples = stage.get("samples")
    if not isinstance(staged_samples, dict) or set(staged_samples) != {
        str(index) for index in range(sample_count)
    }:
        raise _checkpoint_error(example_id, "sample stages")
    if [sample["index"] for sample in record["samples"]] != list(range(sample_count)):
        raise _checkpoint_error(example_id, "sample order")
    for sample in record["samples"]:
        index = sample["index"]
        staged_sample = staged_samples[str(index)]
        generation = staged_sample.get("generation") if isinstance(staged_sample, dict) else None
        expected_seed = derive_sample_seed(seed, example_id, index)
        if (
            not isinstance(staged_sample, dict)
            or staged_sample.get("seed") != expected_seed
            or sample["seed"] != staged_sample.get("seed")
            or not isinstance(generation, dict)
            or not {"text", "token_ids", "eos_generated", "was_truncated", "reason"}.issubset(generation)
        ):
            raise _checkpoint_error(example_id, "sample stage identity")
        if any(
            sample[output_field] != generation[stage_field]
            for output_field, stage_field in (
                ("text", "text"), ("token_ids", "token_ids"),
                ("eos_generated", "eos_generated"), ("was_truncated", "was_truncated"),
                ("termination_reason", "reason"),
            )
        ):
            raise _checkpoint_error(example_id, "sample generation differs from its stage")

    greedy_answer = extract_final_answer_span(checkpoint_greedy["text"])
    if (
        checkpoint_greedy["predicted_answer"] != (greedy_answer.value if greedy_answer else None)
        or checkpoint_greedy["answer_method"] != (greedy_answer.method if greedy_answer else None)
    ):
        raise _checkpoint_error(example_id, "greedy answer differs from generated text")
    for sample in record["samples"]:
        sample_answer = extract_final_answer_span(sample["text"])
        if (
            sample["answer"] != (sample_answer.value if sample_answer else None)
            or sample["answer_method"] != (sample_answer.method if sample_answer else None)
        ):
            raise _checkpoint_error(example_id, "sample answer differs from generated text")


def _save_progress(
    path: Path,
    manifest: dict[str, Any],
    *,
    active_request: str | None,
    generated_tokens: int,
    memory: dict[str, Any] | None = None,
    last_progress_at: str | None = None,
) -> dict[str, Any]:
    if memory is None and path.is_file():
        try:
            memory = json.loads(path.read_text(encoding="utf-8")).get("memory")
        except (OSError, json.JSONDecodeError):
            memory = None
    try:
        started = datetime.fromisoformat(manifest["started_at"])
        elapsed_hours = max(1e-9, (_utc_now() - started).total_seconds() / 3600)
    except (KeyError, TypeError, ValueError):
        elapsed_hours = None
    local_completed = manifest.get("completed_example_count", 0)
    progress = {
        "run_id": manifest["run_id"],
        "status": manifest.get("status", "running"),
        "completed_rows": manifest.get("completed_example_count", 0),
        "failed_rows": manifest.get("failed_example_count", 0),
        "pending_rows": manifest.get("pending_example_count", 0),
        "active_request": active_request,
        "generated_tokens": generated_tokens,
        "local_completed_rows_per_hour": local_completed / elapsed_hours if elapsed_hours else 0.0,
        "memory": memory or {},
        "last_progress_at": last_progress_at or _utc_now().isoformat(timespec="seconds"),
    }
    atomic_write_json(path, progress)
    return progress


def run_local(
    run_dir: Path,
    profile: RunProfile,
    seed: int = 42,
    *,
    examples=None,
    model_revision: str | None = None,
    dataset_revisions: dict[str, str] | None = None,
    tokenizer: Any,
    model_vocab_size: int = MODEL_OUTPUT_VOCAB_SIZE,
    greedy_execute: Callable,
    sample_execute: Callable,
    max_runtime_hours: float | None = None,
    greedy_profiles: list[MemoryProfile] | None = None,
    sample_profiles: list[MemoryProfile] | None = None,
    package_versions: dict[str, str] | None = None,
    hardware: dict[str, Any] | None = None,
    worker_runtime: dict[str, Any] | None = None,
    now: Callable[[], datetime] = _utc_now,
    resume: bool = False,
    retry_failed: bool = False,
    stop_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run or resume an immutable local selection with per-component checkpoints."""
    if max_runtime_hours is not None and (
        isinstance(max_runtime_hours, bool)
        or not isinstance(max_runtime_hours, (int, float))
        or not math.isfinite(max_runtime_hours)
        or max_runtime_hours <= 0
    ):
        raise ValueError("max_runtime_hours must be a positive number when supplied")
    if model_vocab_size != MODEL_OUTPUT_VOCAB_SIZE:
        raise ValueError("model output vocabulary must be 152064 for Qwen2.5-Math-7B-Instruct")

    run_dir = Path(run_dir).resolve()
    root = Path(__file__).resolve().parents[1]
    manifest_path = run_dir / "run_manifest.json"
    predictions_path = run_dir / "predictions.jsonl"
    progress_path = run_dir / "progress.json"
    greedy_profiles = list(greedy_profiles or DEFAULT_GREEDY_PROFILES)
    sample_profiles = list(sample_profiles or DEFAULT_SAMPLE_PROFILES)
    if not greedy_profiles or not sample_profiles:
        raise ValueError("local memory profiles cannot be empty")
    if any(item.batch_size < 1 for item in greedy_profiles + sample_profiles):
        raise ValueError("local batch sizes must be positive")

    with run_lock(run_dir):
        # Acquiring the lock proves there is no active coordinator, so a prior
        # stop marker belonged to a paused/interrupted invocation.
        (run_dir / "stop.requested").unlink(missing_ok=True)
        old_manifest = None
        if manifest_path.exists():
            if not resume:
                raise FileExistsError(f"{run_dir} already contains a run; use --resume")
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            model_revision = model_revision or old_manifest["model_revision"]
            dataset_revisions = dataset_revisions or old_manifest["dataset_revisions"]
            if max_runtime_hours is None:
                max_runtime_hours = old_manifest["identity"].get("max_runtime_hours")
        elif predictions_path.exists():
            raise ValueError("prediction checkpoints exist without a run manifest")

        if examples is None:
            examples, dataset_revisions = load_examples(
                profile, seed, dataset_revisions=dataset_revisions
            )
        if dataset_revisions is None or model_revision is None:
            raise ValueError("model and dataset revisions are required")
        _validate_revisions(model_revision, dataset_revisions)
        if not examples:
            raise ValueError("selected run profile contains no examples")
        ids = [example.example_id for example in examples]
        if len(set(ids)) != len(ids):
            raise ValueError("selected examples contain duplicate IDs")
        for example in examples:
            repository = DATASET_REPOSITORIES.get(example.dataset)
            if repository is None or example.source_revision != dataset_revisions[repository]:
                raise ValueError(f"{example.example_id} source revision differs from dataset manifest")

        example_by_id = {example.example_id: example for example in examples}

        versions = package_versions or _package_versions()
        runtime_hardware = hardware or _hardware_info()
        identity = _run_identity(
            root, profile, seed, examples, model_revision, dataset_revisions,
            versions, runtime_hardware,
        )
        identity.update({
            "backend": "transformers-local",
            "device_preset": RTX4060_DEVICE,
            "model_output_vocab_size": MODEL_OUTPUT_VOCAB_SIZE,
            "max_runtime_hours": max_runtime_hours,
            "greedy_memory_profiles": [_execution_profile(row) for row in greedy_profiles],
            "sampling_memory_profiles": [_execution_profile(row) for row in sample_profiles],
            "torch_allocator_fraction": 0.85,
            "minimum_global_gpu_free_mib": 1024,
            "minimum_offloaded_cache_gpu_free_mib": 512,
            "maximum_context_tokens": MAX_CONTEXT_TOKENS,
            "window_size": WINDOW_SIZE,
            # Kept with empty values so manifests keep the committed mini run's
            # schema; this project imports no external rows and has no qualification.
            "import": {
                "archive_sha256": None,
                "source_run_id": None,
                "source_run_signature": None,
                "imported_ids_sha256": _canonical_digest([]),
            },
            "qualification": None,
        })
        signature = _canonical_digest(identity)
        if old_manifest and (
            old_manifest.get("run_signature") != signature
            or old_manifest.get("selected_example_ids") != ids
        ):
            raise ValueError("run configuration differs from the existing local manifest")

        completed = load_completed_predictions(predictions_path)
        if set(completed) - set(ids):
            raise ValueError("checkpoint contains IDs outside the local selection")
        stages = StageStore(run_dir, signature, ids)
        for example in examples:
            validate_stage_payload(
                stages.get(example.example_id), example_id=example.example_id,
                base_seed=seed, sample_count=profile.sample_count,
                max_new_tokens=profile.max_new_tokens, tokenizer=tokenizer,
            )
        for example_id, row in completed.items():
            example = example_by_id[example_id]
            if (
                row.get("example_id") != example_id
                or row.get("status") != "complete"
                or row.get("question") != example.question
                or row.get("gold_answer") != example.gold_answer
                or row.get("source") != _source_identity(example)
            ):
                raise _checkpoint_error(example_id, "identity or source content")
            _validate_local_record_integrity(run_dir, signature, row)
            _validate_local_checkpoint_record(
                row, example, seed, profile.sample_count, profile.max_new_tokens
            )
            _validate_local_checkpoint_stage(
                row, stages.get(example_id), example, seed, profile.sample_count
            )

        start = now()
        if start.tzinfo is None:
            raise ValueError("now() must return a timezone-aware datetime")
        if old_manifest is None:
            deadline = start + timedelta(hours=max_runtime_hours) if max_runtime_hours is not None else None
        else:
            deadline = (
                datetime.fromisoformat(old_manifest["deadline_utc"])
                if old_manifest.get("deadline_utc") else None
            )
        manifest = old_manifest or {
            "schema_version": 3,
            "backend": "transformers-local",
            "run_id": run_dir.name,
            "started_at": start.isoformat(),
            "deadline_utc": deadline.isoformat() if deadline else None,
            "run_signature": signature,
            "identity": identity,
            "qualification": None,
            "estimated_remaining_hours": None,
            "profile": profile.as_dict(),
            "seed": seed,
            "model_repository": "Qwen/Qwen2.5-Math-7B-Instruct",
            "model_revision": model_revision,
            "tokenizer_repository": "Qwen/Qwen2.5-Math-7B-Instruct",
            "tokenizer_revision": model_revision,
            "dataset_revisions": dataset_revisions,
            "selected_example_ids": ids,
            "selected_example_count": len(ids),
            "scorable_source_count": sum(example.scorable for example in examples),
            "unscorable_source_count": sum(not example.scorable for example in examples),
            "git_commit": _git_commit(root),
            "source_snapshot_sha256": identity["source_snapshot_sha256"],
            "software": versions,
            "hardware": runtime_hardware,
            "execution_origins": {
                "rtx4060-local": {"gpu_name": runtime_hardware.get("gpu_name", "RTX 4060 Laptop GPU")},
            },
            "imported_example_ids": [],
            "imported_example_count": 0,
            "study_scope": "synthetic-or-subset",
            "expected_remaining_rows_by_dataset": None,
            "import_provenance": {},
            "worker_runtime": worker_runtime if worker_runtime is not None else {},
            "adaptations": [],
            "active_settings": {
                "greedy_profile_index": 0,
                "sampling_profile_index": 0,
                "greedy_batch_size": greedy_profiles[0].batch_size,
                "sampling_batch_size": sample_profiles[0].batch_size,
                "greedy_cache_implementation": greedy_profiles[0].cache_implementation,
                "sampling_cache_implementation": sample_profiles[0].cache_implementation,
            },
        }
        active_settings = manifest.setdefault("active_settings", {})

        def active_cache_implementation(kind: str, profiles: list[MemoryProfile]):
            index = active_settings.get(f"{kind}_profile_index", 0)
            if isinstance(index, int) and 0 <= index < len(profiles):
                return profiles[index].cache_implementation
            return None

        active_settings.setdefault(
            "greedy_cache_implementation",
            active_cache_implementation("greedy", greedy_profiles),
        )
        active_settings.setdefault(
            "sampling_cache_implementation",
            active_cache_implementation("sampling", sample_profiles),
        )
        # The former generic field mixed two independent worker stages and
        # could retain stale cache provenance after one stage adapted.
        active_settings.pop("cache_implementation", None)
        if old_manifest is not None:
            manifest.pop("completed_at", None)
            manifest.pop("last_error", None)
            manifest.pop("failed_at", None)
            manifest["resumed_at"] = start.isoformat()
            if worker_runtime:
                manifest["worker_runtime"] = worker_runtime

        failed_ids = {
            example_id for example_id in ids
            if example_id not in completed and stages.get(example_id)["failure"]
        }
        if retry_failed:
            for example_id in sorted(failed_ids):
                stages.clear_failure(example_id)
            failed_ids.clear()

        def refresh(status: str, *, flush: bool = True) -> None:
            manifest["status"] = status
            if status == "running":
                manifest["coordinator_pid"] = os.getpid()
            else:
                manifest.pop("coordinator_pid", None)
            manifest["completed_example_count"] = len(completed)
            manifest["imported_example_count"] = 0
            manifest["failed_example_count"] = len(failed_ids)
            manifest["pending_example_count"] = len(ids) - len(completed) - len(failed_ids)
            if flush:
                atomic_write_json(manifest_path, manifest)

        active_request: str | None = None
        generated_tokens = 0

        def emit_progress(event: dict[str, Any] | None = None) -> None:
            nonlocal active_request, generated_tokens
            event = event or {}
            active_request = event.get("active_request", active_request)
            generated_tokens = int(event.get("generated_tokens", generated_tokens))
            if event.get("worker_runtime"):
                manifest["worker_runtime"] = event["worker_runtime"]
            progress = _save_progress(
                progress_path, manifest, active_request=active_request,
                generated_tokens=generated_tokens, memory=event.get("memory"),
                last_progress_at=event.get("last_progress_at"),
            )
            if progress_callback is not None:
                progress_callback(progress)

        def stop_or_expired() -> bool:
            if stop_check is not None and stop_check():
                return True
            return deadline is not None and now() >= deadline

        def on_adapt(kind: str):
            def adapt(new_profile: MemoryProfile, error: Exception) -> None:
                manifest["adaptations"].append({
                    "stage": kind, "profile": asdict(new_profile),
                    "reason": str(error)[:500], "at": now().isoformat(),
                })
                key = "greedy_profile_index" if kind == "greedy" else "sampling_profile_index"
                rows = greedy_profiles if kind == "greedy" else sample_profiles
                manifest["active_settings"][key] = rows.index(new_profile)
                manifest["active_settings"][
                    "greedy_batch_size" if kind == "greedy" else "sampling_batch_size"
                ] = new_profile.batch_size
                manifest["active_settings"][f"{kind}_cache_implementation"] = (
                    new_profile.cache_implementation
                )
                refresh("running")
            return adapt

        refresh("running")
        emit_progress({"active_request": None, "generated_tokens": 0})

        model = SimpleNamespace(config=SimpleNamespace(vocab_size=model_vocab_size))

        def fail_request(request: dict[str, Any], error: Exception) -> None:
            example_id = request["example_id"]
            worker_error_type = getattr(error, "worker_error_type", None)
            error_type = (
                worker_error_type
                if isinstance(worker_error_type, str) and worker_error_type
                else type(error).__name__
            )
            if stages.get(example_id)["failure"] is None:
                stages.mark_failed(example_id, f"{error_type}: {error}")
                failed_ids.add(example_id)
            refresh("running")
            emit_progress({"active_request": request["request_id"]})

        def check_before_batch(_batch, _memory_profile):
            if stop_or_expired():
                raise StopRequested("stop requested or runtime deadline reached")

        def worker_progress(batch, event):
            if stop_or_expired():
                raise StopRequested("stop requested or runtime deadline reached during generation")
            request_index = event.get("request_index")
            request_id = (
                batch[request_index]["request_id"]
                if isinstance(request_index, int) and 0 <= request_index < len(batch)
                else batch[0]["request_id"] if batch else None
            )
            emit_progress({
                "active_request": request_id,
                "generated_tokens": event.get("generated_tokens", 0),
                "memory": event.get("memory"),
                "last_progress_at": event.get("last_progress_at"),
            })

        def greedy_call(batch, memory_profile, *, on_result, on_progress):
            check_before_batch(batch, memory_profile)
            parameters = inspect.signature(greedy_execute).parameters
            callbacks = {}
            if "on_result" in parameters:
                callbacks["on_result"] = on_result
            if "on_progress" in parameters:
                callbacks["on_progress"] = on_progress
            return greedy_execute(batch, memory_profile, **callbacks)

        def sample_call(batch, memory_profile, *, on_result, on_progress):
            check_before_batch(batch, memory_profile)
            parameters = inspect.signature(sample_execute).parameters
            callbacks = {}
            if "on_result" in parameters:
                callbacks["on_result"] = on_result
            if "on_progress" in parameters:
                callbacks["on_progress"] = on_progress
            return sample_execute(batch, memory_profile, **callbacks)

        was_stopped = False
        try:
            greedy_level = int(manifest["active_settings"].get("greedy_profile_index", 0))
            sample_level = int(manifest["active_settings"].get("sampling_profile_index", 0))
            if not 0 <= greedy_level < len(greedy_profiles) or not 0 <= sample_level < len(sample_profiles):
                raise ValueError("persisted local memory profile is outside configured profiles")

            for offset in range(0, len(examples), WINDOW_SIZE):
                if stop_or_expired():
                    was_stopped = True
                    break
                window = [
                    example for example in examples[offset:offset + WINDOW_SIZE]
                    if example.example_id not in completed
                    and not stages.get(example.example_id)["failure"]
                ]
                if not window:
                    continue
                prompts: dict[str, list[int]] = {}
                for example in window:
                    prompt_ids = tokenizer.apply_chat_template(
                        build_chat_messages(example.question), tokenize=True,
                        add_generation_prompt=True,
                    )
                    if len(prompt_ids) + profile.max_new_tokens > MAX_CONTEXT_TOKENS:
                        fail_request(
                            {"request_id": example.example_id, "example_id": example.example_id},
                            ValueError(f"prompt plus output exceeds {MAX_CONTEXT_TOKENS} token context"),
                        )
                    else:
                        prompts[example.example_id] = list(prompt_ids)
                window = [example for example in window if example.example_id in prompts]

                greedy_requests = [
                    {"request_id": example.example_id, "example_id": example.example_id,
                     "question": example.question, "max_new_tokens": profile.max_new_tokens}
                    for example in window if stages.get(example.example_id)["greedy"] is None
                ]

                def save_greedy(request, result):
                    if not isinstance(result, GreedyGeneration):
                        raise TypeError("greedy worker must return GreedyGeneration")
                    if stages.get(request["example_id"])["greedy"] is None:
                        stages.checkpoint_greedy(request["example_id"], asdict(result))
                    emit_progress({"active_request": request["request_id"], "generated_tokens": len(result.token_ids)})

                used_greedy = run_with_oom_backoff(
                    greedy_requests, greedy_profiles[greedy_level:], greedy_call,
                    on_complete=save_greedy, on_failure=fail_request,
                    on_adapt=on_adapt("greedy"),
                    skip_request=lambda row: bool(stages.get(row["example_id"])["failure"]),
                    on_progress=worker_progress,
                )
                greedy_level = greedy_profiles.index(used_greedy)
                manifest["active_settings"]["greedy_profile_index"] = greedy_level
                refresh("running")
                if stop_or_expired():
                    was_stopped = True
                    break

                sample_requests = []
                for example in window:
                    stage = stages.get(example.example_id)
                    if stage["failure"]:
                        continue
                    for index in range(profile.sample_count):
                        if str(index) not in stage["samples"]:
                            sample_requests.append({
                                "request_id": f"{example.example_id}::sample::{index}",
                                "example_id": example.example_id,
                                "sample_index": index,
                                "question": example.question,
                                "prompt_token_ids": prompts[example.example_id],
                                "max_new_tokens": profile.max_new_tokens,
                                "seed": derive_sample_seed(seed, example.example_id, index),
                            })

                def save_sample(request, result):
                    if not isinstance(result, SampleGeneration):
                        raise TypeError("sampling worker must return SampleGeneration")
                    stage = stages.get(request["example_id"])
                    key = str(request["sample_index"])
                    if key not in stage["samples"]:
                        stages.checkpoint_sample(
                            request["example_id"], request["sample_index"],
                            {"seed": request["seed"], "generation": asdict(result)},
                        )
                    emit_progress({"active_request": request["request_id"], "generated_tokens": len(result.token_ids)})

                run_with_oom_backoff(
                    sample_requests, sample_profiles[sample_level:], sample_call,
                    on_complete=save_sample, on_failure=fail_request,
                    on_adapt=on_adapt("sampling"),
                    skip_request=lambda row: bool(stages.get(row["example_id"])["failure"]),
                    on_progress=worker_progress,
                )
                sample_level = int(manifest["active_settings"].get("sampling_profile_index", sample_level))
                manifest["active_settings"]["sampling_profile_index"] = sample_level
                refresh("running")

                for example in window:
                    example_id = example.example_id
                    stage = stages.get(example_id)
                    validate_stage_payload(
                        stage, example_id=example_id, base_seed=seed,
                        sample_count=profile.sample_count,
                        max_new_tokens=profile.max_new_tokens, tokenizer=tokenizer,
                    )
                    if stage["failure"] or stage["greedy"] is None:
                        continue
                    if set(stage["samples"]) != {str(index) for index in range(profile.sample_count)}:
                        continue
                    record = _prediction_record(
                        example, GreedyGeneration(**stage["greedy"]),
                        [SampleGeneration(**stage["samples"][str(index)]["generation"])
                         for index in range(profile.sample_count)],
                        [derive_sample_seed(seed, example_id, index)
                         for index in range(profile.sample_count)],
                        tokenizer, model, profile.sample_count,
                    )
                    record["execution_origin"] = "rtx4060-local"
                    _write_local_record_integrity(run_dir, signature, record)
                    append_complete_prediction(predictions_path, record)
                    completed[example_id] = record
                    failed_ids.discard(example_id)
                    refresh("running")
                    emit_progress({"active_request": example_id, "generated_tokens": 0})

            status = (
                "partial"
                if was_stopped or len(completed) < len(ids)
                else "complete"
            )
            refresh(status)
            if status == "complete":
                manifest["completed_at"] = now().isoformat()
            else:
                manifest["paused_at"] = now().isoformat()
            if stop_check is not None and stop_check():
                (run_dir / "stop.requested").unlink(missing_ok=True)
            atomic_write_json(manifest_path, manifest)
            _save_progress(
                progress_path, manifest, active_request=active_request,
                generated_tokens=generated_tokens,
            )
            return manifest
        except (StopRequested, TimeoutError) as exc:
            if isinstance(exc, TimeoutError) and not (deadline is not None and now() >= deadline):
                manifest["status"] = "failed"
                manifest["failed_at"] = now().isoformat()
                manifest["last_error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
                refresh("failed")
                raise
            was_stopped = True
            manifest["pause_reason"] = str(exc)
            manifest["paused_at"] = now().isoformat()
            refresh("partial")
            if stop_check is not None and stop_check():
                (run_dir / "stop.requested").unlink(missing_ok=True)
            _save_progress(progress_path, manifest, active_request=active_request, generated_tokens=generated_tokens)
            return manifest
        except BaseException as exc:
            manifest["failed_at"] = now().isoformat()
            manifest["status"] = "failed"
            manifest["last_error"] = {
                "type": type(exc).__name__,
                "worker_error_type": getattr(exc, "worker_error_type", None),
                "message": str(exc)[:1000],
            }
            refresh("failed")
            raise
