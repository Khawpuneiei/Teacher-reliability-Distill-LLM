"""CPU coordinator for independently staged hybrid inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from .hybrid import MemoryProfile, StageStore, run_with_oom_backoff, validate_stage_payload
from .hub_data import DATASET_REPOSITORIES, load_examples
from .run import (
    _canonical_digest, _git_commit, _hardware_info, _package_versions,
    _prediction_record, _run_identity, _validate_revisions,
)
from .selection import RunProfile
from .state import atomic_write_json, append_complete_prediction, load_completed_predictions, run_lock
from .teacher import GreedyGeneration, SampleGeneration, build_chat_messages, derive_sample_seed


GREEDY_PROFILES = [MemoryProfile(f"greedy-{size}", size) for size in (8, 4, 2, 1)]
SAMPLE_PROFILES = [
    MemoryProfile("vllm-16", 16, 4096, 0.60),
    MemoryProfile("vllm-8", 8, 2048, 0.50, True),
    MemoryProfile("vllm-4", 4, 1024, 0.40, True),
    MemoryProfile("vllm-2", 2, 1024, 0.40, True),
    MemoryProfile("vllm-1", 1, 1024, 0.40, True),
]
MAX_MODEL_LEN = 4096


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _lockfile_sha(root: Path) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("requirements-vllm.in", "requirements-vllm.lock"):
        path = root / name
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return result


def _stage_to_record(example, stage, seed, tokenizer, model, sample_count):
    greedy = GreedyGeneration(**stage["greedy"])
    generations = []
    seeds = []
    for index in range(sample_count):
        saved = stage["samples"][str(index)]
        expected_seed = derive_sample_seed(seed, example.example_id, index)
        if saved["seed"] != expected_seed:
            raise ValueError("staged sample seed differs from run identity")
        generations.append(SampleGeneration(**saved["generation"]))
        seeds.append(expected_seed)
    return _prediction_record(example, greedy, generations, seeds, tokenizer, model, sample_count)


def run_hybrid(
    run_dir: Path,
    profile: RunProfile,
    seed: int = 42,
    *,
    examples=None,
    model_revision: str | None = None,
    dataset_revisions: dict[str, str] | None = None,
    artifact: dict[str, Any],
    tokenizer: Any,
    greedy_execute: Callable,
    sample_execute: Callable,
    max_runtime_hours: float = 12,
    worker_package_versions: dict[str, str] | None = None,
    worker_runtime: dict[str, Any] | None = None,
    now: Callable[[], datetime] = _utc_now,
    resume: bool = False,
) -> dict[str, Any]:
    """Run one immutable selection, checkpointing each finished component."""
    if max_runtime_hours <= 0:
        raise ValueError("max_runtime_hours must be positive")
    run_dir = Path(run_dir).resolve()
    root = Path(__file__).resolve().parents[1]
    manifest_path = run_dir / "run_manifest.json"
    predictions_path = run_dir / "predictions.jsonl"
    with run_lock(run_dir):
        old_manifest = None
        if manifest_path.exists():
            if not resume:
                raise FileExistsError(f"{run_dir} already contains a run; use --resume")
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            model_revision = model_revision or old_manifest["model_revision"]
            dataset_revisions = dataset_revisions or old_manifest["dataset_revisions"]
        elif predictions_path.exists():
            raise ValueError("prediction checkpoints exist without a run manifest")
        if examples is None:
            examples, dataset_revisions = load_examples(profile, seed, dataset_revisions=dataset_revisions)
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
        if artifact.get("model_revision") != model_revision or not artifact.get("aggregate_sha256"):
            raise ValueError("NF4 artifact identity differs from the model revision")

        versions = _package_versions()
        hardware = _hardware_info()
        identity = _run_identity(root, profile, seed, examples, model_revision, dataset_revisions, versions, hardware)
        identity["backend"] = "vllm-hybrid"
        identity["nf4_artifact_sha256"] = artifact["aggregate_sha256"]
        identity["worker_package_versions"] = worker_package_versions or {}
        identity["vllm_lockfiles_sha256"] = _lockfile_sha(root)
        identity["memory_profiles"] = {
            "greedy": [asdict(item) for item in GREEDY_PROFILES],
            "sampling": [asdict(item) for item in SAMPLE_PROFILES],
            "transformers_allocator_fraction": 0.25,
            "max_model_len": MAX_MODEL_LEN,
        }
        identity["max_runtime_hours"] = max_runtime_hours
        signature = _canonical_digest(identity)
        if old_manifest and (old_manifest.get("run_signature") != signature or old_manifest.get("selected_example_ids") != ids):
            raise ValueError("run configuration differs from existing hybrid manifest")
        completed = load_completed_predictions(predictions_path)
        if set(completed) - set(ids):
            raise ValueError("checkpoint contains IDs outside selection")
        stages = StageStore(run_dir, signature, ids)
        for example in examples:
            validate_stage_payload(
                stages.get(example.example_id), example_id=example.example_id,
                base_seed=seed, sample_count=profile.sample_count,
                max_new_tokens=profile.max_new_tokens, tokenizer=tokenizer,
            )
        failed_ids = {
            example_id for example_id in ids
            if example_id not in completed and stages.get(example_id)["failure"]
        }
        start = now()
        if start.tzinfo is None:
            raise ValueError("now() must return a timezone-aware datetime")
        deadline = datetime.fromisoformat(old_manifest["deadline_utc"]) if old_manifest else start + timedelta(hours=max_runtime_hours)
        manifest = old_manifest or {
            "schema_version": 2,
            "backend": "vllm-hybrid",
            "run_id": run_dir.name,
            "started_at": start.isoformat(),
            "deadline_utc": deadline.isoformat(),
            "run_signature": signature,
            "identity": identity,
            "profile": profile.as_dict(),
            "seed": seed,
            "model_revision": model_revision,
            "dataset_revisions": dataset_revisions,
            "nf4_artifact": artifact,
            "selected_example_ids": ids,
            "selected_example_count": len(ids),
            "scorable_source_count": sum(example.scorable for example in examples),
            "unscorable_source_count": sum(not example.scorable for example in examples),
            "git_commit": _git_commit(root),
            "source_snapshot_sha256": identity["source_snapshot_sha256"],
            "software": versions,
            "worker_software": worker_package_versions or {},
            "worker_runtime": worker_runtime if worker_runtime is not None else {},
            "hardware": hardware,
            "adaptations": [],
        }
        if worker_runtime is not None:
            manifest["worker_runtime"] = worker_runtime
        if old_manifest is not None:
            if manifest.get("last_error"):
                manifest.setdefault("previous_failures", []).append({
                    "failed_at": manifest.get("failed_at"),
                    "error": manifest["last_error"],
                })
            manifest.pop("last_error", None)
            manifest.pop("failed_at", None)
            manifest.pop("completed_at", None)
            manifest["resumed_at"] = start.isoformat()

        def refresh(status: str, *, flush: bool = True) -> None:
            manifest["status"] = status
            manifest["completed_example_count"] = len(completed)
            manifest["failed_example_count"] = len(failed_ids)
            manifest["pending_example_count"] = len(ids) - len(completed) - len(failed_ids)
            if flush:
                atomic_write_json(manifest_path, manifest)

        runtime_signature = _canonical_digest(worker_runtime or {})

        def persist_runtime_change() -> None:
            nonlocal runtime_signature
            latest_signature = _canonical_digest(worker_runtime or {})
            if latest_signature != runtime_signature:
                runtime_signature = latest_signature
                refresh("running")

        refresh("running")
        model = SimpleNamespace(config=SimpleNamespace(vocab_size=getattr(tokenizer, "vocab_size", 0)))

        def adapt(profile: MemoryProfile, error: Exception) -> None:
            manifest["adaptations"].append({"profile": asdict(profile), "reason": str(error)[:500], "at": now().isoformat()})
            refresh("running")

        def fail_request(request, error: Exception) -> None:
            example_id = request["example_id"]
            stages.mark_failed(example_id, str(error))
            failed_ids.add(example_id)
            refresh("running", flush=len(failed_ids) % 25 == 0)

        try:
            greedy_level = 0
            sample_level = 0
            for offset in range(0, len(examples), 8):
                if now() >= deadline:
                    break
                window = [example for example in examples[offset:offset + 8] if example.example_id not in completed and not stages.get(example.example_id)["failure"]]
                if not window:
                    continue
                prompts: dict[str, list[int]] = {}
                for example in window:
                    prompt_ids = tokenizer.apply_chat_template(build_chat_messages(example.question), tokenize=True, add_generation_prompt=True)
                    if len(prompt_ids) + profile.max_new_tokens > MAX_MODEL_LEN:
                        stages.mark_failed(example.example_id, f"prompt plus output exceeds {MAX_MODEL_LEN} token context")
                        failed_ids.add(example.example_id)
                    else:
                        prompts[example.example_id] = list(prompt_ids)
                window = [example for example in window if example.example_id in prompts]
                greedy_requests = [
                    {"request_id": example.example_id, "example_id": example.example_id, "question": example.question, "max_new_tokens": profile.max_new_tokens}
                    for example in window if stages.get(example.example_id)["greedy"] is None
                ]

                def save_greedy(request, generation):
                    if not isinstance(generation, GreedyGeneration):
                        raise TypeError("greedy worker must return GreedyGeneration")
                    stages.checkpoint_greedy(request["example_id"], asdict(generation))
                    persist_runtime_change()

                used_greedy_profile = run_with_oom_backoff(
                    greedy_requests, GREEDY_PROFILES[greedy_level:], greedy_execute,
                    on_complete=save_greedy,
                    on_failure=fail_request,
                    on_adapt=adapt,
                )
                greedy_level = GREEDY_PROFILES.index(used_greedy_profile)
                if now() >= deadline:
                    break
                sample_requests = []
                for example in window:
                    if stages.get(example.example_id)["failure"]:
                        continue
                    saved = stages.get(example.example_id)["samples"]
                    for index in range(profile.sample_count):
                        if str(index) not in saved:
                            sample_requests.append({
                                "request_id": f"{example.example_id}::sample::{index}",
                                "example_id": example.example_id,
                                "sample_index": index,
                                "prompt_token_ids": prompts[example.example_id],
                                "seed": derive_sample_seed(seed, example.example_id, index),
                                "max_new_tokens": profile.max_new_tokens,
                            })

                def save_sample(request, generation):
                    if not isinstance(generation, SampleGeneration):
                        raise TypeError("sample worker must return SampleGeneration")
                    stages.checkpoint_sample(request["example_id"], request["sample_index"], {
                        "seed": request["seed"], "generation": asdict(generation),
                    })
                    persist_runtime_change()

                used_sample_profile = run_with_oom_backoff(
                    sample_requests, SAMPLE_PROFILES[sample_level:], sample_execute,
                    on_complete=save_sample,
                    on_failure=fail_request,
                    on_adapt=adapt,
                    skip_request=lambda request: bool(stages.get(request["example_id"])["failure"]),
                )
                sample_level = SAMPLE_PROFILES.index(used_sample_profile)
                for example in window:
                    stage = stages.get(example.example_id)
                    validate_stage_payload(
                        stage, example_id=example.example_id, base_seed=seed,
                        sample_count=profile.sample_count,
                        max_new_tokens=profile.max_new_tokens, tokenizer=tokenizer,
                    )
                    if stage["failure"] or stage["greedy"] is None or len(stage["samples"]) != profile.sample_count:
                        continue
                    record = _stage_to_record(example, stage, seed, tokenizer, model, profile.sample_count)
                    append_complete_prediction(predictions_path, record)
                    completed[example.example_id] = record
                    refresh("running", flush=len(completed) % 25 == 0)
            failed_count = len(failed_ids)
            status = "complete" if len(completed) == len(ids) else "partial" if now() >= deadline else "incomplete" if failed_count else "partial"
            manifest["completed_at"] = now().isoformat()
            refresh(status)
            return manifest
        except TimeoutError:
            if now() >= deadline:
                manifest["completed_at"] = now().isoformat()
                refresh("partial")
                return manifest
            manifest["last_error"] = {
                "type": "TimeoutError", "message": "worker timed out before the run deadline"
            }
            manifest["failed_at"] = now().isoformat()
            refresh("failed")
            raise
        except BaseException as exc:
            manifest["last_error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
            manifest["failed_at"] = now().isoformat()
            refresh("failed")
            raise
