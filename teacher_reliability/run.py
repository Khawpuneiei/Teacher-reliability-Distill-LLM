"""Run a reproducible greedy-confidence and self-consistency evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

from .answers import extract_final_answer_span
from .confidence import aggregate_confidence
from .consistency import summarize_consistency
from .data import Example
from .grading import grade_prediction
from .hub_data import DATASET_REPOSITORIES, load_examples
from .selection import PROFILES, RunProfile, get_profile
from .state import atomic_write_json, append_complete_prediction, load_completed_predictions, run_lock
from .teacher import (
    GREEDY_DECODING,
    MODEL_REPOSITORY,
    SAMPLE_TEMPERATURE,
    GreedyGeneration,
    SampleGeneration,
    build_chat_messages,
    derive_sample_seed,
    generate_greedy,
    generate_sample,
    load_teacher,
    resolve_model_revision,
)


MANIFEST_SCHEMA_VERSION = 1
N_ECE_BINS = 15
FULL_STUDY_ORIGIN_DATASET_COUNTS = {
    "a100-import": {"gsm8k": 110, "math": 0, "gsm_plus": 0},
    "rtx4060-local": {"gsm8k": 1209, "math": 350, "gsm_plus": 10552},
}
FULL_STUDY_SELECTED_DATASET_COUNTS = {
    dataset: sum(origin_counts[dataset] for origin_counts in FULL_STUDY_ORIGIN_DATASET_COUNTS.values())
    for dataset in ("gsm8k", "math", "gsm_plus")
}
FULL_STUDY_SELECTED_ROW_COUNT = sum(FULL_STUDY_SELECTED_DATASET_COUNTS.values())
_PACKAGE_NAMES = (
    "torch",
    "transformers",
    "tokenizers",
    "accelerate",
    "bitsandbytes",
    "psutil",
    "datasets",
    "huggingface-hub",
    "hf_xet",
    "math-verify",
    "sympy",
    "latex2sympy2-extended",
    "antlr4-python3-runtime",
    "numpy",
    "pandas",
    "matplotlib",
)


def _validate_full_study_counts(
    selected_by_dataset: dict[str, int], imported_by_dataset: dict[str, int]
) -> dict[str, int]:
    """Enforce the pinned production selection and A100/RTX execution split."""
    if selected_by_dataset != FULL_STUDY_SELECTED_DATASET_COUNTS:
        raise ValueError(
            "full study requires the pinned 12,221-row dataset selection "
            f"{FULL_STUDY_SELECTED_DATASET_COUNTS}; found {selected_by_dataset}"
        )
    expected_imported = FULL_STUDY_ORIGIN_DATASET_COUNTS["a100-import"]
    if imported_by_dataset != expected_imported:
        raise ValueError(
            "full study requires exactly 110 validated A100 imports, all from GSM8K; "
            f"expected {expected_imported}, found {imported_by_dataset}"
        )
    return dict(FULL_STUDY_ORIGIN_DATASET_COUNTS["rtx4060-local"])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in _PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _source_snapshot(root: Path) -> str:
    digester = hashlib.sha256()
    package_root = root / "teacher_reliability"
    included = sorted(package_root.rglob("*.py"))
    included.extend(
        path for path in (root / "requirements.txt", root / "pyproject.toml") if path.exists()
    )
    for path in sorted(set(included)):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digester.update(len(relative).to_bytes(4, "big"))
        digester.update(relative)
        digester.update(len(content).to_bytes(8, "big"))
        digester.update(content)
    return digester.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _hardware_info() -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        result.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_total_memory_bytes": int(properties.total_memory),
                "gpu_compute_capability": [properties.major, properties.minor],
            }
        )
    return result


def _validate_revisions(model_revision: str, dataset_revisions: dict[str, str]) -> None:
    if not isinstance(model_revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", model_revision):
        raise ValueError("model_revision must be an immutable 40- or 64-character commit SHA")
    missing = set(DATASET_REPOSITORIES.values()) - set(dataset_revisions)
    if missing:
        raise ValueError(f"missing dataset commit revisions: {', '.join(sorted(missing))}")
    for repository in DATASET_REPOSITORIES.values():
        revision = dataset_revisions[repository]
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
            raise ValueError(f"{repository} revision must be an immutable commit SHA")


def _run_identity(
    root: Path,
    profile: RunProfile,
    seed: int,
    examples: Sequence[Example],
    model_revision: str,
    dataset_revisions: dict[str, str],
    package_versions: dict[str, str],
    hardware: dict[str, Any],
) -> dict[str, Any]:
    prompt_template = build_chat_messages("<QUESTION>")
    return {
        "profile": profile.as_dict(),
        "base_seed": seed,
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": model_revision,
            "quantization": {
                "load_in_4bit": True,
                "quant_type": "nf4",
                "double_quant": True,
                "compute_dtype": "float16",
            },
        },
        "datasets": dataset_revisions,
        "selected_ids_sha256": _canonical_digest([example.example_id for example in examples]),
        "selected_examples_sha256": _canonical_digest([asdict(example) for example in examples]),
        "prompt_template_sha256": _canonical_digest(prompt_template),
        "greedy_decoding": GREEDY_DECODING,
        "sample_temperature": SAMPLE_TEMPERATURE,
        "ece_bins": N_ECE_BINS,
        "packages": package_versions,
        "runtime": hardware,
        "source_snapshot_sha256": _source_snapshot(root),
    }


def _prediction_record(
    example: Example,
    greedy: GreedyGeneration,
    sample_generations: list[SampleGeneration],
    sample_seeds: list[int],
    tokenizer: Any,
    model: Any,
    requested_sample_count: int,
) -> dict[str, Any]:
    grade = grade_prediction(greedy.text, example.gold_answer)
    confidence = aggregate_confidence(
        token_entropies=greedy.token_entropies_nats,
        token_logprobs=greedy.token_logprobs,
        token_char_spans=greedy.token_char_spans,
        content_token_mask=greedy.content_token_mask,
        answer_span=grade.prediction_span,
        vocab_size=int(
            getattr(getattr(model, "config", None), "vocab_size", 0)
            or getattr(tokenizer, "vocab_size", 0)
        ),
    )
    sample_texts = [sample.text for sample in sample_generations]
    sample_spans = [extract_final_answer_span(text) for text in sample_texts]
    sample_answers = [span.value if span is not None else None for span in sample_spans]
    consistency = summarize_consistency(
        grade.prediction_span.value if grade.prediction_span else None,
        sample_answers,
        example.gold_answer,
    )
    return {
        "example_id": example.example_id,
        "status": "complete",
        "source": {
            "dataset": example.dataset,
            "config": example.source_config,
            "split": example.split,
            "row_id": example.source_row_id,
            "revision": example.source_revision,
            "gold_status": example.gold_status,
            "subject": example.subject,
            "perturbation_type": example.perturbation_type,
            "seed_question_id": example.seed_question_id,
        },
        "question": example.question,
        "gold_answer": example.gold_answer,
        "greedy": {
            "text": greedy.text,
            "token_ids": greedy.token_ids,
            "token_entropies_nats": greedy.token_entropies_nats,
            "token_logprobs": greedy.token_logprobs,
            "token_char_spans": greedy.token_char_spans,
            "content_token_mask": greedy.content_token_mask,
            "token_alignment_status": greedy.token_alignment_status,
            "eos_generated": greedy.eos_generated,
            "was_truncated": greedy.was_truncated,
            "termination_reason": greedy.termination_reason,
            "predicted_answer": grade.prediction_span.value if grade.prediction_span else None,
            "answer_method": grade.prediction_span.method if grade.prediction_span else None,
            "prediction_parseable": grade.prediction_parseable,
            "gold_parseable": grade.gold_parseable,
            "correct": grade.correct,
            "grading_status": grade.status,
            "content_token_count": confidence.content_token_count,
            "mean_entropy_nats": confidence.mean_entropy_nats,
            "entropy_confidence": confidence.entropy_confidence,
            "sequence_logprob_sum": confidence.sequence_logprob_sum,
            "sequence_geometric_probability": confidence.sequence_geometric_probability,
            "answer_token_count": confidence.answer_token_count,
            "answer_token_geometric_probability": confidence.answer_token_geometric_probability,
            "answer_alignment_status": confidence.answer_alignment_status,
        },
        "samples": [
            {
                "index": index,
                "seed": sample_seeds[index],
                "text": text,
                "token_ids": sample_generations[index].token_ids,
                "answer": sample_answers[index],
                "answer_method": sample_spans[index].method if sample_spans[index] else None,
                "eos_generated": sample_generations[index].eos_generated,
                "was_truncated": sample_generations[index].was_truncated,
                "termination_reason": sample_generations[index].reason,
            }
            for index, text in enumerate(sample_texts)
        ],
        "self_consistency": {
            "sample_count": consistency.sample_count,
            "sample_answer_parseable_count": consistency.sample_answer_parseable_count,
            "sample_answer_parse_unknown_count": consistency.sample_answer_parse_unknown_count,
            "greedy_agreement_count": consistency.greedy_agreement_count,
            "greedy_agreement_unknown_count": consistency.greedy_agreement_unknown_count,
            "greedy_agreement_share": consistency.greedy_agreement_share,
            "majority_answer": consistency.majority_answer,
            "majority_vote_count": consistency.majority_vote_count,
            "majority_vote_share": consistency.majority_vote_share,
            "majority_correct": consistency.majority_correct,
            "majority_grading_status": consistency.majority_grade_status,
            "majority_comparison_unknown_count": consistency.majority_comparison_unknown_count,
        },
        "requested_sample_count": requested_sample_count,
    }


def run_experiment(
    run_dir: Path,
    profile: RunProfile,
    seed: int = 42,
    *,
    examples: list[Example] | None = None,
    model_revision: str | None = None,
    dataset_revisions: dict[str, str] | None = None,
    tokenizer: Any = None,
    model: Any = None,
    greedy_generator: Callable[..., GreedyGeneration] | None = None,
    sample_generator: Callable[..., SampleGeneration] | None = None,
    resume: bool = False,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Execute or resume one immutable run directory, checkpointing each row."""
    run_dir = Path(run_dir).resolve()
    root = Path(__file__).resolve().parents[1]
    predictions_path = run_dir / "predictions.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    if (tokenizer is None) != (model is None):
        raise ValueError("tokenizer and model must either both be supplied or both be omitted")

    with run_lock(run_dir):
        existing_manifest = None
        if manifest_path.exists():
            if not resume:
                raise FileExistsError(
                    f"{run_dir} already contains a run; use --resume to continue it"
                )
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if model_revision is None:
                model_revision = existing_manifest["model_revision"]
            if dataset_revisions is None:
                dataset_revisions = existing_manifest["dataset_revisions"]
            _validate_revisions(model_revision, dataset_revisions)
        elif predictions_path.exists():
            raise ValueError("prediction checkpoints exist without a run manifest")

        if examples is None:
            examples, dataset_revisions = load_examples(
                profile, seed, dataset_revisions=dataset_revisions
            )
        if dataset_revisions is None:
            raise ValueError("dataset_revisions are required with preloaded examples")
        if model_revision is None:
            model_revision = resolve_model_revision()
        _validate_revisions(model_revision, dataset_revisions)
        if not examples:
            raise ValueError("the selected run profile contains no examples")

        selected_ids = [example.example_id for example in examples]
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("selected examples contain duplicate IDs")
        for example in examples:
            repository = DATASET_REPOSITORIES.get(example.dataset)
            if repository is None or example.source_revision != dataset_revisions[repository]:
                raise ValueError(f"{example.example_id} source revision differs from dataset manifest")
        package_versions = _package_versions()
        hardware = _hardware_info()
        identity = _run_identity(
            root,
            profile,
            seed,
            examples,
            model_revision,
            dataset_revisions,
            package_versions,
            hardware,
        )
        signature = _canonical_digest(identity)
        if existing_manifest is not None:
            if existing_manifest.get("run_signature") != signature:
                raise ValueError("run configuration differs from the existing manifest")
            if existing_manifest.get("selected_example_ids") != selected_ids:
                raise ValueError("selected IDs differ from the existing manifest")

        completed = load_completed_predictions(predictions_path)
        unknown_checkpoints = set(completed) - set(selected_ids)
        if unknown_checkpoints:
            raise ValueError(
                "checkpoint contains IDs outside this selection: "
                + ", ".join(sorted(unknown_checkpoints)[:5])
            )
        manifest = existing_manifest or {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": run_dir.name,
            "started_at": _now(),
            "run_signature": signature,
            "identity": identity,
            "profile": profile.as_dict(),
            "seed": seed,
            "model_repository": MODEL_REPOSITORY,
            "model_revision": model_revision,
            "tokenizer_repository": MODEL_REPOSITORY,
            "tokenizer_revision": model_revision,
            "dataset_revisions": dataset_revisions,
            "selected_example_ids": selected_ids,
            "selected_example_count": len(selected_ids),
            "scorable_source_count": sum(example.scorable for example in examples),
            "unscorable_source_count": sum(not example.scorable for example in examples),
            "git_commit": _git_commit(root),
            "source_snapshot_sha256": identity["source_snapshot_sha256"],
            "software": package_versions,
            "hardware": hardware,
        }
        manifest["status"] = "running"
        manifest["completed_example_count"] = len(completed)
        manifest["resumed_at"] = _now() if existing_manifest else None
        manifest["last_error"] = None
        atomic_write_json(manifest_path, manifest)

        try:
            if len(completed) < len(examples):
                if tokenizer is None:
                    tokenizer, model = load_teacher(
                        model_revision,
                        repository=MODEL_REPOSITORY,
                        local_files_only=local_files_only,
                    )
                greedy_fn = greedy_generator or generate_greedy
                sample_fn = sample_generator or generate_sample
                model_load_info = getattr(model, "_teacher_reliability_load_info", None)
                if model_load_info is not None:
                    manifest["model_load"] = model_load_info
                    atomic_write_json(manifest_path, manifest)

                for index, example in enumerate(examples, start=1):
                    if example.example_id in completed:
                        continue
                    greedy = greedy_fn(
                        tokenizer,
                        model,
                        example.question,
                        profile.max_new_tokens,
                    )
                    sample_generations: list[SampleGeneration] = []
                    sample_seeds: list[int] = []
                    for sample_index in range(profile.sample_count):
                        sample_seed = derive_sample_seed(seed, example.example_id, sample_index)
                        sample_generation = sample_fn(
                            tokenizer,
                            model,
                            example.question,
                            profile.max_new_tokens,
                            sample_seed,
                        )
                        if not isinstance(sample_generation, SampleGeneration):
                            raise TypeError("sample generator must return SampleGeneration")
                        sample_seeds.append(sample_seed)
                        sample_generations.append(sample_generation)

                    record = _prediction_record(
                        example,
                        greedy,
                        sample_generations,
                        sample_seeds,
                        tokenizer,
                        model,
                        profile.sample_count,
                    )
                    append_complete_prediction(predictions_path, record)
                    completed[example.example_id] = record
                    if len(completed) % 25 == 0:
                        manifest["completed_example_count"] = len(completed)
                        atomic_write_json(manifest_path, manifest)

            manifest["completed_example_count"] = len(completed)
            manifest["completed_at"] = _now()
            manifest["status"] = "complete"
            atomic_write_json(manifest_path, manifest)
            return manifest
        except BaseException as exc:
            manifest["completed_example_count"] = len(completed)
            manifest["failed_at"] = _now()
            manifest["status"] = "failed"
            manifest["last_error"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:1000],
            }
            atomic_write_json(manifest_path, manifest)
            raise


def _default_run_dir(profile_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("outputs") / f"{profile_name}-{timestamp}-{uuid.uuid4().hex[:6]}"


LOCAL_MODEL_REVISION = "ef9926d75ab1d54532f6a30dd5e760355eb9aa4d"
FULL_LOCAL_STUDY_SEED = 42
FULL_LOCAL_DATASET_REVISIONS = {
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "EleutherAI/hendrycks_math": "21a5633873b6a120296cce3e2df9d5550074f4a3",
    "qintongli/GSM-Plus": "3b708db57b96a16e8e3368ed2956990c0809440e",
}
FULL_LOCAL_A100_ARCHIVE_SHA256 = "7fa37f4a6e083e293e4f29606f7643f7ede940569d531410c47e8162130a4d82"


def _default_local_run_dir(profile_name: str) -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    root = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / "TeacherReliability" / "runs" / f"{profile_name}-{timestamp}-{uuid.uuid4().hex[:6]}"


def _local_status(args) -> int:
    if args.run_dir is None:
        raise ValueError("--status and --stop require --run-dir")
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("backend") != "transformers-local":
        raise ValueError("lifecycle commands only apply to transformers-local runs")
    progress_path = run_dir / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else None
    if args.status:
        fields = (
            "run_id", "backend", "status", "profile", "started_at", "resumed_at",
            "deadline_utc", "selected_example_count", "completed_example_count",
            "imported_example_count", "failed_example_count", "pending_example_count",
            "execution_origins", "active_settings", "qualification",
            "estimated_remaining_hours", "coordinator_pid", "last_error",
        )
        compact_manifest = {key: manifest[key] for key in fields if key in manifest}
        print(json.dumps({"manifest": compact_manifest, "progress": progress}, ensure_ascii=False, indent=2))
        return 0
    if manifest.get("status") != "running" or not manifest.get("coordinator_pid"):
        raise ValueError("--stop requires an active local run")
    atomic_write_json(run_dir / "stop.requested", {
        "requested_at": _now(), "requested_by_pid": os.getpid(),
        "coordinator_pid": manifest["coordinator_pid"],
    })
    print(f"Stop requested for {manifest.get('run_id', run_dir.name)}; checkpoints will be preserved.")
    return 0


def _read_import_manifest(archive_path: Path) -> dict[str, Any]:
    from .archive_import import _member_names

    import zipfile

    with zipfile.ZipFile(archive_path) as archive:
        members = _member_names(archive)
        manifest = json.loads(archive.read(members["run_manifest.json"]))
    if not isinstance(manifest, dict):
        raise ValueError("source run manifest must be a JSON object")
    return manifest


def _local_dataset_revisions(revisions: dict[str, str] | None) -> dict[str, str]:
    """Resolve every missing source revision once before enforcing offline data loads."""
    if revisions is not None:
        return dict(revisions)
    from .hub_data import DATASET_REPOSITORIES, _resolve_dataset_revision

    return {
        repository: _resolve_dataset_revision(repository)
        for repository in DATASET_REPOSITORIES.values()
    }


def _recovery_profiles_from_qualification(configuration: dict[str, Any]):
    """Start at the profile actually qualified, retaining only lower-pressure fallbacks."""
    from .local_benchmark import memory_profiles

    selected_profiles = []
    for stage, key in (("greedy", "actual_greedy_profile"), ("sample", "actual_sample_profile")):
        actual = configuration.get(key)
        if not isinstance(actual, dict) or not isinstance(actual.get("batch_size"), int):
            raise ValueError(f"qualification config omits its actual {stage} profile")
        profiles = memory_profiles(actual["batch_size"], stage)
        matches = [
            index for index, profile in enumerate(profiles)
            if profile.batch_size == actual["batch_size"]
            and profile.cache_implementation == actual.get("cache_implementation")
            and (not actual.get("name") or profile.name == actual["name"])
        ]
        if len(matches) != 1:
            raise ValueError(f"qualification {stage} profile is outside the supported memory sequence")
        selected_profiles.append(profiles[matches[0]:])
    return tuple(selected_profiles)


def _fast_start_qualification_record() -> dict[str, Any]:
    """Record an owner-authorized start using the conservative local defaults."""
    from .local_run import DEFAULT_GREEDY_PROFILES, DEFAULT_SAMPLE_PROFILES

    return {
        "status": "unqualified-fast-start",
        "qualified": False,
        "configuration": "default-greedy-1-sample-2",
        "actual_greedy_profile": asdict(DEFAULT_GREEDY_PROFILES[0]),
        "actual_sample_profile": asdict(DEFAULT_SAMPLE_PROFILES[0]),
        "rows_per_hour": None,
        "rows_per_hour_by_dataset": None,
        "dataset_weighted_eta_hours": None,
        "reason": "Owner-authorized fast start; the full qualification matrix was skipped.",
    }


def _load_local_qualification(
    path: Path,
    *,
    model_revision: str,
    dataset_revisions: dict[str, str],
    seed: int,
    source_archive_sha256: str,
    package_versions: dict[str, str],
    hardware: dict[str, Any],
    expected_remaining_rows_by_dataset: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Verify the measured configuration and bind it to this full local run."""
    from .local_benchmark import is_qualified_configuration, qualification_matrix

    path = Path(path).resolve(strict=True)
    raw_manifest = path.read_bytes()
    try:
        manifest = json.loads(raw_manifest)
    except json.JSONDecodeError as exc:
        raise ValueError("qualification manifest is not valid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        raise ValueError("qualification manifest must be complete")
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("qualification manifest omits its identity")
    if identity.get("source_snapshot_sha256") != _source_snapshot(Path(__file__).resolve().parents[1]):
        raise ValueError("qualification source code snapshot differs from the full-run candidate")
    if identity.get("model_revision") != model_revision:
        raise ValueError("qualification model revision differs from the full run")
    if identity.get("dataset_revisions") != dataset_revisions:
        raise ValueError("qualification dataset revisions differ from the full run")
    if identity.get("seed") != seed:
        raise ValueError("qualification seed differs from the full run")
    if identity.get("source_archive_sha256") != source_archive_sha256:
        raise ValueError("qualification archive differs from the imported A100 source")
    if identity.get("software") != package_versions:
        raise ValueError("qualification package versions differ from the full run")
    measured_hardware = identity.get("hardware")
    if not isinstance(measured_hardware, dict):
        raise ValueError("qualification omits its hardware identity")
    if (
        measured_hardware.get("platform") != hardware.get("platform")
        or measured_hardware.get("gpu_name") != hardware.get("gpu_name")
        or round(int(measured_hardware.get("gpu_total_memory_bytes", 0)) / (1024**2))
        != int(hardware.get("gpu_total_memory_mib", 0))
    ):
        raise ValueError("qualification hardware differs from the current RTX 4060 host")
    signature = _canonical_digest(identity)
    if manifest.get("qualification_signature") != signature:
        raise ValueError("qualification identity signature is invalid")
    name = manifest.get("recommended_configuration")
    allowed_names = {f"g{greedy}-s{sample}" for greedy, sample in qualification_matrix()}
    if name not in allowed_names or manifest.get("settings", {}).get(name, {}).get("status") != "complete":
        raise ValueError("qualification has no completed recommended configuration")
    config_path = path.parent / name / "config.json"
    config_bytes = config_path.read_bytes()
    try:
        config = json.loads(config_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("recommended qualification config is not valid JSON") from exc
    expected_config_signature = _canonical_digest({"qualification": signature, "config": name})
    if config.get("run_signature") != expected_config_signature:
        raise ValueError("recommended qualification config signature is invalid")
    if (
        config.get("status") != "complete"
        or len(config.get("greedy", {})) != 24
        or len(config.get("samples", {})) != 192
        or config.get("failures")
        or not is_qualified_configuration(config)
    ):
        raise ValueError("recommended qualification config is incomplete or unsafe")
    rates = config.get("rows_per_hour_by_dataset")
    remaining = manifest.get("remaining_rows_by_dataset")
    eta = manifest.get("dataset_weighted_eta_hours")
    if not isinstance(rates, dict) or not isinstance(remaining, dict):
        raise ValueError("qualification omits dataset throughput or remaining counts")
    datasets = {"gsm8k", "math", "gsm_plus"}
    if set(rates) != datasets or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or value <= 0
        for value in rates.values()
    ):
        raise ValueError("qualification dataset throughput is incomplete or invalid")
    if set(remaining) != datasets or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in remaining.values()
    ):
        raise ValueError("qualification remaining row counts are incomplete or invalid")
    if expected_remaining_rows_by_dataset is not None and remaining != expected_remaining_rows_by_dataset:
        raise ValueError("qualification remaining row counts differ from the imported full selection")
    if not isinstance(eta, (int, float)) or isinstance(eta, bool) or not math.isfinite(eta) or eta < 0:
        raise ValueError("qualification omits its dataset-weighted ETA")
    from .local_benchmark import estimate_dataset_weighted_eta_hours
    from .local_worker import MINIMUM_GLOBAL_FREE_MIB

    expected_eta = estimate_dataset_weighted_eta_hours(remaining, rates)
    if not math.isclose(float(eta), expected_eta, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("qualification dataset-weighted ETA does not match its rates and counts")
    stress = manifest.get("stress_test")
    if not isinstance(stress, dict) or stress.get("status") != "complete":
        raise ValueError("qualification did not pass the 1,024-token stress check")
    stress_memory = stress.get("global_memory")
    if not isinstance(stress_memory, dict) or int(stress_memory.get("free_mib", 0)) < MINIMUM_GLOBAL_FREE_MIB:
        raise ValueError("qualification 1,024-token stress check did not retain 1 GiB GPU headroom")
    return {
        "manifest_path": str(path),
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "configuration": name,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "qualification_signature": signature,
        "source_snapshot_sha256": identity["source_snapshot_sha256"],
        "actual_greedy_profile": config["actual_greedy_profile"],
        "actual_sample_profile": config["actual_sample_profile"],
        "rows_per_hour": config.get("rows_per_hour"),
        "rows_per_hour_by_dataset": rates,
        "remaining_rows_by_dataset": remaining,
        "dataset_weighted_eta_hours": float(eta),
    }


def _load_cached_examples(profile, seed, revisions):
    from datasets import DownloadConfig, load_dataset

    def cached_loader(*, repository, config, split, revision):
        kwargs = {"split": split, "revision": revision,
                  "download_config": DownloadConfig(local_files_only=True)}
        if config is not None:
            kwargs["name"] = config
        return load_dataset(repository, **kwargs)

    return load_examples(
        profile, seed, dataset_revisions=revisions, dataset_loader=cached_loader
    )


def _execute_local_cli(args) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from .archive_import import import_a100_run
    from .hybrid_workers import ManagedWorker
    from .local_run import (
        DEFAULT_GREEDY_PROFILES,
        DEFAULT_SAMPLE_PROFILES,
        MODEL_OUTPUT_VOCAB_SIZE,
        RTX4060_DEVICE,
        StopRequested,
        run_local,
    )

    if args.device_preset != RTX4060_DEVICE:
        raise ValueError("the local backend supports only --device-preset rtx4060-8gb")
    if args.profile == "full" and not args.import_run_zip:
        raise ValueError("the full local run requires --import-run-zip for the 110 A100 source rows")
    if args.fast_start_unqualified and args.qualification_manifest:
        raise ValueError("--fast-start-unqualified cannot be combined with --qualification-manifest")
    if args.profile == "full" and not args.qualification_manifest and not args.fast_start_unqualified:
        raise ValueError(
            "the full local run requires --qualification-manifest unless "
            "--fast-start-unqualified is explicitly selected"
        )
    if args.profile == "full" and args.seed != FULL_LOCAL_STUDY_SEED:
        raise ValueError("full local study seed must be 42")
    run_dir = args.run_dir.resolve()
    old_manifest_path = run_dir / "run_manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8")) if old_manifest_path.is_file() and args.resume else None
    import_manifest = _read_import_manifest(args.import_run_zip.resolve(strict=True)) if args.import_run_zip else None
    if old_manifest and old_manifest.get("import_provenance", {}).get("archive_sha256") and not args.import_run_zip:
        raise ValueError("resume this imported run with the same --import-run-zip archive")
    model_revision = (
        import_manifest.get("model_revision") if import_manifest else
        old_manifest.get("model_revision") if old_manifest else LOCAL_MODEL_REVISION
    )
    dataset_revisions = (
        import_manifest.get("dataset_revisions") if import_manifest else
        old_manifest.get("dataset_revisions") if old_manifest else None
    )
    dataset_revisions = _local_dataset_revisions(dataset_revisions)
    profile = get_profile(args.profile)
    seed = args.seed if not old_manifest else old_manifest["seed"]
    if old_manifest and args.seed != old_manifest["seed"]:
        raise ValueError("resume seed differs from the local manifest")
    if args.profile == "full":
        if model_revision != LOCAL_MODEL_REVISION:
            raise ValueError("full local study model revision differs from the pinned revision")
        if dataset_revisions != FULL_LOCAL_DATASET_REVISIONS:
            raise ValueError("full local study dataset revisions differ from the pinned revisions")
        if seed != FULL_LOCAL_STUDY_SEED:
            raise ValueError("full local study seed must be 42")
        if old_manifest and (
            old_manifest.get("model_revision") != LOCAL_MODEL_REVISION
            or old_manifest.get("dataset_revisions") != FULL_LOCAL_DATASET_REVISIONS
        ):
            raise ValueError("resume manifest differs from the pinned full-study revisions")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    examples, dataset_revisions = _load_cached_examples(profile, seed, dataset_revisions)
    _validate_revisions(model_revision, dataset_revisions)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPOSITORY, revision=model_revision, use_fast=True, local_files_only=True
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("the local backend requires the pinned fast tokenizer")

    imported = None
    if args.import_run_zip:
        imported = import_a100_run(
            args.import_run_zip, examples=examples, profile=profile, seed=seed,
            model_revision=model_revision, dataset_revisions=dataset_revisions,
            expected_archive_sha256=(
                FULL_LOCAL_A100_ARCHIVE_SHA256 if args.profile == "full" else None
            ),
        )
    stderr_path = run_dir / "local_worker.stderr.log"
    command = [
        sys.executable, "-m", "teacher_reliability.local_worker",
        "--model-revision", model_revision, "--memory-fraction", "0.85",
    ]
    worker = ManagedWorker(lambda _resident: command, stderr_path)
    resident_worker_profile = "single-resident-local-model"
    worker_runtime: dict[str, Any] = {}
    manifest_path = run_dir / "run_manifest.json"
    stop_path = run_dir / "stop.requested"

    def current_deadline():
        if not manifest_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw = manifest.get("deadline_utc")
        return datetime.fromisoformat(raw) if raw else None

    def request(stage, requests, memory_profile, *, on_result, on_progress):
        deadline = current_deadline()
        payload = {
            "op": "generate", "stage": stage,
            "cache_implementation": memory_profile.cache_implementation,
            "requests": requests,
        }

        def result_callback(request_id, row):
            if stage == "greedy":
                generation = GreedyGeneration(**row["greedy"])
            else:
                generation = SampleGeneration(**row["sample"])
            on_result(request_id, generation)

        response = worker.request(
            payload, resident_worker_profile, deadline=deadline,
            on_result=result_callback, on_progress=on_progress,
            unbounded=deadline is None,
        )
        runtime = dict(worker.runtime or {})
        if runtime:
            if runtime.get("model_revision") != model_revision:
                raise ValueError("local worker model revision differs from the selected revision")
            if runtime.get("vocab_size") != MODEL_OUTPUT_VOCAB_SIZE:
                raise ValueError("local worker output vocabulary differs from the confidence protocol")
            worker_runtime.clear()
            worker_runtime.update(runtime)
        rows = response.get("results")
        expected = {item["request_id"] for item in requests}
        mapping = {row.get("request_id"): row for row in rows if isinstance(row, dict)}
        if len(mapping) != len(rows) or set(mapping) != expected:
            raise ValueError("local worker returned incomplete or duplicate request IDs")
        return {
            request_id: GreedyGeneration(**row["greedy"]) if stage == "greedy"
            else SampleGeneration(**row["sample"])
            for request_id, row in mapping.items()
        }

    def greedy_execute(requests, memory_profile, *, on_result, on_progress):
        return request("greedy", requests, memory_profile, on_result=on_result, on_progress=on_progress)

    def sample_execute(requests, memory_profile, *, on_result, on_progress):
        return request("sample", requests, memory_profile, on_result=on_result, on_progress=on_progress)

    def local_hardware():
        import subprocess

        output = subprocess.check_output(
            ["nvidia-smi", "--id=0", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10, stderr=subprocess.PIPE,
        ).strip()
        parts = [part.strip() for part in output.split(",")]
        if len(parts) != 3 or "RTX 4060" not in parts[0]:
            raise RuntimeError(f"device preset expects RTX 4060, found: {output}")
        memory_mib = int(parts[1])
        if not 7000 <= memory_mib <= 9000:
            raise RuntimeError(f"device preset expects an 8 GiB RTX 4060, found {memory_mib} MiB")
        return {"gpu_name": parts[0], "gpu_total_memory_mib": memory_mib,
                "driver_version": parts[2], "platform": platform.platform(),
                "python": platform.python_version(), "cuda_available": True}

    try:
        runtime_hardware = local_hardware()
        package_versions = _package_versions()
        qualification = None
        greedy_profiles = DEFAULT_GREEDY_PROFILES
        sample_profiles = DEFAULT_SAMPLE_PROFILES
        if args.profile == "full":
            if imported is None:
                raise ValueError("a full local run requires the validated A100 import")
            example_by_id = {example.example_id: example for example in examples}
            imported_ids = {row["example_id"] for row in imported.rows}
            imported_counts = {
                dataset: sum(
                    example_by_id[example_id].dataset == dataset
                    for example_id in imported_ids
                )
                for dataset in ("gsm8k", "math", "gsm_plus")
            }
            selected_counts = {
                dataset: sum(example.dataset == dataset for example in examples)
                for dataset in ("gsm8k", "math", "gsm_plus")
            }
            expected_remaining = _validate_full_study_counts(selected_counts, imported_counts)
            if args.fast_start_unqualified:
                qualification = _fast_start_qualification_record()
            else:
                qualification = _load_local_qualification(
                    args.qualification_manifest.resolve(strict=True),
                    model_revision=model_revision,
                    dataset_revisions=dataset_revisions,
                    seed=seed,
                    source_archive_sha256=imported.metadata["archive_sha256"],
                    package_versions=package_versions,
                    hardware=runtime_hardware,
                    expected_remaining_rows_by_dataset=expected_remaining,
                )
                greedy_profiles, sample_profiles = _recovery_profiles_from_qualification(qualification)
        return run_local(
            run_dir, profile, seed=seed, examples=examples,
            model_revision=model_revision, dataset_revisions=dataset_revisions,
            tokenizer=tokenizer, model_vocab_size=MODEL_OUTPUT_VOCAB_SIZE,
            greedy_execute=greedy_execute, sample_execute=sample_execute,
            imported_run=imported, source_archive_path=args.import_run_zip,
            max_runtime_hours=args.max_runtime_hours,
            greedy_profiles=greedy_profiles,
            sample_profiles=sample_profiles,
            package_versions=package_versions, hardware=runtime_hardware,
            qualification=qualification,
            worker_runtime=worker_runtime, resume=args.resume, retry_failed=args.retry_failed,
            stop_check=stop_path.exists,
        )
    finally:
        try:
            worker.close()
        finally:
            if imported is not None:
                imported.close()


def _vllm_package_versions(python_path: Path) -> dict[str, str]:
    """Read package metadata in the separate worker interpreter without importing CUDA."""
    script = (
        "import importlib.metadata as metadata,json; "
        "print(json.dumps({d.metadata['Name'].lower().replace('_','-'):d.version "
        "for d in metadata.distributions() if d.metadata.get('Name')},sort_keys=True))"
    )
    try:
        result = subprocess.check_output(
            [str(python_path), "-c", script], text=True, timeout=30, stderr=subprocess.PIPE
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"vLLM Python environment is unavailable: {python_path}") from exc
    versions = json.loads(result)
    required = {"vllm": "0.30.0", "vllm-bnb-plugin": "0.0.3", "bitsandbytes": "0.50.2"}
    for package, expected in required.items():
        if versions.get(package) != expected:
            raise ValueError(f"{package} must be {expected} in the vLLM environment")
    return versions


def _require_vllm_headroom(snapshot: dict[str, int], profile) -> None:
    """Require global free memory for vLLM's configured total-device budget plus reserve."""
    from .hybrid import CudaOutOfMemory, WorkerStartupOutOfMemory

    required = int(profile.gpu_memory_utilization * snapshot["total_mib"]) + 2048
    if snapshot["free_mib"] < required:
        error_type = WorkerStartupOutOfMemory if profile.name == "vllm-1" else CudaOutOfMemory
        raise error_type(
            f"global GPU free memory {snapshot['free_mib']} MiB is below "
            f"the requested {required} MiB for profile {profile.name}"
        )


def _execute_hybrid_cli(args) -> dict[str, Any]:
    from .hybrid_run import MAX_MODEL_LEN, run_hybrid
    from .hybrid_workers import ManagedWorker, gpu_memory_snapshot
    from .model_artifact import validate_artifact

    if args.model_dir is None or args.vllm_python is None:
        raise ValueError("--model-dir and --vllm-python are required for vllm-hybrid")
    from transformers import AutoTokenizer

    model_dir = args.model_dir.resolve()
    artifact = validate_artifact(model_dir)
    vllm_python = args.vllm_python.resolve()
    vllm_versions = _vllm_package_versions(vllm_python)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("a fast tokenizer is required for confidence alignment")
    run_dir = args.run_dir or _default_run_dir(args.profile)
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    worker_runtime: dict[str, Any] = {}

    def deadline() -> datetime:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return datetime.fromisoformat(manifest["deadline_utc"])

    def verify_worker(worker, engine):
        runtime = dict(worker.runtime or {})
        if runtime.get("model_revision") != artifact["model_revision"]:
            raise ValueError(f"{engine} worker model revision differs from the NF4 artifact")
        hash_key = "artifact_sha256" if engine == "greedy" else "aggregate_sha256"
        if runtime.get(hash_key) != artifact["aggregate_sha256"]:
            raise ValueError(f"{engine} worker artifact hash differs from the coordinator")
        if engine == "sampling" and runtime.get("version") != vllm_versions["vllm"]:
            raise ValueError("vLLM worker runtime version differs from environment metadata")
        runtime["active_batch_size"] = worker.active_profile.batch_size
        worker_runtime[engine] = runtime

    def verified_result(response, requests, worker, engine):
        verify_worker(worker, engine)
        results = response.get("results")
        if not isinstance(results, list):
            raise ValueError(f"{engine} worker omitted result list")
        mapping = {row.get("request_id"): row for row in results if isinstance(row, dict)}
        expected = {row["request_id"] for row in requests}
        if len(mapping) != len(results) or set(mapping) != expected:
            raise ValueError(f"{engine} worker returned incorrect request IDs")
        return mapping

    greedy_worker = ManagedWorker(
        lambda _profile: [
            sys.executable, "-m", "teacher_reliability.greedy_worker",
            "--model-dir", str(model_dir), "--memory-fraction", "0.25",
        ], run_dir / "greedy_worker.stderr.log",
    )
    sample_worker = ManagedWorker(
        lambda profile: [
            str(vllm_python), "-m", "teacher_reliability.vllm_worker",
            "--model-dir", str(model_dir),
            "--memory-utilization", str(profile.gpu_memory_utilization),
            "--max-num-seqs", str(profile.batch_size),
            "--max-num-batched-tokens", str(profile.max_num_batched_tokens),
            "--max-model-len", str(MAX_MODEL_LEN),
            *(["--enforce-eager"] if profile.enforce_eager else []),
        ], run_dir / "vllm_worker.stderr.log",
    )

    def greedy_execute(batch, profile):
        worker_deadline = deadline()
        if datetime.now(timezone.utc) >= worker_deadline:
            raise TimeoutError("worker deadline expired")
        if greedy_worker.active_profile is None:
            snapshot = gpu_memory_snapshot()
            if snapshot["free_mib"] < 8192:
                from .hybrid import WorkerStartupOutOfMemory

                raise WorkerStartupOutOfMemory(f"global GPU free memory {snapshot['free_mib']} MiB is below 8192 MiB for greedy model startup")
            worker_runtime["greedy_memory_preflight"] = snapshot
        response = greedy_worker.request({"op": "generate", "requests": batch}, profile, deadline=worker_deadline)
        mapping = verified_result(response, batch, greedy_worker, "greedy")
        return {key: GreedyGeneration(**row["greedy"]) for key, row in mapping.items()}

    def sample_execute(batch, profile, *, on_result=None):
        worker_deadline = deadline()
        if datetime.now(timezone.utc) >= worker_deadline:
            raise TimeoutError("worker deadline expired")
        if sample_worker.active_profile is None:
            snapshot = gpu_memory_snapshot()
            _require_vllm_headroom(snapshot, profile)
            worker_runtime["sampling_memory_preflight"] = {
                "profile": profile.name, **snapshot,
            }
            sample_worker.start(profile, deadline=worker_deadline)
            verify_worker(sample_worker, "sampling")
        def convert_sample(row):
            return SampleGeneration(
                row["text"], row["token_ids"], row["eos_generated"], row["was_truncated"], row["reason"]
            )

        def stream_sample(request_id, row):
            if on_result is not None:
                on_result(request_id, convert_sample(row))

        response = sample_worker.request(
            {"op": "generate", "requests": batch}, profile,
            deadline=worker_deadline, on_result=stream_sample,
        )
        mapping = verified_result(response, batch, sample_worker, "sampling")
        return {key: convert_sample(row) for key, row in mapping.items()}

    try:
        return run_hybrid(
            run_dir, get_profile(args.profile), seed=args.seed,
            model_revision=artifact["model_revision"], artifact=artifact,
            tokenizer=tokenizer, greedy_execute=greedy_execute,
            sample_execute=sample_execute,
            max_runtime_hours=args.max_runtime_hours if args.max_runtime_hours is not None else 12,
            worker_package_versions=vllm_versions, worker_runtime=worker_runtime,
            resume=args.resume,
        )
    finally:
        greedy_worker.close()
        sample_worker.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("transformers", "transformers-local", "vllm-hybrid"),
        default="transformers",
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--vllm-python", type=Path)
    parser.add_argument("--device-preset", choices=("rtx4060-8gb",), default="rtx4060-8gb")
    parser.add_argument("--import-run-zip", type=Path)
    parser.add_argument("--qualification-manifest", type=Path)
    parser.add_argument(
        "--fast-start-unqualified",
        action="store_true",
        help="skip the full local profile matrix and record an unqualified start using conservative batch defaults",
    )
    parser.add_argument("--max-runtime-hours", type=float)
    lifecycle = parser.add_mutually_exclusive_group()
    lifecycle.add_argument("--status", action="store_true")
    lifecycle.add_argument("--stop", action="store_true")
    args = parser.parse_args(argv)
    if args.fast_start_unqualified and (
        args.backend != "transformers-local"
        or args.profile != "full"
        or args.qualification_manifest is not None
    ):
        print(
            "--fast-start-unqualified applies only to a full transformers-local run without --qualification-manifest.",
            file=sys.stderr,
        )
        return 1
    if args.status or args.stop:
        try:
            return _local_status(args)
        except Exception as exc:
            print(f"Run control failed: {exc}", file=sys.stderr)
            return 1
    if args.backend == "transformers-local" and args.profile == "full":
        if not args.import_run_zip:
            print("Full local runs require --import-run-zip for the 110 A100 source rows.", file=sys.stderr)
            return 1
        if not args.qualification_manifest and not args.fast_start_unqualified:
            print(
                "Full local runs require --qualification-manifest or the explicit --fast-start-unqualified option.",
                file=sys.stderr,
            )
            return 1
    if args.backend == "transformers-local":
        run_dir = args.run_dir or _default_local_run_dir(args.profile)
        pointer_path = run_dir.parent / "current_run_dir.txt"
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        pointer_path.write_text(str(run_dir.resolve()), encoding="utf-8")
    else:
        run_dir = args.run_dir or _default_run_dir(args.profile)
    args.run_dir = run_dir
    try:
        if args.backend == "vllm-hybrid":
            manifest = _execute_hybrid_cli(args)
        elif args.backend == "transformers-local":
            manifest = _execute_local_cli(args)
        else:
            manifest = run_experiment(
                run_dir,
                get_profile(args.profile),
                seed=args.seed,
                resume=args.resume,
                local_files_only=args.local_files_only,
            )
    except Exception as exc:
        print(f"Run failed: {exc}", file=sys.stderr)
        return 1
    summary = (
        f"Run {manifest['run_id']} {manifest['status']}: "
        f"{manifest['completed_example_count']}/{manifest['selected_example_count']} examples; "
        f"manifest: {run_dir / 'run_manifest.json'}"
    )
    eta = manifest.get("estimated_remaining_hours")
    if eta is not None:
        summary += f"; qualification ETA at launch: {float(eta):.1f} hours"
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
