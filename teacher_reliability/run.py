"""Run a reproducible greedy-confidence and self-consistency evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
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
_PACKAGE_NAMES = (
    "torch",
    "transformers",
    "tokenizers",
    "accelerate",
    "bitsandbytes",
    "datasets",
    "huggingface-hub",
    "hf_xet",
    "math-verify",
    "numpy",
    "pandas",
    "matplotlib",
)


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
            "greedy_agreement_count": consistency.greedy_agreement_count,
            "greedy_agreement_share": consistency.greedy_agreement_share,
            "majority_answer": consistency.majority_answer,
            "majority_vote_count": consistency.majority_vote_count,
            "majority_vote_share": consistency.majority_vote_share,
            "majority_correct": consistency.majority_correct,
            "majority_grading_status": consistency.majority_grade_status,
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    run_dir = args.run_dir or _default_run_dir(args.profile)
    try:
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
    print(
        f"Run {manifest['run_id']} {manifest['status']}: "
        f"{manifest['completed_example_count']}/{manifest['selected_example_count']} examples; "
        f"manifest: {run_dir / 'run_manifest.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
