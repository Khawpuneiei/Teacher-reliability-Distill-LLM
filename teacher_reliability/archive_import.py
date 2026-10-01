"""Validation for importing complete A100 rows into a separately identified run."""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .answers import extract_final_answer_span
from .confidence import aggregate_confidence
from .consistency import summarize_consistency
from .data import Example
from .grading import grade_prediction
from .run import _canonical_digest
from .selection import RunProfile
from .teacher import MODEL_REPOSITORY, derive_sample_seed


MODEL_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")
ARCHIVE_SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MODEL_OUTPUT_VOCAB_SIZE = 152_064
EXPECTED_MEMBERS = {
    "run_manifest.json",
    "predictions.jsonl",
    "runner.log",
    "runner.pid",
}
MAX_ARCHIVE_BYTES = 2 * 1024**3
MAX_MEMBER_BYTES = 1024**3
MAX_RECORD_BYTES = 64 * 1024**2
LEGACY_UNAVAILABLE_CONSISTENCY_FIELDS = {
    "sample_answer_parse_unknown_count",
    "greedy_agreement_unknown_count",
    "majority_comparison_unknown_count",
}


@dataclass(frozen=True)
class ImportedRun:
    rows: list[dict[str, Any]]
    metadata: dict[str, Any]
    archive_snapshot: Any = field(default=None, repr=False, compare=False)

    def close(self) -> None:
        if self.archive_snapshot is not None:
            self.archive_snapshot.close()


def _snapshot_archive(path: Path):
    """Spool and hash one open source handle before parsing any ZIP members."""
    snapshot = tempfile.TemporaryFile(mode="w+b")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            if stream.seekable() and stream.seek(0, 2) > MAX_ARCHIVE_BYTES:
                raise ValueError("A100 archive is missing or exceeds the import size limit")
            stream.seek(0)
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(block)
                if size > MAX_ARCHIVE_BYTES:
                    raise ValueError("A100 archive is missing or exceeds the import size limit")
                digest.update(block)
                snapshot.write(block)
        snapshot.seek(0)
        return snapshot, digest.hexdigest()
    except BaseException:
        snapshot.close()
        raise


def _member_names(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    files: dict[str, zipfile.ZipInfo] = {}
    roots: set[str | None] = set()
    for info in archive.infolist():
        name = info.filename
        if not name or "\\" in name:
            raise ValueError(f"unsafe ZIP member path: {name!r}")
        path = PurePosixPath(name)
        if path.is_absolute() or any(part in {".", ".."} for part in path.parts):
            raise ValueError(f"unsafe ZIP member path: {name!r}")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f"ZIP member cannot be a symbolic link: {name!r}")
        if info.flag_bits & 0x1:
            raise ValueError(f"encrypted ZIP members are unsupported: {name!r}")
        if info.is_dir():
            if len(path.parts) != 1:
                raise ValueError(f"nested ZIP directories are unsupported: {name!r}")
            roots.add(path.parts[0])
            continue
        if len(path.parts) == 1:
            roots.add(None)
            relative = path.name
        elif len(path.parts) == 2:
            roots.add(path.parts[0])
            relative = path.parts[1]
        else:
            raise ValueError(f"nested ZIP members are unsupported: {name!r}")
        if relative in files:
            raise ValueError(f"duplicate ZIP member path: {relative}")
        if relative not in EXPECTED_MEMBERS:
            raise ValueError(f"unexpected ZIP member: {relative}")
        if info.file_size > MAX_MEMBER_BYTES:
            raise ValueError(f"ZIP member exceeds the import size limit: {relative}")
        files[relative] = info
    if len(roots) != 1:
        raise ValueError("ZIP members must share one archive root directory")
    if not {"run_manifest.json", "predictions.jsonl"}.issubset(files):
        raise ValueError("archive must contain run_manifest.json and predictions.jsonl")
    return files


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    examples: list[Example],
    profile: RunProfile,
    seed: int,
    model_revision: str,
    dataset_revisions: dict[str, str],
) -> None:
    if profile.name != "full" or profile.sample_count != 8 or profile.max_new_tokens != 1024:
        raise ValueError("A100 imports require the full eight-sample, 1,024-token profile")
    if not MODEL_REVISION_PATTERN.fullmatch(model_revision):
        raise ValueError("model_revision must be an immutable model commit SHA")
    if manifest.get("profile") != profile.as_dict() or manifest.get("seed") != seed:
        raise ValueError("A100 run profile or seed differs from the local run")
    if manifest.get("model_repository") != MODEL_REPOSITORY:
        raise ValueError("A100 run uses a different model repository")
    if manifest.get("model_revision") != model_revision:
        raise ValueError("A100 model revision differs from the local run")
    if manifest.get("dataset_revisions") != dataset_revisions:
        raise ValueError("A100 dataset revisions differ from the local run")
    hardware = manifest.get("hardware")
    if not isinstance(hardware, dict):
        raise ValueError("source run manifest omits top-level hardware metadata")
    signature = manifest.get("run_signature")
    identity = manifest.get("identity")
    if (
        not isinstance(signature, str)
        or not ARCHIVE_SHA_PATTERN.fullmatch(signature)
        or not isinstance(identity, dict)
        or _canonical_digest(identity) != signature
    ):
        raise ValueError("A100 manifest run signature does not match its identity")
    runtime = identity.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("signed A100 identity omits its runtime hardware claim")
    if runtime != hardware:
        raise ValueError("signed runtime hardware disagrees with top-level hardware metadata")
    if "A100" not in str(runtime.get("gpu_name", "")):
        raise ValueError("signed runtime does not identify an A100 GPU")

    ids = [example.example_id for example in examples]
    if len(ids) != len(set(ids)) or manifest.get("selected_example_ids") != ids:
        raise ValueError("A100 selected IDs differ from the pinned local selection")
    if manifest.get("selected_example_count") != len(ids):
        raise ValueError("A100 selected-example count is inconsistent")
    if identity.get("profile") != profile.as_dict() or identity.get("base_seed") != seed:
        raise ValueError("A100 signed identity has a different profile or seed")
    if identity.get("datasets") != dataset_revisions:
        raise ValueError("A100 signed identity has different dataset revisions")
    model = identity.get("model")
    if (
        not isinstance(model, dict)
        or model.get("repository") != MODEL_REPOSITORY
        or model.get("revision") != model_revision
        or model.get("quantization")
        != {
            "load_in_4bit": True,
            "quant_type": "nf4",
            "double_quant": True,
            "compute_dtype": "float16",
        }
    ):
        raise ValueError("A100 signed identity does not specify the required NF4 model")
    if identity.get("selected_ids_sha256") != _canonical_digest(ids):
        raise ValueError("A100 selected-ID digest differs from the local selection")
    if identity.get("selected_examples_sha256") != _canonical_digest(
        [asdict(example) for example in examples]
    ):
        raise ValueError("A100 normalized-input digest differs from local dataset content")


def _validate_token_ids(value: Any, field: str, maximum: int) -> list[int]:
    if not isinstance(value, list) or len(value) > maximum or any(
        not isinstance(token, int) or isinstance(token, bool)
        or token < 0 or token >= MODEL_OUTPUT_VOCAB_SIZE
        for token in value
    ):
        raise ValueError(f"invalid or over-limit {field}")
    return value


def _semantic_value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return (
            isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-10)
        )
    return type(actual) is type(expected) and actual == expected


def _require_semantic_fields(
    value: dict[str, Any], expected: dict[str, Any], field: str, example_id: str
) -> None:
    for key, expected_value in expected.items():
        if key not in value or not _semantic_value_matches(value.get(key), expected_value):
            raise ValueError(f"archive {field} is inconsistent for {example_id}: {key}")


def _validate_termination_fields(
    generation: dict[str, Any], *, field: str, example_id: str
) -> None:
    eos_generated = generation.get("eos_generated")
    was_truncated = generation.get("was_truncated")
    reason = generation.get("termination_reason")
    if not isinstance(eos_generated, bool) or not isinstance(was_truncated, bool):
        raise ValueError(f"archive {field} termination flags are invalid for {example_id}")
    if not isinstance(reason, str) or reason not in {"eos", "max_new_tokens", "other_stop"}:
        raise ValueError(f"archive {field} termination reason is invalid for {example_id}")
    if (
        (eos_generated and (reason != "eos" or was_truncated))
        or (was_truncated and (reason != "max_new_tokens" or eos_generated))
        or (not eos_generated and not was_truncated and reason != "other_stop")
    ):
        raise ValueError(f"archive {field} termination fields conflict for {example_id}")


def _validate_derived_content(
    record: dict[str, Any],
    example: Example,
    example_id: str,
    *,
    unavailable_consistency_fields: set[str] | None = None,
) -> None:
    greedy = record["greedy"]
    grade = grade_prediction(greedy["text"], example.gold_answer)
    if grade.status not in {
        "parse_timeout", "verifier_timeout", "grading_worker_error", "parser_error"
    }:
        _require_semantic_fields(
            greedy,
            {
                "predicted_answer": grade.prediction_span.value if grade.prediction_span else None,
                "answer_method": grade.prediction_span.method if grade.prediction_span else None,
                "prediction_parseable": grade.prediction_parseable,
                "gold_parseable": grade.gold_parseable,
                "correct": grade.correct,
                "grading_status": grade.status,
            },
            "greedy grading",
            example_id,
        )

    spans = greedy.get("token_char_spans")
    token_count = len(greedy["token_ids"])
    if spans is not None and any(
        span is not None and (
            not isinstance(span, list) or len(span) != 2
            or any(not isinstance(offset, int) or isinstance(offset, bool) for offset in span)
            or span[0] < 0 or span[1] < span[0] or span[1] > len(greedy["text"])
        )
        for span in spans
    ):
        raise ValueError(f"archive greedy token spans exceed generated text for {example_id}")
    confidence = aggregate_confidence(
        token_entropies=greedy["token_entropies_nats"],
        token_logprobs=greedy["token_logprobs"],
        token_char_spans=(
            [None if span is None else (span[0], span[1]) for span in spans]
            if spans is not None else None
        ),
        content_token_mask=greedy["content_token_mask"],
        answer_span=grade.prediction_span,
        vocab_size=MODEL_OUTPUT_VOCAB_SIZE,
    )
    _require_semantic_fields(
        greedy,
        {
            "content_token_count": confidence.content_token_count,
            "mean_entropy_nats": confidence.mean_entropy_nats,
            "entropy_confidence": confidence.entropy_confidence,
            "sequence_logprob_sum": confidence.sequence_logprob_sum,
            "sequence_geometric_probability": confidence.sequence_geometric_probability,
            "answer_token_count": confidence.answer_token_count,
            "answer_token_geometric_probability": confidence.answer_token_geometric_probability,
            "answer_alignment_status": confidence.answer_alignment_status,
        },
        "greedy confidence",
        example_id,
    )

    samples = sorted(record["samples"], key=lambda sample: sample["index"])
    sample_answers: list[str | None] = []
    for sample in samples:
        _validate_termination_fields(sample, field="sample", example_id=example_id)
        span = extract_final_answer_span(sample["text"])
        answer = span.value if span is not None else None
        answer_method = span.method if span is not None else None
        _require_semantic_fields(
            sample,
            {"answer": answer, "answer_method": answer_method},
            "sample answer/text",
            example_id,
        )
        sample_answers.append(answer)

    consistency = summarize_consistency(
        grade.prediction_span.value if grade.prediction_span else None,
        sample_answers,
        example.gold_answer,
    )
    expected_consistency = {
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
    }
    for field_name in unavailable_consistency_fields or ():
        expected_consistency.pop(field_name, None)
    _require_semantic_fields(
        record["self_consistency"],
        expected_consistency,
        "self-consistency",
        example_id,
    )


def _validate_record(
    record: Any,
    examples_by_id: dict[str, Example],
    *,
    seed: int,
) -> dict[str, Any]:
    if not isinstance(record, dict) or record.get("status") != "complete":
        raise ValueError("archive contains a non-complete prediction row")
    example_id = record.get("example_id")
    example = examples_by_id.get(example_id)
    if example is None:
        raise ValueError("archive prediction ID is outside the selected local inputs")
    if record.get("question") != example.question or record.get("gold_answer") != example.gold_answer:
        raise ValueError(f"archive question or reference answer differs for {example_id}")
    expected_source = {
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
    if record.get("source") != expected_source:
        raise ValueError(f"archive source identity differs for {example_id}")
    if record.get("requested_sample_count") != 8:
        raise ValueError(f"archive row does not request eight samples for {example_id}")
    greedy = record.get("greedy")
    if not isinstance(greedy, dict):
        raise ValueError(f"archive row has no greedy answer for {example_id}")
    required_greedy_fields = {
        "text", "token_ids", "token_entropies_nats", "token_logprobs",
        "content_token_mask", "token_char_spans", "predicted_answer", "answer_method",
        "eos_generated", "was_truncated", "termination_reason",
        "prediction_parseable", "gold_parseable", "correct", "grading_status",
        "content_token_count", "mean_entropy_nats", "entropy_confidence",
        "sequence_logprob_sum", "sequence_geometric_probability", "answer_token_count",
        "answer_token_geometric_probability", "answer_alignment_status",
    }
    if not required_greedy_fields.issubset(greedy):
        raise ValueError(f"archive greedy derived fields are incomplete for {example_id}")
    token_ids = _validate_token_ids(greedy.get("token_ids"), "greedy token IDs", 1024)
    entropies = greedy.get("token_entropies_nats")
    logprobs = greedy.get("token_logprobs")
    content_mask = greedy.get("content_token_mask")
    if not all(isinstance(row, list) and len(row) == len(token_ids) for row in (entropies, logprobs, content_mask)):
        raise ValueError(f"archive greedy scores do not align with token IDs for {example_id}")
    if any(not isinstance(value, bool) for value in content_mask):
        raise ValueError(f"archive content-token mask is invalid for {example_id}")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 for value in entropies):
        raise ValueError(f"archive entropy score is invalid for {example_id}")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value > 1e-6 for value in logprobs):
        raise ValueError(f"archive log probability is invalid for {example_id}")
    spans = greedy.get("token_char_spans")
    if spans is not None and (
        not isinstance(spans, list) or len(spans) != len(token_ids)
    ):
        raise ValueError(f"archive character spans do not align for {example_id}")
    if not isinstance(greedy.get("text"), str):
        raise ValueError(f"archive greedy text is invalid for {example_id}")
    _validate_termination_fields(greedy, field="greedy", example_id=example_id)

    samples = record.get("samples")
    if not isinstance(samples, list) or len(samples) != 8:
        raise ValueError(f"archive row must contain eight complete samples for {example_id}")
    seen_indices: set[int] = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError(f"archive sample is invalid for {example_id}")
        required_sample_fields = {
            "index", "seed", "token_ids", "text", "answer", "answer_method",
            "eos_generated", "was_truncated", "termination_reason",
        }
        if not required_sample_fields.issubset(sample):
            raise ValueError(f"archive sample fields are incomplete for {example_id}")
        index = sample.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 8 or index in seen_indices:
            raise ValueError(f"archive sample indices are invalid for {example_id}")
        seen_indices.add(index)
        sample_seed = sample.get("seed")
        if (
            not isinstance(sample_seed, int) or isinstance(sample_seed, bool)
            or sample_seed != derive_sample_seed(seed, example_id, index)
        ):
            raise ValueError(f"archive sample seed is invalid for {example_id}")
        _validate_token_ids(sample.get("token_ids"), "sample token IDs", 1024)
        if (
            not isinstance(sample.get("text"), str)
            or (sample.get("answer") is not None and not isinstance(sample.get("answer"), str))
            or (sample.get("answer_method") is not None and not isinstance(sample.get("answer_method"), str))
            or not isinstance(sample.get("eos_generated"), bool)
            or not isinstance(sample.get("was_truncated"), bool)
            or not isinstance(sample.get("termination_reason"), str)
        ):
            raise ValueError(f"archive sample text is invalid for {example_id}")
    if seen_indices != set(range(8)):
        raise ValueError(f"archive sample indices are incomplete for {example_id}")
    consistency = record.get("self_consistency")
    if not isinstance(consistency, dict) or consistency.get("sample_count") != 8:
        raise ValueError(f"archive consistency summary is incomplete for {example_id}")
    required_consistency_fields = {
        "sample_answer_parseable_count", "sample_answer_parse_unknown_count",
        "greedy_agreement_count", "greedy_agreement_unknown_count",
        "greedy_agreement_share", "majority_answer", "majority_vote_count",
        "majority_vote_share", "majority_correct", "majority_grading_status",
        "majority_comparison_unknown_count",
    }
    unavailable_consistency_fields = {
        field_name
        for field_name in LEGACY_UNAVAILABLE_CONSISTENCY_FIELDS
        if field_name not in consistency
    }
    if not (required_consistency_fields - unavailable_consistency_fields).issubset(consistency):
        raise ValueError(f"archive consistency fields are incomplete for {example_id}")
    for field_name in unavailable_consistency_fields:
        consistency[field_name] = None
    _validate_derived_content(
        record,
        example,
        example_id,
        unavailable_consistency_fields=unavailable_consistency_fields,
    )
    return record


def import_a100_run(
    archive_path: str | Path,
    *,
    examples: list[Example],
    profile: RunProfile,
    seed: int,
    model_revision: str,
    dataset_revisions: dict[str, str],
    expected_archive_sha256: str | None = None,
) -> ImportedRun:
    """Read and validate complete A100 predictions without extracting ZIP files."""
    archive_path = Path(archive_path).resolve(strict=True)
    if not archive_path.is_file():
        raise ValueError("A100 archive is missing or exceeds the import size limit")
    snapshot, archive_digest = _snapshot_archive(archive_path)
    try:
        if expected_archive_sha256 is not None and (
            not isinstance(expected_archive_sha256, str)
            or not ARCHIVE_SHA_PATTERN.fullmatch(expected_archive_sha256)
            or archive_digest != expected_archive_sha256
        ):
            raise ValueError("pinned A100 archive SHA-256 does not match the imported bytes")
        with zipfile.ZipFile(snapshot) as archive:
            members = _member_names(archive)
            manifest_bytes = archive.read(members["run_manifest.json"])
            if len(manifest_bytes) > MAX_MEMBER_BYTES:
                raise ValueError("A100 run manifest exceeds the import size limit")
            manifest = json.loads(manifest_bytes)
            if not isinstance(manifest, dict):
                raise ValueError("A100 run manifest must be a JSON object")
            _validate_manifest(
                manifest,
                examples=examples,
                profile=profile,
                seed=seed,
                model_revision=model_revision,
                dataset_revisions=dataset_revisions,
            )
            examples_by_id = {example.example_id: example for example in examples}
            rows: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            predictions_hash = hashlib.sha256()
            with archive.open(members["predictions.jsonl"]) as stream:
                for line_number, line in enumerate(stream, start=1):
                    predictions_hash.update(line)
                    if len(line) > MAX_RECORD_BYTES:
                        raise ValueError(f"archive prediction line {line_number} exceeds the import limit")
                    if not line.endswith(b"\n"):
                        raise ValueError("A100 predictions file contains a torn final row")
                    try:
                        row = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ValueError(f"A100 prediction row {line_number} is invalid JSON") from exc
                    row = _validate_record(row, examples_by_id, seed=seed)
                    if row["example_id"] in seen_ids:
                        raise ValueError(f"duplicate A100 prediction for {row['example_id']}")
                    seen_ids.add(row["example_id"])
                    rows.append(row)
            if not rows:
                raise ValueError("A100 archive has no complete prediction rows")
    except (zipfile.BadZipFile, OSError, json.JSONDecodeError) as exc:
        snapshot.close()
        raise ValueError(f"cannot read A100 run archive: {exc}") from exc
    except BaseException:
        snapshot.close()
        raise

    return ImportedRun(
        rows=rows,
        metadata={
            "archive_sha256": archive_digest,
            "predictions_sha256": predictions_hash.hexdigest(),
            "source_run_id": manifest.get("run_id"),
            "source_run_signature": manifest["run_signature"],
            "source_run_status": manifest.get("status"),
            "source_completed_count_in_manifest": manifest.get("completed_example_count"),
            "imported_count": len(rows),
            "source_gpu": manifest["hardware"]["gpu_name"],
            "source_model_revision": model_revision,
            "source_dataset_revisions": dict(dataset_revisions),
            "source_manifest": manifest,
        },
        archive_snapshot=snapshot,
    )
