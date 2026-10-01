"""Export and verify the pinned teacher's serialized NF4 model artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from .teacher import MODEL_REPOSITORY


MANIFEST_NAME = "artifact_manifest.json"
SCHEMA_VERSION = 1
MODELS_DIRECTORY = Path(__file__).resolve().parents[1] / "models"
EXPECTED_QUANTIZATION = {
    "load_in_4bit": True,
    "quant_type": "nf4",
    "double_quant": True,
    "compute_dtype": "float16",
}
_REVISION_RE = re.compile(r"[0-9a-fA-F]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _revision_sha(revision: Any) -> str:
    if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
        raise ValueError("revision must be an immutable 40-character commit SHA")
    return revision.lower()


def _target_below_models(output_dir: str | Path, models_directory: str | Path) -> tuple[Path, Path]:
    models_root = Path(models_directory).resolve()
    target = Path(output_dir).resolve(strict=False)
    if target == models_root or models_root not in target.parents:
        raise ValueError(f"output directory must be below the ignored models directory: {models_root}")
    if target.exists():
        raise FileExistsError(f"output directory must be new: {target}")
    return models_root, target


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files_under(root: Path) -> dict[str, Path]:
    files = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"model artifact must not contain symlinks: {path}")
        relative_path = path.relative_to(root).as_posix()
        if path.is_file() and relative_path != MANIFEST_NAME:
            files[relative_path] = path
    return files


def _read_packed_nf4_state(root: Path) -> dict[str, int]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required to inspect packed NF4 state") from error

    evidence = {}
    for path in sorted(root.rglob("*.safetensors")):
        if path.is_symlink():
            raise ValueError(f"model artifact must not contain symlinks: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            count = sum("quant_state.bitsandbytes__nf4" in key for key in handle.keys())
        if count:
            evidence[path.relative_to(root).as_posix()] = count
    return evidence


def _read_quantization(root: Path) -> dict[str, Any]:
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValueError("model artifact is missing config.json")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("model config.json is unreadable") from error
    raw = config.get("quantization_config") if isinstance(config, dict) else None
    if not isinstance(raw, dict) or raw.get("quant_method") != "bitsandbytes":
        raise ValueError("model config quantization must use bitsandbytes NF4")
    try:
        actual = {
            "load_in_4bit": raw["load_in_4bit"],
            "quant_type": str(raw["bnb_4bit_quant_type"]).lower(),
            "double_quant": raw["bnb_4bit_use_double_quant"],
            "compute_dtype": str(raw["bnb_4bit_compute_dtype"]).lower(),
        }
    except KeyError as error:
        raise ValueError("model config has incomplete NF4 quantization metadata") from error
    if actual != EXPECTED_QUANTIZATION:
        raise ValueError("model config quantization must be NF4 with double quant and float16 compute")
    return actual


def _aggregate_hash(files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative_path, file_hash in sorted(files.items()):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_artifact(model_dir: str | Path) -> dict[str, Any]:
    """Rehash and verify every model/tokenizer file and the quantized model state."""
    root = Path(model_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"model artifact directory does not exist: {root}")
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"model artifact is missing {MANIFEST_NAME}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("model artifact manifest is unreadable") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported model artifact manifest schema")

    revision = _revision_sha(manifest.get("revision"))
    if manifest.get("repository") != MODEL_REPOSITORY:
        raise ValueError("model artifact repository does not match the pinned teacher")
    listed = manifest.get("files")
    if not isinstance(listed, dict) or not listed:
        raise ValueError("model artifact manifest has no checksummed files")

    actual = _files_under(root)
    if set(actual) != set(listed):
        missing = sorted(set(listed) - set(actual))
        extra = sorted(set(actual) - set(listed))
        raise ValueError(f"model artifact file set differs from manifest (missing={missing}, extra={extra})")

    verified_files = {}
    for relative_path, expected_hash in listed.items():
        if (
            not isinstance(relative_path, str)
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
            or not isinstance(expected_hash, str)
            or not _SHA256_RE.fullmatch(expected_hash)
        ):
            raise ValueError("model artifact manifest contains an invalid path or SHA-256")
        actual_hash = _digest_file(actual[relative_path])
        if actual_hash != expected_hash:
            raise ValueError(f"model artifact checksum mismatch: {relative_path}")
        verified_files[relative_path] = actual_hash

    quantization = _read_quantization(root)
    if manifest.get("quantization") != EXPECTED_QUANTIZATION:
        raise ValueError("manifest quantization differs from the required NF4 configuration")
    if quantization != manifest["quantization"]:
        raise ValueError("model config quantization differs from manifest quantization")

    packed_state = _read_packed_nf4_state(root)
    if not packed_state:
        raise ValueError("model artifact contains no serialized packed NF4 state")
    if manifest.get("packed_nf4_state") != packed_state:
        raise ValueError("packed NF4 state evidence differs from the manifest")

    return {
        "model_revision": revision,
        "aggregate_sha256": _aggregate_hash(verified_files),
        "files": verified_files,
        "quantization": quantization,
        "packed_nf4_state": packed_state,
        "repository": MODEL_REPOSITORY,
        "schema_version": SCHEMA_VERSION,
    }


def validate_model_artifact(model_dir: str | Path) -> dict[str, Any]:
    """Descriptive alias for callers that name the local artifact explicitly."""
    return validate_artifact(model_dir)


def _require_loaded_nf4_model(model: Any) -> None:
    load_info = getattr(model, "_teacher_reliability_load_info", None)
    actual = load_info.get("quantization") if isinstance(load_info, dict) else None
    expected = {
        "load_in_4bit": True,
        "quant_type": "nf4",
        "double_quant": True,
        "compute_dtype": "float16",
    }
    if actual != expected:
        raise ValueError("teacher.load_teacher did not return the expected NF4 model")


def export_model_artifact(
    output_dir: str | Path,
    revision: str,
    *,
    loader: Callable[..., tuple[Any, Any]] | None = None,
    models_directory: str | Path = MODELS_DIRECTORY,
) -> Path:
    """Serialize an existing pinned teacher load into a new ignored models/ directory."""
    pinned_revision = _revision_sha(revision)
    models_root, target = _target_below_models(output_dir, models_directory)
    if loader is None:
        from .teacher import load_teacher

        loader = load_teacher
    tokenizer, model = loader(
        revision=pinned_revision,
        repository=MODEL_REPOSITORY,
        local_files_only=False,
    )
    _require_loaded_nf4_model(model)

    models_root.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".teacher-nf4-", dir=models_root))
    try:
        model.save_pretrained(stage, safe_serialization=True)
        tokenizer.save_pretrained(stage)
        files = _files_under(stage)
        hashes = {relative: _digest_file(path) for relative, path in sorted(files.items())}
        quantization = _read_quantization(stage)
        if quantization != EXPECTED_QUANTIZATION:
            raise ValueError("saved config lost the required NF4 quantization metadata")
        packed_state = _read_packed_nf4_state(stage)
        if not packed_state:
            raise ValueError("saved model has no packed NF4 state")

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "repository": MODEL_REPOSITORY,
            "revision": pinned_revision,
            "quantization": quantization,
            "packed_nf4_state": packed_state,
            "files": hashes,
        }
        (stage / MANIFEST_NAME).write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        validate_artifact(stage)
        os.rename(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="immutable 40-character model commit SHA")
    parser.add_argument("--output-dir", required=True, type=Path, help="new destination below models/")
    args = parser.parse_args(argv)
    output = export_model_artifact(args.output_dir, args.revision)
    artifact = validate_artifact(output)
    print(
        json.dumps(
            {
                "model_dir": str(output),
                "model_revision": artifact["model_revision"],
                "aggregate_sha256": artifact["aggregate_sha256"],
                "quantization": artifact["quantization"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
