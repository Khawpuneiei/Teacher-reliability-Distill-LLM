"""Atomic manifests, exclusive run ownership, and append-only checkpoints."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from filelock import FileLock, Timeout


class RunAlreadyActive(RuntimeError):
    """Raised when another process owns this run directory."""


def _normalize_json_object_keys(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise TypeError("JSON object keys must be strings or integers")
            string_key = str(key)
            if string_key in normalized:
                raise ValueError(f"duplicate JSON object key after string conversion: {string_key}")
            normalized[string_key] = _normalize_json_object_keys(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json_object_keys(item) for item in value]
    return value


@contextmanager
def run_lock(run_dir: Path) -> Iterator[None]:
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(run_dir / ".run.lock"), timeout=0)
    try:
        lock.acquire()
    except Timeout as exc:
        raise RunAlreadyActive(f"another process is writing {run_dir}") from exc
    try:
        yield
    finally:
        lock.release()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_json_object_keys(value)
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def append_complete_prediction(path: Path, record: dict[str, Any]) -> None:
    if not _is_complete_prediction(record):
        raise ValueError("only complete prediction records may be checkpointed")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _is_complete_prediction(record: Any) -> bool:
    return (
        isinstance(record, dict)
        and isinstance(record.get("example_id"), str)
        and bool(record["example_id"])
        and record.get("status") == "complete"
        and isinstance(record.get("greedy"), dict)
        and isinstance(record.get("samples"), list)
    )


def load_completed_predictions(
    path: Path, *, repair_torn_tail: bool = True
) -> dict[str, dict[str, Any]]:
    """Load complete checkpoints and truncate only a torn final append."""
    if not path.exists():
        return {}
    contents = path.read_bytes()
    lines = contents.splitlines(keepends=True)
    records: dict[str, dict[str, Any]] = {}
    offset = 0
    for index, line in enumerate(lines):
        line_end = offset + len(line)
        is_final = index == len(lines) - 1
        if not line.endswith(b"\n"):
            if repair_torn_tail:
                with path.open("r+b") as handle:
                    handle.truncate(offset)
            break
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if is_final:
                raise ValueError("newline-terminated final checkpoint row is corrupt") from exc
            raise ValueError(f"checkpoint row {index + 1} is corrupt") from exc
        if not _is_complete_prediction(record):
            raise ValueError(f"checkpoint row {index + 1} is not a complete prediction")
        example_id = record["example_id"]
        if example_id in records:
            raise ValueError(f"duplicate checkpoint for {example_id}")
        records[example_id] = record
        offset = line_end
    return records
