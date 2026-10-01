"""Isolated NDJSON GPU workers with deadline-bound reads and safe restarts."""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .hybrid import CudaOutOfMemory, WorkerRestartRequired, WorkerStartupOutOfMemory


class WorkerExecutionError(RuntimeError):
    """A worker-side failure whose original exception class is preserved."""

    def __init__(self, worker_error_type: str, message: str):
        super().__init__(message)
        self.worker_error_type = worker_error_type


def gpu_memory_snapshot() -> dict[str, int]:
    """Read global GPU 0 free/total memory without creating a CUDA context."""
    result = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
        text=True, timeout=10, stderr=subprocess.PIPE,
    )
    rows = [line.strip() for line in result.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("nvidia-smi did not return exactly one GPU row")
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != 2:
        raise ValueError("nvidia-smi did not return free and total memory")
    free_mib, total_mib = map(int, values)
    if free_mib < 0 or total_mib <= 0 or free_mib > total_mib:
        raise ValueError("nvidia-smi returned invalid GPU memory values")
    return {"free_mib": free_mib, "total_mib": total_mib}


class LineWorker:
    def __init__(self, command, stderr_path, *, startup_deadline=None):
        self.runtime = {}
        self.process = None
        self._stderr = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        path = Path(stderr_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr = path.open("a", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                list(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._stderr, text=True, encoding="utf-8", bufsize=1,
                start_new_session=(os.name != "nt"),
            )
            threading.Thread(target=self._read_lines, daemon=True).start()
            ready = self._receive(startup_deadline or datetime.now(timezone.utc) + timedelta(minutes=10))
            if ready.get("error_type") == "cuda_oom":
                raise WorkerStartupOutOfMemory(ready.get("message", "CUDA OOM while loading worker"))
            if ready.get("ready") is not True or not isinstance(ready.get("runtime"), dict):
                raise WorkerExecutionError(
                    str(ready.get("error_type", "WorkerStartupError")),
                    ready.get("message", f"worker did not report readiness: {ready}"),
                )
            self.runtime = ready["runtime"]
        except BaseException:
            self.close()
            raise

    def _read_lines(self):
        try:
            for line in self.process.stdout:
                self._lines.put(line)
        finally:
            self._lines.put(None)

    def _receive(self, deadline):
        if deadline is not None and deadline.tzinfo is None:
            raise ValueError("deadline must be timezone-aware")
        seconds = None if deadline is None else (deadline - datetime.now(timezone.utc)).total_seconds()
        if seconds is not None and seconds <= 0:
            self.close()
            raise TimeoutError("worker deadline expired")
        try:
            line = self._lines.get(timeout=seconds)
        except queue.Empty as exc:
            self.close()
            raise TimeoutError("worker deadline expired") from exc
        if line is None:
            raise WorkerRestartRequired(
                f"worker exited before replying (code {self.process.poll()})"
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("worker emitted malformed JSON") from exc
        if not isinstance(response, dict):
            raise ValueError("worker emitted a non-object response")
        return response

    def request(self, payload, *, deadline=None, on_result=None, on_progress=None, unbounded=False):
        if self.process is None or self.process.poll() is not None:
            raise WorkerRestartRequired("worker process is not running")
        if unbounded and deadline is not None:
            raise ValueError("unbounded worker requests cannot also have a deadline")
        if unbounded:
            deadline = None
        elif deadline is None:
            deadline = datetime.now(timezone.utc) + timedelta(minutes=10)
        if deadline is not None and deadline <= datetime.now(timezone.utc):
            self.close()
            raise TimeoutError("worker deadline expired")
        try:
            self.process.stdin.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.close()
            raise WorkerRestartRequired("worker pipe closed while sending request") from exc
        streamed_results = {}
        while True:
            try:
                response = self._receive(deadline)
            except WorkerRestartRequired as exc:
                exc.completed_results = streamed_results
                raise
            if response.get("ok") is True and isinstance(response.get("result"), dict):
                result = response["result"]
                request_id = result.get("request_id")
                if not isinstance(request_id, str) or request_id in streamed_results:
                    raise ValueError("worker emitted an invalid or duplicate streamed request ID")
                streamed_results[request_id] = result
                if on_result is not None:
                    on_result(request_id, result)
                continue
            if response.get("ok") is True and isinstance(response.get("progress"), dict):
                if on_progress is not None:
                    on_progress(response["progress"])
                continue
            if response.get("ok") is True and response.get("done") is True:
                return {"ok": True, "results": list(streamed_results.values())}
            if response.get("ok") is True and isinstance(response.get("results"), list):
                return response
            if response.get("error_type") == "cuda_oom":
                error = CudaOutOfMemory(response.get("message", "CUDA OOM"))
                error.completed_results = streamed_results
                raise error
            raise WorkerExecutionError(
                str(response.get("error_type", "WorkerError")),
                response.get("message", "worker request failed"),
            )

    def close(self):
        process = self.process
        if process is not None and process.poll() is None:
            try:
                if os.name == "nt":
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired, OSError):
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
        if process is not None:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None


class ManagedWorker:
    def __init__(self, command_for_profile, stderr_path):
        self.command_for_profile = command_for_profile
        self.stderr_path = stderr_path
        self._worker = None
        self._profile = None
        self.runtime = None

    @property
    def active_profile(self):
        return self._profile

    def start(self, profile, *, deadline=None):
        if self._worker is None or self._profile != profile:
            self.close()
            self._worker = LineWorker(self.command_for_profile(profile), self.stderr_path, startup_deadline=deadline)
            self._profile = profile
            self.runtime = self._worker.runtime
        return self.runtime

    def request(self, payload, profile, *, deadline=None, on_result=None, on_progress=None, unbounded=False):
        self.start(profile, deadline=deadline)
        try:
            return self._worker.request(
                payload, deadline=deadline, on_result=on_result,
                on_progress=on_progress, unbounded=unbounded,
            )
        except (CudaOutOfMemory, TimeoutError, RuntimeError, ValueError):
            self.close()
            raise

    def close(self):
        if self._worker is not None:
            self._worker.close()
        self._worker = None
        self._profile = None
