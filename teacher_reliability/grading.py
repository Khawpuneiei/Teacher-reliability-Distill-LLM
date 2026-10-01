"""Math-Verify-backed answer parsing, equivalence, and correctness labels."""

from __future__ import annotations

import atexit
import logging
import multiprocessing
import os
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

from .answers import AnswerSpan, extract_final_answer_span

MAX_ANSWER_CHARACTERS = 2048
DEFAULT_GRADING_TIMEOUT_SECONDS = 30.0
_POLL_INTERVAL_SECONDS = 0.05
_WORKER_SHUTDOWN_SECONDS = 1.0


class _ExternalTimeoutWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (
            record.levelno == logging.WARNING
            and message.startswith("Timeout is disabled as ")
        )


def _configure_external_timeout_logging() -> None:
    warning_filter = _ExternalTimeoutWarningFilter()
    logging.getLogger("math_verify.parser").addFilter(warning_filter)
    logging.getLogger("math_verify.grader").addFilter(warning_filter)


@dataclass(frozen=True, slots=True)
class GradeOutcome:
    correct: bool | None
    prediction_parseable: bool | None
    gold_parseable: bool | None
    status: str
    prediction_span: AnswerSpan | None = None
    failure_stage: str | None = None


class GradingTimeoutError(TimeoutError):
    """A Math-Verify operation exceeded its external worker deadline."""

    def __init__(self, stage: str, progress: dict[str, Any] | None = None):
        super().__init__(f"Math-Verify worker timed out during {stage}")
        self.stage = stage
        self.progress = progress or {}


class GradingWorkerError(RuntimeError):
    """The isolated Math-Verify worker exited or returned an invalid response."""


def _parse_boxed(value: str):
    from math_verify import parse

    if len(value) > MAX_ANSWER_CHARACTERS:
        return None
    # Windows Math-Verify cannot use its positive timeout: its nested worker is
    # not spawn-picklable. The parent process bounds this entire worker request.
    timeout = 0 if os.name == "nt" else 5
    parsed = parse(
        r"\boxed{" + value + "}",
        fallback_mode="no_fallback",
        parsing_timeout=timeout,
    )
    return parsed if parsed else None


@lru_cache(maxsize=8192)
def _try_parse(value: str | None):
    if value is None or not value.strip():
        return None
    try:
        return _parse_boxed(value.strip())
    except TimeoutError:
        # Preserve timeouts as a separate state; they are not malformed answers.
        raise


def _send_progress(
    connection,
    request_id: int,
    stage: str,
    prediction_parseable: bool | None = None,
    gold_parseable: bool | None = None,
) -> None:
    connection.send(
        (
            request_id,
            "progress",
            {
                "stage": stage,
                "prediction_parseable": prediction_parseable,
                "gold_parseable": gold_parseable,
            },
        )
    )


def _evaluate_grade(connection, request_id: int, payload: dict[str, Any]):
    prediction = payload.get("prediction")
    gold = payload.get("gold")

    _send_progress(connection, request_id, "parse_prediction")
    try:
        parsed_prediction = _try_parse(prediction)
    except TimeoutError:
        raise
    except Exception:
        return {
            "correct": None,
            "prediction_parseable": None,
            "gold_parseable": None,
            "status": "parser_error",
        }
    prediction_parseable = parsed_prediction is not None

    _send_progress(
        connection,
        request_id,
        "parse_gold",
        prediction_parseable,
        None,
    )
    try:
        parsed_gold = _try_parse(gold)
    except TimeoutError:
        raise
    except Exception:
        return {
            "correct": None,
            "prediction_parseable": prediction_parseable,
            "gold_parseable": None,
            "status": "parser_error",
        }
    gold_parseable = parsed_gold is not None

    if gold is None or not gold.strip():
        return {
            "correct": None,
            "prediction_parseable": prediction_parseable,
            "gold_parseable": False,
            "status": "missing_gold",
        }
    if parsed_gold is None:
        return {
            "correct": None,
            "prediction_parseable": prediction_parseable,
            "gold_parseable": False,
            "status": "unparseable_gold",
        }
    if prediction is None:
        return {
            "correct": False,
            "prediction_parseable": False,
            "gold_parseable": True,
            "status": "missing_prediction_answer",
        }
    if parsed_prediction is None:
        return {
            "correct": False,
            "prediction_parseable": False,
            "gold_parseable": True,
            "status": "unparseable_prediction",
        }

    _send_progress(
        connection,
        request_id,
        "verify",
        prediction_parseable,
        gold_parseable,
    )
    from math_verify import verify

    timeout = 0 if os.name == "nt" else 5
    try:
        correct = bool(verify(parsed_gold, parsed_prediction, timeout_seconds=timeout))
    except TimeoutError:
        raise
    except Exception:
        return {
            "correct": None,
            "prediction_parseable": True,
            "gold_parseable": True,
            "status": "verifier_error",
        }
    return {
        "correct": correct,
        "prediction_parseable": True,
        "gold_parseable": True,
        "status": "graded",
    }


def _evaluate_equivalence(connection, request_id: int, payload: dict[str, Any]):
    left = payload.get("left")
    right = payload.get("right")
    _send_progress(connection, request_id, "parse_left")
    parsed_left = _try_parse(left)
    _send_progress(
        connection,
        request_id,
        "parse_right",
        parsed_left is not None,
        None,
    )
    parsed_right = _try_parse(right)
    if parsed_left is None or parsed_right is None:
        return {"equivalent": False}

    _send_progress(
        connection,
        request_id,
        "verify_equivalence",
        True,
        True,
    )
    from math_verify import verify

    timeout = 0 if os.name == "nt" else 5
    try:
        equivalent = bool(
            verify(parsed_left, parsed_right, timeout_seconds=timeout)
            or verify(parsed_right, parsed_left, timeout_seconds=timeout)
        )
    except TimeoutError:
        raise
    except Exception:
        equivalent = None
    return {"equivalent": equivalent}


def _math_verify_worker_main(connection) -> None:
    """Serve serialized Math-Verify requests in one restartable child process."""
    _configure_external_timeout_logging()
    while True:
        try:
            request_id, operation, payload = connection.recv()
        except (EOFError, OSError):
            return
        if operation == "shutdown":
            return
        stage = operation
        try:
            if operation == "parse":
                _send_progress(connection, request_id, "parse")
                result = {"parseable": _try_parse(payload) is not None}
            elif operation == "equivalent":
                result = _evaluate_equivalence(connection, request_id, payload)
            elif operation == "grade":
                result = _evaluate_grade(connection, request_id, payload)
            else:
                raise ValueError(f"unsupported grading operation: {operation}")
            connection.send((request_id, "result", result))
        except TimeoutError as exc:
            connection.send(
                (
                    request_id,
                    "error",
                    {"kind": "timeout", "stage": stage, "message": str(exc)},
                )
            )
        except Exception as exc:
            connection.send(
                (
                    request_id,
                    "error",
                    {
                        "kind": type(exc).__name__,
                        "stage": stage,
                        "message": str(exc),
                    },
                )
            )


class MathVerifyWorker:
    """Persistent Math-Verify process with a hard parent-side request deadline.

    A timeout terminates the stuck process. The next request starts a fresh
    worker, while successful requests reuse the existing process and its parse
    cache. ``worker_target`` is injectable so tests can exercise real process
    termination and restart without depending on pathological Math-Verify data.
    """

    def __init__(
        self,
        timeout_seconds: float = DEFAULT_GRADING_TIMEOUT_SECONDS,
        *,
        worker_target: Callable[..., None] | None = None,
        worker_args: tuple[Any, ...] = (),
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = float(timeout_seconds)
        self.worker_target = worker_target or _math_verify_worker_main
        self.worker_args = worker_args
        self._context = multiprocessing.get_context("spawn")
        self._connection = None
        self._process = None
        self._last_process = None
        self._lock = threading.Lock()
        self._next_request_id = 0
        self._needs_restart = False
        self.start_count = 0
        self.restart_count = 0
        self.timeout_count = 0

    @property
    def worker_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def current_worker_pid(self) -> int | None:
        if self.worker_alive:
            return self._process.pid
        return None

    def _start_worker(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._discard_worker(mark_restart=self._process is not None)
        if self._needs_restart:
            self.restart_count += 1
            self._needs_restart = False
        parent_connection, child_connection = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=self.worker_target,
            args=(child_connection, *self.worker_args),
            name="teacher-reliability-math-verify",
        )
        # This worker has no child processes of its own. Daemon status ensures
        # interpreter shutdown can never wait indefinitely for an idle grader.
        process.daemon = True
        process.start()
        child_connection.close()
        self._connection = parent_connection
        self._process = process
        self._last_process = process
        self.start_count += 1

    def _discard_worker(self, *, mark_restart: bool) -> None:
        process = self._process
        connection = self._connection
        self._process = None
        self._connection = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=_WORKER_SHUTDOWN_SECONDS)
            if process.is_alive():
                process.kill()
                process.join(timeout=_WORKER_SHUTDOWN_SECONDS)
            if mark_restart:
                self._needs_restart = True

    def request(self, operation: str, payload: Any) -> dict[str, Any]:
        with self._lock:
            self._start_worker()
            self._next_request_id += 1
            request_id = self._next_request_id
            deadline = time.monotonic() + self.timeout_seconds
            connection = self._connection
            process = self._process
            try:
                connection.send((request_id, operation, payload))
            except (BrokenPipeError, EOFError, OSError) as exc:
                self._discard_worker(mark_restart=True)
                raise GradingWorkerError("Math-Verify worker could not accept a request") from exc

            progress: dict[str, Any] = {"stage": operation}
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.timeout_count += 1
                    self._discard_worker(mark_restart=True)
                    raise GradingTimeoutError(str(progress.get("stage", operation)), progress)
                try:
                    has_message = connection.poll(min(remaining, _POLL_INTERVAL_SECONDS))
                except (EOFError, OSError) as exc:
                    self._discard_worker(mark_restart=True)
                    raise GradingWorkerError("Math-Verify worker connection was lost") from exc
                if not has_message:
                    if process is None or not process.is_alive():
                        self._discard_worker(mark_restart=True)
                        raise GradingWorkerError("Math-Verify worker exited before replying")
                    continue
                try:
                    response_id, response_kind, response = connection.recv()
                except (EOFError, OSError) as exc:
                    self._discard_worker(mark_restart=True)
                    raise GradingWorkerError("Math-Verify worker exited before replying") from exc
                if response_id != request_id:
                    self._discard_worker(mark_restart=True)
                    raise GradingWorkerError("Math-Verify worker returned an unexpected request id")
                if response_kind == "progress":
                    progress = response
                    continue
                if response_kind == "result":
                    return response
                if response_kind == "error":
                    if response.get("kind") == "timeout":
                        raise GradingTimeoutError(
                            str(response.get("stage", operation)), progress
                        )
                    raise GradingWorkerError(
                        f"Math-Verify worker failed during {response.get('stage', operation)}: "
                        f"{response.get('message', 'unknown worker error')}"
                    )
                self._discard_worker(mark_restart=True)
                raise GradingWorkerError("Math-Verify worker returned an unknown response")

    def close(self) -> None:
        with self._lock:
            connection = self._connection
            process = self._process
            if connection is not None and process is not None and process.is_alive():
                try:
                    connection.send((0, "shutdown", None))
                    process.join(timeout=_WORKER_SHUTDOWN_SECONDS)
                except (BrokenPipeError, EOFError, OSError):
                    pass
            self._discard_worker(mark_restart=False)


_DEFAULT_WORKER: MathVerifyWorker | None = None
_DEFAULT_WORKER_LOCK = threading.Lock()


def _default_worker() -> MathVerifyWorker:
    global _DEFAULT_WORKER
    with _DEFAULT_WORKER_LOCK:
        if _DEFAULT_WORKER is None:
            _DEFAULT_WORKER = MathVerifyWorker()
        return _DEFAULT_WORKER


def _close_default_worker() -> None:
    if _DEFAULT_WORKER is not None:
        _DEFAULT_WORKER.close()


atexit.register(_close_default_worker)


def check_answer_parseability(
    value: str | None,
    *,
    worker: MathVerifyWorker | None = None,
) -> bool | None:
    """Return parseability, or ``None`` when the bounded check is unknown."""
    if value is None or not value.strip():
        return False
    try:
        return bool((worker or _default_worker()).request("parse", value.strip())["parseable"])
    except (GradingTimeoutError, GradingWorkerError):
        return None


def answer_parseable(
    value: str | None,
    *,
    worker: MathVerifyWorker | None = None,
) -> bool:
    """Compatibility boolean: true only when the bounded check succeeds."""
    return check_answer_parseability(value, worker=worker) is True


def check_answers_equivalence(
    left: str | None,
    right: str | None,
    *,
    worker: MathVerifyWorker | None = None,
) -> bool | None:
    """Compare answers, returning ``None`` when the bounded check is unknown."""
    if left is None or right is None or not left.strip() or not right.strip():
        return False
    try:
        result = (worker or _default_worker()).request(
            "equivalent", {"left": left, "right": right}
        )
        equivalent = result.get("equivalent")
        return equivalent if isinstance(equivalent, bool) else None
    except (GradingTimeoutError, GradingWorkerError):
        return None


def answers_equivalent(
    left: str | None,
    right: str | None,
    *,
    worker: MathVerifyWorker | None = None,
) -> bool:
    """Compatibility boolean for callers that cannot represent unknown."""
    return check_answers_equivalence(left, right, worker=worker) is True


def grade_prediction(
    prediction_text: str,
    gold_text: str | None,
    *,
    worker: MathVerifyWorker | None = None,
) -> GradeOutcome:
    """Grade the extracted final answer and retain parse coverage and timeout status."""
    span = extract_final_answer_span(prediction_text)
    payload = {
        "prediction": span.value if span is not None else None,
        "gold": gold_text,
    }
    grading_worker = worker or _default_worker()
    try:
        result = grading_worker.request("grade", payload)
    except GradingTimeoutError as exc:
        progress = exc.progress
        progress_stage = progress.get("stage")
        stage = progress_stage if isinstance(progress_stage, str) and progress_stage else exc.stage
        status = "verifier_timeout" if stage == "verify" else "parse_timeout"
        return GradeOutcome(
            None,
            progress.get("prediction_parseable"),
            progress.get("gold_parseable"),
            status,
            span,
            stage,
        )
    except GradingWorkerError as exc:
        return GradeOutcome(None, None, None, "grading_worker_error", span, str(exc))
    return GradeOutcome(
        result["correct"],
        result["prediction_parseable"],
        result["gold_parseable"],
        result["status"],
        span,
    )
