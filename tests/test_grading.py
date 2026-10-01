import unittest
import time
from unittest.mock import patch

import teacher_reliability.grading as grading
from teacher_reliability.grading import answer_parseable, answers_equivalent, grade_prediction
from teacher_reliability.run import _package_versions


def _stalling_grading_worker(connection, stage):
    while True:
        request_id, operation, _payload = connection.recv()
        if operation == "shutdown":
            return
        if operation == "grade":
            prediction_parseable = True if stage == "verify" else None
            gold_parseable = True if stage == "verify" else None
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
            time.sleep(30)
        elif operation == "parse":
            connection.send((request_id, "result", {"parseable": True}))


class MathAnswerGradingTests(unittest.TestCase):
    def test_parser_exception_is_unknown_instead_of_an_incorrect_prediction(self):
        class Connection:
            def send(self, message):
                pass

        grading._try_parse.cache_clear()
        with patch.object(grading, "_parse_boxed", side_effect=ValueError("parser crash")):
            result = grading._evaluate_grade(
                Connection(), 17, {"prediction": "parser-crash-unique", "gold": "12"}
            )

        self.assertIsNone(result["correct"])
        self.assertIsNone(result["prediction_parseable"])
        self.assertEqual(result["status"], "parser_error")

    def test_verifier_exception_remains_unknown_through_equivalence_api(self):
        class EvaluatingWorker:
            def request(self, operation, payload):
                self.assert_operation = operation
                return grading._evaluate_equivalence(Connection(), 18, payload)

        class Connection:
            def send(self, message):
                pass

        worker = EvaluatingWorker()
        with (
            patch.object(grading, "_try_parse", return_value=("parsed",)),
            patch("math_verify.verify", side_effect=RuntimeError("verifier crash")),
        ):
            result = grading.check_answers_equivalence("left", "right", worker=worker)

        self.assertEqual(worker.assert_operation, "equivalent")
        self.assertIsNone(result)

    def test_run_identity_records_the_installed_math_parser_stack(self):
        versions = _package_versions()

        self.assertEqual(versions.get("sympy"), "1.13.1")
        self.assertEqual(versions.get("latex2sympy2-extended"), "1.11.0")
        self.assertEqual(versions.get("antlr4-python3-runtime"), "4.13.2")

    def test_check_apis_keep_timeouts_unknown_for_consistency_metrics(self):
        class TimedOutWorker:
            def request(self, operation, payload):
                raise grading.GradingTimeoutError("parse")

        parse_check = getattr(grading, "check_answer_parseability", None)
        equivalence_check = getattr(grading, "check_answers_equivalence", None)
        self.assertTrue(callable(parse_check))
        self.assertTrue(callable(equivalence_check))
        self.assertIsNone(parse_check("12", worker=TimedOutWorker()))
        self.assertIsNone(equivalence_check("12", "13", worker=TimedOutWorker()))

    def test_grade_classifies_internal_timeout_from_latest_progress_stage(self):
        class TimedOutDuringVerificationWorker:
            def request(self, operation, payload):
                self.assert_operation = operation
                raise grading.GradingTimeoutError(
                    operation,
                    {
                        "stage": "verify",
                        "prediction_parseable": True,
                        "gold_parseable": True,
                    },
                )

        worker = TimedOutDuringVerificationWorker()

        result = grading.grade_prediction(r"\boxed{12}", "12", worker=worker)

        self.assertEqual(worker.assert_operation, "grade")
        self.assertEqual(result.status, "verifier_timeout")

    def test_stuck_parse_and_verify_are_bounded_and_worker_is_restarted(self):
        worker_type = getattr(grading, "MathVerifyWorker", None)
        self.assertIsNotNone(
            worker_type,
            "grading needs a persistent isolated worker with a request deadline",
        )

        for stage, expected_status in (
            ("parse_prediction", "parse_timeout"),
            ("verify", "verifier_timeout"),
        ):
            with self.subTest(stage=stage):
                worker = worker_type(
                    timeout_seconds=1.0,
                    worker_target=_stalling_grading_worker,
                    worker_args=(stage,),
                )
                started = time.monotonic()
                try:
                    result = grading.grade_prediction(
                        r"\boxed{812345}", "812345", worker=worker
                    )
                    elapsed = time.monotonic() - started

                    self.assertLess(elapsed, 5.0)
                    self.assertIsNone(result.correct)
                    self.assertEqual(result.status, expected_status)
                    self.assertEqual(worker.start_count, 1)
                    self.assertFalse(worker.worker_alive)

                    self.assertTrue(worker.request("parse", "812345")["parseable"])
                    self.assertEqual(worker.start_count, 2)
                    self.assertEqual(worker.restart_count, 1)
                finally:
                    worker.close()

    def test_equivalent_fraction_and_decimal_are_mathematically_equal(self):
        self.assertTrue(answers_equivalent(r"\frac{1}{2}", "0.5"))

    def test_greedy_answer_is_compared_with_the_extracted_final_box(self):
        result = grade_prediction(
            "First I got \\boxed{9}. Rechecking gives \\boxed{12}.", "12"
        )

        self.assertTrue(result.correct)
        self.assertTrue(result.prediction_parseable)
        self.assertTrue(result.gold_parseable)
        self.assertEqual(result.status, "graded")

    def test_unparseable_prediction_is_incorrect_when_gold_is_parseable(self):
        result = grade_prediction("No final answer is provided.", "12")

        self.assertFalse(result.correct)
        self.assertFalse(result.prediction_parseable)
        self.assertTrue(result.gold_parseable)
        self.assertEqual(result.status, "missing_prediction_answer")

    def test_unparseable_gold_is_unscorable_instead_of_counted_wrong(self):
        result = grade_prediction(r"\boxed{12}", r"\frac{1}{")

        self.assertIsNone(result.correct)
        self.assertTrue(result.prediction_parseable)
        self.assertFalse(result.gold_parseable)
        self.assertEqual(result.status, "unparseable_gold")

    def test_missing_reference_is_unscorable(self):
        result = grade_prediction(r"\boxed{12}", None)

        self.assertIsNone(result.correct)
        self.assertEqual(result.status, "missing_gold")

    def test_parse_coverage_distinguishes_valid_expression_from_missing_or_malformed(self):
        self.assertTrue(answer_parseable("12"))
        self.assertFalse(answer_parseable(r"\frac{1}{"))
        self.assertFalse(answer_parseable(None))


if __name__ == "__main__":
    unittest.main()
