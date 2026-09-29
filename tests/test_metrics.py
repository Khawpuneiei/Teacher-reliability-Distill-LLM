import unittest

from teacher_reliability.metrics import (
    binary_roc_auc,
    calibration_bins,
    expected_calibration_error,
)


class CalibrationMetricTests(unittest.TestCase):
    def test_ece_weights_each_equal_width_bin_by_its_row_count(self):
        confidences = [0.9, 0.7, 0.2, 0.1]
        outcomes = [1, 0, 1, 0]

        observed = expected_calibration_error(confidences, outcomes, n_bins=2)

        self.assertAlmostEqual(observed, 0.325)

    def test_ece_includes_confidence_one_in_the_last_bin(self):
        observed = expected_calibration_error([0.0, 1.0], [1, 1], n_bins=2)

        self.assertAlmostEqual(observed, 0.5)

    def test_ece_rejects_confidence_outside_unit_interval(self):
        with self.assertRaises(ValueError):
            expected_calibration_error([1.1], [1], n_bins=2)

    def test_calibration_bins_are_auditable_and_final_bin_includes_one(self):
        observed = calibration_bins(
            [0.0, 0.25, 0.5, 0.75, 1.0], [0, 0, 1, 1, 1], n_bins=2
        )

        self.assertEqual(observed[0]["count"], 2)
        self.assertAlmostEqual(observed[0]["mean_confidence"], 0.125)
        self.assertEqual(observed[0]["accuracy"], 0.0)
        self.assertEqual(observed[1]["count"], 3)
        self.assertAlmostEqual(observed[1]["mean_confidence"], 0.75)
        self.assertEqual(observed[1]["accuracy"], 1.0)

    def test_empty_calibration_bin_has_null_means(self):
        observed = calibration_bins([0.1], [0], n_bins=3)

        self.assertEqual(observed[2]["count"], 0)
        self.assertIsNone(observed[2]["mean_confidence"])
        self.assertIsNone(observed[2]["accuracy"])


class BinaryRocAucTests(unittest.TestCase):
    def test_auc_counts_a_tied_positive_negative_pair_as_half(self):
        labels = [1, 0, 1, 0]
        confidence_scores = [0.9, 0.9, 0.2, 0.1]

        observed = binary_roc_auc(labels, confidence_scores)

        self.assertAlmostEqual(observed, 0.625)

    def test_auc_is_undefined_when_the_rows_have_one_class(self):
        self.assertIsNone(binary_roc_auc([1, 1, 1], [0.1, 0.2, 0.3]))

    def test_auc_rejects_different_label_and_score_counts(self):
        with self.assertRaises(ValueError):
            binary_roc_auc([1, 0], [0.5])


if __name__ == "__main__":
    unittest.main()
