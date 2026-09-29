import unittest

from teacher_reliability.env_check import build_environment_report


class EnvironmentCheckTests(unittest.TestCase):
    def test_cuda_runtime_and_required_packages_are_reported_as_ready(self):
        report = build_environment_report(
            python_version=(3, 10, 4),
            cuda_available=True,
            cuda_runtime="12.1",
            gpu_name="Test GPU",
            gpu_memory_bytes=8 * 1024**3,
            package_versions={
                "torch": "2.5.1+cu121",
                "transformers": "4.52.4",
                "accelerate": "1.8.1",
                "bitsandbytes": "0.46.1",
                "hf_xet": "1.6.0",
            },
            bitsandbytes_import_error=None,
        )

        self.assertTrue(report["ready"])
        self.assertEqual(report["gpu_name"], "Test GPU")
        self.assertEqual(report["errors"], [])

    def test_cuda_or_bitsandbytes_failures_are_actionable_errors(self):
        report = build_environment_report(
            python_version=(3, 10, 4),
            cuda_available=False,
            cuda_runtime=None,
            gpu_name=None,
            gpu_memory_bytes=None,
            package_versions={"torch": "2.5.1", "bitsandbytes": "0.46.1"},
            bitsandbytes_import_error="DLL load failed",
        )

        self.assertFalse(report["ready"])
        self.assertIn("CUDA-enabled PyTorch", " ".join(report["errors"]))
        self.assertIn("bitsandbytes", " ".join(report["errors"]))

    def test_less_than_six_gib_gpu_is_a_warning_not_a_false_ready_failure(self):
        report = build_environment_report(
            python_version=(3, 10, 4),
            cuda_available=True,
            cuda_runtime="12.1",
            gpu_name="Small GPU",
            gpu_memory_bytes=5 * 1024**3,
            package_versions={
                "torch": "2.5.1+cu121",
                "transformers": "4.52.4",
                "accelerate": "1.8.1",
                "bitsandbytes": "0.46.1",
                "hf_xet": "1.6.0",
            },
            bitsandbytes_import_error=None,
        )

        self.assertTrue(report["ready"])
        self.assertTrue(any("VRAM" in warning for warning in report["warnings"]))

    def test_missing_hf_xet_is_reported_as_a_setup_error(self):
        report = build_environment_report(
            python_version=(3, 10, 4),
            cuda_available=True,
            cuda_runtime="12.1",
            gpu_name="Test GPU",
            gpu_memory_bytes=8 * 1024**3,
            package_versions={
                "torch": "2.5.1+cu121",
                "transformers": "4.52.4",
                "accelerate": "1.8.1",
                "bitsandbytes": "0.46.1",
            },
            bitsandbytes_import_error=None,
        )

        self.assertFalse(report["ready"])
        self.assertIn("hf_xet", " ".join(report["errors"]))


if __name__ == "__main__":
    unittest.main()
