import hashlib
import importlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

try:
    model_artifact = importlib.import_module("teacher_reliability.model_artifact")
except ModuleNotFoundError:
    model_artifact = None


QUANTIZATION = {
    "load_in_4bit": True,
    "quant_type": "nf4",
    "double_quant": True,
    "compute_dtype": "float16",
}


def _config_quantization():
    return {
        "quant_method": "bitsandbytes",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "float16",
    }
REVISION = "0123456789abcdef0123456789abcdef01234567"


def _write_tiny_safetensors(path, tensor_key):
    header = json.dumps(
        {tensor_key: {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}
    ).encode("utf-8")
    header += b" " * ((-len(header)) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x01")


def _write_manifest(root):
    files = {
        file.relative_to(root).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in root.rglob("*")
        if file.is_file() and file.name != "artifact_manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "repository": "Qwen/Qwen2.5-Math-7B-Instruct",
        "revision": REVISION,
        "quantization": QUANTIZATION,
        "packed_nf4_state": {"model.safetensors": 1},
        "files": files,
    }
    (root / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _make_fixture(root):
    root.mkdir(parents=True)
    config = {
        "model_type": "qwen2",
        "quantization_config": _config_quantization(),
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_tiny_safetensors(root / "model.safetensors", "layer.weight.quant_state.bitsandbytes__nf4")
    _write_manifest(root)


class ModelArtifactAvailabilityTests(unittest.TestCase):
    def test_artifact_module_is_available_without_vllm(self):
        self.assertIsNotNone(model_artifact, "the model artifact module is missing")
        importlib.import_module("teacher_reliability")
        self.assertNotIn("vllm", __import__("sys").modules)


@unittest.skipIf(model_artifact is None, "model artifact module is not implemented yet")
class ModelArtifactTests(unittest.TestCase):

    def test_cli_requires_an_immutable_revision_sha(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "immutable"):
                model_artifact.main(
                    [
                        "--revision", "main",
                        "--output-dir", str(Path(temporary_directory) / "models" / "artifact"),
                    ]
                )

    def test_validator_accepts_a_checksummed_packed_nf4_fixture(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = Path(temporary_directory) / "model"
            _make_fixture(artifact)

            validated = model_artifact.validate_artifact(artifact)

        self.assertEqual(validated["model_revision"], REVISION)
        self.assertEqual(validated["quantization"], QUANTIZATION)
        self.assertEqual(validated["packed_nf4_state"], {"model.safetensors": 1})
        self.assertEqual(set(validated), {
            "model_revision",
            "aggregate_sha256",
            "files",
            "quantization",
            "packed_nf4_state",
            "repository",
            "schema_version",
        })
        self.assertEqual(set(validated["files"]), {
            "config.json",
            "tokenizer.json",
            "model.safetensors",
        })
        self.assertEqual(len(validated["aggregate_sha256"]), 64)

    def test_validator_rejects_a_file_whose_sha256_changed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = Path(temporary_directory) / "model"
            _make_fixture(artifact)
            (artifact / "tokenizer.json").write_text('{"changed":true}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "checksum"):
                model_artifact.validate_artifact(artifact)

    def test_validator_rejects_an_extra_unmanifested_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = Path(temporary_directory) / "model"
            _make_fixture(artifact)
            (artifact / "unexpected.bin").write_bytes(b"extra")

            with self.assertRaisesRegex(ValueError, "file set"):
                model_artifact.validate_artifact(artifact)

    def test_validator_rejects_nested_manifest_named_file_as_extra(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = Path(temporary_directory) / "model"
            _make_fixture(artifact)
            nested = artifact / "subdir"
            nested.mkdir()
            (nested / "artifact_manifest.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "file set"):
                model_artifact.validate_artifact(artifact)

    def test_validator_rejects_altered_nf4_quantization_even_with_updated_checksum(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = Path(temporary_directory) / "model"
            _make_fixture(artifact)
            config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
            config["quantization_config"]["bnb_4bit_quant_type"] = "fp4"
            (artifact / "config.json").write_text(json.dumps(config), encoding="utf-8")
            _write_manifest(artifact)

            with self.assertRaisesRegex(ValueError, "quantization"):
                model_artifact.validate_artifact(artifact)

    def test_export_uses_pinned_teacher_loader_and_writes_a_validated_artifact(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            models_directory = Path(temporary_directory) / "models"
            output_directory = models_directory / "qwen-nf4"
            loader_calls = []

            class Tokenizer:
                def save_pretrained(self, directory):
                    Path(directory, "tokenizer.json").write_text("{}", encoding="utf-8")

            class Model:
                _teacher_reliability_load_info = {
                    "quantization": {
                        "load_in_4bit": True,
                        "quant_type": "nf4",
                        "double_quant": True,
                        "compute_dtype": "float16",
                    }
                }

                def save_pretrained(self, directory, safe_serialization=False):
                    self_safe_serialization.append(safe_serialization)
                    Path(directory, "config.json").write_text(
                        json.dumps(
                            {
                                "model_type": "qwen2",
                                "quantization_config": _config_quantization(),
                            }
                        ),
                        encoding="utf-8",
                    )
                    _write_tiny_safetensors(
                        Path(directory, "model.safetensors"),
                        "layer.weight.quant_state.bitsandbytes__nf4",
                    )

            self_safe_serialization = []

            def loader(revision, repository, local_files_only):
                loader_calls.append((revision, repository, local_files_only))
                return Tokenizer(), Model()

            result = model_artifact.export_model_artifact(
                output_directory,
                REVISION,
                loader=loader,
                models_directory=models_directory,
            )

            self.assertEqual(result, output_directory)
            self.assertEqual(
                loader_calls,
                [(REVISION, "Qwen/Qwen2.5-Math-7B-Instruct", False)],
            )
            self.assertEqual(self_safe_serialization, [True])
            self.assertTrue((output_directory / "artifact_manifest.json").is_file())
            self.assertEqual(
                model_artifact.validate_artifact(output_directory)["model_revision"], REVISION
            )

    def test_export_rejects_mutable_revision_before_loading_teacher(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            models_directory = Path(temporary_directory) / "models"
            loader_calls = []

            with self.assertRaisesRegex(ValueError, "immutable"):
                model_artifact.export_model_artifact(
                    models_directory / "bad-revision",
                    "main",
                    loader=lambda *args, **kwargs: loader_calls.append((args, kwargs)),
                    models_directory=models_directory,
                )

            self.assertEqual(loader_calls, [])

    def test_export_rejects_existing_output_directory_without_overwriting_it(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            models_directory = Path(temporary_directory) / "models"
            output_directory = models_directory / "existing"
            output_directory.mkdir(parents=True)
            marker = output_directory / "keep.txt"
            marker.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "new"):
                model_artifact.export_model_artifact(
                    output_directory,
                    REVISION,
                    loader=lambda *args, **kwargs: self.fail("must reject before loading"),
                    models_directory=models_directory,
                )

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
