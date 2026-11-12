"""Tests for backend registry, supported backend creation, and fail-fast rejection (Task 2.5).

Verifies:
1. create_backend("mace") creates a functional MACEBatchBackend.
2. create_backend() on unimplemented placeholder backends (sevennet, matgl)
   immediately raises NotImplementedError with clear guidance.
3. create_backend() on unknown backend raises ValueError.
4. Direct instantiation of stub classes raises NotImplementedError.
5. Model identifier grammar family[:spec]: parsing, checkpoint resolution,
   preflight validation, and loader routing (mace_off / mace_mp).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from batchase.potentials import (
    SUPPORTED_BACKENDS,
    UNIMPLEMENTED_BACKENDS,
    create_backend,
    parse_model_id,
    resolve_checkpoint,
    validate_model_id,
    MACEBatchBackend,
    SevenNetBatchBackend,
    CHGNetBatchBackend,
    MatRISBatchBackend,
    EqV3BatchBackend,
)


class MockMACEModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.r_max = torch.tensor(4.5)


class MockMACECalculator:
    def __init__(self):
        self.models = [MockMACEModel()]
        self.z_table = [1, 6, 8]


class TestBackendRegistry(unittest.TestCase):
    def test_supported_backend_creation(self):
        """create_backend('mace') returns a MACEBatchBackend instance."""
        calc = MockMACECalculator()
        backend = create_backend("mace", calculator=calc, device="cpu")
        self.assertIsInstance(backend, MACEBatchBackend)
        self.assertEqual(backend.kind, "mace")

    def test_unimplemented_backends_fail_fast(self):
        """create_backend on placeholder backends immediately raises NotImplementedError."""
        for name in ("sevennet", "matgl"):
            with self.subTest(backend=name):
                with self.assertRaises(NotImplementedError) as ctx:
                    create_backend(name)
                err_msg = str(ctx.exception)
                self.assertIn(name, err_msg)
                self.assertIn("mace", err_msg)

    def test_unknown_backend_raises_value_error(self):
        """create_backend on unrecognized backend name raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            create_backend("totally_unknown_mlip_model")
        self.assertIn("totally_unknown_mlip_model", str(ctx.exception))
        self.assertIn("Supported", str(ctx.exception))

    def test_stub_classes_direct_instantiation_raises(self):
        """Direct instantiation of placeholder classes raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            SevenNetBatchBackend()


class TestParseModelId(unittest.TestCase):
    def test_plain_family(self):
        self.assertEqual(parse_model_id("mace"), ("mace", None))
        self.assertEqual(parse_model_id("mock"), ("mock", None))

    def test_family_with_spec(self):
        self.assertEqual(parse_model_id("mace_mp:medium-mpa-0"), ("mace_mp", "medium-mpa-0"))
        self.assertEqual(parse_model_id("mace_off:small"), ("mace_off", "small"))

    def test_family_case_insensitive(self):
        self.assertEqual(parse_model_id("MACE_MP:X"), ("mace_mp", "X"))

    def test_empty_spec_yields_none(self):
        self.assertEqual(parse_model_id("mace_mp:"), ("mace_mp", None))

    def test_empty_identifier_raises(self):
        with self.assertRaises(ValueError):
            parse_model_id("")
        with self.assertRaises(ValueError):
            parse_model_id(None)

    def test_empty_family_raises(self):
        with self.assertRaises(ValueError):
            parse_model_id(":some_spec")


class TestResolveCheckpoint(unittest.TestCase):
    def test_resolves_direct_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "model.model"
            ckpt.write_bytes(b"x")
            self.assertEqual(resolve_checkpoint(str(ckpt)), ckpt)

    def test_resolves_in_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "mace-mpa-0-medium.model"
            ckpt.write_bytes(b"x")
            self.assertEqual(resolve_checkpoint("mace-mpa-0-medium.model", cache_dir=Path(tmp)), ckpt)

    def test_missing_spec_returns_none(self):
        self.assertIsNone(resolve_checkpoint("no-such-file-xyz.model"))

    def test_empty_spec_returns_none(self):
        self.assertIsNone(resolve_checkpoint(""))
        self.assertIsNone(resolve_checkpoint(None))


class TestValidateModelId(unittest.TestCase):
    def test_mock_passes(self):
        self.assertIsNone(validate_model_id("mock"))

    def test_placeholder_raises(self):
        for name in UNIMPLEMENTED_BACKENDS:
            with self.subTest(family=name):
                with self.assertRaises(NotImplementedError):
                    validate_model_id(name)

    def test_unknown_family_raises(self):
        with self.assertRaises(ValueError):
            validate_model_id("totally_unknown_mlip_model")

    def test_legacy_mace_without_spec_passes(self):
        self.assertIsNone(validate_model_id("mace"))

    def test_resolved_local_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "mace-mpa-0-medium.model"
            ckpt.write_bytes(b"x")
            self.assertEqual(validate_model_id(f"mace_mp:{ckpt}"), ckpt)

    def test_builtin_name_passes_without_file(self):
        # "medium-mpa-0" is a built-in mace_mp name: no local file needed.
        self.assertIsNone(validate_model_id("mace_mp:medium-mpa-0"))

    def test_missing_checkpoint_raises(self):
        with self.assertRaises(FileNotFoundError):
            validate_model_id("mace_mp:definitely-not-a-model-xyz.model")


class TestLoaderRouting(unittest.TestCase):
    def _fake_loader(self, calls):
        def _loader(**kwargs):
            calls.append(kwargs)
            return MockMACECalculator()

        return _loader

    def test_mace_mp_routes_to_mace_mp_loader(self):
        calls = []
        with patch("mace.calculators.mace_mp", side_effect=self._fake_loader(calls)):
            backend = create_backend("mace_mp:medium-mpa-0", device="cpu")
        self.assertIsInstance(backend, MACEBatchBackend)
        self.assertEqual(backend.loader_hint, "mp")
        self.assertEqual(calls[0]["model"], "medium-mpa-0")

    def test_mace_mp_resolved_path_forwarded_as_str(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "mace-mpa-0-medium.model"
            ckpt.write_bytes(b"x")
            with patch("mace.calculators.mace_mp", side_effect=self._fake_loader(calls)):
                backend = create_backend(f"mace_mp:{ckpt}", device="cpu")
        self.assertIsInstance(backend, MACEBatchBackend)
        self.assertEqual(backend.loader_hint, "mp")
        self.assertEqual(calls[0]["model"], str(ckpt))

    def test_mace_off_routes_to_mace_off_loader(self):
        calls = []
        with patch("mace.calculators.mace_off", side_effect=self._fake_loader(calls)):
            backend = create_backend("mace_off:MACE-OFF23_small.model", device="cpu")
        self.assertIsInstance(backend, MACEBatchBackend)
        self.assertEqual(backend.loader_hint, "off")

    def test_legacy_mace_defaults_to_off_small(self):
        calls = []
        with patch("mace.calculators.mace_off", side_effect=self._fake_loader(calls)):
            backend = create_backend("mace", device="cpu")
        self.assertEqual(backend.loader_hint, "off")
        self.assertEqual(calls[0]["model"], "small")

    def test_explicit_model_kwarg_overrides_spec(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "other.model"
            ckpt.write_bytes(b"x")
            with patch("mace.calculators.mace_off", side_effect=self._fake_loader(calls)):
                backend = create_backend("mace", model=str(ckpt), device="cpu")
        self.assertEqual(backend.loader_hint, "off")
        self.assertEqual(calls[0]["model"], str(ckpt))

    def test_calculator_short_circuits_loading(self):
        calc = MagicMock()
        calc.models = [MockMACEModel()]
        calc.z_table = [1, 6, 8]
        backend = create_backend("mace_mp:mace-mpa-0-medium.model", calculator=calc, device="cpu")
        self.assertIs(backend.calculator, calc)
        self.assertEqual(backend.loader_hint, "mp")


class MockMatRISModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.config = {"pairwise_cutoff": 6.0, "three_body_cutoff": 4.0}
        self.is_intensive = True
        self.calls = []

    def forward(self, graphs, task, **kwargs):
        self.calls.append((task, kwargs))
        prediction = {
            "e": torch.tensor([2.0, 3.0], dtype=torch.float32),
            "atoms_per_graph": torch.tensor([2, 1], dtype=torch.int32),
            "f": [
                torch.arange(6, dtype=torch.float32).reshape(2, 3),
                torch.arange(3, dtype=torch.float32).reshape(1, 3),
            ],
        }
        if "s" in task:
            prediction["s"] = [
                torch.eye(3, dtype=torch.float32) * 160.21766208,
                torch.eye(3, dtype=torch.float32) * 320.43532416,
            ]
        return prediction


class TestMatRISBackend(unittest.TestCase):
    def setUp(self):
        self.model = MockMatRISModel()
        self.backend = create_backend(
            "matris",
            model=self.model,
            device="cpu",
            mode="torch",
            enable_cueq=True,
            use_fasteq=True,
            compile_mode=None,
        )
        self.inputs = {
            "graphs": [object(), object()],
            "pbc": torch.tensor([[True, True, True], [False, False, False]]),
        }

    def test_factory_and_standardized_outputs(self):
        self.assertIsInstance(self.backend, MatRISBatchBackend)
        self.assertEqual(self.backend.kind, "matris")
        self.assertEqual(self.backend.r_max, 6.0)
        results = self.backend._forward(self.inputs, compute_stress=True)
        self.assertEqual(self.model.calls[-1][0], "efs")
        self.assertEqual(results["energy"].shape, (2, 1))
        self.assertEqual(results["forces"].shape, (3, 3))
        self.assertEqual(results["stress"].shape, (2, 3, 3))
        for value in results.values():
            self.assertEqual(value.dtype, torch.float64)
        torch.testing.assert_close(
            results["energy"], torch.tensor([[4.0], [3.0]], dtype=torch.float64)
        )
        torch.testing.assert_close(
            results["stress"][0],
            torch.eye(3, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            results["stress"][1], torch.zeros((3, 3), dtype=torch.float64)
        )

    def test_force_only_prediction_omits_stress(self):
        results = self.backend._forward(self.inputs, compute_stress=False)
        self.assertEqual(self.model.calls[-1][0], "ef")
        self.assertNotIn("stress", results)

    def test_matris_model_validation(self):
        self.assertIsNone(validate_model_id("matris"))
        self.assertIsNone(validate_model_id("matris:matris_4m_matpes_pbev2"))
        self.assertIsNone(validate_model_id("matris:https://example.com/model.pth.tar"))
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "model.pth.tar"
            ckpt.write_bytes(b"checkpoint")
            self.assertEqual(validate_model_id(f"matris:{ckpt}"), ckpt)
        with self.assertRaises(FileNotFoundError):
            validate_model_id("matris:definitely-missing-model.pth.tar")


if __name__ == "__main__":
    unittest.main()
