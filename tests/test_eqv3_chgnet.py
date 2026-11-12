"""Unit tests for the CHGNet and eqV3 backend adapters."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from batchase.potentials import (
    CHGNetBatchBackend,
    EqV3BatchBackend,
    create_backend,
    validate_model_id,
)


class MockCHGNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.graph_converter = SimpleNamespace(atom_graph_cutoff=6.0)

    def predict_structure(self, structures, task, batch_size):
        return [
            {
                "e": np.asarray(1.5),
                "f": np.arange(len(structure) * 3, dtype=np.float32).reshape(-1, 3),
                "s": np.eye(3, dtype=np.float32) * 160.21766208,
            }
            for structure in structures
        ]


class TestCHGNetBackend(unittest.TestCase):
    def test_forward_standardizes_units_and_shapes(self):
        backend = create_backend("chgnet", model=MockCHGNet(), device="cpu")
        structures = [[0, 0], [0]]
        result = backend._forward({"structures": structures}, compute_stress=True)
        self.assertIsInstance(backend, CHGNetBatchBackend)
        self.assertEqual(result["energy"].shape, (2, 1))
        self.assertEqual(result["forces"].shape, (3, 3))
        self.assertEqual(result["stress"].shape, (2, 3, 3))
        torch.testing.assert_close(
            result["energy"], torch.tensor([[3.0], [1.5]], dtype=torch.float64)
        )
        torch.testing.assert_close(
            result["stress"], torch.ones(2, 3, 3, dtype=torch.float64) * torch.eye(3).double(),
            rtol=1e-5,
            atol=1e-5,
        )
        self.assertTrue(all(value.dtype == torch.float64 for value in result.values()))


class TestEqV3Backend(unittest.TestCase):
    def test_factory_accepts_injected_calculator(self):
        calculator = SimpleNamespace(config={"model": {"max_radius": 6.0}})
        backend = create_backend("eqv3", calculator=calculator, device="cpu")
        self.assertIsInstance(backend, EqV3BatchBackend)
        self.assertEqual(backend.kind, "eqv3")
        self.assertEqual(backend.r_max, 6.0)

    def test_checkpoint_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "eqv3.pt"
            checkpoint.write_bytes(b"checkpoint")
            self.assertEqual(validate_model_id(f"eqv3:{checkpoint}"), checkpoint)
        with self.assertRaises(FileNotFoundError):
            validate_model_id("eqv3:definitely-missing-eqv3.pt")

    def test_chgnet_validation(self):
        self.assertIsNone(validate_model_id("chgnet"))
        self.assertIsNone(validate_model_id("chgnet:0.3.0"))


if __name__ == "__main__":
    unittest.main()
