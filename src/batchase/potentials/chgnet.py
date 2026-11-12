"""CHGNet batched potential adapter for batchASE."""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np
import torch

from .base import BatchPotential


_GPA_TO_EV_PER_A3 = 0.006241509125883258


def _atoms_from_gbatch(gbatch) -> list:
    from ase import Atoms

    natoms = gbatch["natoms"].detach().cpu().tolist()
    positions = gbatch["pos"].detach().cpu().numpy()
    numbers = gbatch["atomic_numbers"].detach().cpu().numpy()
    cells = gbatch["cell"].detach().cpu().numpy().reshape(-1, 3, 3)
    atoms_list = []
    start = 0
    for index, count in enumerate(natoms):
        stop = start + count
        atoms_list.append(
            Atoms(
                numbers=numbers[start:stop],
                positions=positions[start:stop],
                cell=cells[index],
                pbc=True,
            )
        )
        start = stop
    return atoms_list


class CHGNetBatchBackend:
    """CHGNet potential adapter implementing :class:`BatchPotential`."""

    def __init__(
        self,
        model=None,
        model_name: str = "0.3.0",
        device: str = "cuda",
        batch_size: Optional[int] = None,
        enable_cueq: bool = False,
        use_fasteq: bool = False,
        compile_mode: Optional[str] = None,
        **kwargs,
    ):
        self.kind = "chgnet"
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.batch_size = batch_size
        if model is None or isinstance(model, (str, bytes)):
            from chgnet.model import CHGNet

            model = CHGNet.load(
                model_name=str(model or model_name),
                use_device=str(self.device),
                **kwargs,
            )
        self.model = model
        self.model.to(self.device)
        self.model.eval()
        graph_converter = getattr(self.model, "graph_converter", None)
        cutoff = getattr(graph_converter, "atom_graph_cutoff", None)
        self.r_max = float(cutoff if cutoff is not None else 6.0)
        self.chgnet_time = 0.0
        self.graph_time = 0.0
        self.forward_calls = 0

    @property
    def mace_time(self) -> float:
        """Worker-compatible cumulative model inference time."""
        return self.chgnet_time

    def build_inputs(self, gbatch) -> dict:
        return self.build_inputs_from_atoms(_atoms_from_gbatch(gbatch))

    def build_inputs_from_atoms(self, atoms_list: Sequence) -> dict:
        from pymatgen.io.ase import AseAtomsAdaptor

        structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in atoms_list]
        return {"structures": structures}

    def _forward(self, inputs: dict, compute_stress: bool) -> dict[str, torch.Tensor]:
        structures = inputs["structures"]
        predictions = self.model.predict_structure(
            structures,
            task="efs" if compute_stress else "ef",
            batch_size=self.batch_size or max(1, len(structures)),
        )
        if isinstance(predictions, dict):
            predictions = [predictions]
        counts = [len(structure) for structure in structures]
        energies = torch.tensor(
            [float(prediction["e"]) * count for prediction, count in zip(predictions, counts)],
            dtype=torch.float64,
            device=self.device,
        ).reshape(-1, 1)
        forces = torch.as_tensor(
            np.concatenate([np.asarray(prediction["f"]) for prediction in predictions]),
            dtype=torch.float64,
            device=self.device,
        ).reshape(-1, 3)
        results = {"energy": energies, "forces": forces}
        if compute_stress:
            stress = torch.as_tensor(
                np.stack([np.asarray(prediction["s"]) for prediction in predictions]),
                dtype=torch.float64,
                device=self.device,
            ).reshape(-1, 3, 3)
            results["stress"] = stress * _GPA_TO_EV_PER_A3
        return results

    def predict(self, gbatch, compute_stress: bool = False) -> dict[str, torch.Tensor]:
        t0 = time.perf_counter()
        inputs = self.build_inputs(gbatch)
        t1 = time.perf_counter()
        results = self._forward(inputs, compute_stress)
        t2 = time.perf_counter()
        self.graph_time += t1 - t0
        self.chgnet_time += t2 - t1
        self.forward_calls += 1
        return results

    def predict_from_atoms(
        self, atoms_list: Sequence, compute_stress: bool = False
    ) -> dict[str, torch.Tensor]:
        t0 = time.perf_counter()
        inputs = self.build_inputs_from_atoms(atoms_list)
        t1 = time.perf_counter()
        results = self._forward(inputs, compute_stress)
        t2 = time.perf_counter()
        self.graph_time += t1 - t0
        self.chgnet_time += t2 - t1
        self.forward_calls += 1
        return results
