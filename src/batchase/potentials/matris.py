"""MatRIS batched potential adapter for batchASE."""

from __future__ import annotations

import logging
import os
import time
from typing import Sequence

import numpy as np
import torch
from ase import units

from .base import BatchPotential

logger = logging.getLogger("batchase.potentials.matris")

DEFAULT_MATRIS_MODEL = "matris_4m_matpes_pbev2"


class MatRISBatchBackend:
    """Batched MatRIS inference implementing the :class:`BatchPotential` protocol.

    Graph construction follows MatRIS's native TorchSim adapter, but consumes
    batchASE graphs directly and therefore does not require ``torch-sim``.
    MatRIS currently uses float32 model weights; standardized outputs are
    returned in float64, with total energies and stresses in ASE units.
    """

    def __init__(
        self,
        model=None,
        model_name: str | None = None,
        device: str = "cuda",
        mode: str = "fast",
        activation_checkpoint: bool = True,
        tf32: bool = False,
        **kwargs,
    ):
        # Worker forwards these MACE-specific options to every backend.
        kwargs.pop("enable_cueq", None)
        kwargs.pop("use_fasteq", None)
        kwargs.pop("compile_mode", None)
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(f"Unexpected MatRIS backend argument: {unexpected!r}")
        if mode not in ("fast", "torch"):
            raise ValueError(f"Unknown MatRIS mode {mode!r}; choose 'fast' or 'torch'.")
        if model is not None and model_name is not None:
            raise ValueError("Pass only one of 'model' and 'model_name'.")
        if model_name == "default":
            model_name = None

        self.kind = "matris"
        self.mode = mode
        self.activation_checkpoint = activation_checkpoint
        self.tf32 = tf32
        self.dtype = torch.float32
        self.default_dtype = "float32"
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        model = model if model is not None else model_name
        if model is None:
            model = DEFAULT_MATRIS_MODEL
        if isinstance(model, (str, os.PathLike)):
            from matris.model import MatRIS

            model = MatRIS.load(model_name=os.fspath(model), device=str(self.device))
        self.model = model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        parameter = next(self.model.parameters())
        if parameter.dtype != self.dtype:
            raise ValueError("MatRIS batch inference currently requires float32 weights.")
        self.device = parameter.device
        self.r_max = float(self.model.config["pairwise_cutoff"])
        self.line_cutoff = float(self.model.config["three_body_cutoff"])

        self.matris_time = 0.0
        # The engine's shared metrics still use this historical field name.
        self.mace_time = 0.0
        self.graph_time = 0.0
        self.forward_calls = 0
        logger.info(
            "MatRISBatchBackend initialized: device=%s dtype=%s r_max=%.2f mode=%s",
            self.device,
            self.dtype,
            self.r_max,
            self.mode,
        )

    def _build_graph(self, index: int, atomic_numbers, positions, cell, pbc):
        """Build one native MatRIS RadiusGraph from Cartesian geometry."""
        from matris.graph import RadiusGraph

        periodic_dims = int(pbc.sum().item())
        if periodic_dims == 0:
            positions = positions - positions.mean(dim=0)
            length = 2 * (positions.abs().max() + self.r_max)
            cell = torch.eye(3, device=self.device, dtype=positions.dtype) * length
        elif periodic_dims < 3:
            missing = (cell.norm(dim=1) == 0) & ~pbc
            if missing.any():
                basis = torch.linalg.svd(cell).Vh
                cell = cell.clone()
                length = 2 * (positions.abs().max() + self.r_max)
                cell[missing] = basis[-int(missing.sum()) :] * length

        if self.device.type == "cuda" and self.mode == "fast":
            from matris.model.op.graph import build_graph_tensors_gpu, gpu_neighbor_list

            edges, ptr, images, distances = gpu_neighbor_list(
                positions, cell, self.r_max, device=self.device, pbc=pbc
            )
            tensors = build_graph_tensors_gpu(
                edges, ptr, images, distances, line_cutoff=self.line_cutoff
            )
        else:
            from ase.neighborlist import primitive_neighbor_list
            from matris.graph.converter import build_graph_tensors

            centers, neighbors, images, distances = primitive_neighbor_list(
                "ijSd",
                pbc.cpu().numpy(),
                cell.cpu().numpy(),
                positions.cpu().numpy(),
                self.r_max + 1e-8,
            )
            tensors = build_graph_tensors(
                len(positions),
                centers,
                neighbors,
                images,
                distances,
                line_cutoff=self.line_cutoff,
            )

        atom_graph, d2u, u2d, line_graph = tensors
        return RadiusGraph(
            graph_id=index,
            mp_id=None,
            composition=None,
            atomic_number=atomic_numbers.to(torch.int32, copy=True),
            atom_frac_coord=torch.linalg.solve(cell.mT, positions.mT).mT.to(self.dtype),
            lattice=cell.to(self.dtype),
            neighbor_image=torch.as_tensor(
                images, device=self.device, dtype=self.dtype
            ),
            atom_graph=atom_graph.to(self.device),
            directed2undirected=d2u.to(self.device),
            undirected2directed=u2d.to(self.device),
            line_graph=line_graph.to(self.device),
            atom_graph_cutoff=self.r_max,
            line_graph_cutoff=self.line_cutoff,
        )

    def _build_inputs(self, pos, cell, natoms, atomic_numbers, batch, ptr, pbc=None):
        del batch, ptr  # contiguous atom ranges are sufficient for MatRIS graphs
        pos = pos.detach().to(self.device, torch.float64)
        cell = cell.detach().to(self.device, torch.float64).view(-1, 3, 3)
        atomic_numbers = atomic_numbers.detach().to(self.device)
        counts = [int(value) for value in natoms.detach().cpu().tolist()]
        if not counts:
            raise ValueError("MatRIS prediction requires at least one structure.")
        if sum(counts) != len(pos):
            raise ValueError("MatRIS atom counts do not match the positions tensor.")
        if len(cell) != len(counts):
            raise ValueError("MatRIS cell count does not match the structure count.")

        if pbc is None:
            pbc = torch.ones((len(counts), 3), dtype=torch.bool, device=self.device)
        else:
            pbc = pbc.detach().to(self.device, torch.bool).view(-1, 3)
        if len(pbc) != len(counts):
            raise ValueError("MatRIS PBC count does not match the structure count.")

        graphs = []
        start = 0
        for index, count in enumerate(counts):
            stop = start + count
            graph = self._build_graph(
                index,
                atomic_numbers[start:stop],
                pos[start:stop],
                cell[index],
                pbc[index],
            )
            graphs.append(graph)
            start = stop
        return {"graphs": graphs, "pbc": pbc}

    def build_inputs(self, gbatch) -> dict:
        return self._build_inputs(
            pos=gbatch["pos"],
            cell=gbatch["cell"],
            natoms=gbatch["natoms"],
            atomic_numbers=gbatch["atomic_numbers"],
            batch=gbatch["batch"],
            ptr=gbatch["ptr"],
            pbc=gbatch["pbc"] if "pbc" in gbatch else None,
        )

    def build_inputs_from_atoms(self, atoms_list: Sequence) -> dict:
        if not atoms_list:
            raise ValueError("MatRIS prediction requires at least one structure.")
        natoms = torch.tensor([len(atoms) for atoms in atoms_list], dtype=torch.long)
        positions = torch.from_numpy(
            np.concatenate([np.asarray(atoms.get_positions()) for atoms in atoms_list])
        )
        cell = torch.from_numpy(
            np.stack(
                [np.asarray(atoms.get_cell(complete=True)) for atoms in atoms_list]
            )
        )
        atomic_numbers = torch.from_numpy(
            np.concatenate(
                [np.asarray(atoms.get_atomic_numbers()) for atoms in atoms_list]
            )
        )
        batch = torch.repeat_interleave(
            torch.arange(len(atoms_list), dtype=torch.long), natoms
        )
        ptr = torch.cat([torch.tensor([0], dtype=torch.long), natoms.cumsum(0)])
        pbc = torch.from_numpy(np.stack([np.asarray(atoms.pbc) for atoms in atoms_list]))
        return self._build_inputs(
            positions, cell, natoms, atomic_numbers, batch, ptr, pbc
        )

    def _forward(self, inputs: dict, compute_stress: bool) -> dict[str, torch.Tensor]:
        prediction = self.model(
            inputs["graphs"],
            task="efs" if compute_stress else "ef",
            is_training=False,
            mode=self.mode,
            activation_checkpoint=self.activation_checkpoint,
            tf32=self.tf32,
        )
        energy = prediction["e"]
        if self.model.is_intensive:
            energy = energy * prediction["atoms_per_graph"]
        results = {
            "energy": energy.reshape(-1, 1).detach().to(torch.float64),
            "forces": torch.cat(prediction["f"]).detach().to(torch.float64),
        }
        if compute_stress:
            stress = torch.stack(prediction["s"]).detach() * units.GPa
            periodic = inputs["pbc"].any(dim=1)[:, None, None]
            results["stress"] = torch.where(periodic, stress, 0).to(torch.float64)
        return results

    def predict(self, gbatch, compute_stress: bool = False) -> dict[str, torch.Tensor]:
        start = time.perf_counter()
        inputs = self.build_inputs(gbatch)
        graph_done = time.perf_counter()
        results = self._forward(inputs, compute_stress)
        forward_done = time.perf_counter()
        self.graph_time += graph_done - start
        self.matris_time += forward_done - graph_done
        self.mace_time = self.matris_time
        self.forward_calls += 1
        return results

    def predict_from_atoms(
        self, atoms_list: Sequence, compute_stress: bool = False
    ) -> dict[str, torch.Tensor]:
        start = time.perf_counter()
        inputs = self.build_inputs_from_atoms(atoms_list)
        graph_done = time.perf_counter()
        results = self._forward(inputs, compute_stress)
        forward_done = time.perf_counter()
        self.graph_time += graph_done - start
        self.matris_time += forward_done - graph_done
        self.mace_time = self.matris_time
        self.forward_calls += 1
        return results


__all__ = ["DEFAULT_MATRIS_MODEL", "MatRISBatchBackend"]
