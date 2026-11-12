"""eqV3 batched potential adapter for batchASE.

eqV3 is distributed in the Equiformer V3 FairChem source tree rather than the
public fairchem-core wheels.  Imports are intentionally local so installing
batchASE does not make FairChem a requirement for the other backends.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from .base import BatchPotential


DEFAULT_EQV3_MODEL = "/batchase/models/eqv3_omat24_direct.pt"


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


class EqV3BatchBackend:
    """FairChem eqV3 adapter implementing :class:`BatchPotential`."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        seed: int = 0,
        calculator=None,
        enable_cueq: bool = False,
        use_fasteq: bool = False,
        compile_mode: Optional[str] = None,
        **kwargs,
    ):
        self.kind = "eqv3"
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.checkpoint_path = str(
            Path(checkpoint_path or DEFAULT_EQV3_MODEL).expanduser()
        )
        if calculator is None:
            from fairchem.core.common.relaxation.ase_utils import OCPCalculator

            cpu = self.device.type == "cpu"
            checkpoint = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=False
            )
            config_yml = checkpoint["config"]
            dataset = config_yml.get("dataset")
            if isinstance(dataset, dict) and "train" not in dataset:
                # The config_yml API expects a split mapping, whereas this
                # checkpoint stores its training dataset configuration directly.
                config_yml["dataset"] = {"train": dataset}
            # OCPCalculator otherwise passes local_rank=0 to its trainer,
            # which would place every worker on cuda:0.
            config_yml["local_rank"] = self.device.index or 0
            if self.device.index is not None:
                torch.cuda.set_device(self.device)
            state_dict = checkpoint["state_dict"]
            for prefix in ("_orig_mod.module.", "od.module."):
                if state_dict and all(key.startswith(prefix) for key in state_dict):
                    checkpoint["state_dict"] = {
                        key.removeprefix(prefix): value
                        for key, value in state_dict.items()
                    }
                    break

            prepared_checkpoint = checkpoint

            class CheckpointOCPCalculator(OCPCalculator):
                def load_checkpoint(self, checkpoint_path, checkpoint=None):
                    # Reuse the normalized in-memory checkpoint during __init__.
                    return super().load_checkpoint(
                        checkpoint_path, checkpoint=prepared_checkpoint
                    )

            calculator = CheckpointOCPCalculator(
                config_yml=config_yml,
                checkpoint_path=self.checkpoint_path,
                cpu=cpu,
                seed=seed,
                **kwargs,
            )
        self.calculator = calculator
        model_config = calculator.config["model"]
        self.r_max = float(model_config.get("max_radius", 6.0))
        self.eqv3_time = 0.0
        self.graph_time = 0.0
        self.forward_calls = 0

    @property
    def mace_time(self) -> float:
        """Worker-compatible cumulative model inference time."""
        return self.eqv3_time

    def build_inputs(self, gbatch) -> dict:
        return {"atoms_list": _atoms_from_gbatch(gbatch)}

    def build_inputs_from_atoms(self, atoms_list: Sequence) -> dict:
        return {"atoms_list": list(atoms_list)}

    def _forward(self, inputs: dict, compute_stress: bool) -> dict[str, torch.Tensor]:
        from fairchem.core.datasets import data_list_collater

        data = [self.calculator.a2g.convert(atoms) for atoms in inputs["atoms_list"]]
        batch = data_list_collater(data, otf_graph=True)
        predictions = self.calculator.trainer.predict(
            batch, per_image=False, disable_tqdm=True
        )
        results = {
            "energy": predictions["energy"]
            .detach()
            .reshape(-1, 1)
            .to(torch.float64),
            "forces": predictions["forces"]
            .detach()
            .reshape(-1, 3)
            .to(torch.float64),
        }
        if compute_stress:
            results["stress"] = (
                predictions["stress"].detach().reshape(-1, 3, 3).to(torch.float64)
            )
        return results

    def predict(self, gbatch, compute_stress: bool = False) -> dict[str, torch.Tensor]:
        t0 = time.perf_counter()
        inputs = self.build_inputs(gbatch)
        t1 = time.perf_counter()
        results = self._forward(inputs, compute_stress)
        t2 = time.perf_counter()
        self.graph_time += t1 - t0
        self.eqv3_time += t2 - t1
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
        self.eqv3_time += t2 - t1
        self.forward_calls += 1
        return results
