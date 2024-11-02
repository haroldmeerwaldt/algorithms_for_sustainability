from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from qiskit_nature.second_q.formats import MoleculeInfo
from scipy.optimize import minimize

from quantum_algorithms.vqe.energy_calculation import GroundStateEnergyCalculation


class DistanceFinder:
    def __init__(
        self,
        iteration_callback: Callable[["IterationInfo"], None] | None = None,
        init_distance: float = 3,  # Angstroms
        atoms: tuple[str, str] = ("H", "H"),
    ):
        self._iteration_callback = iteration_callback
        self._last_atoms_init_coords = (0.0, 0.0, init_distance)
        molecule = MoleculeInfo(atoms, [(0.0, 0.0, 0.0), self._last_atoms_init_coords])
        self._calculation = GroundStateEnergyCalculation(molecule)

    def _minimize_ground_state_energy(self, coords: np.ndarray):
        coords = cast(tuple[float, float, float], tuple(float(c) for c in coords))
        self._calculation.update_last_atoms_coords(coords)
        ground_state_energy = self._calculation.run()
        if self._iteration_callback is not None:
            # the COBYLA method does not support a callback with the evaluated function value, so I put it here
            self._iteration_callback(IterationInfo(coords, ground_state_energy))
        return ground_state_energy

    def run(self):
        result = minimize(
            self._minimize_ground_state_energy, np.array(self._last_atoms_init_coords), method="COBYLA", tol=0.01
        )
        return np.linalg.norm(result.x)


@dataclass
class IterationInfo:
    coords: tuple[float, float, float]
    ground_state_energy: float


if __name__ == "__main__":
    distance_finder = DistanceFinder(iteration_callback=print)
    print(distance_finder.run())
