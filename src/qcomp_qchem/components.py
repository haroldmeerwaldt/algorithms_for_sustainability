from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from qiskit.algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit_nature.second_q.algorithms import VQEUCCFactory, GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats import MoleculeInfo
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer


@dataclass
class AtomInfo:
    symbol: str
    coords: tuple[float, float, float]


class GroundStateEnergyCalculation:
    def __init__(self):
        # self._molecule = MoleculeInfo(["C", "H", "H", "H"], [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (-0.1, 0.0, 0.0), (0.0, 0.1, 0.0)], charge=0, multiplicity=1)
        self._molecule = MoleculeInfo(["H"], [(0.0, 0.0, 0.0)], charge=0, multiplicity=1)

        converter = JordanWignerMapper()  # also ParityMapper, BravyiKitaevMapper
        self._qubit_converter = QubitConverter(converter, two_qubit_reduction=True)
        self._solver_factory = VQEUCCFactory(estimator=Estimator(), ansatz=UCCSD(), optimizer=SLSQP())

    def add_atom(self, atom: AtomInfo):
        self._molecule.symbols.append(atom.symbol)
        self._molecule.coords.append(atom.coords)

    def update_last_atoms_coords(self, coords: tuple[float, float, float]):
        self._molecule.coords[-1] = coords

    def __call__(self):
        print(".", end="")
        driver = PySCFDriver.from_molecule(self._molecule, basis="sto3g")
        problem = driver.run()

        n_electrons = problem.num_particles
        n_spatial_orbitals = problem.num_spatial_orbitals
        transformer = ActiveSpaceTransformer(num_electrons=n_electrons, num_spatial_orbitals=n_spatial_orbitals)
        problem_transformed = transformer.transform(problem)

        solver = self._solver_factory.get_solver(problem_transformed, self._qubit_converter)
        ground_state_eigen_solver = GroundStateEigensolver(self._qubit_converter, solver)
        ground_state = ground_state_eigen_solver.solve(problem_transformed)
        electronic_ground_state_energy = ground_state.groundenergy
        nuclear_repulsion_energy = problem.nuclear_repulsion_energy
        total_ground_state_energy = electronic_ground_state_energy + nuclear_repulsion_energy
        return total_ground_state_energy

box_size = 3
def minimize_ground_state_energy():
    ax = setup_plot()
    plot_point((0, 0, 0), ax, symbol="*", size=100)
    calc = GroundStateEnergyCalculation()
    for _ in range(2):
        coords = np.random.default_rng().uniform(-box_size, box_size, 3)
        # coords = (0, 0, 0.1)
        atom = AtomInfo("H", tuple(coords))
        calc.add_atom(atom)
        ground_state_energy = calc()
        print("\n", coords, ground_state_energy)
        ax.scatter(*coords, marker="o", c="b")
        learning_rate = 0.1
        epsilon = 0.01
        update_directions = [sign * learning_rate * np.array([int(n==0), int(n==1), int(n==2)]) for n in range(3) for sign in [1, -1]]
        ground_state_energy_j = np.inf
        last_direction_updated = -1
        while True:
            coords_j = coords.copy()
            for k, update_direction in enumerate(update_directions):
                ground_state_energy_k, coords = try_single_step(calc, coords, update_direction, ground_state_energy, ax)
                if ground_state_energy_k == ground_state_energy:
                    if last_direction_updated==k:
                        break
                else:
                    last_direction_updated = k
                    ground_state_energy = ground_state_energy_k

            # acceleration step
            ground_state_energy_l, coords = line_search(calc, coords, coords - coords_j, ground_state_energy, ax)
            if ground_state_energy_l < ground_state_energy:
                ground_state_energy = ground_state_energy_l

            if abs(ground_state_energy_j - ground_state_energy) < epsilon:
                break
            else:
                ground_state_energy_j = ground_state_energy

        print(f"done at position {coords} at distance {np.linalg.norm(coords)} with ground state energy {ground_state_energy}")

    plt.show()
    return ground_state_energy


def try_single_step(calc, coords, update_direction, ground_state_energy, ax):
    coords_i = coords + update_direction
    calc.update_last_atoms_coords(coords_i)
    ground_state_energy_i = calc()
    print(update_direction, ground_state_energy_i - ground_state_energy, end="")
    if ground_state_energy_i < ground_state_energy:
        ground_state_energy = ground_state_energy_i
        coords = coords_i
        print(coords, ground_state_energy)
        plot_point(coords, ax)
    else:
        print()
    return ground_state_energy, coords


def line_search(calc, coords, update_direction, ground_state_energy, ax):
    while True:
        ground_state_energy_i, coords = try_single_step(calc, coords, update_direction, ground_state_energy, ax)
        if ground_state_energy_i < ground_state_energy:
            ground_state_energy = ground_state_energy_i
        else:
            break
    return ground_state_energy, coords


def setup_plot():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim([-box_size, box_size])
    ax.set_ylim([-box_size, box_size])
    ax.set_zlim([-box_size, box_size])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return ax

def plot_point(coords, ax, symbol="o", size=None):
    size = size or 20
    ax.scatter(*coords, marker=symbol, s=size, c="b")
    x, y, z = tuple(coords)
    markersize = size / 10 if size is not None else None
    ax.plot(x, z, f'r{symbol}', markersize=markersize, zdir='y', zs=box_size, markerfacecolor="none")
    ax.plot(y, z, f'g{symbol}', markersize=markersize, zdir='x', zs=-box_size, markerfacecolor="none")
    ax.plot(x, y, f'k{symbol}', markersize=markersize, zdir='z', zs=-box_size, markerfacecolor="none")
    plt.draw()
    plt.pause(0.05)