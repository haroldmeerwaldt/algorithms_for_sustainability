from qiskit.primitives import Estimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats import MoleculeInfo
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem


class GroundStateEnergyCalculation:
    def __init__(self, molecule: MoleculeInfo, n_max_iterations: int = 100):
        """Calculates the ground state energy of a molecule.

        Args:
            molecule (MoleculeInfo): The molecular structure information, including atom types and coordinates.
            n_max_iterations (int): The maximum number of iterations allowed for the optimization algorithm.
                Defaults to 100.
        """
        self._molecule = molecule
        # todo: move to V2 Estimator once https://github.com/qiskit-community/qiskit-algorithms/issues/136 is fixed
        self._estimator = Estimator()
        self._optimizer = COBYLA(maxiter=n_max_iterations)

    def update_last_atoms_coords(self, coords: tuple[float, float, float]):
        new_coords = list(self._molecule.coords)
        new_coords[-1] = coords
        self._molecule.coords = new_coords

    def run(self):
        driver = PySCFDriver.from_molecule(self._molecule)
        es_problem: ElectronicStructureProblem = driver.run()

        fermionic_operator = es_problem.hamiltonian.second_q_op()
        mapper = ParityMapper(num_particles=es_problem.num_particles)

        qubit_operator = mapper.map(fermionic_operator)

        initial_state = HartreeFock(es_problem.num_spatial_orbitals, es_problem.num_particles, mapper)
        ansatz = UCCSD(es_problem.num_spatial_orbitals, es_problem.num_particles, mapper, initial_state=initial_state)

        algorithm = VQE(self._estimator, ansatz, self._optimizer)
        result = algorithm.compute_minimum_eigenvalue(qubit_operator)

        electronic_ground_state_energy = result.eigenvalue
        total_ground_state_energy = electronic_ground_state_energy + es_problem.nuclear_repulsion_energy
        return total_ground_state_energy
