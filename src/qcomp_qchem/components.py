import itertools

from qiskit.algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit_nature.second_q.algorithms import VQEUCCFactory, GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats import MoleculeInfo
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer


def create_h2_molecule(distance: float):
    # simple function until I find a replacement for BOPESSampler
    return MoleculeInfo(["H", "H"], [(0.0, 0.0, 0.0), (0.0, 0.0, distance)], charge=0, multiplicity=1)

def calculate_ground_state_energy(molecule):
    print(".", end="")
    driver = PySCFDriver.from_molecule(molecule, basis="sto3g")
    problem = driver.run()

    converter = JordanWignerMapper()  # also ParityMapper, BravyiKitaevMapper
    qubit_converter = QubitConverter(converter, two_qubit_reduction=True)
    vqe_ucc_solver_factory = VQEUCCFactory(estimator=Estimator(), ansatz=UCCSD(), optimizer=SLSQP())

    n_electrons = problem.num_particles
    n_spatial_orbitals = problem.num_spatial_orbitals

    transformer = ActiveSpaceTransformer(num_electrons=n_electrons, num_spatial_orbitals=n_spatial_orbitals)
    problem_transformed = transformer.transform(problem)
    vqe_ucc_solver = vqe_ucc_solver_factory.get_solver(problem, qubit_converter)
    ground_state_eigen_solver_vqe_ucc = GroundStateEigensolver(qubit_converter, vqe_ucc_solver)
    ground_state = ground_state_eigen_solver_vqe_ucc.solve(problem_transformed)
    electronic_ground_state_energy = ground_state.groundenergy
    nuclear_repulsion_energy = problem.nuclear_repulsion_energy
    total_ground_state_energy = electronic_ground_state_energy + nuclear_repulsion_energy
    return total_ground_state_energy
