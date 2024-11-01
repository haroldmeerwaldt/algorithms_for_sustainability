import pytest
from qiskit_nature.second_q.formats import MoleculeInfo
from quantum_algorithms.vqe.energy_calculation import AtomInfo, GroundStateEnergyCalculation


def test_calculate_ground_state_energy():
    distance = 0.735
    molecule = MoleculeInfo(["H", "H"], [(0.0, 0.0, 0.0), (0.0, 0.0, distance)], charge=0, multiplicity=1)
    calc = GroundStateEnergyCalculation(molecule)
    ground_state_energy = calc.run()
    assert ground_state_energy == pytest.approx(-1.1373060356959406)

def test_minimize_ground_state_energy():
    # ground_state_energy = minimize_ground_state_energy()
    print(ground_state_energy)
