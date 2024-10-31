import pytest

from qcomp_qchem.components import AtomInfo, GroundStateEnergyCalculation, minimize_ground_state_energy


def test_calculate_ground_state_energy():
    atom = AtomInfo("H", (0.0, 0.0, 0.735))
    calc = GroundStateEnergyCalculation()
    calc.add_atom(atom)
    ground_state_energy = calc()
    assert ground_state_energy == pytest.approx(-1.1373060356959406)

def test_minimize_ground_state_energy():
    ground_state_energy = minimize_ground_state_energy()
    print(ground_state_energy)
