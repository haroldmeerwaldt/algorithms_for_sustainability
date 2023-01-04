import pytest

from qcomp_qchem.components import AtomInfo, GroundStateEnergyCalculation


def test_calculate_ground_state_energy():
    atom = AtomInfo("H", (0.0, 0.0, 0.735))
    calc = GroundStateEnergyCalculation()
    calc.add_atom(atom)
    ground_state_energy = calc()
    assert ground_state_energy == pytest.approx(-1.1373060356959406)
