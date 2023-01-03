import pytest
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo

from qcomp_qchem.components import create_h2_molecule, calculate_ground_state_energy


def test_create_h2_molecule():
    molecule = create_h2_molecule(0.735)
    assert molecule == MoleculeInfo(["H", "H"], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.735)], charge=0, multiplicity=1)


def test_calculate_ground_state_energy():
    molecule = MoleculeInfo(["H", "H"], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.735)], charge=0, multiplicity=1)
    ground_state_energy = calculate_ground_state_energy(molecule)
    assert ground_state_energy == pytest.approx(-1.8572750301449956)