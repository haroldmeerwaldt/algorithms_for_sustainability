import pytest
from qiskit_nature.second_q.formats import MoleculeInfo
from quantum_algorithms.vqe.energy_calculation import GroundStateEnergyCalculation


@pytest.fixture
def molecule_info():
    distance = 0.735
    return MoleculeInfo(["H", "H"], [(0.0, 0.0, 0.0), (0.0, 0.0, distance)])


class TestGroundStateEnergyCalculation:
    def test_initialization(self, molecule_info):
        _ = GroundStateEnergyCalculation(molecule_info)

    def test_update_last_atoms_coords(self, molecule_info):
        calculation = GroundStateEnergyCalculation(molecule_info)
        new_coords = (0.0, 0.0, 1.0)
        calculation.update_last_atoms_coords(new_coords)

        assert calculation._molecule.coords[-1] == new_coords

    def test_run(self, mocker, molecule_info):
        mock_vqe = mocker.patch("quantum_algorithms.vqe.energy_calculation.VQE")
        mock_vqe_calculation = mock_vqe.return_value
        mock_vqe_calculation.compute_minimum_eigenvalue.return_value.eigenvalue = -1.0

        calculation = GroundStateEnergyCalculation(molecule_info)
        result = calculation.run()

        assert result == pytest.approx(-0.28, rel=1e-3)
