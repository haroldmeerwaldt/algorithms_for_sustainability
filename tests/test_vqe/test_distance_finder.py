import pytest
import numpy as np
from quantum_algorithms.vqe.distance_finder import DistanceFinder, IterationInfo



class TestDistanceFinder:
    @pytest.fixture
    def mock_ground_state_calculation(self, mocker):
        mock_class = mocker.patch("quantum_algorithms.vqe.distance_finder.GroundStateEnergyCalculation")
        mock_instance = mock_class.return_value
        mock_instance.run.return_value = -1.8
        return mock_instance

    def test_initialization(self, mock_ground_state_calculation):
        _ = DistanceFinder()

    def test_run_method(self, mocker, mock_ground_state_calculation):
        mock_ground_state_calculation.run.side_effect = lambda: -1.8
        finder = DistanceFinder()
        mock_minimize = mocker.patch("quantum_algorithms.vqe.distance_finder.minimize")
        mock_minimize.return_value.x = np.array([0.0, 0.0, 1.5])

        result = finder.run()
        assert np.isclose(result, np.linalg.norm([0.0, 0.0, 1.5]))
        mock_minimize.assert_called_once()

    def test_iteration_callback_execution(self, mocker, mock_ground_state_calculation):
        mock_callback = mocker.Mock()
        finder = DistanceFinder(iteration_callback=mock_callback)
        mock_ground_state_calculation.run.return_value = -1.5

        finder._minimize_ground_state_energy(np.array([0.0, 0.0, 2.0]))

        mock_callback.assert_called_once()
        call_args = mock_callback.call_args[0][0]
        assert isinstance(call_args, IterationInfo)
        assert call_args.coords == (0.0, 0.0, 2.0)
        assert call_args.ground_state_energy == -1.5
