import unittest
from unittest.mock import MagicMock

from experiments import ParameterSweeper


class TestParameterSweeper(unittest.TestCase):
    def test_sweep(self):
        # Mock function to be called with parameters
        mock_fn = MagicMock()

        # Sample parameter sets
        parameter_sets = [{"param1": 1, "param2": 2}, {"param1": 3, "param2": 4}]

        # Create an instance of ParameterSweeper
        sweeper = ParameterSweeper(mock_fn, parameter_sets, pool_size=2)

        # Run the sweep method
        sweeper.sweep()

        # Check if the mock function was called with the correct parameters
        mock_fn.assert_any_call(param1=1, param2=2)
        mock_fn.assert_any_call(param1=3, param2=4)
        self.assertEqual(mock_fn.call_count, 2)
