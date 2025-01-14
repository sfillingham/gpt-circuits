from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


class ParameterSweeper:
    """
    Runs a sweep over a list of parameter sets.
    """

    def __init__(self, fn: Callable, parameter_sets: list[dict[str, Any]], pool_size: int = 4):
        self.parameter_sets = parameter_sets
        self.pool_size = pool_size
        self.fn = fn

    def execute_fn(self, parameters):
        self.fn(**parameters)

    def sweep(self):
        with ThreadPoolExecutor(max_workers=self.pool_size) as executor:
            executor.map(self.execute_fn, self.parameter_sets)
