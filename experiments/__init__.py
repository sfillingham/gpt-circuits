from typing import Any, Callable

from torch import multiprocessing


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
        # CUDA requirement: https://stackoverflow.com/questions/72779926
        multiprocessing.set_start_method("spawn")
        with multiprocessing.Pool(processes=self.pool_size) as pool:
            pool.map(self.execute_fn, self.parameter_sets)
