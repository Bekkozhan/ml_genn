import numpy as np
from .readout import Readout
from ..utils.model import NeuronModel
from copy import deepcopy


class WindowedSumVar(Readout):
    """Read out per-neuron sum of neuron model's output variable,
    accumulating only from start_timestep onwards.

    Used to restrict readout to a specific time window (e.g. recall window
    in the evidence accumulation task), matching NEST's behaviour where
    the output is evaluated only during the decision period.

    Args:
        start_timestep: Timestep from which to start accumulating [timesteps].
                        Default 0 reproduces standard SumVar behaviour.
        dt:             Simulation timestep [ms]. Default 1.0.
    """

    def __init__(self, start_timestep: int = 0, dt: float = 1.0):
        self.start_timestep = start_timestep
        self.dt = dt

    def add_readout_logic(self, model: NeuronModel, **kwargs) -> NeuronModel:
        self.output_var_name = model.output_var_name
        if "vars" not in model.model:
            raise RuntimeError("WindowedSumVar readout can only be used "
                               "with models with state variables")
        if self.output_var_name is None:
            raise RuntimeError("WindowedSumVar readout requires that models "
                               "specify an output variable name")
        try:
            output_var = next(v for v in model.model["vars"]
                              if v[0] == self.output_var_name)
        except StopIteration:
            raise RuntimeError(f"Model does not have variable "
                               f"{self.output_var_name} to sum")

        model_copy = deepcopy(model)
        sum_var_name = self.output_var_name + "Sum"
        self.output_var_type = output_var[1]
        start_time = self.start_timestep * self.dt

        model_copy.append_sim_code(
            f"if(t >= {start_time}) {{ {sum_var_name} += {self.output_var_name}; }}")
        model_copy.add_var(sum_var_name, self.output_var_type, 0)
        return model_copy

    def get_readout(self, genn_pop, batch_size: int, shape) -> np.ndarray:
        sum_var = genn_pop.vars[self.output_var_name + "Sum"]
        sum_var.pull_from_device()
        return np.reshape(sum_var.view, (batch_size,) + shape)

    @property
    def reset_vars(self):
        return [(self.output_var_name + "Sum", self.output_var_type, 0.0)]
