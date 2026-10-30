# TODO:
# 1) observation-to-simulation time alignment
# 2) Allow for calculating some params / ICs pre simulation "derived knowns"

"""
Data classes to specify the inference problem to be solved
"""


from dataclasses import dataclass, field
import numpy as np
from numbers import Number
from ..model.pygom_model import Model
from ..simulation.src.simulation.data_classes import SimulationResult
from .priors import Prior
from .error_models import ErrorModel

# ---------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------

class Observable:
    """
    Map a SimulationResult to observable quantities
    """
    name = None

    def evaluate(self, simulation_output:SimulationResult):
        """
        Overloaded by child classes
        """
        raise NotImplementedError

@dataclass
class Target:
    """
    - Target value of observation
    - How to evaluate observation from simulation output
    - Error model and weight when combined in grand loss funtion
    - TODO: times?
    """
    observable: Observable
    error_model: ErrorModel
    value: float|np.ndarray
    weight: float = 1.0

@dataclass
class TargetSpec:
    """
    List of targets
    """
    targets: list[Target]

# ---------------------------------------------------------------------
# Unknowns
# ---------------------------------------------------------------------

@dataclass
class Unknown:
    # TODO: For now, we have scalar unknowns
    id: str
    lower: float = -np.inf
    upper: float = np.inf

    # Method specific attributes, TODO: inherit from Unknown to have ABCUnknown, MLEUnknown?
    initial_guess: float | None = None
    prior: Prior | None = None

    def __post_init__(self):
        if self.lower >= self.upper:
            raise ValueError(
                f"Unknown '{self.id}': lower bound ({self.lower}) "
                f"must be less than upper bound ({self.upper})"
            )
        if self.initial_guess is not None:
            if not (self.lower <= self.initial_guess <= self.upper):
                raise ValueError(
                    f"Unknown '{self.id}': initial guess ({self.initial_guess}) "
                    f"must be within upper and lower bounds."
                )
        # TODO: add check for prior: what if prior is zero across the entire allowed range?

@dataclass
class UnknownSpec:
    """
    Describe inferable parameters and initial states, including domains.
    Default to empty list
    """
    parameters: list[Unknown] = field(default_factory=list)
    initial_states: list[Unknown] = field(default_factory=list)

    # TODO: check that params / initial_states haven't been added multiple times

# ---------------------------------------------------------------------
# Knowns
# ---------------------------------------------------------------------

# NOTE: previously, known information was smuggled into the model object,
#       probably better to declare explicitly instead like here:

@dataclass
class KnownSpec:
    """
    Describe inferable parameters and initial states, including domains.
    """
    parameters: dict[str: Number]
    initial_states: dict[str: Number]

    # TODO: Sometimes e.g. initial conditions are known, but a function of parameters.
    #       e.g. an SIS system is in endemic equilibrium (a known feature), but
    #       R0 is not known. Thus, S0 = N/R0, is something we declare to be known
    #       but do not provide a value for. This is not the same as a derived parameter.
    #       We need a way to calculate knowns after proposal of the unknowns.
    #       Maybe we call these "derived knowns"

# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------

# TODO: 
# 1) Currently these do not return SimulationResult but take the[0].result
# 2) Bit annoying for user to set solver kwargs

class Simulator:
    """
    Evaluate a configured model and return a defined SimulationResult. 
    """
    def simulate(self, model):
        """
        Overloaded by child classes
        """
        raise NotImplementedError

class DeterministicSimulator(Simulator):
    def __init__(self, times):
        self.is_stochastic = False
        self.times = times

    def simulate(self, model) -> SimulationResult:
        return model.solve_deterministic(self.times, method="RK45")[0].result

class StochasticSimulator(Simulator):
    """
    A non working example of what we might want from a stochastic simulator
    where we discard those that go extinct 
    """
    def __init__(self, times, dt, seed=123):
        self.is_stochastic = True
        self.times = times
        self.dt = dt
        self.seed = seed

    def simulate(self, model) -> SimulationResult:

        out = model.solve_stochastic(
            t=self.times,
            method="fixed_tau",
            tau=self.dt,
            iteration=1,
            seed=self.seed,
            checker = "forbidden_state"
        )

        return out[0].result

# ---------------------------------------------------------------------
# Fitting Problem 
# ---------------------------------------------------------------------

@dataclass
class FittingProblem:
    """
    Combine the model, simulator, unknowns and targets without running inference.

    Perform some consistency checks
    """
    model: Model
    simulator: Simulator
    unknowns: UnknownSpec
    knowns: KnownSpec
    targets: TargetSpec

    def __post_init__(self):
        """
        Check that knowns and unknowns are properly specified
        """

        ## Params ## 

        unknown_params = {p.id for p in self.unknowns.parameters}
        known_params = set(self.knowns.parameters.keys())
        model_params = set(self.model.parameter_ids.keys())

        self.check_consistency(unknown_params, known_params, model_params, "Parameter")

        ## ICs ## 

        unknown_states = {p.id for p in self.unknowns.initial_states}
        known_states = set(self.knowns.initial_states.keys())
        model_states = set(self.model.state_ids.keys())

        self.check_consistency(unknown_states, known_states, model_states, "State")

    def check_consistency(self, unknown_vars, known_vars, model_vars, var_type):
        # Variables cannot be both known and unknown
        overlap = unknown_vars & known_vars
        if overlap:
            raise ValueError(
                f"{var_type}s specified as both known and unknown: {sorted(overlap)}"
            )

        # Every specified variable must exist in the model
        extra_vars = (unknown_vars | known_vars) - model_vars
        if extra_vars:
            raise ValueError(
                f"{var_type}s not found in model: {sorted(extra_vars)}"
            )

        # Every model variable must be accounted for
        missing_vars = model_vars - (unknown_vars | known_vars)
        if missing_vars:
            raise ValueError(
                f"Model {var_type}s not specified as known or unknown: {sorted(missing_vars)}"
            )