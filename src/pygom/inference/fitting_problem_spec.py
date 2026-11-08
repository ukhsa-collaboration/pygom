# TODO:
# 1) observation-to-simulation time alignment (time is special in ODEs, so should be simple)
# 2) Allow for calculating some parameters / ICs pre simulation "derived knowns"

"""
Data classes to specify the inference problem to be solved
"""

from dataclasses import dataclass, field
import numpy as np
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

# ---------------------------------------------------------------------
# Variable classes
# ---------------------------------------------------------------------

## Unknown ##

@dataclass
class Unknown:
    id: str
    lower: float = -np.inf              # TODO: For now, we have scalar unknowns
    upper: float = np.inf

    # Method specific attributes
    # TODO: Maybe inherit from Unknown to have ABCUnknown, MLEUnknown?
    initial_guess: float | None = None
    prior: Prior | None = None

    def __post_init__(self):

        # Lower limit must be strictly less than upper limit
        if self.lower >= self.upper:
            raise ValueError(
                f"Unknown '{self.id}': lower bound ({self.lower}) "
                f"must be less than upper bound ({self.upper})"
            )

        # Initial guess must fall between limits
        if self.initial_guess is not None:
            if not (self.lower <= self.initial_guess <= self.upper):
                raise ValueError(
                    f"Unknown '{self.id}': initial guess ({self.initial_guess}) "
                    f"must be within upper and lower bounds."
                )
        
        # TODO: prior must be defined between limits

# ---------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------

@dataclass
class UnknownSpec:
    parameters: list[Unknown] = field(default_factory=list)
    initial_conditions: list[Unknown] = field(default_factory=list)

    def __post_init__(self):

        # Check conflicts within each list
        self._check_duplicates(self.param_ids, "parameter")
        self._check_duplicates(self.initial_condition_ids, "initial condition")

        # Check conflicts between lists
        overlap = set(self.param_ids) & set(self.initial_condition_ids)
        if overlap:
            raise ValueError(
                f"Unknown specified as both parameter and initial condition: {sorted(overlap)}"
            )

    @staticmethod
    def _check_duplicates(ids, name):

        duplicates = [
            x
            for x in set(ids)
            if ids.count(x) > 1
        ]

        if duplicates:
            raise ValueError(
                f"Duplicate {name} ids: {sorted(duplicates)}"
            )

    @property
    def param_ids(self):
        return [x.id for x in self.parameters]

    @property
    def initial_condition_ids(self):
        return [x.id for x in self.initial_conditions]

## Generic container for variables ##

@dataclass
class ModelVariables:
    parameters: dict[str, float] = field(default_factory=dict)
    initial_conditions: dict[str, float] = field(default_factory=dict)

    @property
    def param_ids(self):
        return self.parameters.keys()

    @property
    def initial_condition_ids(self):
        return self.initial_conditions.keys()
    

@dataclass
class FixedSpec(ModelVariables):
    def __post_init__(self):
        overlap = self.param_ids & self.initial_condition_ids

        if overlap:
            raise ValueError(
                f"Fixed specified as both parameter and initial condition: {sorted(overlap)}"
            )

## Derived (but not estimated) ##

@dataclass(frozen=True)
class ResolverVariables:
    parameters: set[str] = field(default_factory=set)
    initial_conditions: set[str] = field(default_factory=set)

class Resolver:
    inputs = ResolverVariables()
    outputs = ResolverVariables()

    def resolve(self, values: ModelVariables) -> ModelVariables:
        raise NotImplementedError    


# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------

# TODO: 
# 1) Currently these do not return SimulationResult but take [0].result
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
# Specify the fitting problem (not the method)
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
    fixed: FixedSpec # = FixedSpec()
    targets: list[Target]
    resolvers: list[Resolver] = field(default_factory=list)

    def __post_init__(self):

        # Check parameter consistency

        self._check_consistency(
            unknown_vars=set(self.unknowns.param_ids),
            fixed_vars=set(self.fixed.parameters),
            resolver_vars=set().union( *(r.outputs.parameters for r in self.resolvers) ),
            model_vars=set(self.model.parameter_ids),
            var_type="Parameter"
        )

        # Check initial condition consistency

        self._check_consistency(
            unknown_vars=set(self.unknowns.initial_condition_ids),
            fixed_vars=set(self.fixed.initial_conditions),
            resolver_vars=set().union( *(r.outputs.initial_conditions for r in self.resolvers) ),
            model_vars=set(self.model.state_ids),
            var_type="State"
        )


    def _check_consistency(
        self,
        unknown_vars: set[str],
        fixed_vars: set[str],
        resolver_vars: set[str],
        model_vars: set[str],
        var_type: str
    ):
        provided = unknown_vars | fixed_vars | resolver_vars

        # Every specified variable must exist in the model
        extra = provided - model_vars
        if extra:
            raise ValueError(f"{var_type}s not found in model: {sorted(extra)}")

        # Every model variable must be accounted for
        missing = model_vars - provided
        if missing:
            raise ValueError(f"Model {var_type}s not accounted for: {sorted(missing)}")