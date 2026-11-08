import numpy as np
from dataclasses import dataclass
from .priors import Prior

# ---------------------------------------------------------------------
# Variable classes
# ---------------------------------------------------------------------

## Unknown ##

@dataclass
class Unknown:
    id: str
    lower: float = -np.inf              # TODO: For now, we have scalar unknowns
    upper: float = np.inf

    # Method specific attributes.         TODO: Maybe inherit from Unknown to have ABCUnknown, MLEUnknown?
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

## Known ##

@dataclass
class Fixed:
    id: str
    value: float

## Derived (but not estimated) ##

class Resolver:
    inputs: set[str] = set()
    outputs: set[str] = set()

    def resolve(self, values: dict[str, float]) -> dict[str, float]:
        raise NotImplementedError

# ---------------------------------------------------------------------
# Variable store
# ---------------------------------------------------------------------

class VariableSpec:
    """
    Specify unknowns, fixed and resolved
    """

    # unknowns: list[Unknown] = field(default_factory=list)
    # fixed: list[Fixed] = field(default_factory=list)
    # resolvers: list[Resolver] = field(default_factory=list)

    def __init__(self, unknowns, fixed, resolvers):

        self._unknown_ids = None
        self._fixed_ids = None
        self._resolver_ids = None

        self.unknowns = unknowns
        self.fixed = fixed
        self.resolvers = resolvers

        # Checks:

        # Variables cannot be both unknown and fixed
        overlap = self.unknown_ids & self.fixed_ids
        if overlap:
            raise ValueError(
                f"Variables specified as both known and unknown: {sorted(overlap)}"
            )

        # Variables cannot be both unknown and resolver
        overlap = self.unknown_ids & self.resolver_ids
        if overlap:
            raise ValueError(
                f"Variables specified as both unknown and derived: {sorted(overlap)}"
            )

        # Variables cannot be both fixed and resolver
        overlap = self.fixed_ids & self.resolver_ids
        if overlap:
            raise ValueError(
                f"Variables specified as both fixed and derived: {sorted(overlap)}"
            )

    ## Unknowns ##

    @property
    def unknowns(self):
        return self._unknowns

    @unknowns.setter
    def unknowns(self, unknowns):
        self._unknowns = unknowns

        ids = self.unknown_ids
        duplicates = [id for id in set(ids) if ids.count(id) > 1]
        if duplicates:
            raise ValueError(f"Duplicate unknown ids: {sorted(duplicates)}")

    ## Fixed ##

    @property
    def fixed(self):
        return self._fixed

    @fixed.setter
    def fixed(self, fixed):
        self._fixed = fixed

        ids = self.fixed_ids
        duplicates = [id for id in set(ids) if ids.count(id) > 1]
        if duplicates:
            raise ValueError(f"Duplicate fixed ids: {sorted(duplicates)}")

    @property
    def fixed_ids(self):
        if self._fixed_ids is None:
            self._fixed_ids = [u.id for u in self.fixed]

        return self._fixed_ids

    ## Resolver ##

    @property
    def resolver(self):
        return self._resolver

    @resolver.setter
    def resolver(self, resolver):
        self._resolver = resolver

        ids = self.resolver_ids
        duplicates = [id for id in set(ids) if ids.count(id) > 1]
        if duplicates:
            raise ValueError(f"Duplicate resolver ids: {sorted(duplicates)}")

    @property
    def resolver_ids(self):
        if self._resolver_ids is None:
            resolver_ids = []
            for r in self.resolvers:
                resolver_ids.extend(r.outputs)
            self._resolver_ids = resolver_ids

        return self._resolver_ids

    
class PyGOMConfigurator:
    """
    
    """

    def __init__(self, model):

        self.model = model

        self.parameter_ids = set(model.parameter_ids.keys())
        self.state_ids = set(model.state_ids.keys())

    # def _unpack_theta(self, theta):
    #     """
    #     Safely unpack unknown variables from list to dict which
    #     can then be fed into the right place in the model.
    #     """

    #     n_params = len(self._unknowns.parameters)

    #     params = {
    #         u.id: value
    #         for u, value
    #         in zip(self._unknowns.parameters, theta[:n_params])
    #     }

    #     initial_states = {
    #         u.id: value
    #         for u, value
    #         in zip(self._unknowns.initial_states, theta[n_params:])
    #     }

    #     return params, initial_states

    def resolve(self, proposal):
        """
        Get values ready for the model
        """

        values = {}

        # fixed values
        for v in self.fixed:
            values[v.id] = v.value

        # sampled values
        values.update(proposal)

        # derived values
        for d in self.derived:
            values[d.id] = d.compute(values)

        return values

    def configure(self, values):

        params = {}
        states = {}

        for k, v in values.items():
            if k in self.parameter_ids:
                params[k] = v
            elif k in self.state_ids:
                states[k] = v

        missing_params = self.parameter_ids - set(params)
        if missing_params:
            raise ValueError(f"Missing parameters: {sorted(missing_params)}")

        missing_states = self.state_ids - set(states)
        if missing_states:
            raise ValueError(f"Missing states: {sorted(missing_states)}")

        model = self.model.copy()
        model.parameters = params
        model.initial_states = states

        return model


















@dataclass
class VariableSpec:

    unknowns: list[Unknown] = field(default_factory=list)
    fixed: list[Fixed] = field(default_factory=list)
    derived: list[Derived] = field(default_factory=list)

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
        """
        Check ....
        """

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

    def resolve(self, proposal):
        """
        Get values ready for the model
        """

        values = {}

        # fixed values
        for v in self.fixed:
            values[v.id] = v.value

        # sampled values
        values.update(proposal)

        # derived values
        for d in self.derived:
            values[d.id] = d.compute(values)

        return values

        # states = {
        #     k: v
        #     for k, v in values.items()
        #     if k in model.state_list
        # }

        # parameters = {
        #     k: v
        #     for k, v in values.items()
        #     if k in model.param_list
        # }