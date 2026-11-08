import numpy as np
from .fitting_problem_spec import FittingProblem
from dataclasses import dataclass
from ..model._model_errors import InputError
from .fitting_problem_spec import ModelVariables

# TODO:
# 1) Ability to set constraints, e.g. N=S+I+R=const
# 2) Full separate method spec
# 3) Improve model output class
# 4) Setting parameters via dunder is misleading
# 5) Fit via param transforms (recovery period instead of gamma, log(x) insted of x)
# 6) copy pygom model, don't use dunders to srt parameters and ICs

# ---------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------

@dataclass
class MethodSpec:
    """
    Parameters for the specific method (ABC, MLE etc...)
    """
    pass

@dataclass
class FitResult:
    """
    Output of fitting
    """
    pass

class FittingMethod:
    """
    Execute a specific optimisation or inference algorithm.
    """
    def __init__(self, fitting_problem:FittingProblem, method_spec:MethodSpec):

        self._model = fitting_problem.model         # TODO: copy()
        self._simulator = fitting_problem.simulator
        self._target_spec = fitting_problem.targets
        self._method_spec = method_spec

        self._unknowns = fitting_problem.unknowns
        self.n_unknowns = len(self._unknowns.parameters)
        self._all_unknowns = self._unknowns.parameters + self._unknowns.initial_conditions
        self._fixed = fitting_problem.fixed
        self._resolvers = fitting_problem.resolvers

    def _unpack_theta(self, theta):
        """
        Safely unpack unknown variables from list to dict which
        can then be fed into the right place in the model.
        """

        ## Parameters ##

        known_parameters = self._fixed.parameters

        unknown_parameters = {
            u.id: value
            for u, value
            in zip(self._unknowns.parameters, theta[:self.n_unknowns])
        }

        parameters = known_parameters | unknown_parameters

        ## Initial statess ##

        known_states = self._fixed.initial_conditions

        unknown_states = {
            u.id: value
            for u, value
            in zip(self._unknowns.initial_conditions, theta[self.n_unknowns:])
        }

        states = known_states | unknown_states

        ## Resolve ##

        for resolver in self._resolvers:
            resolved = resolver.resolve(parameters, states)

            parameters = parameters | resolved.parameters
            states = states | resolved.initial_conditions

        return ModelVariables(
            parameters=parameters,
            initial_conditions=states
        )


    def _configure_model(self, theta):
        """
        Parameterise the model, ready for simulation
        """

        model_vars = self._unpack_theta(theta)

        # TODO: change from dunder method to model.set_parameters()-like function
        self._model.parameters = model_vars.parameters
        self._model.initial_states = model_vars.initial_conditions

        return self._model

    def _objective(self, theta):
        """
        Calculate loss, given trial parameters
        """

        sim = self._simulator.simulate(self._configure_model(theta))

        loss = 0
        for target in self._target_spec:
            predicted = target.observable.evaluate(sim)
            loss += target.weight * target.error_model.evaluate(target.value, predicted)

        return loss

    def fit(self) -> FitResult:
        """
        Overloaded by child class
        """
        raise NotImplementedError

# ---------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------

from .approximate_bayesian_computation.abc import ABC, Generation

@dataclass
class ABCMethodSpec(MethodSpec):
    n_particles: int
    quantile: float
    initial_tolerance: float=np.inf
    covariance_method: str = "global"
    distance_method: str|None = None
    k: int|None = None
    lengthscales=None
    regularisation=None
    max_attempts: int=1_000_000
    rng: np.random.Generator | None = None

@dataclass
class ABCResult(FitResult):
    """
    Output of fitting
    """
    generations: list[Generation]

class ABCFit(FittingMethod):
    
    def fit(self, n_generations, progress=True, previous_generations=None):

        solver = ABC(
            n_particles=self._method_spec.n_particles,
            quantile=self._method_spec.quantile,
            initial_tolerance=self._method_spec.initial_tolerance,
            covariance_method=self._method_spec.covariance_method,
            distance_method=self._method_spec.distance_method,
            k=self._method_spec.k,
            lengthscales=self._method_spec.lengthscales,
            regularisation=self._method_spec.regularisation,
            max_attempts=self._method_spec.max_attempts,
            rng=self._method_spec.rng,
            progress=progress
        )

        out = solver.refine_posterior(
            objective=self._objective,
            unknowns=self._all_unknowns,
            n_generations=n_generations,
            previous_generations=previous_generations
        )

        return ABCResult(
            generations = out
        )

# ---------------------------------------------------------------------
# MLE
# ---------------------------------------------------------------------

# TODO: add CI estimates to MLE

from scipy.optimize import minimize

@dataclass
class MLEMethodSpec(MethodSpec):
    method: str='BFGS'
    tol: float|None = None
    options: dict|None = None

    def __post_init__(self):
        if self.method in ['dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact']:
            raise(InputError("Cannot currently provide Jacobian to these methods"))

@dataclass
class MLEResult(FitResult):
    """
    Output of fitting
    """
    parameters: dict
    initial_conditions: dict
    success: bool
    loss: float

class MLEFit(FittingMethod):
    def __init__(self, fitting_problem:FittingProblem, method_spec:MethodSpec):
        super.__init__()

        if self._simulator.is_stochastic:
            raise(InputError("Stochastic objective used within deterministic optimiser."))

    def fit(self):
        """
        Optimise loss function to obtain MLE parameters and/or initial states
        """

        x0 = []
        bounds = []
        for u in self._unknowns:
            x0.append(u.initial_guess)
            bounds.append( (u.lower, u.upper) )

        res = minimize(
            fun=self.objective,
            x0=x0,
            bounds=bounds
        )

        parameters, initial_conditions = self.unpack_theta(res.x)

        return MLEResult(
            parameters=parameters,
            initial_conditions=initial_conditions,
            success=res.success,
            loss=res.fun
        )