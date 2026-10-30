import numpy as np
from .fitting_problem_spec import FittingProblem
from dataclasses import dataclass
from ..model._model_errors import InputError

# TODO:
# 1) Ability to set constraints, e.g. N=S+I+R=const
# 2) Full separate method spec
# 3) Improve model output class
# 4) Setting params via dunder is misleading
# 5) Fit via param transforms (recovery period instead of gamma, log(x) insted of x)

# ---------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------

@dataclass
class MethodSpec:
    """
    Parameters for the method spec
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
        self._unknowns = fitting_problem.unknowns
        self._all_unknowns = self._unknowns.parameters + self._unknowns.initial_states
        self._simulator = fitting_problem.simulator
        self._target_spec = fitting_problem.targets

        self._method_spec = method_spec

        # Set knowns right away (no point in resetting every iteraiton)
        # NOTE: this will probably change when we allow "derived knowns"
        # TODO: change from dunder method to model.set_params()-like function
        #       to make less confusing looking
        self._model.parameters = fitting_problem.knowns.parameters
        self._model.initial_states = fitting_problem.knowns.initial_states

    def _unpack_theta(self, theta):
        """
        Safely unpack unknown variables from list to dict which
        can then be fed into the right place in the model.
        """

        n_params = len(self._unknowns.parameters)

        params = {
            u.id: value
            for u, value
            in zip(self._unknowns.parameters, theta[:n_params])
        }

        initial_states = {
            u.id: value
            for u, value
            in zip(self._unknowns.initial_states, theta[n_params:])
        }

        return params, initial_states

    def _configure_model(self, theta):
        """
        Parameterise the model, ready for simulation
        """

        params, states = self._unpack_theta(theta)

        # TODO: change from dunder method to model.set_params()-like function
        self._model.parameters = params
        self._model.initial_states = states

        return self._model

    def _objective(self, theta):
        """
        Calculate loss, given trial parameters
        """

        sim = self._simulator.simulate(self._configure_model(theta))

        loss = 0
        for target in self._target_spec.targets:
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

from .ABC import ABC, Generation

    # TODO: n_generations is a parameter you may want to declare at start,
    #       or to refine a pre-existing output
@dataclass
class ABCMethodSpec(MethodSpec):
    n_particles: int
    quantile: float
    initial_tolerance: float=np.inf
    max_attempts: int=1_000_000
    seed: int=123

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
            max_attempts=self._method_spec.max_attempts,
            seed=self._method_spec.seed,
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
    initial_states: dict
    success: bool
    loss: float

class MLEFit(FittingMethod):
    def __init__(self, fitting_problem:FittingProblem, method_spec:MethodSpec):
        super.__init__()

        if self._simulator.is_stochastic:
            raise(InputError("Stochastic objective used within deterministic optimiser."))

    def fit(self):
        """
        Optimise loss function to obtain MLE params and/or initial states
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

        params, initial_states = self.unpack_theta(res.x)

        return MLEResult(
            parameters=params,
            initial_states=initial_states,
            success=res.success,
            loss=res.fun
        )