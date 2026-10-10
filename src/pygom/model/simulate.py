"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    Module/class that carries out different type of simulation
    on an ode formulation

"""

__all__ = ['SimulateOde']
import logging
import warnings

import copy
from numbers import Number

import numpy as np
import sympy
#import sympy.matrices.matrices
import scipy.stats

# from dask.distributed import Client
from .parallel_backends import run_dask, run_multiprocessing

from . import ode_utils
from .deterministic import DeterministicOde
from .stochastic_simulation import cle, exact, firstReaction, tauLeap, hybrid
from .transition import TransitionType, Transition
from ._model_errors import InputError, SimulationError
from ._model_verification import checkEquation, simplifyEquation
from . import _ode_composition

from .maths import (StateChangeMatrix, 
                    TransitionMean, 
                    TransitionVariance,
                    TransitionJacobian)

# from ..simulation.src.simulation.api import solve_stochastic, solve_deterministic
# from ..simulation.src.simulation.stochastic.config_api import build_config

from ..simulation.src.simulation.stochastic.solve_stochastic import solve_stochastic
from ..simulation.src.simulation.deterministic.solve_deterministic import solve_deterministic


from .ode_utils.variable_store import CallableParameter

class SimulateOde(DeterministicOde):
    '''
    This builds on top of :class:`DeterministicOde` which we
    simulate the outcome instead of solving it deterministically

    Parameters
    ----------
    state: list
        A list of states (string) or (string, (numeric, numeric)) if specifying limits
    param: list
        A list of the parameters (string)
    derived_param: list
        A list of the derived parameters (tuple of (string,string))
    transition: list
        A list of transition (:class:`Transition`) #TODO Now this might actually only be deterministic ODE objects. Check.
    transition: list
        A list of events (:class:`Event`)
    birth_death: list                              #TODO Now these are wrapped in events. Should try to make work for back compat.
        A list of birth or death process (:class:`Transition`)
    ode: list
        A list of ode (:class:`Transition`)

    '''
    # The compiled math functions within this class
    _maths_methods = DeterministicOde._maths_methods + [
        StateChangeMatrix,
        TransitionMean,
        TransitionVariance,
        TransitionJacobian
    ]

    def __init__(self,
                 state=None,
                 param=None,
                 derived_param=None,
                 transition=None,
                 event=None,
                 birth_death=None,
                 ode=None,
                 backend='lambda'
                 ):
        '''
        Constructor that is built on top of DeterministicOde
        '''
        logging.debug(self._maths_methods)
        super(SimulateOde, self).__init__(state,
                                          param,
                                          derived_param,
                                          transition,
                                          event,
                                          birth_death,
                                          ode,
                                          backend)

        # self.pre_tau=None       # If tau is set, then this overrides the adaptive tau leap.
        # self._epsilon=0.03      # Default parameter recommended by Cao et al.

        #self._stochasticParam=None


        # Add templates of compiled sympy functions with:
        # 1) Name of the compiled version of the sympy object
        # 2) The function used to generate the underlying sympy object
        #    (convention: starts with "get_", in previous versions have
        #     started with get_ or _compute)


    # # TODO 1: Deterministic solver


    def solve_deterministic(
            self,
            t,
            method="LSODA",
            dense_output=False,
            events=None,
            vectorized=False,
            args=None,
            perf=False,
            iteration=1,
            seed=None,
            parallel=False,
            **options):
    
        if (not self._parameter_store.has_stochastic_parameters) and (iteration>1):
            warnings.warn("System only has deterministic parameters, but multiple"
                          "iterations specified. Perhaps you wish to set iteration=1 instead?")

        # Normalise time input
        t0 = self.initial_time
        if isinstance(t, Number):
            t_span = (t0, t)
            t_eval = None
        elif isinstance(t, np.ndarray):
            t_span = (t0, t.max())
            t_eval = t
        else:
            raise InputError("Unknown dtype for time. Expected Number or numpy.array")
        

        def _run_single_iteration(iter_ss):
            params_ss = iter_ss.spawn(1)[0]
            stochastic_params = self._parameter_store.stochastic_parameters
            param_seqs = params_ss.spawn(len(stochastic_params))

            for seq, param in zip(param_seqs, stochastic_params.values()):
                if isinstance(param.value, CallableParameter):
                    param.value.rng = np.random.default_rng(seq)

            self._parameter_store.new_realisation()

            jacobian_methods = ['LSODA', 'Radau', 'BDF']

            if method in jacobian_methods:
                return solve_deterministic(
                        ode_eqns=self.ode.T,
                        n_state=self.num_state,
                        event_rates=self.event_rate_vector.T,
                        n_event=self.num_events,
                        t_span=t_span,
                        y0=self.initial_state,
                        jac_ode=self.jacobian.T,
                        jac_events=self.rates_jacobian.T,
                        method=method,
                        t_eval=t_eval,
                        dense_output=dense_output,
                        events=events,
                        vectorized=vectorized,
                        args=args,
                        perf=perf,
                        **options)
            else:
                return solve_deterministic(
                        ode_eqns=self.ode.T,
                        n_state=self.num_state,
                        event_rates=self.event_rate_vector.T,
                        n_event=self.num_events,
                        t_span=t_span,
                        y0=self.initial_state,
                        method=method,
                        t_eval=t_eval,
                        dense_output=dense_output,
                        events=events,
                        vectorized=vectorized,
                        args=args,
                        perf=perf,
                        **options)
        
        master_ss = np.random.SeedSequence(seed)
        iteration_seqs = master_ss.spawn(iteration)

        if not parallel:
            logging.debug("Performing serial simulation")

            out = []
            for i in range(iteration):
                result = _run_single_iteration(iteration_seqs[i])
                out.append(result)

            return out

    def solve_stochastic(
            self,
            t,
            method,
            seed=None,
            proceed_if_rates_zero=False,
            perf=False,
            iteration=1,
            parallel=False,
            client=None,
            parallel_backend="dask",
            **options):
        
        y_min = np.zeros(self.num_state, dtype=int)
        y_max = np.full(self.num_state, np.inf)

        if method == "cao2006":
            options["transition_mean_func"] = self.transition_mean.T
            options["transition_var_func"] = self.transition_variance.T

        # Normalise time input
        t0 = self.initial_time
        if isinstance(t, Number):
            t_span = (t0, t)
            t_eval = None
        elif isinstance(t, np.ndarray):
            t_span = (t0, t.max())
            t_eval = t
        else:
            raise InputError("Unknown dtype for time. Expected Number or numpy.array")
        
        # Evaluate stoichiometry_matrix at initial conditions and assume constant
        stoichiometry_matrix = self.state_change_matrix(self.initial_state, self.initial_time)

        def _run_single_iteration(iter_ss):
            solver_ss, params_ss = iter_ss.spawn(2)
            solver_rng = np.random.default_rng(solver_ss)
            stochastic_params = self._parameter_store.stochastic_parameters
            param_seqs = params_ss.spawn(len(stochastic_params))

            for seq, param in zip(param_seqs, stochastic_params.values()):
                if isinstance(param.value, CallableParameter):
                    param.value.rng = np.random.default_rng(seq)

            self._parameter_store.new_realisation()

            return solve_stochastic(
                    event_rates=self.event_rate_vector.T,
                    stoichiometry_matrix=stoichiometry_matrix,
                    t_span=t_span,
                    y0=self.initial_state,
                    y_min=y_min,
                    y_max=y_max,
                    method=method,
                    t_eval=t_eval,
                    proceed_if_rates_zero=proceed_if_rates_zero,
                    perf=perf,
                    rng=solver_rng,
                    **options)
        
        master_ss = np.random.SeedSequence(seed)
        iteration_seqs = master_ss.spawn(iteration)


        if not parallel:
            logging.debug("Performing serial simulation")

            out = []
            for i in range(iteration):
                # model = self.copy()
                result = _run_single_iteration(iteration_seqs[i])
                out.append(result)

            return out

        # ----------------------------------------------------
        # Parallel execution
        # ----------------------------------------------------
        
        if parallel_backend == "dask":
            logging.debug(f"Using {parallel_backend} for parallel simulation")

            if client is None:
                from dask.distributed import Client
                client = Client()
            
            futures = [
                client.submit(_run_single_iteration, iteration_seqs[i], self.copy())
                for i in range(iteration)
            ]
            return client.gather(futures)

        raise ValueError(f"Unknown backend: {parallel_backend}")


    ###########################################################################
    #
    # Unrolling of ode to transitions
    #
    # TODO: I doubt any of this works with the event based framework
    #       but it didn't work perfectly anyway. Will be a challenge
    #       now we are dealing with more general systems.
    ###########################################################################

    def get_unrolled_obj(self):
        '''
        Returns a :class:`SimulateOde` with the same state and parameters
        as the current object but with the equations defined by a set of
        transitions and birth death process instead of say, odes
        '''
        transition = self.get_transitions_from_ode()
        bdList = self.get_bd_from_ode()

        return SimulateOde(
                           [str(s) for s in self._stateList],
                           [str(p) for p in self._paramList],
                           derived_param=self._derivedParamEqn,
                           transition=transition,
                           birth_death=bdList
                           )

    def get_transitions_from_ode(self):
        '''
        Returns a list of :class:`Transition` from this object by unrolling
        the odes.  All the elements are of TransitionType.T
        '''
        M = self._generateTransitionMatrix()

        transition = list()
        for i, s1 in enumerate(self._stateList):
            for j, s2 in enumerate(self._stateList):
                if M[i,j] != 0:
                    t = Transition(origin=str(s1),
                                   destination=str(s2),
                                   equation=str(M[i,j]),
                                   transition_type=TransitionType.T)
                    transition.append(t)

        return transition

    def _get_A(self, A=None):
        if A is None:
            if not ode_utils.none_or_empty_list(self._odeList):
                eqn_list = [t.equation for t in self._odeList]
                A = sympy.Matrix(checkEquation(eqn_list,
                                               *self._getListOfVariablesDict(),
                                               subs_derived=False))
                return A
            else:
                raise Exception("Object was not initialized using a set of ode")
        else:
            return A

    def get_bd_from_ode(self, A=None):
        '''
        Returns a list of:class:`Transition` from this object by unrolling
        the odes.  All the elements are of TransitionType.B or
        TransitionType.D
        '''

        A=self._get_A(A)

        bdList, _term = _ode_composition.getUnmatchedExpressionVector(A, True)
        if len(bdList) > 0:
            M = self._generateTransitionMatrix(A)

            A1 = _ode_composition.pureTransitionToOde(M)
            diffA = sympy.simplify(A - A1)

            # get our birth and death process
            bdUnroll = list()
            states = [str(i) for i in self.state_list]

            for i, a in enumerate(diffA):
                for b in bdList:
                    if _ode_composition._hasExpression(a, b):
                        if sympy.Integer(-1) in _ode_composition.getLeafs(b):
                            bdUnroll.append(Transition(origin=states[i],
                                            equation=str(b*-1),
                                            transition_type=TransitionType.D))
                        else:
                            bdUnroll.append(Transition(origin=states[i],
                                            equation=str(b),
                                            transition_type=TransitionType.B))
                        a -= b

            return bdUnroll
        else:
            return []

    def _generateTransitionMatrix(self, A=None):#, transitionExpressionList=None):
        '''
        Finds the transition matrix from the set of ode.  It is
        important to note that although some of the functions used
        in this method appear to be the same as _getReactantMatrix
        and _getStateChangeMatrix, they are different in the sense
        that the functions called here is focused on the terms of
        the equation rather than the states.
        '''
        A=self._get_A(A)
        bdList, _term = _ode_composition.getUnmatchedExpressionVector(A, True)
        fx = _ode_composition.stripBDFromOde(A, bdList)
        states = [s for s in self._iterStateList()]
        M, _remain = _ode_composition.odeToPureTransition(fx, states, True)
        return M


    def plot(self, sim_X=None, sim_T=None):
        '''
        Plot the results of a simulation

        Takes the output of a function like `solve_stochast`

        Parameters
        ----------
        sim_X: list
            of length iteration each with (len(t),len(state)) if t is a vector,
            else it outputs unequal shape that was record of all the jumps
        sim_T: list or :class:`numpy.ndarray`
            if t is a single value, it outputs unequal shape that was
            record of all the jumps.  if t is a vector, it outputs t so that
            it is a :class:`numpy.ndarray` instead

        Notes
        -----
        If either sim_X or sim_T are None the this function will attempt to
        plot the deterministic ODE

        If we have 3 states or more, it will always be arrange such
        that it has 3 columns.  Uses the operation from
        :mod:`odeutils`
        '''
        if (sim_X is None) or (sim_T is None):
            return super(SimulateOde, self).plot()
        ode_utils.plot_stoc(sim_X, sim_T, self)
