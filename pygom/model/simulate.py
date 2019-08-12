"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    Module/class that carries out different type of simulation
    on an ode formulation

"""

__all__ = ['SimulateOde']

import copy
from numbers import Number

import numpy as np
import sympy
import scipy.stats

from .deterministic import DeterministicOde
from .stochastic_simulation import cle, exact, firstReaction, tauLeap, hybrid
from .transition import TransitionType, Transition
from ._model_errors import InputError, SimulationError
from ._model_verification import checkEquation, simplifyEquation
from . import _ode_composition
from . import ode_utils


class HasNewTransition(ode_utils.CompileCanary):
    states = ['ode',
              'Jacobian',
              'diffJacobian',
              'grad',
              'GradJacobian',
              'transitionMatrixCompile',
              'transitionVector',
              'birthDeathRateCompile',
              'computeTransitionMeanVar',
              'transitionJacobian']

class SimulateOde(DeterministicOde):
    '''
    This builds on top of :class:`DeterministicOde` which we
    simulate the outcome instead of solving it deterministically

    Parameters
    ----------
    state: list
        A list of states (string)
    param: list
        A list of the parameters (string)
    derived_param: list
        A list of the derived parameters (tuple of (string,string))
    transition: list
        A list of transition (:class:`Transition`)
    birth_death: list
        A list of birth or death process (:class:`Transition`)
    ode: list
        A list of ode (:class:`Transition`)

    '''

    def __init__(self,
                 state=None,
                 param=None,
                 derived_param=None,
                 transition=None,
                 birth_death=None,
                 ode=None):
        '''
        Constructor that is built on top of DeterministicOde
        '''

        super(SimulateOde, self).__init__(state,
                                          param,
                                          derived_param,
                                          transition,
                                          birth_death,
                                          ode)

        self._hasNewTransition = HasNewTransition()

        # need a manual override because it is possible that we
        # want to perform simulation in a parallel/distributed manner
        # and there are issues with pickling fortran objects
        self._SC = ode_utils.compileCode(backend='cython')

        self._birthDeathRate = None
        self._birthDeathRateCompile = None

        # information required for the tau leap
        self._transitionJacobian = None

        self._transitionMean = None
        self._transitionMeanCompile = None

        self._transitionVar = None
        self._transitionVarCompile = None

        self._transitionVectorCompile = None
        self._transitionMatrixCompile = None

        # micro times for jumps
        self._tau = None
        self._tauDict = None

    def __repr__(self):
        return "SimulateOde" + self._get_model_str()

    def exact(self, x0, t0, t1, output_time=False):
        '''
        Stochastic simulation using an exact method starting from time
        t0 to t1 with the starting state values of x0

        Parameters
        ----------
        x: array like
            state vector
        t0: double
            start time
        t1: double
            final time
        '''
        return(exact(x0, t0, t1, self._vMat, self.transition_vector,
                     output_time=output_time))

    def cle(self, x0, t0, t1, output_time=False):
        '''
        Stochastic simulation using the CLE approximation starting from time
        t0 to t1 with the starting state values of x0.  The CLE approximation
        is performed using a simple Euler-Maruyama method with step size h.
        We assume that the input parameter transition_func provides
        :math:`f(x,t)` while the CLE is defined as
        :math:`dx = x + V*h*f(x,t) + \\sqrt(f(x,t))*Z*\\sqrt(h)`
        with :math:`Z` being standard normal random variables.

        Parameters
        ----------
        x: array like
            state vector
        t0: double
            start time
        t1: double
            final time
        '''
        return(cle(x0, t0, t1, self._vMat, self.transition_vector,
                   output_time=output_time))

    def hybrid(self, x0, t0, t1, output_time=False):
        '''
        Stochastic simulation using an hybrid method that uses either the
        first reaction method or the :math:`\\tau`-leap depending on the
        size of the states and transition rates.  Starting from time
        t0 to t1 with the starting state values of x0.

        Parameters
        ----------
        x: array like
            state vector
        t0: double
            start time
        t1: double
            final time
        '''
        return(hybrid(x0, t0, t1, self._vMat, self._lambdaMat,
                      self.transition_vector,
                      self.transition_mean,
                      self.transition_var,
                      output_time=output_time))

    def simulate_param(self, t, iteration, parallel=True, full_output=False):
        '''
        Simulate the ode by generating new realization of the stochastic
        parameters and integrate the system deterministically.

        Parameters
        ----------
        t: array like
            the range of time points which we want to see the result of
        iteration: int
            number of iterations you wish to simulate
        parallel: bool, optional
            Defaults to True
        full_output: bool, optional
            if we want additional information, Y_all in the return,
            defaults to false

        Returns
        -------
        Y: :class:`numpy.ndarray`
            of shape (len(t), len(state)), mean of all the simulation
        Y_all: :class:`np.ndarray`
            of shape (iteration, len(t), len(state))
        '''

        # if our parameters not stochastic, then we are going to
        # throw a warning because simulating a deterministic system is
        # just plain stupid
        if self._stochasticParam is None:
            raise InputError("Deterministic parameters.")
        if iteration is None:
            raise InputError("Need to specify the number of iterations")
        if t is None:
            raise InputError("Need to specify the time we wish to observe")

        self._odeSolution = self.integrate(t)

        # try to compute the simulation in parallel
        if parallel:
            try:
                for i in self._stochasticParam:
                    if isinstance(i, scipy.stats._distn_infrastructure.rv_frozen):
                        raise Exception("Cannot perform parallel simulation "
                                        +"using a serialized object as distribution")
                # check the type of parameter we have as input
                import dask.bag
                y = list()
                for i in range(iteration):
                    y_i = list()
                    for key, rv in self._stochasticParam.items():
                        y_i += [{key:rv.rvs(1)[0]}]
                    y += [y_i]
                # y = [rv.rvs(iteration) for rv in self._stochasticParam.values()]
                # y = np.array(list(zip(*y)))
                def sim(x):
                    self.parameters = x
                    return self.integrate(t)

                # def sim(t1): return(self.integrate(t1))

                # xtmp = dask.bag.from_sequence([t]*iteration)
                xtmp = dask.bag.from_sequence(y)
                solutionList = xtmp.map(sim).compute()
            except Exception: # as e:
                # print(e)
                # print("Serial")
                solutionList = [self.integrate(t) for i in range(iteration)]
        else:
            solutionList = [self.integrate(t) for i in range(iteration)]

        # now make our 3D array
        # the first dimension is the number of iteration
        Y = np.dstack(solutionList).mean(axis=2)

        if full_output:
            return Y, solutionList
        else:
            return Y

    def simulate_jump(self, t, iteration, parallel=True,
                      exact=False, full_output=False):
        '''
        Simulate the ode using stochastic simulation.  It switches
        between a first reaction method and a :math:`\\tau`-leap
        algorithm internally. When a parallel backend exists, then a new random
        state (seed) will be used for each processor.  This is due to a lack
        of appropriate parallel seed random number generator in python.

        Parameters
        ----------
        t: array like
            the range of time points which we want to see the result of
            or the final time point
        iteration: int
            number of iterations you wish to simulate
        parallel: bool, optional
            Defaults to True
        exact: bool, optional
            True if exact simulation is desired, defaults to False
        full_output: bool, optional
            if we want additional information, sim_T

        Returns
        -------
        sim_X: list
            of length iteration each with (len(t),len(state)) if t is a vector,
            else it outputs unequal shape that was record of all the jumps
        sim_T: list or :class:`numpy.ndarray`
            if t is a single value, it outputs unequal shape that was
            record of all the jumps.  if t is a vector, it outputs t so that
            it is a :class:`numpy.ndarray` instead

        '''

        assert len(self._odeList) == 0, \
            "Currently only able to simulate when only transitions are present"
        assert np.all(np.mod(self._x0, 1) == 0), \
            "Can only simulate a jump process with integer initial values"

        # this determines what type of output we want
        timePoint = False

        if isinstance(t, Number):#, (int, float, np.int64, np.float64)):
            finalT = t
        elif isinstance(t, (list, tuple)):
            t = np.array(t)
            if len(t) == 1:
                finalT = t
            else:
                finalT = t[-1:]
                timePoint = True
        elif isinstance(t, np.ndarray):
            finalT = t[-1:]
            timePoint = True
        else:
            raise InputError("Unknown data type for time")

        if self._transitionVectorCompile is None:
            self._compileTransitionVector()

        if parallel:
            try:
                import dask.bag
                print("Parallel simulation")
                def jump_partial(final_t): return(self._jump(final_t,
                                                             exact=exact,
                                                             full_output=True,
                                                             seed=True))

                xtmp = dask.bag.from_sequence(np.ones(iteration)*finalT)
                xtmp = xtmp.map(jump_partial).compute()
            except Exception as e:
                # print(e)
                print("Revert to serial")
                xtmp = [self._jump(finalT, exact=exact, full_output=True) for _i in range(iteration)]
        else:
            print("Serial computation")
            xtmp = [self._jump(finalT, exact=exact, full_output=True) for _i in range(iteration)]

        xmat = list(zip(*xtmp))
        simXList, simTList = list(xmat[0]), list(xmat[1])
        ## print("Finish computation")
        # now we want to fix our simulation, if they need fixing that is
        # print timePoint
        if timePoint:
            for _i in range(len(simXList)):
                # unroll, always the first element
                # it is easy to remember that we are accessing the first
                # element because pop is spelt similar to poop and we
                # all know that your execute first in first out when you
                # poop!
                simX = simXList.pop(0)
                simT = simTList.pop(0)

                x = self._extractObservationAtTime(simX, simT, t)
                simTList.append(t)
                simXList.append(x)
        # note that we have to remain in list form because the number of
        # simulation will be different if we are not dealing with
        # a specific set of time points

        if full_output:
            if timePoint:
                return simXList, t
            else:
                return simXList, simTList
        else:
            return simXList

    def _jump(self, finalT, exact=False, full_output=True, seed=None):
        '''
        Jumps from the initial time self._t0 to the input time finalT
        '''
        # initial time
        assert self._t0 is not None, "No initial time"
        assert self._x0 is not None, "No initial state"

        t = self._t0.tolist()
        x = copy.deepcopy(self._x0)

        # holders and record information
        xList = [x.copy()]
        tList = [t]

        # we want to construct some jump times
        if self._GMat is None:
            self._computeDependencyMatrix()

        # keep jumping, Whoop Whoop (put your hands up!).
        f = firstReaction
        while t < finalT:
            # print(t)
            try:
                if exact:
                    x, t, success = f(x, t, self._vMat,
                                      self.transition_vector, seed=seed)
                else:
                    if np.min(x) > 10:
                        x_tmp, t_tmp, success = tauLeap(x, t,
                                                self._vMat, self._lambdaMat,
                                                self.transition_vector,
                                                self.transition_mean,
                                                self.transition_var,
                                                seed=seed)
                        if success:
                            x, t = x_tmp, t_tmp
                        else:
                            x, t, success = f(x, t, self._vMat,
                                              self.transition_vector, seed=seed)
                    else:
                        x, t, success = f(x, t, self._vMat,
                                          self.transition_vector, seed=seed)
                if success:
                    xList.append(x.copy())
                    tList.append(t)
                else:
                    break
                    ## print("x: %s, t: %s" % (x, t))
                    ## raise Exception('WTF')
            except SimulationError:
                break

        return np.array(xList), np.array(tList)

    def _extractObservationAtTime(self, X, t, targetTime):
        '''
        Given simulation and a set of time points which we would like to
        observe, we extract the observations :math:`x_{t}` with
        :math:`\\min\\{ \\abs( t - targetTime) \\}`
        '''
        y = list()
        # maxTime = max(t)
        index = 0
        for i in targetTime:
            if np.any(t == i):
                index = np.where(t == i)[0][0]
            else:
                index = np.searchsorted(t, i) - 1
            y.append(X[index])

        return np.array(y)

    ########################################################################
    #
    # Operators with regard to the transition matrix
    #
    ########################################################################

    def get_transition_matrix(self):
        '''
        Returns the transition matrix in algebraic form.

        Returns
        -------
        :class:`sympy.matrices.matrices`
            A matrix of dimension [number of state x number of state]

        '''
        if self._transitionMatrix is None:
            super(SimulateOde, self)._computeTransitionMatrix()

        if self._transitionMatrixCompile is not None \
           or not self._hasNewTransition.transitionMatrixCompile:
            return self._transitionMatrix
        else:
            return super(SimulateOde, self)._computeTransitionMatrix()

    def get_transition_vector(self):
        '''
        Returns the set of transitions in a single vector, transitions
        between state to state first then the birth and death process

        Returns
        -------
        :class:`sympy.matrices.matrices`
            A matrix of dimension [total number of transitions x 1]

        '''
        if self._transitionVectorCompile is not None \
           or not self._hasNewTransition.transitionVector:
            return self._transitionVector
        else:
            return super(SimulateOde, self)._computeTransitionVector()

    def transition_matrix(self, state, t):
        '''
        Evaluate the transition matrix given state and time

        Parameters
        ----------
        state: array like
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`
        t: double
            The current time

        Returns
        -------
        :class:`numpy.ndarray`
            a 2d array of size (M,M) where M is the number
            of transitions

        '''
        return self.eval_transition_matrix(time=t, state=state)

    def eval_transition_matrix(self, parameters=None, time=None, state=None):
        '''
        Evaluate the transition matrix given parameters, state and time. Note
        that the output is not in sparse format

        Parameters
        ----------
        parameters: list
            see :meth:`.setParameters`
        time: double
            The current time
        state: array list
            The current numerical value for the states which can
            :class:`numpy.ndarray` or :class:`list`

        Returns
        -------
        :class:`numpy.matrix` or :class:`mpmath.matrix`
            Matrix of dimension [number of state x number of state]

        '''
        if self._transitionMatrixCompile is None \
            or self._hasNewTransition.transitionMatrixCompile:
            self._compileTransitionMatrix()

        eval_param = self._getEvalParam(state, time, parameters)
        return self._transitionMatrixCompile(eval_param)

    def _compileTransitionMatrix(self):
        '''
        We would also need to compile the function so that
        it can be evaluated faster.
        '''
        if self._transitionMatrix is None \
            or self._hasNewTransition.transitionMatrixCompile:
            super(SimulateOde, self)._computeTransitionMatrix()

        f = self._SC.compileExprAndFormat
        if self._isDifficult:
            self._transitionMatrixCompile = f(self._sp,
                                              self._transitionMatrix,
                                              modules='mpmath')
        else:
            self._transitionMatrixCompile = f(self._sp,
                                              self._transitionMatrix)

        self._hasNewTransition.reset('transitionMatrixCompile')

        return None

    def transition_vector(self, state, t):
        '''
        Evaluate the transition vector given state and time

        Parameters
        ----------
        state: array like
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`
        t: double
            The current time

        Returns
        -------
        :class:`numpy.ndarray`
            a 1d array of size K where K is the number of between
            states transitions and the number of birth death
            processes
        '''
        return self.eval_transition_vector(time=t, state=state)

    def eval_transition_vector(self, parameters=None, time=None, state=None):
        '''
        Evaluate the transition vector given parameters, state and time. Note
        that the output is not in sparse format

        Parameters
        ----------
        parameters: list
            see :meth:`.setParameters`
        time: double
            The current time
        state: array list
            The current numerical value for the states which can
            :class:`numpy.ndarray` or :class:`list`

        Returns
        -------
        :class:`numpy.matrix` or :class:`mpmath.matrix`
            vector of dimension [total number of transitions]

        '''
        if self._transitionVectorCompile is None \
           or self._hasNewTransition.transitionVector:
            self._compileTransitionVector()

        eval_param = self._getEvalParam(state, time, parameters)
        return self._transitionVectorCompile(eval_param)

    def _compileTransitionVector(self):
        '''
        We would also need to compile the function so that
        it can be evaluated faster.
        '''
        if self._transitionVector is None \
            or self._hasNewTransition.transitionVector:
            super(SimulateOde, self)._computeTransitionVector()

        f = self._SC.compileExprAndFormat
        if self._isDifficult:
            self._transitionVectorCompile = f(self._sp,
                                              self._transitionVector,
                                              modules='mpmath')
        else:
            self._transitionVectorCompile = f(self._sp,
                                              self._transitionVector)

        self._hasNewTransition.reset('transitionVector')

        return

    def get_birth_death_rate(self):
        '''
        Find the algebraic equations of birth and death processes

        Returns
        -------
        :class:`sympy.matrices.matrices`
            birth death process in matrix form
        '''
        if self._birthDeathRate is None or self._hasNewTransition:
            self._computeBirthDeathRate()

        return self._birthDeathRate

    def birth_death_rate(self, state, t):
        '''
        Evaluate the birth death rates given state and time

        Parameters
        ----------
        state: array like
            The current numerical value for the states which can be
            :class:`np.ndarray` or :class:`list`
        t: double
            The current time

        Returns
        -------
        :class:`numpy.ndarray`
            an array of size (M,M) where M is the number
            of birth and death actions

        '''
        return self.eval_birth_death_rate(time=t, state=state)

    def eval_birth_death_rate(self, parameters=None, time=None, state=None):
        '''
        Evaluate the birth and death rates given parameters, state and time.

        Parameters
        ----------
        parameters: list
            see :meth:`.setParameters`
        time: double
            The current time
        state: array list
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`

        Returns
        -------
        :class:`numpy.matrix` or :class:`mpmath.matrix`
            Matrix of dimension [number of birth and death rates x 1]

        '''
        if self._birthDeathRateCompile is None \
            or self._hasNewTransition.birthDeathRateCompile:
            self._computeBirthDeathRate()

        eval_param = self._getEvalParam(state, time, parameters)
        return self._birthDeathRateCompile(eval_param)

    def _computeBirthDeathRate(self):
        '''
        Note that this is different to _birthDeathVector because
        this is of length (number of birth and death process) while
        _birthDeathVector in baseOdeModel has the same length as
        the number of states
        '''
        if self.num_birth_deaths == 0:
            A = sympy.zeros(1, 1)
        else:
            A = sympy.zeros(self.num_birth_deaths, 1)

            # go through all the transition objects
            for i, bd in enumerate(self.birth_death_list):
                A[i] += eval(self._checkEquation(bd.equation()))

        # assign back
        self._birthDeathRate = A
        # compilation of the symbolic calculation.  Note here that we are
        # going to recompile the total transitions because it might
        # have changed
        f = self._SC.compileExprAndFormat
        if self._isDifficult:
            self._birthDeathRateCompile = f(self._sp,
                                            self._birthDeathRate,
                                            modules='mpmath')
        else:
            self._birthDeathRateCompile = f(self._sp,
                                            self._birthDeathRate)

        self._hasNewTransition.reset('birthDeathRateCompile')

        return None

    def total_transition(self, state, t):
        '''
        Evaluate the total transition rate given state and time

        Parameters
        ----------
        state: array like
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`
        t: double
            The current time

        Returns
        -------
        float
            total rate

        '''
        return sum(self.transition_vector(time=t, state=state))

    def transition_mean(self, state, t):
        '''
        Evaluate the mean of the transitions given state and time.  For
        m transitions and n states, we have

        .. math::
            f_{j,k} &= \\sum_{i=1}^{n} \\frac{\\partial a_{j}(x)}{\\partial x_{i}} v_{i,k} \\\\
            \\mu_{j} &= \\sum_{k=1}^{m} f_{j,k}(x)a_{k}(x) \\\\
            \\sigma^{2}_{j}(x) &= \\sum_{k=1}^{m} f_{j,k}^{2}(x) a_{k}(x)

        where :math:`v_{i,k}` is the state change matrix.

        Parameters
        ----------
        state: array like
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`
        t: double
            The current time

        Returns
        -------
        :class:`numpy.ndarray`
            an array of size m where m is the number of transition

        '''
        return self.eval_transition_mean(time=t, state=state)

    def eval_transition_mean(self, parameters=None, time=None, state=None):
        '''
        Evaluate the transition mean given parameters, state and time.

        Parameters
        ----------
        parameters: list
            see :meth:`.setParameters`
        time: double
            The current time
        state: array list
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`

        Returns
        -------
        :class:`numpy.matrix` or :class:`mpmath.matrix`
            Matrix of dimension [number of state x number of state]

        '''
        if self._transitionMeanCompile is None \
            or self._hasNewTransition.computeTransitionMeanVar:
            self._computeTransitionMeanVar()

        eval_param = self._getEvalParam(state, time, parameters)
        return self._transitionMeanCompile(eval_param)

    def transition_var(self, state, t):
        '''
        Evaluate the variance of the transitions given state and time

        Parameters
        ----------
        state: array like
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`
        t: double
            The current time

        Returns
        -------
        :class:`numpy.ndarray`
            an array of size M where M is the number of transition

        '''
        return self.eval_transition_var(time=t, state=state)

    def eval_transition_var(self, parameters=None, time=None, state=None):
        '''
        Evaluate the transition variance given parameters, state and time.

        Parameters
        ----------
        parameters: list
            see :meth:`.setParameters`
        time: double
            The current time
        state: array list
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`

        Returns
        -------
        :class:`numpy.matrix` or :class:`mpmath.matrix`
            Matrix of dimension [number of state x number of state]

        '''
        if self._transitionVarCompile is None \
            or self._hasNewTransition.computeTransitionMeanVar:
            self._computeTransitionMeanVar()

        eval_param = self._getEvalParam(state, time, parameters)
        return self._transitionVarCompile(eval_param)

    def _computeTransitionJacobian(self):
        if self._GMat is None:
            self._computeDependencyMatrix()

        F = sympy.zeros(self.num_transitions, self.num_transitions)
        for i in range(self.num_transitions):
            for j, eqn in enumerate(self._transitionVector):
                for k, state in enumerate(self._iterStateList()):
                    diffEqn = sympy.diff(eqn, state, 1)
                    tempEqn, isDifficult = simplifyEquation(diffEqn)
                    F[i,j] += tempEqn*self._vMat[k,i]
                    self._isDifficult = self._isDifficult or isDifficult

        self._transitionJacobian = F

        self._hasNewTransition.reset('transitionJacobian')
        return F

    def _computeTransitionMeanVar(self):
        '''
        This is the mean and variance information that we need
        for the :math:`\\tau`-Leap
        '''

        if self._transitionJacobian is None or self._hasNewTransition.transitionJacobian:
            self._computeTransitionJacobian()

        F = self._transitionJacobian
        # holders
        mu = sympy.zeros(self.num_transitions, 1)
        sigma2 = sympy.zeros(self.num_transitions, 1)
        # we calculate the mean and variance
        for i in range(self.num_transitions):
            for j, eqn in enumerate(self._transitionVector):
                mu[i] += F[i,j] * eqn
                sigma2[i] += F[i,j] * F[i,j] * eqn

        self._transitionMean = mu
        self._transitionVar = sigma2

        # now we are going to compile them
        f = self._SC.compileExprAndFormat
        if self._isDifficult:
            self._transitionMeanCompile = f(self._sp,
                                            self._transitionMean,
                                            modules='mpmath')
            self._transitionVarCompile = f(self._sp,
                                           self._transitionVar,
                                           modules='mpmath')
        else:
            self._transitionMeanCompile = f(self._sp, self._transitionMean)
            self._transitionVarCompile = f(self._sp, self._transitionVar)

        self._hasNewTransition.reset('computeTransitionMeanVar')

        return None

    ########################################################################
    #
    # Unrolling of ode to transitions
    #
    ########################################################################

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

    def get_bd_from_ode(self, A=None):
        '''
        Returns a list of:class:`Transition` from this object by unrolling
        the odes.  All the elements are of TransitionType.B or
        TransitionType.D
        '''
        if A is None:
            if not ode_utils.none_or_empty_list(self._odeList):
                eqn = [t.equation for t in self._odeList]
                A = sympy.Matrix(checkEquation(eqn,
                                               *self._getListOfVariablesDict(),
                                               subs_derived=False))
            else:
                raise Exception("Object was not initialized using a set of ode")
            # A = super(SimulateOde, self).getOde()

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
        if A is None:
            if not ode_utils.none_or_empty_list(self._odeList):
                eqn_list = [t.equation for t in self._odeList]
                A = sympy.Matrix(checkEquation(eqn_list,
                                               *self._getListOfVariablesDict(),
                                               subs_derived=False))
            else:
                raise Exception("Object was not initialized using a set of ode")

        bdList, _term = _ode_composition.getUnmatchedExpressionVector(A, True)
        fx = _ode_composition.stripBDFromOde(A, bdList)
        states = [s for s in self._iterStateList()]
        M, _remain = _ode_composition.odeToPureTransition(fx, states, True)
        return M

    def plot(self, sim_X=None, sim_T=None):
        '''
        Plot the results of a simulation

        Takes the output of a function like `simulate_jump`

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
